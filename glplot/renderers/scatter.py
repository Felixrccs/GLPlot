from __future__ import annotations
import ctypes as C
import numpy as np
from OpenGL.GL import *
from typing import TYPE_CHECKING, Optional, Tuple

from .base import GLScatterBuffers
from ..utils.shaders import SCATTER_VS, SCATTER_FS, DENSITY_POINTS_VS, DENSITY_POINTS_FS
from ..utils.gl_utils import link_program

if TYPE_CHECKING:
    from ..core.layers import ScatterLayer
    from ..core.context import RenderContext
    from ..options import EngineOptions

class ScatterRenderer:
    """
    Primitive renderer for ScatterLayer.
    Specialized for point clouds (GL_POINTS).
    """
    def __init__(self, options: EngineOptions):
        self.options = options
        self.prog = 0
        
        # Uniform locations
        self.u_mvp = -1
        self.u_size = -1
        self.u_alpha = -1
        self.u_offset = -1
        
        # Accumulation uniforms
        self.accum_prog = 0
        self.u_accum_mvp = -1
        self.u_accum_size = -1
        self.u_accum_alpha = -1
        self.u_accum_offset = -1

    def initialize(self) -> None:
        """Link shaders and setup uniform locations."""
        self.prog = link_program(SCATTER_VS, SCATTER_FS)
        self.u_mvp = glGetUniformLocation(self.prog, "u_mvp")
        self.u_size = glGetUniformLocation(self.prog, "u_size")
        self.u_alpha = glGetUniformLocation(self.prog, "u_alpha")
        self.u_offset = glGetUniformLocation(self.prog, "u_layer_offset")
        self.u_point_size_px = glGetUniformLocation(self.prog, "u_point_size_px")
        self.u_outline_enabled = glGetUniformLocation(self.prog, "u_outline_enabled")
        self.u_outline_color = glGetUniformLocation(self.prog, "u_outline_color")
        self.u_outline_width_px = glGetUniformLocation(self.prog, "u_outline_width_px")

        # Density Accumulation Program
        self.accum_prog = link_program(DENSITY_POINTS_VS, DENSITY_POINTS_FS)
        self.u_accum_mvp = glGetUniformLocation(self.accum_prog, "u_mvp")
        self.u_accum_size = glGetUniformLocation(self.accum_prog, "u_size")
        self.u_accum_alpha = glGetUniformLocation(self.accum_prog, "u_alpha")
        self.u_accum_offset = glGetUniformLocation(self.accum_prog, "u_layer_offset")

    def _create_buffers(self, layer: ScatterLayer) -> GLScatterBuffers:
        """Create and initialize GPU buffers for a scatter layer."""
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        
        # 1. Point Positions
        vbo_pts = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pts)
        glBufferData(GL_ARRAY_BUFFER, 16, None, GL_STATIC_DRAW) # Pre-allocate to avoid segfault
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
        
        # 2. Point Colors
        vbo_col = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_col)
        glBufferData(GL_ARRAY_BUFFER, 16, None, GL_STATIC_DRAW) # Pre-allocate to avoid segfault
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
        
        glBindVertexArray(0)
        return GLScatterBuffers(vao=vao, vbo_pts=vbo_pts, vbo_col=vbo_col)

    def update_gpu_data(self, layer: ScatterLayer, bufs: GLScatterBuffers) -> None:
        """Upload semantic points and colors to GPU buffers."""
        if layer.pts is None or len(layer.pts) == 0: return
        
        # Upload Positions
        glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_pts)
        glBufferData(GL_ARRAY_BUFFER, layer.pts.nbytes, layer.pts, GL_STATIC_DRAW)
        
        # Upload Colors
        if layer.colors is not None:
            # If colors are provided as a single color, broadcast it
            if layer.colors.ndim == 1 and len(layer.colors) == 4:
                cols = np.tile(layer.colors, (len(layer.pts), 1)).astype(np.float32)
            else:
                cols = layer.colors.astype(np.float32)
                
            glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_col)
            glBufferData(GL_ARRAY_BUFFER, cols.nbytes, cols, GL_STATIC_DRAW)
        
        bufs.count = len(layer.pts)
        layer.dirty.gpu_dirty = False

    def draw(self, layer: ScatterLayer, ctx: RenderContext) -> None:
        """Draw the scatter layer using current context."""
        if layer.pts is None or len(layer.pts) == 0: return

        # 1. Resource Management
        if not hasattr(layer, "_gl") or layer._gl is None:
            layer._gl = self._create_buffers(layer)
            layer.dirty.gpu_dirty = True
        
        if layer.dirty.gpu_dirty:
            self.update_gpu_data(layer, layer._gl)

        # 2. Setup OpenGL State & Shaders
        glUseProgram(self.prog)
        glEnable(GL_PROGRAM_POINT_SIZE)
        
        glUniformMatrix4fv(self.u_mvp, 1, GL_TRUE, ctx.mvp)
        
        # Style Resolution: Base * Multipliers
        overrides = self.options.visual.overrides
        effective_size = float(layer.style.point_size) * ctx.dpr * overrides.point_size_multiplier
        effective_alpha = ctx.global_alpha * layer.style.alpha * overrides.alpha_multiplier
        
        glUniform1f(self.u_size, effective_size)
        glUniform1f(self.u_alpha, float(effective_alpha))
        glUniform1f(self.u_point_size_px, effective_size)
        
        # Outline logic
        glUniform1i(self.u_outline_enabled, 1 if layer.style.point_outline_enabled else 0)
        if layer.style.point_outline_enabled:
            glUniform4f(self.u_outline_color, *layer.style.point_outline_color)
            glUniform1f(self.u_outline_width_px, float(layer.style.point_outline_width) * ctx.dpr)
        
        glUniform2f(self.u_offset, *layer.translation)

        # 3. Draw call
        glBindVertexArray(layer._gl.vao)
        glDrawArrays(GL_POINTS, 0, layer._gl.count)
        
        # Cleanup
        glBindVertexArray(0)
        glUseProgram(0)
        glDisable(GL_PROGRAM_POINT_SIZE)

    def draw_density(self, layer: ScatterLayer, ctx: RenderContext) -> None:
        """Accumulate point density into the current R32F target."""
        if layer.pts is None or len(layer.pts) == 0: return

        # 1. Resource Management
        if not hasattr(layer, "_gl") or layer._gl is None:
            layer._gl = self._create_buffers(layer)
            layer.dirty.gpu_dirty = True
        
        if layer.dirty.gpu_dirty:
            self.update_gpu_data(layer, layer._gl)

        # 2. Setup Shaders
        glUseProgram(self.accum_prog)
        glEnable(GL_PROGRAM_POINT_SIZE)
        
        glUniformMatrix4fv(self.u_accum_mvp, 1, GL_TRUE, ctx.mvp)
        
        overrides = self.options.visual.overrides
        # Use a slightly smaller size for density to prevent over-blurring
        # unless user explicitly requested massive points.
        effective_size = max(1.0, float(layer.style.point_size) * ctx.dpr * 0.5 * overrides.point_size_multiplier)
        glUniform1f(self.u_accum_size, effective_size)

        # Weighted Accumulation
        if self.options.density_weighted:
            alpha = ctx.global_alpha * layer.style.alpha * overrides.alpha_multiplier
            glUniform1f(self.u_accum_alpha, float(alpha))
        else:
            glUniform1f(self.u_accum_alpha, 1.0) # Simple counting mode
        
        glUniform2f(self.u_accum_offset, *layer.translation)

        # 3. Draw call
        glBindVertexArray(layer._gl.vao)
        glDrawArrays(GL_POINTS, 0, layer._gl.count)
        
        # Cleanup
        glBindVertexArray(0)
        glUseProgram(0)
        glDisable(GL_PROGRAM_POINT_SIZE)
