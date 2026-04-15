import sys
import ctypes as C
import numpy as np
import glfw
from OpenGL.GL import *
import matplotlib.pyplot as plt

import imgui
from imgui.integrations.glfw import GlfwRenderer

# --------------------------- utils ---------------------------

def ortho(l, r, b, t, n=-1.0, f=1.0):
    rl, tb, fn = (r-l), (t-b), (f-n)
    return np.array([
        [2.0/rl, 0.0,    0.0,    -(r+l)/rl],
        [0.0,    2.0/tb, 0.0,    -(t+b)/tb],
        [0.0,    0.0,   -2.0/fn, -(f+n)/fn],
        [0.0,    0.0,    0.0,     1.0     ],
    ], dtype=np.float32)

def compile_shader(src, stype):
    sid = glCreateShader(stype)
    glShaderSource(sid, src)
    glCompileShader(sid)
    if not glGetShaderiv(sid, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(sid).decode(errors="ignore"))
    return sid

def link_program(vs_src, fs_src):
    vs = compile_shader(vs_src, GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL_FRAGMENT_SHADER)
    pid = glCreateProgram()
    glAttachShader(pid, vs); glAttachShader(pid, fs)
    glLinkProgram(pid)
    glDeleteShader(vs); glDeleteShader(fs)
    if not glGetProgramiv(pid, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(pid).decode(errors="ignore"))
    return pid

# --------------------------- models ---------------------------

class ViewState:
    """Manages camera, selection, and HUD status."""
    def __init__(self):
        self.cx = 0.0
        self.cy = 0.0
        self.zoom = 1.0
        self.zoom_min = 0.02
        self.zoom_max = 500.0
        
        self.drag = False
        self.last_mouse = (0.0, 0.0)
        self.dirty = True
        
        # Selection / Picking
        self.hover_idx = -1         # Index of hovered element
        self.hover_type = None      # 'line', 'scatter', 'strip'
        self.selected_idx = -1      # Persistent selection
        self.selected_type = None
        
        # Image-based Rendering Cache
        self.is_caching = False
        self.cache_timer = 0.0      # Seconds until release
        self.cache_state = None     # (l, r, b, t)
        
        # UI Status
        self.hud_visible = True
        self.hud_verbosity = 1      # 1: Basic, 2: Full
        self.show_density = False

class DataState:
    """Handles raw data buffers and metadata."""
    def __init__(self):
        # Primary lines (a*x + b)
        self.ab = None
        self.line_colors = None
        self.xrange = (-3.0, 3.0)
        
        # Scatters
        self.scatter_pts = None
        self.scatter_cols = None
        self.scatter_size = 5.0
        
        # Metadata / Labels
        self.labels = None
        
        # Spatial indexing (updated on-demand)
        self.grid_res = 100         # resolution for picking grid

# --------------------------- shaders ---------------------------

VS_SRC = r"""
#version 330 core
layout(location=0) in float a_t;     // 0 or 1 (two base vertices)
layout(location=1) in vec2  a_ab;    // (a,b) per instance (raw FP16/FP32)
layout(location=2) in vec4  a_col;   // per-instance color (RGBA8 norm or FP32)

uniform mat4  u_mvp;
uniform vec2  u_xrange;              // [xmin, xmax] for line domain
uniform vec4  u_window;              // (l, r, b, t) world window
uniform int   u_use_color;           // 0: black, 1: a_col
uniform float u_alpha;               // global alpha multiplier
uniform int   u_enable_subsample;    // 0/1
uniform float u_keep_prob;           // in (0,1]
out vec4 v_col;

void main() {
    float x = mix(u_xrange.x, u_xrange.y, a_t);
    float y = a_ab.x * x + a_ab.y;
    vec2  w = vec2(x, y);

    gl_Position = u_mvp * vec4(w, 0.0, 1.0);

    gl_ClipDistance[0] =  w.x - u_window.x;
    gl_ClipDistance[1] =  u_window.y - w.x;
    gl_ClipDistance[2] =  w.y - u_window.z;
    gl_ClipDistance[3] =  u_window.w - w.y;

    float l = u_window.x, r = u_window.y;
    float xmin = u_xrange.x, xmax = u_xrange.y;
    float xA = max(l, xmin);
    float xB = min(r, xmax);
    bool noOverlapX = (xA > xB);
    float yA = a_ab.x * xA + a_ab.y;
    float yB = a_ab.x * xB + a_ab.y;
    float bottom = u_window.z, top = u_window.w;
    bool outsideY = (yA > top && yB > top) || (yA < bottom && yB < bottom);

    // On-GPU probabilistic LOD
    uint id = uint(gl_InstanceID);
    id ^= id >> 17; id *= 0xed5ad4bbu; id ^= id >> 11;
    id *= 0xac4c1b51u; id ^= id >> 15; id *= 0x31848babu;
    float rnd = float(id & 0x00FFFFFFu) * (1.0/16777215.0);
    bool drop = (u_enable_subsample == 1) && (rnd > u_keep_prob);

    if (noOverlapX || outsideY || drop) {
        gl_ClipDistance[0] = -1.0;
        gl_ClipDistance[1] = -1.0;
        gl_ClipDistance[2] = -1.0;
        gl_ClipDistance[3] = -1.0;
    }

    v_col = (u_use_color == 1) ? a_col : vec4(0.0, 0.0, 0.0, 1.0);
    v_col.a *= u_alpha;
}
"""

FS_SRC = r"""
#version 330 core
in vec4 v_col;
out vec4 FragColor;
uniform int u_is_density;
void main(){ 
    if (u_is_density == 1) {
        FragColor = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        FragColor = v_col; 
    }
}
"""

# Simple shader for standard line strips
STRIP_VS_SRC = r"""
#version 330 core
layout(location=0) in vec2 a_pos;
uniform mat4  u_mvp;
uniform vec4  u_color;
uniform float u_alpha;
out vec4 v_col;

void main() {
    gl_Position = u_mvp * vec4(a_pos, 0.0, 1.0);
    v_col = u_color;
    v_col.a *= u_alpha;
}
"""

SCATTER_VS_SRC = r"""
#version 330 core
layout(location=0) in vec2 a_pos;
layout(location=1) in vec4 a_color;

uniform mat4  u_mvp;
uniform float u_size;
uniform float u_alpha;

out vec4 v_col;

void main() {
    gl_Position = u_mvp * vec4(a_pos, 0.0, 1.0);
    gl_PointSize = u_size;
    v_col = a_color;
    v_col.a *= u_alpha;
}
"""

SCATTER_FS_SRC = r"""
#version 330 core
in vec4 v_col;
out vec4 FragColor;
uniform int u_is_density;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (dot(coord, coord) > 0.25) discard;
    if (u_is_density == 1) {
        FragColor = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        FragColor = v_col;
    }
}
"""

PICKING_VS_SRC = r"""
#version 330 core
layout(location=0) in float a_t;
layout(location=1) in vec2 a_ab;
uniform mat4  u_mvp;
uniform vec2  u_xrange;
uniform float u_keep_prob;
uniform int   u_enable_subsample;
flat out int v_id;

uint pcg_hash(uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

void main() {
    float x = (a_t == 0.0) ? u_xrange[0] : u_xrange[1];
    float y = a_ab.x * x + a_ab.y;
    gl_Position = u_mvp * vec4(x, y, 0.0, 1.0);
    
    // ID encoding (obj_id = gl_InstanceID + 1. 0 reserved for background)
    v_id = gl_InstanceID + 1;

    if (u_enable_subsample == 1 && u_keep_prob < 1.0) {
        uint h = pcg_hash(uint(gl_InstanceID));
        float p = float(h & 0xFFFFFFu) / 16777216.0;
        if (p > u_keep_prob) {
            gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        }
    }
}
"""

PICKING_FS_SRC = r"""
#version 330 core
flat in int v_id;
out int FragData;
void main() { FragData = v_id; }
"""

PICKING_SCATTER_VS_SRC = r"""
#version 330 core
layout(location=0) in vec2 a_pos;
uniform mat4  u_mvp;
uniform int   u_id_offset;
flat out int v_id;

void main() {
    gl_Position = u_mvp * vec4(a_pos, 0.0, 1.0);
    v_id = gl_VertexID + u_id_offset + 1;
}
"""

PICKING_SCATTER_FS_SRC = r"""
#version 330 core
flat in int v_id;
out int FragData;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (dot(coord, coord) > 0.25) discard;
    FragData = v_id;
}
"""

class GPULinePlot:
    def __init__(self, width=1280, height=800, title="GLPlot", density_gain=1.0):
        # Window & FB Resolution
        self.width  = int(width)
        self.height = int(height)
        self.fb_width = int(width)
        self.fb_height = int(height)
        self.title  = title

        # Internal State managers
        self.view = ViewState()
        self.data = DataState()

        # Rendering options
        self.use_fp16_ab      = True
        self.use_packed_color = True
        self.global_alpha     = 0.25
        self.density_gain     = density_gain

        # GPU resources
        self.window   = None
        self.vao      = None
        self.vbo_base = self.vbo_ab = self.vbo_col = None
        self.prog     = None

        # Uniform locations
        self.u_mvp = self.u_xrange = self.u_window = None
        self.u_use_color = self.u_alpha = None
        self.u_enable_sub = self.u_keep_prob = self.u_is_density = None

        # LOD controls
        self.enable_subsample = True
        self.max_lines_per_px = 10
        
        # Interaction delays and thresholds
        self._mouse_press_pos = (0.0, 0.0)
        self._drag_started = False
        self._drag_threshold_px = 4.0
        
        self._cache_refresh_requested = False
        self._last_cache_capture_time = 0.0
        self._cache_refresh_interval = 1.0 / 12.0 # 12Hz max refresh
        
        self._hover_resume_time = 0.0
        self._last_hover_pick_time = 0.0
        self._hover_pick_interval = 1.0 / 6.0 # 6Hz pick rate for smoothness

        # Legacy / Compatibility attributes (shared with data state)
        self.N = 0
        self._has_color = False
        
        # Pending buffers
        self._pending_ab = None
        
        self._picking_dirty = True
        self._pending_colors = None

        # Density FBO resources
        self._density_fbo = self._density_tex = self._density_prog = None
        
        # Primitive storage
        self._line_strips = []
        self._strip_prog = None
        self._scatters = []
        self._scatter_prog = None
        self.imgui_ctx = None
        self.imgui_impl = None
        self._hud_timer = 0.0
        
        self._is_test_mode = False

    def _init_density_fbo(self):
        if self._density_fbo:
            glDeleteFramebuffers(1, [self._density_fbo])
            glDeleteTextures(1, [self._density_tex])

        self._density_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._density_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, self.fb_width, self.fb_height, 0, GL_RED, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        self._density_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._density_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._density_tex, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _init_picking_fbo(self):
        if hasattr(self, '_picking_fbo') and self._picking_fbo:
            glDeleteFramebuffers(1, [self._picking_fbo])
            glDeleteTextures(1, [self._picking_tex])
            
        self._picking_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._picking_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, self.fb_width, self.fb_height, 0, GL_RED_INTEGER, GL_INT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        
        self._picking_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._picking_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._picking_tex, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _init_cache_fbo(self):
        if hasattr(self, '_cache_fbo') and self._cache_fbo:
            glDeleteFramebuffers(1, [self._cache_fbo])
            glDeleteTextures(1, [self._cache_tex])
            
        self._cache_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._cache_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.fb_width, self.fb_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        self._cache_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._cache_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._cache_tex, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def set_global_alpha(self, alpha: float):
        self.global_alpha = float(np.clip(alpha, 0.0, 1.0))
        self.view.dirty = True

    def set_lines_ab(self, ab: np.ndarray, x_range=(-3.0, 3.0), colors: np.ndarray | None = None):
        ab = np.ascontiguousarray(ab, np.float32)
        if ab.ndim != 2 or ab.shape[1] != 2:
            raise ValueError("ab must be (N,2) float32")
        self.N = ab.shape[0]
        self.data.xrange = (float(x_range[0]), float(x_range[1]))
        
        self._cpu_ab = ab
        self._picking_dirty = True
        if colors is not None:
             self._cpu_cols = np.ascontiguousarray(colors, np.float32)

        if self.vao is None:
            self._pending_ab = ab
            self._pending_xr = self.data.xrange
            self._pending_colors = self._cpu_cols
            return

        self._upload_ab_and_colors(ab, colors)
        self.view.dirty = True

    def add_text(self, x: float, y: float, text_str: str, fontsize: int = 12, color="k"):
        """Stores spatial text to be rendered by ImGui's draw list in screen space."""
        if not hasattr(self, '_spatial_texts'):
            self._spatial_texts = []
        self._spatial_texts.append({'x': x, 'y': y, 'str': text_str})
        self.view.dirty = True

    def add_line_strip(self, x: np.ndarray, y: np.ndarray, color: tuple):
        """Add a standard sequential line strip."""
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        pts = np.column_stack([x, y]).astype(np.float32)
        
        if self.vao is not None:
            vao = glGenVertexArrays(1)
            vbo = glGenBuffers(1)
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, pts.nbytes, pts, GL_STATIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
            glBindVertexArray(0)
            self._line_strips.append({'vao': vao, 'vbo': vbo, 'count': len(pts), 'color': color, 'pts': pts})
        else:
            self._line_strips.append({'pts': pts, 'color': color})
            
        self.view.dirty = True

    def add_scatter(self, x: np.ndarray, y: np.ndarray, color: np.ndarray, size: float):
        """Add scatter points array."""
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        pts = np.column_stack([x, y]).astype(np.float32)
        c = np.asarray(color, dtype=np.float32)
        if c.ndim == 1 and len(c) == 4:
            c = np.tile(c, (len(pts), 1))
            
        if self.vao is not None:
            vao = glGenVertexArrays(1)
            # Pts
            vbo_pts = glGenBuffers(1)
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_pts)
            glBufferData(GL_ARRAY_BUFFER, pts.nbytes, pts, GL_STATIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
            # Colors
            vbo_col = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_col)
            glBufferData(GL_ARRAY_BUFFER, c.nbytes, c, GL_STATIC_DRAW)
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
            
            glBindVertexArray(0)
            self._scatters.append({'vao': vao, 'vbo_p': vbo_pts, 'vbo_c': vbo_col, 'count': len(pts), 'size': size, 'pts': pts, 'color': c})
        else:
            self._scatters.append({'pts': pts, 'color': c, 'size': size})
        self.view.dirty = True

    def run(self):
        self._init_window()
        self._init_gl()
        self._init_shaders()
        self._init_buffers()

        if self._pending_ab is not None:
            self._upload_ab_and_colors(self._pending_ab, self._pending_colors)
            self.data.xrange = self._pending_xr
            self._pending_ab = None
            self._pending_colors = None
            
        # Init pending line strips
        for i, strip in enumerate(self._line_strips):
            if 'pts' in strip:
                pts = strip['pts']
                vao = glGenVertexArrays(1)
                vbo = glGenBuffers(1)
                glBindVertexArray(vao)
                glBindBuffer(GL_ARRAY_BUFFER, vbo)
                glBufferData(GL_ARRAY_BUFFER, pts.nbytes, pts, GL_STATIC_DRAW)
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
                glBindVertexArray(0)
                self._line_strips[i] = {'vao': vao, 'vbo': vbo, 'count': len(pts), 'color': strip['color'], 'pts': pts}
                
        # Init pending scatters
        for i, scat in enumerate(self._scatters):
            if 'pts' in scat:
                pts = scat['pts']
                c = scat['color']
                vao = glGenVertexArrays(1)
                vbo_p = glGenBuffers(1)
                glBindVertexArray(vao)
                glBindBuffer(GL_ARRAY_BUFFER, vbo_p)
                glBufferData(GL_ARRAY_BUFFER, pts.nbytes, pts, GL_STATIC_DRAW)
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
                vbo_c = glGenBuffers(1)
                glBindBuffer(GL_ARRAY_BUFFER, vbo_c)
                glBufferData(GL_ARRAY_BUFFER, c.nbytes, c, GL_STATIC_DRAW)
                glEnableVertexAttribArray(1)
                glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
                glBindVertexArray(0)
                self._scatters[i] = {'vao': vao, 'vbo_p': vbo_p, 'vbo_c': vbo_c, 'count': len(pts), 'size': scat['size'], 'pts': pts, 'color': c}

        if self._is_test_mode:
            # Only draw once in test mode and return
            self._draw()
            return

        glfw.swap_interval(1)
        last_interact = glfw.get_time()
        self._title_timer = 0.0
        
        while not glfw.window_should_close(self.window):
            io = imgui.get_io()
            is_active = (self.view.dirty or 
                         getattr(self.view, 'drag', False) or 
                         getattr(self.view, 'is_caching', False) or 
                         io.want_capture_mouse or 
                         io.want_capture_keyboard)
            
            if is_active:
                last_interact = glfw.get_time()
                
            if glfw.get_time() - last_interact > 0.5:
                glfw.wait_events_timeout(0.05)
                self.imgui_impl.process_inputs()
                # Re-evaluate active state after events
                io = imgui.get_io()
                is_active = (self.view.dirty or 
                             getattr(self.view, 'drag', False) or 
                             getattr(self.view, 'is_caching', False) or 
                             io.want_capture_mouse or 
                             io.want_capture_keyboard)
                if not is_active:
                    continue # Skip drawing to save GPU
            else:
                glfw.poll_events()
                self.imgui_impl.process_inputs()
            
            t0 = glfw.get_time()
            
            # Rate-limited deferred cache capture (12Hz max)
            now = glfw.get_time()
            if getattr(self, '_cache_refresh_requested', False) and self.view.is_caching:
                if now - self._last_cache_capture_time >= self._cache_refresh_interval:
                    self._capture_cache(padding=3.0)
                    self._last_cache_capture_time = now
                    self._cache_refresh_requested = False
            
            self._draw()
            glfw.swap_buffers(self.window)
            t1 = glfw.get_time()
            
            # Update cache timer for scroll-based caching
            if self.view.cache_timer > 0:
                self.view.cache_timer -= (t1 - t0)
                if self.view.cache_timer <= 0:
                    self.view.cache_timer = 0
                    if self.view.is_caching and not self.view.drag:
                        self.view.is_caching = False
                        self.view.dirty = True
                        self._picking_dirty = True

            self.view.dirty = False
            
            # Only update window title twice a second
            if t1 - self._title_timer > 0.5:
                dt = t1 - t0
                fps = 1.0 / dt if dt > 0 else 999.0
                status_title = f"{self.title}  |  {fps:.0f} FPS  |  LOD: {self.max_lines_per_px}"
                glfw.set_window_title(self.window, status_title)
                self._title_timer = t1
            
        self.imgui_impl.shutdown()
        glfw.terminate()

    def _init_window(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
            
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.RED_BITS,   16)
        glfw.window_hint(glfw.GREEN_BITS, 16)
        glfw.window_hint(glfw.BLUE_BITS,  16)
        glfw.window_hint(glfw.ALPHA_BITS, 16)
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)

        if self._is_test_mode:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
            
        glfw.make_context_current(self.window)
        self.width, self.height = glfw.get_window_size(self.window)
        self.fb_width, self.fb_height = glfw.get_framebuffer_size(self.window)

        # Initialize ImGui context
        self.imgui_ctx = imgui.create_context()
        self.imgui_impl = GlfwRenderer(self.window, attach_callbacks=False)

        glfw.set_window_size_callback(self.window, self._on_resize)
        glfw.set_framebuffer_size_callback(self.window, self._on_fb_resize)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_char_callback(self.window, self._on_char)

    def _init_gl(self):
        glViewport(0, 0, self.fb_width, self.fb_height)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        try:
            glDisable(GL_MULTISAMPLE)
        except Exception: pass

        glEnable(GL_CLIP_DISTANCE0)
        glEnable(GL_CLIP_DISTANCE1)
        glEnable(GL_CLIP_DISTANCE2)
        glEnable(GL_CLIP_DISTANCE3)
        self._init_density_fbo()
        self._init_picking_fbo()
        self._init_cache_fbo()

    def _init_shaders(self):
        self.prog = link_program(VS_SRC, FS_SRC)
        self.u_mvp        = glGetUniformLocation(self.prog, "u_mvp")
        self.u_xrange     = glGetUniformLocation(self.prog, "u_xrange")
        self.u_window     = glGetUniformLocation(self.prog, "u_window")
        self.u_use_color  = glGetUniformLocation(self.prog, "u_use_color")
        self.u_alpha      = glGetUniformLocation(self.prog, "u_alpha")
        self.u_enable_sub = glGetUniformLocation(self.prog, "u_enable_subsample")
        self.u_keep_prob  = glGetUniformLocation(self.prog, "u_keep_prob")
        self.u_is_density = glGetUniformLocation(self.prog, "u_is_density")
        
        self._strip_prog = link_program(STRIP_VS_SRC, FS_SRC)
        self.u_strip_mvp = glGetUniformLocation(self._strip_prog, "u_mvp")
        self.u_strip_color = glGetUniformLocation(self._strip_prog, "u_color")
        self.u_strip_alpha = glGetUniformLocation(self._strip_prog, "u_alpha")
        self.u_strip_is_density = glGetUniformLocation(self._strip_prog, "u_is_density")
        self._scatter_prog = link_program(SCATTER_VS_SRC, SCATTER_FS_SRC)
        self.u_scatter_mvp = glGetUniformLocation(self._scatter_prog, "u_mvp")
        self.u_scatter_size = glGetUniformLocation(self._scatter_prog, "u_size")
        self.u_scatter_alpha = glGetUniformLocation(self._scatter_prog, "u_alpha")
        self.u_scatter_is_density = glGetUniformLocation(self._scatter_prog, "u_is_density")
        
        # Pick shaders
        self._pick_prog = link_program(PICKING_VS_SRC, PICKING_FS_SRC)
        self.u_pick_mvp = glGetUniformLocation(self._pick_prog, "u_mvp")
        self.u_pick_xrange = glGetUniformLocation(self._pick_prog, "u_xrange")
        self.u_pick_enable_sub = glGetUniformLocation(self._pick_prog, "u_enable_subsample")
        self.u_pick_keep_prob = glGetUniformLocation(self._pick_prog, "u_keep_prob")
        
        self._pick_scatter_prog = link_program(PICKING_SCATTER_VS_SRC, PICKING_SCATTER_FS_SRC)
        self.u_pick_scatter_mvp = glGetUniformLocation(self._pick_scatter_prog, "u_mvp")
        self.u_pick_scatter_offset = glGetUniformLocation(self._pick_scatter_prog, "u_id_offset")
        
        # Cache Impostor Shader
        vs = compile_shader("""
            #version 330 core
            out vec2 v_uv;
            const vec2 verts[4] = vec2[4](
                vec2(-1.0, -1.0), vec2( 1.0, -1.0),
                vec2(-1.0,  1.0), vec2( 1.0,  1.0)
            );
            void main() {
                gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
                v_uv = verts[gl_VertexID] * 0.5 + 0.5;
            }
        """, GL_VERTEX_SHADER)
        
        fs = compile_shader("""
            #version 330 core
            in vec2 v_uv;
            out vec4 FragColor;
            
            uniform sampler2D u_tex;
            uniform vec4 u_cache_window; // l, r, b, t of the original capture
            uniform vec4 u_cur_window;   // l, r, b, t of the current camera
            
            void main() {
                // Convert current pixel UV to current world coordinate
                float wx = mix(u_cur_window.x, u_cur_window.y, v_uv.x);
                float wy = mix(u_cur_window.z, u_cur_window.w, v_uv.y);
                
                // Map current world coordinate back to the cached UV space
                float cache_u = (wx - u_cache_window.x) / (u_cache_window.y - u_cache_window.x);
                float cache_v = (wy - u_cache_window.z) / (u_cache_window.w - u_cache_window.z);
                
                cache_u = clamp(cache_u, 0.0, 1.0);
                cache_v = clamp(cache_v, 0.0, 1.0);
                
                FragColor = texture(u_tex, vec2(cache_u, cache_v));
            }
        """, GL_FRAGMENT_SHADER)
        
        self._cache_prog = glCreateProgram()
        glAttachShader(self._cache_prog, vs)
        glAttachShader(self._cache_prog, fs)
        glLinkProgram(self._cache_prog)
        glDeleteShader(vs)
        glDeleteShader(fs)
        
        self.u_cache_tex = glGetUniformLocation(self._cache_prog, "u_tex")
        self.u_cache_cw = glGetUniformLocation(self._cache_prog, "u_cache_window")
        self.u_cache_currw = glGetUniformLocation(self._cache_prog, "u_cur_window")
        self._cache_vao = glGenVertexArrays(1)

    def _init_buffers(self):
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo_base = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_base)
        t = np.array([0.0, 1.0], dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, t.nbytes, t, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))

        self.vbo_ab = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_ab)
        glBufferData(GL_ARRAY_BUFFER, 16, None, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        if self.use_fp16_ab:
            glVertexAttribPointer(1, 2, GL_HALF_FLOAT, GL_FALSE, 0, C.c_void_p(0))
        else:
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
        glVertexAttribDivisor(1, 1)

        self.vbo_col = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_col)
        glBufferData(GL_ARRAY_BUFFER, 16, None, GL_STATIC_DRAW)
        glEnableVertexAttribArray(2)
        if self.use_packed_color:
            glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, C.c_void_p(0))
        else:
            glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
        glVertexAttribDivisor(2, 1)

        glBindVertexArray(0)

    def _upload_ab_and_colors(self, ab_f32: np.ndarray, cols_f32: np.ndarray | None):
        glBindVertexArray(self.vao)

        ab_u = ab_f32.astype(np.float16) if self.use_fp16_ab else ab_f32
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_ab)
        glBufferData(GL_ARRAY_BUFFER, ab_u.nbytes, ab_u, GL_STATIC_DRAW)

        self._has_color = cols_f32 is not None
        if self._has_color:
            if cols_f32.shape != (ab_f32.shape[0], 4):
                raise ValueError("colors must be (N,4)")
            if self.use_packed_color:
                cols_u8 = np.clip(cols_f32 * 255.0, 0, 255).astype(np.uint8, copy=False)
                glBindBuffer(GL_ARRAY_BUFFER, self.vbo_col)
                glBufferData(GL_ARRAY_BUFFER, cols_u8.nbytes, cols_u8, GL_STATIC_DRAW)
            else:
                glBindBuffer(GL_ARRAY_BUFFER, self.vbo_col)
                glBufferData(GL_ARRAY_BUFFER, cols_f32.nbytes, cols_f32, GL_STATIC_DRAW)

        glBindVertexArray(0)

    def _world_window(self, padding=1.0):
        aspect = max(self.width, 1) / max(self.height, 1)
        half_h = padding / self.view.zoom
        half_w = half_h * aspect
        l = self.view.cx - half_w; r = self.view.cx + half_w
        b = self.view.cy - half_h; t = self.view.cy + half_h
        return l, r, b, t

    def _mvp(self, window=None):
        if window:
            l, r, b, t = window
        else:
            l, r, b, t = self._world_window()
        return ortho(l, r, b, t)

    def screen_to_world(self, sx, sy):
        l, r, b, t = self._world_window()
        x = l + (sx/self.width) * (r-l)
        y = b + ((self.height - sy)/self.height) * (t-b)
        return x, y

    def _apply_zoom_at_cursor(self, factor, mx, my):
        wx0, wy0 = self.screen_to_world(mx, my)
        self.view.zoom = float(np.clip(self.view.zoom * factor, self.view.zoom_min, self.view.zoom_max))
        wx1, wy1 = self.screen_to_world(mx, my)
        self.view.cx += (wx0 - wx1); self.view.cy += (wy0 - wy1)

    def _get_lod_prob(self):
        if not self.enable_subsample or self.N <= 0:
            return 1.0

        # Much stronger LOD during interaction to ensure 60 FPS
        if getattr(self.view, 'drag', False) or getattr(self.view, 'is_caching', False):
            target_lines = 3 * self.width
        else:
            target_lines = self.max_lines_per_px * self.width

        if target_lines >= self.N:
            return 1.0
        return float(target_lines) / float(self.N)

    def _draw_lines(self, is_density=False, mvp=None, window=None):
        """Draws the core a*x+b lines"""
        if self.N <= 0:
            return
            
        l, r, b, t = window if window else self._world_window()
        m = mvp if mvp is not None else self._mvp()
        xr0, xr1 = self.data.xrange

        glUseProgram(self.prog)
        glUniformMatrix4fv(self.u_mvp, 1, GL_TRUE, m)
        glUniform2f(self.u_xrange, xr0, xr1)
        glUniform4f(self.u_window, l, r, b, t)
        glUniform1i(self.u_is_density, 1 if is_density else 0)
        
        prob = self._get_lod_prob()
        effective_alpha = self.global_alpha
        
        # When removing lines statistically, boost opacity slightly so the plot
        # retains visual weight instead of fading out entirely.
        if prob < 1.0:
            effective_alpha = min(1.0, self.global_alpha / (prob ** 0.5))
            
        glUniform1i(self.u_use_color, 1 if self._has_color else 0)
        glUniform1f(self.u_alpha, float(effective_alpha))
        glUniform1i(self.u_enable_sub, 1 if self.enable_subsample else 0)
        glUniform1f(self.u_keep_prob, prob)

        glBindVertexArray(self.vao)
        glDrawArraysInstanced(GL_LINES, 0, 2, self.N)
        glBindVertexArray(0)
        glUseProgram(0)

    def _draw_strips(self, is_density=False, mvp=None):
        """Draws sequential standard line strips"""
        if not self._line_strips:
            return
            
        m = mvp if mvp is not None else self._mvp()
        glUseProgram(self._strip_prog)
        glUniformMatrix4fv(self.u_strip_mvp, 1, GL_TRUE, m)
        glUniform1f(self.u_strip_alpha, float(self.global_alpha))
        glUniform1i(self.u_strip_is_density, 1 if is_density else 0)
        
        for strip in self._line_strips:
            if 'vao' not in strip: continue # Pending initialization edgecase
            c = strip['color']
            glUniform4f(self.u_strip_color, c[0], c[1], c[2], c[3])
            glBindVertexArray(strip['vao'])
            glDrawArrays(GL_LINE_STRIP, 0, strip['count'])
            glBindVertexArray(0)
            
        glUseProgram(0)

        glUseProgram(0)

    def _build_imgui_hud(self):
        if not self.view.hud_visible:
            return
            
        # Draw floating labels
        dl = imgui.get_background_draw_list()
        for t in getattr(self, '_spatial_texts', []):
            wx, wy = t['x'], t['y']
            l, r, b, top_y = self._world_window()
            # Project to screen
            sx = ((wx - l) / (r - l)) * self.width
            sy = (1.0 - (wy - b) / (top_y - b)) * self.height
            dl.add_text(sx, sy, imgui.get_color_u32_rgba(0, 0, 0, 1), t['str'])
            
        # Analytics Window
        imgui.set_next_window_position(10, 10, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(300, 350, imgui.FIRST_USE_EVER)
        expanded, _ = imgui.begin("GLPlot Analytics Dashboard", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        if expanded:
            imgui.text("Performance")
            imgui.separator()
            if self._hud_timer == 0: self._hud_timer = glfw.get_time()
            dt = glfw.get_time() - self._hud_timer
            self._hud_timer = glfw.get_time()
            fps = 1.0/dt if dt > 0 else 0
            imgui.text(f"FPS: {fps:.1f}  |  N: {self.N:,}")
            imgui.text(f"Camera: {self.view.cx:.2f}, {self.view.cy:.2f} (x{self.view.zoom:.1f})")
            
            imgui.spacing()
            imgui.text("Controls")
            imgui.separator()
            changed, self.view.show_density = imgui.checkbox("Density Mode [D]", self.view.show_density)
            if changed: self.view.dirty = True
            
            changed, self.max_lines_per_px = imgui.slider_int("LOD", self.max_lines_per_px, 1, 1000)
            if changed: self.view.dirty = True
            
            if imgui.button("Reset View [R]"):
                self.view.cx = self.view.cy = 0.0
                self.view.zoom = 1.0
                self.view.dirty = True
            imgui.same_line()
            if imgui.button("Export 4K"):
                self.save_current_view(scale=2.0)
                
            imgui.spacing()
            imgui.text("Inspector")
            imgui.separator()
            if self.view.selected_idx != -1:
                imgui.text_colored(f"Selected: {self.view.selected_type} #{self.view.selected_idx}", 1, 0.8, 0.2)
                stats = self.get_summary_stats('selected')
                imgui.text(f"Values: {stats}")
                if imgui.button("Clear Selection [ESC]"):
                    self.view.selected_idx = -1
                    self.view.dirty = True
            elif self.view.hover_idx != -1:
                imgui.text(f"Hover: {self.view.hover_type} #{self.view.hover_idx}")
            else:
                imgui.text("Hover over data to inspect...")
                
            imgui.spacing()
            if imgui.tree_node("Visible Statistics"):
                stats = self.get_summary_stats('visible')
                if "error" not in stats:
                    imgui.columns(2, "stats")
                    imgui.text("Metric"); imgui.next_column()
                    imgui.text("Value"); imgui.next_column(); imgui.separator()
                    for k, v in stats.items():
                        if k == 'scope': continue
                        imgui.text(str(k)); imgui.next_column()
                        val = f"{v:.4f}" if isinstance(v, float) else str(v)
                        imgui.text(val); imgui.next_column()
                    imgui.columns(1)
                else:
                    imgui.text("No data visible.")
                imgui.tree_pop()
        imgui.end()

    def _draw_scatters(self, is_density=False, mvp=None):
        if not self._scatters: return
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glEnable(GL_PROGRAM_POINT_SIZE)
        glUseProgram(self._scatter_prog)
        m = mvp if mvp is not None else self._mvp()
        glUniformMatrix4fv(self.u_scatter_mvp, 1, GL_TRUE, m)
        glUniform1f(self.u_scatter_alpha, float(self.global_alpha))
        glUniform1i(self.u_scatter_is_density, 1 if is_density else 0)
        
        for scat in self._scatters:
            if 'vao' not in scat: continue
            glUniform1f(self.u_scatter_size, float(scat['size']))
            glBindVertexArray(scat['vao'])
            glDrawArrays(GL_POINTS, 0, scat['count'])
            glBindVertexArray(0)
            
        glUseProgram(0)
        glDisable(GL_PROGRAM_POINT_SIZE)

    def _draw_cached_impostor(self):
        glDisable(GL_BLEND)
        glUseProgram(self._cache_prog)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._cache_tex)
        glUniform1i(self.u_cache_tex, 0)
        
        cl, cr, cb, ct = self.view.cache_state
        glUniform4f(self.u_cache_cw, cl, cr, cb, ct)
        
        l, r, b, t = self._world_window()
        glUniform4f(self.u_cache_currw, l, r, b, t)
        
        glBindVertexArray(self._cache_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)

    def _draw(self):
        glViewport(0, 0, self.fb_width, self.fb_height)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # Skip all UI logic during high-speed drag/cache mode to maximize throughput
        use_ui = not (getattr(self.view, 'drag', False) or getattr(self.view, 'is_caching', False))

        if use_ui:
            imgui.new_frame()
            self._build_imgui_hud()

        # 1. Background Data Layer
        if getattr(self.view, 'is_caching', False) and self.view.cache_state:
            self._draw_cached_impostor()
        elif self.view.show_density:
            self._draw_density_map()
        else:
            glLineWidth(1.0)
            self._draw_lines()
            self._draw_strips()
            self._draw_scatters()
            
        # 2. Tools & Overlay Layer (Always HD / Live)
        self._draw_overlay()
            
        if use_ui:
            imgui.render()
            self.imgui_impl.render(imgui.get_draw_data())

    def _draw_overlay(self):
        if not hasattr(self, "_overlay_prog"):
            vs = compile_shader("""
                #version 330 core
                out vec2 v_uv;
                out vec2 v_screen;
                const vec2 verts[4] = vec2[4](
                    vec2(-1.0, -1.0), vec2( 1.0, -1.0),
                    vec2(-1.0,  1.0), vec2( 1.0,  1.0)
                );
                
                void main() {
                    gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
                    v_uv = verts[gl_VertexID] * 0.5 + 0.5;
                    v_screen = verts[gl_VertexID];
                }
            """, GL_VERTEX_SHADER)
            
            fs = compile_shader("""
                #version 330 core
                in vec2 v_uv;
                in vec2 v_screen;
                out vec4 FragColor;
                
                uniform vec4 u_window; // l, r, b, t
                uniform vec2 u_res;    // resolution width, height
                
                float draw_line(float val, float pixel_size) {
                    return 1.0 - smoothstep(0.0, pixel_size * 1.5, abs(val));
                }
                
                void main() {
                    float l = u_window.x;
                    float r = u_window.y;
                    float b = u_window.z;
                    float t = u_window.w;
                    
                    float wx = mix(l, r, v_uv.x);
                    float wy = mix(b, t, v_uv.y);
                    
                    float px_w = (r - l) / u_res.x;
                    float px_h = (t - b) / u_res.y;
                    
                    // Main axes
                    float axis_x = draw_line(wy, px_h);
                    float axis_y = draw_line(wx, px_w);
                    
                    // Ticks
                    float x_span = r - l;
                    float y_span = t - b;
                    float tick_spacing_x = pow(10.0, floor(log(max(x_span, 1e-6)) / log(10.0)) - 1.0);
                    float tick_spacing_y = pow(10.0, floor(log(max(y_span, 1e-6)) / log(10.0)) - 1.0);
                    
                    // Fix negative modulo properly
                    float mx = mod(wx + tick_spacing_x * 0.5, tick_spacing_x) - tick_spacing_x * 0.5;
                    float my = mod(wy + tick_spacing_y * 0.5, tick_spacing_y) - tick_spacing_y * 0.5;
                    
                    float tick_len_x = y_span * 0.02;
                    float tick_len_y = x_span * 0.02;
                    
                    float tick_x = draw_line(mx, px_w) * step(abs(wy), tick_len_x);
                    float tick_y = draw_line(my, px_h) * step(abs(wx), tick_len_y);
                    
                    float intensity = max(max(axis_x, axis_y), max(tick_x, tick_y));
                    if (intensity < 0.05) discard;
                    
                    FragColor = vec4(0.3, 0.3, 0.3, intensity * 0.8);
                }
            """, GL_FRAGMENT_SHADER)
            
            self._overlay_prog = glCreateProgram()
            glAttachShader(self._overlay_prog, vs)
            glAttachShader(self._overlay_prog, fs)
            glLinkProgram(self._overlay_prog)
            glDeleteShader(vs)
            glDeleteShader(fs)
            
            self.u_overlay_window = glGetUniformLocation(self._overlay_prog, "u_window")
            self.u_overlay_res = glGetUniformLocation(self._overlay_prog, "u_res")
            self._overlay_vao = glGenVertexArrays(1)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glUseProgram(self._overlay_prog)
        l, r, b, t = self._world_window()
        glUniform4f(self.u_overlay_window, l, r, b, t)
        glUniform2f(self.u_overlay_res, float(self.fb_width), float(self.fb_height))

        glBindVertexArray(self._overlay_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)

    def _draw_density_map(self, target_fbo=0, mvp=None, window=None):
        glBindFramebuffer(GL_FRAMEBUFFER, self._density_fbo)
        glViewport(0, 0, self.fb_width, self.fb_height)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE) # Additive blending for density accumulation

        self._draw_lines(is_density=True, mvp=mvp, window=window)
        self._draw_strips(is_density=True, mvp=mvp)
        self._draw_scatters(is_density=True, mvp=mvp)

        glBindFramebuffer(GL_FRAMEBUFFER, target_fbo)
        if not self._density_prog:
            self._density_prog = link_program("""
                #version 330 core
                out vec2 uv;
                const vec2 verts[4] = vec2[4](
                    vec2(-1.0, -1.0),
                    vec2( 1.0, -1.0),
                    vec2(-1.0,  1.0),
                    vec2( 1.0,  1.0)
                );
                void main() {
                    gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
                    uv = verts[gl_VertexID] * 0.5 + 0.5;
                }
            """, """
                #version 330 core
                #define log10(x) (log(x) / log(10.0))
                in vec2 uv;
                out vec4 FragColor;
                uniform sampler2D u_tex;
                uniform float u_gain;

                vec3 heatmap(float x) {
                    x = clamp(x, 0.0, 1.0);
                    return vec3(
                        smoothstep(0.0, 0.3, x),
                        smoothstep(0.3, 0.6, x),
                        smoothstep(0.6, 1.0, x)
                    );
                }

                void main() {
                    float d = texture(u_tex, uv).r * u_gain;
                    float val = log10(1.0 + d) / log10(50.0);
                    FragColor = vec4(heatmap(val), 1.0);
                }
            """)
            
            self.u_density_tex = glGetUniformLocation(self._density_prog, "u_tex")
            self.u_density_gain = glGetUniformLocation(self._density_prog, "u_gain")
            
            self._density_vao = glGenVertexArrays(1)

        glDisable(GL_BLEND)
        glUseProgram(self._density_prog)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._density_tex)
        glUniform1i(self.u_density_tex, 0)
        glUniform1f(self.u_density_gain, self.density_gain)

        glBindVertexArray(self._density_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)

        glUseProgram(0)

    def _capture_cache(self, padding=3.0):
        if not hasattr(self, '_cache_fbo') or not self._cache_fbo:
            return

        capture_window = self._world_window(padding=padding)
        mvp = self._mvp(window=capture_window)

        glBindFramebuffer(GL_FRAMEBUFFER, self._cache_fbo)
        glViewport(0, 0, self.fb_width, self.fb_height)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        if self.view.show_density:
            self._draw_density_map(target_fbo=self._cache_fbo, mvp=mvp, window=capture_window)
        else:
            glLineWidth(1.0)
            self._draw_lines(mvp=mvp, window=capture_window)
            self._draw_strips(mvp=mvp)
            self._draw_scatters(mvp=mvp)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self.view.cache_state = capture_window
    # ---------- callbacks ----------
    def _on_resize(self, win, w, h):
        self.width = max(1, int(w)); self.height = max(1, int(h))
        self.view.dirty = True
        self._picking_dirty = True
        if self.imgui_impl: self.imgui_impl.resize_callback(win, w, h)
        
    def _on_fb_resize(self, win, w, h):
        self.fb_width = max(1, int(w)); self.fb_height = max(1, int(h))
        self.view.dirty = True
        self._picking_dirty = True
        self._init_density_fbo() # Must rebuild texture sizes
        self._init_picking_fbo()
        self._init_cache_fbo()
        self.view.dirty = True

    def _on_scroll(self, win, dx, dy):
        if self.imgui_impl: self.imgui_impl.scroll_callback(win, dx, dy)
        if getattr(self, 'imgui_ctx', None) and imgui.get_io().want_capture_mouse: return
        
        # Trigger deferred impostor cache for zoom
        if not self.view.is_caching:
            self.view.is_caching = True
        self._cache_refresh_requested = True
        self.view.cache_timer = 0.2 
        
        factor = 1.1 if dy > 0 else 1.0/1.1
        mx, my = glfw.get_cursor_pos(self.window)
        self._apply_zoom_at_cursor(factor, mx, my)
        self.view.dirty = True
        self._picking_dirty = True

    def get_summary_stats(self, scope='visible'):
        """Computes summary statistics for the specified scope."""
        
        if not hasattr(self, '_stats_cache'):
            self._stats_cache = {}
            self._stats_cache_time = {}
            self._stats_cache_view = {}
            
        current_time = glfw.get_time()
        view_state = (self.view.cx, self.view.cy, self.view.zoom, self.view.selected_idx)
        
        if getattr(self.view, 'drag', False) or getattr(self.view, 'is_caching', False):
            return self._stats_cache.get(scope, {"error": "Stats paused during interaction", "scope": scope})
            
        if scope in self._stats_cache:
            time_since = current_time - self._stats_cache_time.get(scope, 0)
            
            # 2) Absolute hard limit: never calculate faster than 4 Hz (0.25s) to avoid UI freezing
            if time_since < 0.25:
                return self._stats_cache[scope]
                
            # 3) Cache is perfectly valid forever if the camera hasn't moved
            if self._stats_cache_view.get(scope) == view_state:
                return self._stats_cache[scope]

        data_to_analyze = []
        
        if scope == 'visible':
            l, r, b, t = self._world_window()
            if self.N > 0:
                xr0, xr1 = self.data.xrange
                xm = (xr0 + xr1) * 0.5
                if l <= xm <= r:
                    ym = self._cpu_ab[:, 0] * xm + self._cpu_ab[:, 1]
                    mask = (ym >= b) & (ym <= t)
                    data_to_analyze.extend(ym[mask])
            for scat in self._scatters:
                pts = scat.get('pts')
                if pts is None: continue
                mask = (pts[:, 0] >= l) & (pts[:, 0] <= r) & (pts[:, 1] >= b) & (pts[:, 1] <= t)
                data_to_analyze.extend(pts[mask, 1])
        elif scope == 'selected':
            if self.view.selected_idx != -1:
                if self.view.selected_type == 'line':
                    a, b_val = self._cpu_ab[self.view.selected_idx]
                    data_to_analyze = [a, b_val]
                elif self.view.selected_type == 'scatter':
                    s_idx, p_idx = self.view.selected_idx
                    pts = self._scatters[s_idx]['pts']
                    data_to_analyze = list(pts[p_idx])
        else: # 'all'
            if self.N > 0: data_to_analyze.extend(self._cpu_ab[:, 1])
            for scat in self._scatters:
                pts = scat.get('pts')
                if pts is not None: data_to_analyze.extend(pts[:, 1])
                
        if not data_to_analyze:
            result = {"error": "No data in scope"}
        else:
            arr = np.array(data_to_analyze)
            result = {
                "scope": scope,
                "count": len(arr),
                "mean": np.mean(arr),
                "std": np.std(arr),
                "min": np.min(arr),
                "max": np.max(arr),
                "median": float(np.median(arr))
            }
            
        self._stats_cache[scope] = result
        self._stats_cache_time[scope] = current_time
        self._stats_cache_view[scope] = view_state
        return result

    def _draw_picking(self):
        if not hasattr(self, '_picking_fbo') or not self._picking_fbo: return
        
        glBindFramebuffer(GL_FRAMEBUFFER, self._picking_fbo)
        glViewport(0, 0, self.fb_width, self.fb_height)
        glDisable(GL_BLEND)
        
        # Clear to 0
        glClearBufferiv(GL_COLOR, 0, np.array([0, 0, 0, 0], dtype=np.int32))
        
        mvp = self._mvp()
        
        # Draw Lines
        if self.N > 0:
            glUseProgram(self._pick_prog)
            glUniformMatrix4fv(self.u_pick_mvp, 1, GL_TRUE, mvp)
            glUniform2f(self.u_pick_xrange, self.data.xrange[0], self.data.xrange[1])
            glUniform1i(self.u_pick_enable_sub, 1 if self.enable_subsample else 0)
            glUniform1f(self.u_pick_keep_prob, self._get_lod_prob())
            
            glBindVertexArray(self.vao)
            glDrawArraysInstanced(GL_LINES, 0, 2, self.N)
            glBindVertexArray(0)
            
        # Draw Scatters
        if self._scatters:
            glEnable(GL_PROGRAM_POINT_SIZE)
            glUseProgram(self._pick_scatter_prog)
            glUniformMatrix4fv(self.u_pick_scatter_mvp, 1, GL_TRUE, mvp)
            
            offset = self.N
            for s_idx, scat in enumerate(self._scatters):
                count = scat.get('count', 0)
                if count > 0:
                    glUniform1i(self.u_pick_scatter_offset, offset)
                    glBindVertexArray(scat['vao'])
                    glDrawArrays(GL_POINTS, 0, count)
                    glBindVertexArray(0)
                offset += count
            glDisable(GL_PROGRAM_POINT_SIZE)
            
        glUseProgram(0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glEnable(GL_BLEND)

    def pick_nearest(self, sx, sy, radius=5):
        """Hardware picking natively from the GPU ID Buffer."""
        if not hasattr(self, '_picking_fbo') or not self._picking_fbo:
            return (-1, None)
            
        if getattr(self, '_picking_dirty', True):
            self._draw_picking()
            self._picking_dirty = False
            
        # Adjust Y to OpenGL Origin
        gy = int(self.fb_height - sy)
        gx = int(sx)
        
        if gx < 0 or gx >= self.fb_width or gy < 0 or gy >= self.fb_height:
            return (-1, None)
            
        glBindFramebuffer(GL_FRAMEBUFFER, self._picking_fbo)
        x0 = max(0, gx - radius)
        x1 = min(self.fb_width, gx + radius + 1)
        y0 = max(0, gy - radius)
        y1 = min(self.fb_height, gy + radius + 1)
        w, h = x1 - x0, y1 - y0
        
        if w <= 0 or h <= 0:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            return (-1, None)
            
        pixels = glReadPixels(x0, y0, w, h, GL_RED_INTEGER, GL_INT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        data = np.frombuffer(pixels, dtype=np.int32).reshape((h, w))
        if not np.any(data > 0):
            return (-1, None)
            
        # Find closest ID to center
        cy, cx = h // 2, w // 2
        y_idx, x_idx = np.nonzero(data > 0)
        dists = (y_idx - cy)**2 + (x_idx - cx)**2
        best_id = data[y_idx[np.argmin(dists)], x_idx[np.argmin(dists)]]
        
        real_id = best_id - 1
        
        if real_id < self.N:
            return (real_id, 'line')
            
        offset = self.N
        for s_idx, scat in enumerate(self._scatters):
            count = scat.get('count', 0)
            if offset <= real_id < offset + count:
                return ((s_idx, real_id - offset), 'scatter')
            offset += count
            
        return (-1, None)

    def _on_mouse_button(self, win, button, action, mods):
        if self.imgui_impl: self.imgui_impl.mouse_callback(win, button, action, mods)
        if getattr(self, 'imgui_ctx', None) and imgui.get_io().want_capture_mouse: return
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.view.drag = True
                self._drag_started = False
                self._mouse_press_pos = glfw.get_cursor_pos(self.window)
                self.view.last_mouse = self._mouse_press_pos

            elif action == glfw.RELEASE:
                self.view.drag = False
                self._hover_resume_time = glfw.get_time() + 0.15 # 150ms cooldown after drag

                mx, my = glfw.get_cursor_pos(self.window)
                px, py = self._mouse_press_pos
                dist2 = (mx - px) ** 2 + (my - py) ** 2

                # Click, not drag
                if dist2 <= self._drag_threshold_px ** 2:
                    idx, ptype = self.pick_nearest(mx, my)
                    self.view.selected_idx = idx
                    self.view.selected_type = ptype
                    self.view.dirty = True

                if getattr(self.view, 'is_caching', False):
                    self.view.is_caching = False
                    self.view.dirty = True

    def _on_cursor(self, win, x, y):
        if getattr(self, 'imgui_ctx', None) and imgui.get_io().want_capture_mouse: return

        # Not dragging -> hover path
        if not self.view.drag:
            now = glfw.get_time()
            if now < self._hover_resume_time:
                return

            if now - self._last_hover_pick_time >= self._hover_pick_interval:
                idx, ptype = self.pick_nearest(x, y)
                if idx != self.view.hover_idx or ptype != self.view.hover_type:
                    self.view.hover_idx = idx
                    self.view.hover_type = ptype
                    self.view.dirty = True
                self._last_hover_pick_time = now
            return

        # Dragging
        px, py = self._mouse_press_pos
        dist2 = (x - px) ** 2 + (y - py) ** 2

        if not self._drag_started and dist2 > self._drag_threshold_px ** 2:
            self._drag_started = True
            self.view.is_caching = True
            self._cache_refresh_requested = True # Defer capture to run loop

        lx, ly = self.view.last_mouse
        wx0, wy0 = self.screen_to_world(lx, ly)
        wx1, wy1 = self.screen_to_world(x, y)
        self.view.cx -= (wx1 - wx0)
        self.view.cy -= (wy1 - wy0)
        self.view.last_mouse = (x, y)
        self.view.dirty = True
        self._picking_dirty = True
        
        # Adaptive Re-capture check: Request refresh if we're getting close to any edge
        if self.view.is_caching and self.view.cache_state:
            cl, cr, cb, ct = self.view.cache_state
            l, r, b, t = self._world_window()
            margin = 0.15 
            cw, ch = (cr - cl), (ct - cb)
            if (l < cl + cw*margin) or (r > cr - cw*margin) or \
               (b < cb + ch*margin) or (t > ct - ch*margin):
                self._cache_refresh_requested = True

    def _on_char(self, win, char):
        if self.imgui_impl: self.imgui_impl.char_callback(win, char)

    def _on_key(self, win, key, sc, action, mods):
        if self.imgui_impl: self.imgui_impl.keyboard_callback(win, key, sc, action, mods)
        if getattr(self, 'imgui_ctx', None) and imgui.get_io().want_capture_keyboard: return
        if action != glfw.PRESS: return
        if key == glfw.KEY_ESCAPE:
            if self.view.selected_idx != -1:
                self.view.selected_idx = -1
                self.view.selected_type = None
                self.view.dirty = True
            else:
                glfw.set_window_should_close(self.window, True)
        elif key == glfw.KEY_I:
            if mods & glfw.MOD_SHIFT:
                self.view.hud_verbosity = (self.view.hud_verbosity % 2) + 1
            else:
                self.view.hud_visible = not self.view.hud_visible
            self.view.dirty = True
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            self._apply_zoom_at_cursor(1.1, self.width*0.5, self.height*0.5); self.view.dirty = True
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            self._apply_zoom_at_cursor(1/1.1, self.width*0.5, self.height*0.5); self.view.dirty = True
        elif key == glfw.KEY_R:
            self.view.cx = self.view.cy = 0.0; self.view.zoom = 1.0; self.view.dirty = True
        elif key == glfw.KEY_LEFT_BRACKET:
            self.max_lines_per_px = max(1, int(self.max_lines_per_px*0.8)); self.view.dirty = True
        elif key == glfw.KEY_RIGHT_BRACKET:
            self.max_lines_per_px = max(self.max_lines_per_px + 1, int(self.max_lines_per_px*1.25)); self.view.dirty = True
        elif key == glfw.KEY_S:
            self.save_current_view()
        elif key == glfw.KEY_D:
            self.view.show_density = not self.view.show_density
            self.view.dirty = True
        elif key == glfw.KEY_UP:
            self.density_gain *= 1.5
            self.view.dirty = True
        elif key == glfw.KEY_DOWN:
            self.density_gain /= 1.5
            self.view.dirty = True

    def save_current_view(self, filename=None, scale=2.0):
        """High-level export wrapper that supports supersampling."""
        if filename is None:
            filename = "density_view.png" if self.view.show_density else "view.png"
        
        # Use the specialized high-res export engine
        self.export_high_res(filename, scale=scale)

    def export_high_res(self, filename, scale=2.0):
        """Renders the current view to an offscreen buffer at higher resolution."""
        target_w = int(self.fb_width * scale)
        target_h = int(self.fb_height * scale)
        
        # 1. Create Offscreen FBO
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, target_w, target_h, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
        
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print("Failed to create high-res FBO")
            return
            
        # 2. Render to FBO
        glViewport(0, 0, target_w, target_h)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        
        # Temporarily swap FB dimensions for the draw calls
        old_fbw, old_fbh = self.fb_width, self.fb_height
        self.fb_width, self.fb_height = target_w, target_h
        
        # Draw everything (except HUD usually, but user might want it)
        # We'll include HUD if it's visible
        self._draw_lines()
        self._draw_strips()
        self._draw_scatters()
        self._draw_overlay()
        if self.view.hud_visible:
            # Note: ImGui HUD cannot be easily rendered to a texture in this way
            # without complex modifications. We'll skip it for high-res headless export.
            pass
        
        # 3. Readback and Save
        data = glReadPixels(0, 0, target_w, target_h, GL_RGB, GL_UNSIGNED_BYTE)
        img = np.frombuffer(data, dtype=np.uint8).reshape(target_h, target_w, 3)
        
        # 4. Cleanup
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDeleteFramebuffers(1, [fbo])
        glDeleteTextures(1, [tex])
        self.fb_width, self.fb_height = old_fbw, old_fbh
        glViewport(0, 0, self.fb_width, self.fb_height)
        
        # Save via Matplotlib to handle DPI
        fig, ax = plt.subplots(figsize=(target_w/200, target_h/200), dpi=200)
        ax.imshow(img, origin="lower")
        ax.axis('off')
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Exported: {filename} ({target_w}x{target_h})")
