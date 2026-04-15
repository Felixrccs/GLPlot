from __future__ import annotations
from typing import Tuple, Optional, TYPE_CHECKING
import numpy as np
from .utils.gl_utils import ortho

if TYPE_CHECKING:
    from .core.legacy import CameraState
    from .options import EngineOptions

class CameraController:
    def __init__(self, camera: CameraState, options: EngineOptions):
        self.camera = camera
        self.options = options

    def world_window(self, width: int, height: int, padding: float = 1.0) -> Tuple[float, float, float, float]:
        aspect = max(width, 1) / max(height, 1)
        half_h = padding / self.camera.zoom
        half_w = half_h * aspect
        l = self.camera.cx - half_w
        r = self.camera.cx + half_w
        b = self.camera.cy - half_h
        t = self.camera.cy + half_h
        return l, r, b, t

    def mvp(self, width: int, height: int, window: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
        l, r, b, t = window if window is not None else self.world_window(width, height)
        return ortho(l, r, b, t)

    def screen_to_world(self, sx: float, sy: float, width: int, height: int) -> Tuple[float, float]:
        l, r, b, t = self.world_window(width, height)
        x = l + (sx / width) * (r - l)
        y = b + ((height - sy) / height) * (t - b)
        return x, y

    def apply_zoom_at_cursor(self, factor: float, mx: float, my: float, width: int, height: int) -> None:
        wx0, wy0 = self.screen_to_world(mx, my, width, height)
        self.camera.zoom = float(np.clip(self.camera.zoom * factor, self.camera.zoom_min, self.camera.zoom_max))
        wx1, wy1 = self.screen_to_world(mx, my, width, height)
        self.camera.cx += (wx0 - wx1)
        self.camera.cy += (wy0 - wy1)

    def fit_bounds(self, xmin: float, xmax: float, ymin: float, ymax: float, width: int, height: int) -> None:
        """Calculate best cx, cy, and zoom to fit the given bounds into the current viewport."""
        self.camera.cx = 0.5 * (xmin + xmax)
        self.camera.cy = 0.5 * (ymin + ymax)
        
        span_x = max(1e-9, xmax - xmin)
        span_y = max(1e-9, ymax - ymin)
        
        aspect = max(width, 1) / max(height, 1)
        
        # Calculate zooms required for each axis
        # Zoom = 1.0 / half_h
        # half_h_required = span_y / 2.0
        # half_w_required = span_x / 2.0
        
        # In our world_window logic: half_w = (1.0 / zoom) * aspect
        # So zoom_x = aspect / half_w_req = (2.0 * aspect) / span_x
        # zoom_y = 1.0 / half_h_req = 2.0 / span_y
        
        zx = (2.0 * aspect) / span_x
        zy = 2.0 / span_y
        
        self.camera.zoom = float(np.clip(min(zx, zy), self.camera.zoom_min, self.camera.zoom_max))

    def reset_view(self) -> None:
        self.camera.cx = 0.0
        self.camera.cy = 0.0
        self.camera.zoom = 1.0
