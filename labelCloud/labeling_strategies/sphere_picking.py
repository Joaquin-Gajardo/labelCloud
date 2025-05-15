# labelCloud/labeling_strategies/sphere_picking.py
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..control.config_manager import config
from ..definitions import Mode, Point3D
from ..model.sphere import Sphere
from ..utils import oglhelper as ogl
from . import BaseLabelingStrategy

if TYPE_CHECKING:
    from ..view.gui import GUI


class SpherePickingStrategy(BaseLabelingStrategy):
    POINTS_NEEDED = 1
    PREVIEW = True

    def __init__(self, view: "GUI") -> None:
        super().__init__(view)
        logging.info("Enabled sphere drawing mode.")
        self.view.status_manager.update_status(
            "Please pick the center point for the sphere.",
            mode=Mode.DRAWING,
        )
        self.tmp_p1: Optional[Point3D] = None
        self.sphere_radius: float = config.getfloat(
            "LABEL", "std_sphere_radius", fallback=0.5
        )

    def register_point(self, new_point: Point3D) -> None:
        self.point_1 = new_point
        self.points_registered += 1

    def register_tmp_point(self, new_tmp_point: Point3D) -> None:
        self.tmp_p1 = new_tmp_point

    def register_scrolling(self, distance: float) -> None:
        # Adjust radius with scroll
        delta = distance / 500.0
        self.sphere_radius = max(0.01, self.sphere_radius + delta)

    def draw_preview(self) -> None:
        if self.tmp_p1:
            # Draw sphere preview using OpenGL
            ogl.draw_sphere(
                self.tmp_p1,
                self.sphere_radius,
                draw_wireframe=True,
                color=(1, 1, 0, 0.5),
            )

    def get_sphere(self) -> Sphere:
        """Create a sphere at the selected point."""
        assert self.point_1 is not None
        return Sphere(center=self.point_1, radius=self.sphere_radius)

    def get_bbox(self):
        # For compatibility with abstract base class
        # Either return None or implement sphere-to-bbox conversion if needed
        return None

    def is_bbox_finished(self) -> bool:
        """Check if we have enough points to create a sphere."""
        return self.points_registered >= self.POINTS_NEEDED

    def reset(self) -> None:
        super().reset()
        self.tmp_p1 = None
        self.sphere_radius = config.getfloat("LABEL", "std_sphere_radius", fallback=0.5)
        if hasattr(self.view, "button_pick_sphere"):
            self.view.button_pick_sphere.setChecked(False)
