# labelCloud/control/sphere_controller.py
import logging
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from ..io.labels.config import LabelConfig
from ..model.sphere import Sphere
from .pcd_manager import PointCloudManger

if TYPE_CHECKING:
    from ..view.gui import GUI


class SphereController:
    """Controller for managing sphere labels."""

    def __init__(self) -> None:
        """Initialize the sphere controller."""
        self.view: GUI
        self.pcd_manager: PointCloudManger
        self.spheres: List[Sphere] = []
        self.active_sphere_id: Optional[int] = None

    def set_view(self, view) -> None:
        """Set the view for this controller."""
        self.view = view

    def reset(self) -> None:
        """Reset the sphere controller state."""
        self.spheres = []
        self.active_sphere_id = None
        if self.view:
            self.update_all()

    def add_sphere(self, sphere: Sphere) -> None:
        """Add a sphere to the controller."""
        sphere.id = len(self.spheres)
        self.spheres.append(sphere)
        self.set_active_sphere(sphere.id)
        self.update_all()

    def set_spheres(self, spheres: List[Sphere]) -> None:
        """Set all spheres at once."""
        self.spheres = spheres
        if self.spheres:
            self.set_active_sphere(0)
        else:
            self.active_sphere_id = None

    def set_active_sphere(self, sphere_id: int) -> None:
        """Set the active sphere by ID."""
        if sphere_id is not None and 0 <= sphere_id < len(self.spheres):
            # Deselect previous active sphere
            if self.has_active_sphere():
                self.get_active_sphere().selected = False

            # Set new active sphere
            self.active_sphere_id = sphere_id
            self.get_active_sphere().selected = True  # Make sure this flag is set

            # Update UI
            self.update_all()
        else:
            if self.has_active_sphere():
                self.get_active_sphere().selected = False
            self.active_sphere_id = None

    def has_active_sphere(self) -> bool:
        """Check if there is an active sphere."""
        return self.active_sphere_id is not None and 0 <= self.active_sphere_id < len(
            self.spheres
        )

    def get_active_sphere(self) -> Optional[Sphere]:
        """Get the active sphere if any."""
        if self.has_active_sphere():
            return self.spheres[self.active_sphere_id]
        return None

    def select_sphere_by_ray(self, x: int, y: int) -> None:
        """Select a sphere by casting a ray from screen coordinates."""
        if not self.spheres or not self.view:
            return

        # Get world coordinates from screen position
        world_pos = self.view.gl_widget.get_world_coords(x, y, correction=False)

        # Find closest sphere
        closest_sphere = None
        closest_distance = float("inf")

        for sphere in self.spheres:
            distance = np.linalg.norm(sphere.center - world_pos)
            if distance < closest_distance:
                closest_distance = distance
                closest_sphere = sphere

        # Select sphere only if close enough
        if closest_sphere and closest_distance <= closest_sphere.radius * 1.5:
            self.set_active_sphere(closest_sphere.id)

    def deselect_sphere(self) -> None:
        """Deselect the current sphere."""
        if self.has_active_sphere():
            self.get_active_sphere().selected = False
            self.active_sphere_id = None
            self.update_all()

    def delete_current_sphere(self) -> None:
        """Delete the currently active sphere."""
        if not self.has_active_sphere():
            return

        # Remove the active sphere
        del self.spheres[self.active_sphere_id]

        # Update IDs of all spheres
        for i, sphere in enumerate(self.spheres):
            sphere.id = i

        # Update active sphere ID
        if not self.spheres:
            self.active_sphere_id = None
        elif self.active_sphere_id >= len(self.spheres):
            self.active_sphere_id = len(self.spheres) - 1
            if self.has_active_sphere():
                self.get_active_sphere().selected = True

        # Update UI
        self.update_all()

    def set_center(self, x: float, y: float, z: float) -> None:
        """Set the center of the active sphere."""
        if self.has_active_sphere():
            self.get_active_sphere().set_center(x, y, z)

    def translate_along_x(self, left: bool = False) -> None:
        """Translate the active sphere along the X axis."""
        if self.has_active_sphere():
            from ..control.config_manager import config

            step = config.getfloat("LABEL", "std_translation", fallback=0.03)
            step = -step if left else step
            self.get_active_sphere().translate(np.array([step, 0, 0]))

    def translate_along_y(self, forward: bool = False) -> None:
        """Translate the active sphere along the Y axis."""
        if self.has_active_sphere():
            from ..control.config_manager import config

            step = config.getfloat("LABEL", "std_translation", fallback=0.03)
            step = step if forward else -step
            self.get_active_sphere().translate(np.array([0, step, 0]))

    def translate_along_z(self, down: bool = False) -> None:
        """Translate the active sphere along the Z axis."""
        if self.has_active_sphere():
            from ..control.config_manager import config

            step = config.getfloat("LABEL", "std_translation", fallback=0.03)
            step = -step if down else step
            self.get_active_sphere().translate(np.array([0, 0, step]))

    def adjust_radius(self, increase: bool = True) -> None:
        """Adjust the radius of the active sphere."""
        if self.has_active_sphere():
            from ..control.config_manager import config

            step = config.getfloat("LABEL", "std_sphere_scaling", fallback=0.03)
            delta = step if increase else -step
            self.get_active_sphere().change_radius(delta)

    def set_classname(self, classname: str) -> None:
        """Set the class name of the active sphere."""
        if self.has_active_sphere():
            self.get_active_sphere().set_classname(classname)

    def update_position(self, parameter: str, value: float) -> None:
        """Update a position parameter of the active sphere."""
        if not self.has_active_sphere():
            return

        sphere = self.get_active_sphere()
        center = sphere.get_center().copy()

        if parameter == "pos_x":
            center[0] = value
        elif parameter == "pos_y":
            center[1] = value
        elif parameter == "pos_z":
            center[2] = value

        sphere.set_center(center[0], center[1], center[2])

    def update_radius(self, value: float) -> None:
        """Update the radius of the active sphere."""
        if self.has_active_sphere():
            self.get_active_sphere().set_radius(value)

    def assign_point_label_in_active_sphere(self) -> None:
        """Assign label to points inside the active sphere."""
        if not self.has_active_sphere() or self.pcd_manager.pointcloud is None:
            return

        sphere = self.get_active_sphere()
        points = self.pcd_manager.pointcloud.points
        points_inside = sphere.is_inside(points)

        # Use pcd_manager to assign labels
        self.pcd_manager.assign_point_label_in_sphere(sphere, points_inside)

    def update_all(self) -> None:
        """Update UI to reflect changes in spheres."""
        if self.view:
            # Update sphere list in UI
            self.update_label_list()

            # Update sphere properties display
            if self.has_active_sphere():
                sphere = self.get_active_sphere()
                self.view.update_bbox_stats(sphere)

    def update_label_list(self) -> None:
        """Updates the list of drawn labels and highlights the active label.

        Should be always called if the spheres changed.
        :return: None
        """
        self.view.label_list.blockSignals(True)  # To brake signal loop
        self.view.label_list.clear()
        for sphere in self.spheres:  # Changed from self.bboxes to self.spheres
            self.view.label_list.addItem(sphere.get_classname())
        if (
            self.has_active_sphere()
        ):  # Changed from has_active_bbox to has_active_sphere
            self.view.label_list.setCurrentRow(
                self.active_sphere_id
            )  # Changed from active_bbox_id to active_sphere_id
            current_item = self.view.label_list.currentItem()
            if current_item:
                current_item.setSelected(True)
        self.view.label_list.blockSignals(False)
