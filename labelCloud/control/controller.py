import logging
from typing import Optional

import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QPoint
from PyQt5.QtCore import Qt as Keys

from ..definitions import BBOX_SIDES, Colors, Context, LabelingMode
from ..io.labels.config import LabelConfig
from ..labeling_strategies.picking import PickingStrategy
from ..labeling_strategies.sphere_picking import SpherePickingStrategy
from ..model.sphere import Sphere
from ..utils import oglhelper
from ..utils.math3d import points_in_sphere
from ..view.gui import GUI
from .alignmode import AlignMode
from .bbox_controller import BoundingBoxController
from .config_manager import config
from .drawing_manager import DrawingManager
from .pcd_manager import PointCloudManger
from .sphere_controller import SphereController


class Controller:
    MOVEMENT_THRESHOLD = 0.1

    def __init__(self) -> None:
        """Initializes all controllers and managers."""
        self.view: "GUI"
        self.pcd_manager = PointCloudManger()
        self.bbox_controller = BoundingBoxController()
        self.sphere_controller = SphereController()

        # Drawing states
        self.drawing_mode = DrawingManager(self.bbox_controller, self.sphere_controller)
        self.align_mode = AlignMode(self.pcd_manager)

        # Control states
        self.curr_cursor_pos: Optional[QPoint] = None  # updated by mouse movement
        self.last_cursor_pos: Optional[QPoint] = None  # updated by mouse click
        self.ctrl_pressed = False
        self.scroll_mode = False  # to enable the side-pulling

        # Correction states
        self.side_mode = False
        self.selected_side: Optional[str] = None

        self.primitive_type = "box"  # Either "box" or "sphere"

    def startup(self, view: "GUI") -> None:
        """Sets the view in all controllers and dependent modules; Loads labels from file."""
        self.view = view
        self.bbox_controller.set_view(self.view)
        self.sphere_controller.set_view(self.view)
        self.pcd_manager.set_view(self.view)
        self.drawing_mode.set_view(self.view)
        self.align_mode.set_view(self.view)
        self.view.gl_widget.set_bbox_controller(self.bbox_controller)
        self.view.gl_widget.set_sphere_controller(self.sphere_controller)
        self.bbox_controller.pcd_manager = self.pcd_manager
        self.sphere_controller.pcd_manager = self.pcd_manager

        # Read labels from folders
        self.pcd_manager.read_pointcloud_folder()
        self.next_pcd(save=False)

    def loop_gui(self) -> None:
        """Function collection called during each event loop iteration."""
        self.set_crosshair()
        self.set_selected_side()
        self.view.gl_widget.updateGL()

    def set_primitive_type(self, strategy) -> None:
        """Update controller's primitive type based on strategy type"""

        if isinstance(strategy, SpherePickingStrategy):
            self.primitive_type = "sphere"
        else:  # Default to box for PickingStrategy and SpanningStrategy
            self.primitive_type = "box"

    def set_drawing_strategy(self, strategy):
        """Wrapper to set drawing strategy and synchronize primitive type.

        This ensures the controller and drawing manager are in sync.
        """
        # Set the drawing strategy in the drawing manager
        self.drawing_mode.set_drawing_strategy(strategy)
        # Update the primitive type in the controller
        self.set_primitive_type(strategy)

    # POINT CLOUD METHODS
    def next_pcd(self, save: bool = True) -> None:
        if save:
            self.save()
        if self.pcd_manager.pcds_left():
            previous_bboxes = self.bbox_controller.bboxes
            self.pcd_manager.get_next_pcd()
            self.reset()
            self.bbox_controller.set_bboxes(self.pcd_manager.get_labels_from_file())

            if not self.bbox_controller.bboxes and config.getboolean(
                "LABEL", "propagate_labels"
            ):
                self.bbox_controller.set_bboxes(previous_bboxes)
            self.bbox_controller.set_active_bbox(0)
        else:
            self.view.update_progress(len(self.pcd_manager.pcds))
            self.view.button_next_pcd.setEnabled(False)

    def prev_pcd(self) -> None:
        self.save()
        if self.pcd_manager.current_id > 0:
            self.pcd_manager.get_prev_pcd()
            self.reset()
            self.bbox_controller.set_bboxes(self.pcd_manager.get_labels_from_file())
            self.bbox_controller.set_active_bbox(0)

    def custom_pcd(self, custom: int) -> None:
        self.save()
        self.pcd_manager.get_custom_pcd(custom)
        self.reset()
        self.bbox_controller.set_bboxes(self.pcd_manager.get_labels_from_file())

    # CONTROL METHODS
    def save(self) -> None:
        """Saves all bounding boxes, spheres and optionally segmentation labels in the label file."""
        self.pcd_manager.save_labels_into_file(
            self.bbox_controller.bboxes, self.sphere_controller.spheres
        )

        if LabelConfig().type == LabelingMode.SEMANTIC_SEGMENTATION:
            assert self.pcd_manager.pointcloud is not None
            self.pcd_manager.pointcloud.save_segmentation_labels()

    def reset(self) -> None:
        """Resets the controllers and bounding boxes from the current screen."""
        self.bbox_controller.reset()
        self.sphere_controller.reset()
        self.drawing_mode.reset()
        self.align_mode.reset()

    # CORRECTION METHODS
    def set_crosshair(self) -> None:
        """Sets the crosshair position in the glWidget to the current cursor position."""
        if self.curr_cursor_pos:
            self.view.gl_widget.crosshair_col = Colors.GREEN.value
            self.view.gl_widget.crosshair_pos = (
                self.curr_cursor_pos.x(),
                self.curr_cursor_pos.y(),
            )

    def set_selected_side(self) -> None:
        """Sets the currently hovered bounding box side in the glWidget."""
        if (
            (not self.side_mode)
            and self.curr_cursor_pos
            and self.bbox_controller.has_active_bbox()
            and (not self.scroll_mode)
        ):
            _, self.selected_side = oglhelper.get_intersected_sides(
                self.curr_cursor_pos.x(),
                self.curr_cursor_pos.y(),
                self.bbox_controller.get_active_bbox(),  # type: ignore
                self.view.gl_widget.modelview,
                self.view.gl_widget.projection,
            )
        if (
            self.selected_side
            and (not self.ctrl_pressed)
            and self.bbox_controller.has_active_bbox()
        ):
            self.view.gl_widget.crosshair_col = Colors.RED.value
            side_vertices = self.bbox_controller.get_active_bbox().get_vertices()  # type: ignore
            self.view.gl_widget.selected_side_vertices = side_vertices[
                BBOX_SIDES[self.selected_side]
            ]
            self.view.status_manager.set_message(
                "Scroll to change the bounding box dimension.",
                context=Context.SIDE_HOVERED,
            )
        else:
            self.view.gl_widget.selected_side_vertices = np.array([])
            self.view.status_manager.clear_message(Context.SIDE_HOVERED)

    # EVENT PROCESSING
    def mouse_clicked(self, a0: QtGui.QMouseEvent) -> None:
        """Triggers actions when the user clicks the mouse."""
        self.last_cursor_pos = a0.pos()

        if (
            self.primitive_type == "sphere"
            and self.drawing_mode.is_active()
            and (a0.buttons() & Keys.LeftButton)
            and (not self.ctrl_pressed)
        ):
            self.drawing_mode.register_point(a0.x(), a0.y(), correction=True)

        elif (
            self.primitive_type == "box"
            and self.drawing_mode.is_active()
            and (a0.buttons() & Keys.LeftButton)
            and (not self.ctrl_pressed)
        ):
            self.drawing_mode.register_point(a0.x(), a0.y(), correction=True)

        elif self.align_mode.is_active and (not self.ctrl_pressed):
            self.align_mode.register_point(
                self.view.gl_widget.get_world_coords(a0.x(), a0.y(), correction=False)
            )

        elif self.selected_side:
            self.side_mode = True

        else:
            # Just a click
            pass

    def mouse_double_clicked(self, a0: QtGui.QMouseEvent) -> None:
        """Triggers actions when the user double clicks the mouse."""
        if self.primitive_type == "sphere":
            self.sphere_controller.select_sphere_by_ray(a0.x(), a0.y())
        else:  # Default to box
            self.bbox_controller.select_bbox_by_ray(a0.x(), a0.y())

    def mouse_move_event(self, a0: QtGui.QMouseEvent) -> None:
        """Triggers actions when the user moves the mouse."""
        self.curr_cursor_pos = a0.pos()  # Updates the current mouse cursor position

        # Methods that use absolute cursor position
        if self.drawing_mode.is_active() and (not self.ctrl_pressed):
            self.drawing_mode.register_point(
                a0.x(), a0.y(), correction=True, is_temporary=True
            )

        elif self.align_mode.is_active and (not self.ctrl_pressed):
            self.align_mode.register_tmp_point(
                self.view.gl_widget.get_world_coords(a0.x(), a0.y(), correction=False)
            )

        if self.last_cursor_pos:
            dx = (
                self.last_cursor_pos.x() - a0.x()
            ) / 5  # Calculate relative movement from last click position
            dy = (self.last_cursor_pos.y() - a0.y()) / 5

            if (
                self.ctrl_pressed
                and (not self.drawing_mode.is_active())
                and (not self.align_mode.is_active)
            ):
                if a0.buttons() & Keys.LeftButton:
                    if (
                        self.primitive_type == "box"
                        and self.bbox_controller.has_active_bbox()
                    ):
                        # bbox rotation
                        self.bbox_controller.rotate_with_mouse(-dx, -dy)
                elif a0.buttons() & Keys.RightButton:
                    if (
                        self.primitive_type == "box"
                        and self.bbox_controller.has_active_bbox()
                    ):
                        # bbox translation
                        new_center = self.view.gl_widget.get_world_coords(
                            a0.x(), a0.y(), correction=True
                        )
                        self.bbox_controller.set_center(
                            *new_center
                        )  # absolute positioning
                    elif (
                        self.primitive_type == "sphere"
                        and self.sphere_controller.has_active_sphere()
                    ):
                        # sphere translation
                        new_center = self.view.gl_widget.get_world_coords(
                            a0.x(), a0.y(), correction=True
                        )
                        self.sphere_controller.set_center(
                            *new_center
                        )  # absolute positioning
            else:
                if a0.buttons() & Keys.LeftButton:  # pcd rotation
                    self.pcd_manager.rotate_around_x(dy)
                    self.pcd_manager.rotate_around_z(dx)
                elif a0.buttons() & Keys.RightButton:  # pcd translation
                    self.pcd_manager.translate_along_x(dx)
                    self.pcd_manager.translate_along_y(dy)

            # Reset scroll locks of "side scrolling" for significant cursor movements
            if dx > Controller.MOVEMENT_THRESHOLD or dy > Controller.MOVEMENT_THRESHOLD:
                if self.side_mode:
                    self.side_mode = False
                else:
                    self.scroll_mode = False
        self.last_cursor_pos = a0.pos()

    def mouse_scroll_event(self, a0: QtGui.QWheelEvent) -> None:
        """Triggers actions when the user scrolls the mouse wheel."""
        if self.selected_side:
            self.side_mode = True

        if (
            self.drawing_mode.is_active()
            and (not self.ctrl_pressed)
            and self.drawing_mode.drawing_strategy is not None
        ):
            self.drawing_mode.drawing_strategy.register_scrolling(a0.angleDelta().y())
        elif self.side_mode and self.bbox_controller.has_active_bbox():
            self.bbox_controller.get_active_bbox().change_side(  # type: ignore
                self.selected_side, -a0.angleDelta().y() / 4000  # type: ignore
            )
        elif (
            self.primitive_type == "sphere"
            and self.sphere_controller.has_active_sphere()
            and not self.scroll_mode
        ):
            # Check if cursor is near the active sphere (for bbox it's handled with side_mode)
            sphere = self.sphere_controller.get_active_sphere()
            if self.cursor_near_sphere(sphere):
                # Adjust active sphere's radius with scroll wheel
                delta = a0.angleDelta().y() / 1000  # Smaller value for finer control
                self.sphere_controller.adjust_radius(increase=(delta > 0))
                # Update the UI to reflect the changes
                self.view.update_bbox_stats(self.sphere_controller.get_active_sphere())
            else:
                # If cursor is not near sphere, perform default zoom
                self.pcd_manager.zoom_into(a0.angleDelta().y())
                self.scroll_mode = True
        else:
            self.pcd_manager.zoom_into(a0.angleDelta().y())
            self.scroll_mode = True

    def cursor_near_sphere(self, sphere) -> bool:
        """Check if cursor is near the given sphere."""
        if not self.curr_cursor_pos or not sphere:
            return False

        # Get world coords at cursor position
        world_pos = self.view.gl_widget.get_world_coords(
            self.curr_cursor_pos.x(), self.curr_cursor_pos.y(), correction=False
        )

        # Check if cursor is within certain distance of sphere
        distance = np.linalg.norm(sphere.center - world_pos)
        # Use a slightly larger threshold than the sphere radius for easier interaction
        return distance <= sphere.radius * 1.5

    def key_press_event(self, a0: QtGui.QKeyEvent) -> None:
        """Triggers actions when the user presses a key."""

        # Reset position to intial value
        if a0.key() == Keys.Key_Control:
            self.ctrl_pressed = True
            self.view.status_manager.set_message(
                "Hold right mouse button to translate or left mouse button to rotate "
                "the bounding box.",
                context=Context.CONTROL_PRESSED,
            )
        # Reset point cloud pose to intial rotation and translation
        elif a0.key() in [Keys.Key_P, Keys.Key_Home]:
            self.pcd_manager.reset_transformations()
            logging.info("Reseted position to default.")

        elif a0.key() == Keys.Key_Delete:  # Delete active bbox or sphere
            if (
                self.primitive_type == "sphere"
                and self.sphere_controller.has_active_sphere()
            ):
                self.sphere_controller.delete_current_sphere()
            elif self.primitive_type == "box":
                self.bbox_controller.delete_current_bbox()

        # Save labels to file
        elif a0.key() == Keys.Key_S and self.ctrl_pressed:
            self.save()

        elif a0.key() == Keys.Key_Escape:
            if self.drawing_mode.is_active():
                self.drawing_mode.reset()
                logging.info("Resetted drawn points!")
            elif self.align_mode.is_active:
                self.align_mode.reset()
                logging.info("Resetted selected points!")

        # Toggle between box and sphere mode
        elif a0.key() == Keys.Key_M:
            new_type = (
                SpherePickingStrategy(self.view)
                if self.primitive_type != "sphere"
                else PickingStrategy(self.view)
            )
            self.set_drawing_strategy(new_type)
            logging.info(f"Switched to {self.primitive_type} creation mode")

        # BBOX MANIPULATION
        elif self.primitive_type == "box" and self.bbox_controller.has_active_bbox():
            if a0.key() == Keys.Key_Z:
                # z rotate counterclockwise
                self.bbox_controller.rotate_around_z()
            elif a0.key() == Keys.Key_X:
                # z rotate clockwise
                self.bbox_controller.rotate_around_z(clockwise=True)
            elif a0.key() == Keys.Key_C:
                # y rotate counterclockwise
                self.bbox_controller.rotate_around_y()
            elif a0.key() == Keys.Key_V:
                # y rotate clockwise
                self.bbox_controller.rotate_around_y(clockwise=True)
            elif a0.key() == Keys.Key_B:
                # x rotate counterclockwise
                self.bbox_controller.rotate_around_x()
            elif a0.key() == Keys.Key_N:
                # x rotate clockwise
                self.bbox_controller.rotate_around_x(clockwise=True)
            elif a0.key() == Keys.Key_W:
                # move backward
                self.bbox_controller.translate_along_y()
            elif a0.key() == Keys.Key_S:
                # move forward
                self.bbox_controller.translate_along_y(forward=True)
            elif a0.key() == Keys.Key_A:
                # move left
                self.bbox_controller.translate_along_x(left=True)
            elif a0.key() == Keys.Key_D:
                # move right
                self.bbox_controller.translate_along_x()
            elif a0.key() == Keys.Key_Q:
                # move up
                self.bbox_controller.translate_along_z()
            elif a0.key() == Keys.Key_E:
                # move down
                self.bbox_controller.translate_along_z(down=True)

            # BBOX Scaling
            elif a0.key() == Keys.Key_I:
                # increase length
                self.bbox_controller.scale_along_length()
            elif a0.key() == Keys.Key_O:
                # decrease length
                self.bbox_controller.scale_along_length(decrease=True)
            elif a0.key() == Keys.Key_K:
                # increase width
                self.bbox_controller.scale_along_width()
            elif a0.key() == Keys.Key_L:
                # decrease width
                self.bbox_controller.scale_along_width(decrease=True)
            elif a0.key() == Keys.Key_Comma:
                # increase height
                self.bbox_controller.scale_along_height()
            elif a0.key() == Keys.Key_Period:
                # decrease height
                self.bbox_controller.scale_along_height(decrease=True)

            elif a0.key() in [Keys.Key_R, Keys.Key_Left]:
                # load previous sample
                self.prev_pcd()
            elif a0.key() in [Keys.Key_F, Keys.Key_Right]:
                # load next sample
                self.next_pcd()
            elif a0.key() in [Keys.Key_T, Keys.Key_Up]:
                # select previous bbox
                self.select_relative_bbox(-1)
            elif a0.key() in [Keys.Key_G, Keys.Key_Down]:
                # select previous bbox
                self.select_relative_bbox(1)
            elif a0.key() == Keys.Key_Y:
                # change bbox class to previous available class
                self.select_relative_class(-1)
            elif a0.key() == Keys.Key_H:
                # change bbox class to next available class
                self.select_relative_class(1)
            elif a0.key() in list(range(49, 58)):
                # select bboxes with 1-9 digit keys
                self.bbox_controller.set_active_bbox(int(a0.key()) - 49)

            # SPHERE MANIPULATION
            # Handle translation keys for spheres
            elif (
                self.primitive_type == "sphere"
                and self.sphere_controller.has_active_sphere()
            ):
                if a0.key() == Keys.Key_W:
                    # move backward
                    self.sphere_controller.translate_along_y()
                elif a0.key() == Keys.Key_S:
                    # move forward
                    self.sphere_controller.translate_along_y(forward=True)
                elif a0.key() == Keys.Key_A:
                    # move left
                    self.sphere_controller.translate_along_x(left=True)
                elif a0.key() == Keys.Key_D:
                    # move right
                    self.sphere_controller.translate_along_x()
                elif a0.key() == Keys.Key_Q:
                    # move up
                    self.sphere_controller.translate_along_z()
                elif a0.key() == Keys.Key_E:
                    # move down
                    self.sphere_controller.translate_along_z(down=True)
                elif a0.key() == Keys.Key_Plus or a0.key() == Keys.Key_Equal:
                    # increase radius
                    self.sphere_controller.adjust_radius(increase=True)
                elif a0.key() == Keys.Key_Minus:
                    # decrease radius
                    self.sphere_controller.adjust_radius(increase=False)
                # Add these new controls
                elif a0.key() in [Keys.Key_R, Keys.Key_Left]:
                    # load previous sample
                    self.prev_pcd()
                elif a0.key() in [Keys.Key_F, Keys.Key_Right]:
                    # load next sample
                    self.next_pcd()
                elif a0.key() in [Keys.Key_T, Keys.Key_Up]:
                    # select previous sphere
                    self.select_relative_sphere(-1)
                elif a0.key() in [Keys.Key_G, Keys.Key_Down]:
                    # select next sphere
                    self.select_relative_sphere(1)
                elif a0.key() == Keys.Key_Y:
                    # change sphere class to previous available class
                    self.select_relative_class_for_sphere(-1)
                elif a0.key() == Keys.Key_H:
                    # change sphere class to next available class
                    self.select_relative_class_for_sphere(1)
                elif a0.key() in list(range(49, 58)):
                    # select spheres with 1-9 digit keys
                    if int(a0.key()) - 49 < len(self.sphere_controller.spheres):
                        self.sphere_controller.set_active_sphere(int(a0.key()) - 49)

    def select_relative_class(self, step: int):
        if step == 0:
            return
        curr_class = self.bbox_controller.get_active_bbox().get_classname()  # type: ignore
        new_class = LabelConfig().get_relative_class(curr_class, step)
        self.bbox_controller.get_active_bbox().set_classname(new_class)  # type: ignore
        self.bbox_controller.update_all()  # updates UI in SelectBox

    def select_relative_bbox(self, step: int):
        if step == 0:
            return
        max_id = len(self.bbox_controller.bboxes) - 1
        curr_id = self.bbox_controller.active_bbox_id
        new_id = curr_id + step
        corner_case_id = 0 if step > 0 else max_id
        new_id = new_id if new_id in range(max_id + 1) else corner_case_id
        self.bbox_controller.set_active_bbox(new_id)

    def key_release_event(self, a0: QtGui.QKeyEvent) -> None:
        """Triggers actions when the user releases a key."""
        if a0.key() == Keys.Key_Control:
            self.ctrl_pressed = False
            self.view.status_manager.clear_message(Context.CONTROL_PRESSED)

    def crop_pointcloud_inside_active_bbox(self) -> None:
        bbox = self.bbox_controller.get_active_bbox()
        assert bbox is not None
        assert self.pcd_manager.pointcloud is not None
        points_inside = bbox.is_inside(self.pcd_manager.pointcloud.points)
        pointcloud = self.pcd_manager.pointcloud.get_filtered_pointcloud(points_inside)
        if pointcloud is None:
            logging.warning("No points found inside the box. Ignored.")
            return
        self.view.save_point_cloud_as(pointcloud)

    def select_points_in_sphere(self, center: np.ndarray, radius: float) -> None:
        """Select points within sphere for current point cloud.

        Args:
            center: Center point of sphere (x,y,z)
            radius: Radius of sphere in world coordinates
        """
        if self.pcd_manager.pointcloud is None:
            logging.warning("No point cloud loaded")
            return

        # Get points inside sphere
        points = self.pcd_manager.pointcloud.points
        points_inside = points_in_sphere(points, center, radius)

        # Update labels for selected points
        if self.has_active_sphere():
            sphere = self.get_active_sphere()
            self.pcd_manager.assign_point_label_in_sphere(sphere, points_inside)
            logging.info(f"Selected {np.sum(points_inside)} points in sphere")

    def has_active_sphere(self) -> bool:
        """Check if there is an active sphere."""
        return self.sphere_controller.has_active_sphere()

    def get_active_sphere(self) -> Optional[Sphere]:
        """Get the active sphere if any."""
        return self.sphere_controller.get_active_sphere()

    def select_relative_sphere(self, step: int):
        """Select the previous or next sphere relative to the currently active one."""
        if step == 0 or not self.sphere_controller.spheres:
            return

        max_id = len(self.sphere_controller.spheres) - 1
        curr_id = self.sphere_controller.active_sphere_id

        if curr_id is None:
            # If no sphere is selected, select the first one
            new_id = 0
        else:
            new_id = curr_id + step
            # Handle wrap-around
            if new_id < 0:
                new_id = max_id
            elif new_id > max_id:
                new_id = 0

        self.sphere_controller.set_active_sphere(new_id)

    def select_relative_class_for_sphere(self, step: int):
        """Change the class of the active sphere to the previous or next available class."""
        if step == 0 or not self.sphere_controller.has_active_sphere():
            return

        sphere = self.sphere_controller.get_active_sphere()
        curr_class = sphere.get_classname()
        new_class = LabelConfig().get_relative_class(curr_class, step)

        sphere.set_classname(new_class)
        self.sphere_controller.update_all()  # Update UI

    def handle_deselect_button_clicked(self) -> None:
        """Handle deselect button click for active primitive."""
        if (
            self.primitive_type == "sphere"
            and self.sphere_controller.has_active_sphere()
        ):
            self.sphere_controller.deselect_sphere()
        else:
            self.bbox_controller.deselect_bbox()

    def handle_delete_button_clicked(self) -> None:
        """Handle delete button click for active primitive."""
        if (
            self.primitive_type == "sphere"
            and self.sphere_controller.has_active_sphere()
        ):
            self.sphere_controller.delete_current_sphere()
        else:
            self.bbox_controller.delete_current_bbox()
