# labelCloud/model/sphere.py
import logging
from typing import List, Optional, Tuple

import numpy as np
import OpenGL.GL as GL
from OpenGL import GLU

from ..definitions import Color3f
from ..io.labels.config import LabelConfig


class Sphere:
    """A sphere for labeling points in a point cloud."""

    def __init__(self, center=None, radius=1.0, label=""):
        """Initialize a sphere for labeling.

        Args:
            center: 3D coordinates of sphere center [x, y, z]
            radius: Radius of the sphere
            label: Class label for the sphere
        """
        self.center = (
            np.array(center) if center is not None else np.array([0.0, 0.0, 0.0])
        )
        self.radius = max(radius, 0.01)  # Ensure minimum radius
        self.classname = label  # Using the same attribute name as BoundingBox
        self.id = None  # Will be assigned when added to collection
        self.color = (
            LabelConfig().get_class_color(label) if label else Color3f(0.7, 0.7, 0.7)
        )
        self.selected = False

        # For visualization
        self._sphere_slices = 32
        self._sphere_stacks = 32

        # Create quadric for sphere rendering
        self._quadric = GLU.gluNewQuadric()
        GLU.gluQuadricNormals(self._quadric, GLU.GLU_SMOOTH)
        GLU.gluQuadricTexture(self._quadric, GL.GL_TRUE)

    def draw(self) -> None:
        """Draw the sphere in OpenGL."""
        GL.glPushMatrix()

        try:
            # Set color based on selection state
            if self.selected:
                # Use a highlight color for selected sphere (green)
                GL.glColor3f(0.0, 1.0, 0.0)  # Bright green
                GL.glLineWidth(3.0)  # Thicker lines for selected sphere
            else:
                # Use the normal color for non-selected spheres
                if isinstance(self.color, Color3f):
                    GL.glColor3f(self.color.r, self.color.g, self.color.b)
                elif isinstance(self.color, tuple) and len(self.color) >= 3:
                    GL.glColor3f(*self.color[:3])
                else:
                    # Default color if none set (light blue)
                    GL.glColor3f(0.3, 0.3, 1.0)
                GL.glLineWidth(1.0)

            # Translate to sphere center
            GL.glTranslatef(self.center[0], self.center[1], self.center[2])

            # Create quadric object for sphere
            quadric = GLU.gluNewQuadric()
            GLU.gluQuadricDrawStyle(quadric, GLU.GLU_LINE)  # Wireframe style

            # Draw the sphere
            GLU.gluSphere(
                quadric, self.radius, 16, 16
            )  # 16 slices and stacks for better performance

            # Clean up
            GLU.gluDeleteQuadric(quadric)

        finally:
            # Always restore the matrix state
            GL.glPopMatrix()
            # Reset line width
            GL.glLineWidth(1.0)

    def translate(self, vector: np.ndarray) -> None:
        """Move the sphere center by the given vector."""
        self.center += vector

    def set_center(self, x: float, y: float, z: float) -> None:
        """Set the sphere center coordinates."""
        self.center = np.array([x, y, z])

    def get_center(self) -> np.ndarray:
        """Get the sphere center coordinates."""
        return self.center

    def set_radius(self, radius: float) -> None:
        """Set the sphere radius."""
        self.radius = max(radius, 0.01)  # Ensure minimum radius

    def change_radius(self, delta: float) -> None:
        """Change the sphere radius by a delta amount."""
        self.radius = max(self.radius + delta, 0.01)

    def set_classname(self, classname: str) -> None:
        """Set the class name for this sphere."""
        self.classname = classname
        self.color = LabelConfig().get_class_color(classname)

    def get_classname(self) -> str:
        """Get the class name of this sphere."""
        return self.classname

    def get_volume(self) -> float:
        """Get the volume of the sphere."""
        return (4 / 3) * np.pi * (self.radius**3)

    def is_inside(self, points: np.ndarray) -> np.ndarray:
        """Check which points are inside the sphere.

        Args:
            points: Nx3 array of points

        Returns:
            Boolean mask where True indicates point is inside sphere
        """
        # Calculate squared distances from each point to sphere center
        squared_distances = np.sum((points - self.center) ** 2, axis=1)

        # Return boolean mask (True for points inside sphere)
        return squared_distances <= (self.radius**2)

    def to_dict(self) -> dict:
        """Convert sphere to dictionary for serialization."""
        return {
            "center": self.center.tolist(),
            "radius": float(self.radius),
            "classname": self.classname,  # Match BoundingBox attribute name
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Sphere":
        """Create sphere from dictionary."""
        sphere = cls(
            center=np.array(data["center"]),
            radius=data["radius"],
            label=data["classname"],  # Match BoundingBox attribute name
        )
        sphere.id = data.get("id")
        return sphere

    def __del__(self):
        """Clean up OpenGL resources."""
        try:
            GLU.gluDeleteQuadric(self._quadric)
        except:
            pass
