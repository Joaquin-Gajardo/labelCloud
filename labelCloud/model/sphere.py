# labelCloud/model/sphere.py
import logging
from typing import List, Optional, Tuple

import numpy as np
import OpenGL.GL as GL
from OpenGL import GLU

from ..definitions import Color3f
from ..utils.color import Color


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
        self.color = Color.from_label(label) if label else Color(0.7, 0.7, 0.7)
        self.selected = False

        # For visualization
        self._sphere_slices = 32
        self._sphere_stacks = 32

        # Create quadric for sphere rendering
        self._quadric = GLU.gluNewQuadric()
        GLU.gluQuadricNormals(self._quadric, GLU.GLU_SMOOTH)
        GLU.gluQuadricTexture(self._quadric, GL.GL_TRUE)

    def draw(self) -> None:
        """Draw the sphere using OpenGL."""
        GL.glPushMatrix()

        # Apply transformations
        GL.glTranslatef(self.center[0], self.center[1], self.center[2])

        # Set color based on selection state
        if self.selected:
            GL.glColor3f(1.0, 0.5, 0.0)  # Orange for selected sphere
        else:
            GL.glColor3f(self.color.r, self.color.g, self.color.b)

        # Draw sphere
        GLU.gluSphere(
            self._quadric, self.radius, self._sphere_slices, self._sphere_stacks
        )

        # Draw wireframe if selected
        if self.selected:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            GL.glLineWidth(2.0)
            GL.glColor3f(1.0, 1.0, 1.0)  # White wireframe
            GLU.gluSphere(
                self._quadric,
                self.radius * 1.01,
                self._sphere_slices,
                self._sphere_stacks,
            )
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            GL.glLineWidth(1.0)

        GL.glPopMatrix()

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
        self.color = Color.from_label(classname)

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
