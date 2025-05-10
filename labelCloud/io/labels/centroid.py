import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...model import BBox
from ...model.sphere import Sphere
from . import BaseLabelFormat, abs2rel_rotation, rel2abs_rotation


class CentroidFormat(BaseLabelFormat):
    FILE_ENDING = ".json"

    def import_labels(self, pcd_path: Path) -> List[BBox]:
        labels = []

        label_path = self.label_folder.joinpath(pcd_path.stem + self.FILE_ENDING)
        if label_path.is_file():
            with label_path.open("r") as read_file:
                data = json.load(read_file)

            for label in data["objects"]:
                x = label["centroid"]["x"]
                y = label["centroid"]["y"]
                z = label["centroid"]["z"]
                length = label["dimensions"]["length"]
                width = label["dimensions"]["width"]
                height = label["dimensions"]["height"]
                bbox = BBox(x, y, z, length, width, height)
                rotations = label["rotations"].values()
                if self.relative_rotation:
                    rotations = map(rel2abs_rotation, rotations)
                bbox.set_rotations(*rotations)
                bbox.set_classname(label["name"])
                labels.append(bbox)
            logging.info(
                "Imported %s labels from %s." % (len(data["objects"]), label_path)
            )
        return labels

    def export_labels(self, bboxes: List[BBox], pcd_path: Path) -> None:
        data: Dict[str, Any] = {}
        # Header
        data["folder"] = pcd_path.parent.name
        data["filename"] = pcd_path.name
        data["path"] = str(pcd_path)

        # Labels
        data["objects"] = []
        for bbox in bboxes:
            label: Dict[str, Any] = {}
            label["name"] = bbox.get_classname()
            label["centroid"] = {
                str(axis): self.round_dec(val)
                for axis, val in zip(["x", "y", "z"], bbox.get_center())
            }
            label["dimensions"] = {
                str(dim): self.round_dec(val)
                for dim, val in zip(
                    ["length", "width", "height"], bbox.get_dimensions()
                )
            }
            conv_rotations = bbox.get_rotations()
            if self.relative_rotation:
                conv_rotations = map(abs2rel_rotation, conv_rotations)  # type: ignore

            label["rotations"] = {
                str(axis): self.round_dec(angle)
                for axis, angle in zip(["x", "y", "z"], conv_rotations)
            }
            data["objects"].append(label)

        # Save to JSON
        label_path = self.save_label_to_file(pcd_path, data)
        logging.info(
            f"Exported {len(bboxes)} labels to {label_path} "
            f"in {self.__class__.__name__} formatting!"
        )


class SphereCentroidFormat(CentroidFormat):
    """Extension of CentroidFormat that also supports spheres."""

    def import_labels(self, pcd_path: Path) -> Tuple[List[BBox], List[Sphere]]:
        """Import labels from file.

        Args:
            pcd_path: Path to the point cloud file

        Returns:
            Tuple containing (bboxes, spheres)
        """
        bboxes = super().import_labels(pcd_path)
        spheres = []

        label_path = self.get_label_path(pcd_path)
        if label_path.is_file():
            with label_path.open("r") as read_file:
                data = json.load(read_file)

            # Load spheres if present
            if "spheres" in data:
                for sphere_data in data["spheres"]:
                    center = np.array(
                        [
                            sphere_data["center"]["x"],
                            sphere_data["center"]["y"],
                            sphere_data["center"]["z"],
                        ]
                    )
                    radius = sphere_data["radius"]
                    classname = sphere_data["name"]

                    sphere = Sphere(center=center, radius=radius, label=classname)
                    spheres.append(sphere)

                logging.info(
                    "Imported %s spheres from %s." % (len(data["spheres"]), label_path)
                )

        return bboxes, spheres

    def export_labels(
        self, bboxes: List[BBox], pcd_path: Path, spheres: Optional[List[Sphere]] = None
    ) -> None:
        """Export labels to file.

        Args:
            bboxes: List of bounding boxes
            pcd_path: Path to the point cloud file
            spheres: List of spheres (optional)
        """
        if spheres is None:
            spheres = []

        # First export bounding boxes using parent method
        super().export_labels(bboxes, pcd_path)

        # If we have spheres, add them to the exported data
        if spheres:
            # Read the existing label file
            label_path = self.get_label_path(pcd_path)
            if label_path.is_file():
                with label_path.open("r") as read_file:
                    data = json.load(read_file)

                # Add spheres to the data
                data["spheres"] = []
                for sphere in spheres:
                    sphere_data = {
                        "name": sphere.get_classname(),
                        "center": {
                            "x": self.round_dec(sphere.center[0]),
                            "y": self.round_dec(sphere.center[1]),
                            "z": self.round_dec(sphere.center[2]),
                        },
                        "radius": self.round_dec(sphere.radius),
                    }
                    data["spheres"].append(sphere_data)

                # Save updated data
                with label_path.open("w") as write_file:
                    json.dump(data, write_file, indent=2)

                logging.info(
                    f"Added {len(spheres)} spheres to {label_path} "
                    f"in {self.__class__.__name__} formatting!"
                )
