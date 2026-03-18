# depth data source abstractions

from .common import *
from . import plane, occu

import numpy as np
import h5py

from multiprocessing import Pool
from typing import cast
import random


class DepthSource:
    """Generic data source for for depth segmentation"""

    def timestamp(self) -> int:
        return 0

    def update(self):
        pass

    def image(self) -> np.ndarray:
        return np.empty((1, 1))

    def depth_map(self) -> np.ndarray:
        return np.empty((1, 1))

    def intrinsics(self) -> Intrinsics:
        return Intrinsics(1, 1, 1, 1)

    def about(self) -> str:
        return "generic depth source, using this will break things"


class HDF5Source(DepthSource):
    """Provides a data source from the new multi-camera hdf5 format"""

    def __init__(self, file: h5py.File, dataset_index=0):
        self.file = file

        self.info = cast(h5py.Dataset, self.file['inf' + str(dataset_index)])
        self.timestamps = cast(
            h5py.Dataset, self.file['tim' + str(dataset_index)])
        self.images = cast(h5py.Dataset, self.file['img' + str(dataset_index)])
        self.depth_maps = cast(
            h5py.Dataset, self.file['dep' + str(dataset_index)])

        h, w = self.depth_maps[0].shape
        self._intrinsics = Intrinsics(cx=self.info['cx_left'][()] * w,
                                      cy=self.info['cy_left'][()] * h,
                                      fx=self.info['fx_left'][()] * w,
                                      fy=self.info['fy_left'][()] * h,
                                      tx=self.info['tx'][()])

        self.frame_count = len(self.timestamps)
        self.frame_number = 0
        self.update()

    def update(self):
        self._timestamp = int(self.timestamps[self.frame_number][()])
        self._image = np.array(self.images[self.frame_number])
        self._depth_map = np.array(self.depth_maps[self.frame_number])

    def use_frame(self, frame=-1):
        if frame < 0 or frame >= self.frame_count:
            self.frame_number = random.randint(0, self.frame_count - 1)
        else:
            self.frame_number = frame
        self.update()

        return self.frame_number

    def timestamp(self):
        return self._timestamp

    def image(self):
        return self._image

    def depth_map(self):
        return self._depth_map

    def intrinsics(self):
        return self._intrinsics

    def about(self):
        return f"hdf5 depth source"


class DepthSegementation:
    """Handles all depth segmentation processing. Requires sources to be `.update()`d before calling `process()`"""

    def __init__(self, sources: list[tuple[DepthSource, CameraPosition]], grid_conf: GridConfiguration, processes=4):
        self.grid_conf = grid_conf

        self.sources = sources
        self.guesses = [np.array([0.0, 0.0, 0.0], dtype=float)
                        for _ in sources]
        self.masks = [np.array([]) for _ in sources]
        self.grids = [np.array([]) for _ in sources]

        self.processes = processes

        self.pool = Pool(processes) if processes > 0 else None

    def process(self):
        index = 0
        for source, position in self.sources:
            hsv_mask = plane.hsv_mask(source.image())
            depth_map = plane.clean_depths(source.depth_map())
            ground_mask, px_coeffs = plane.ground_plane(
                depth_map, 200, (1, 16), 0.12, self.guesses[index],
                self.pool, self.processes)
            lane_mask = plane.merge_masks(ground_mask, hsv_mask)

            real_coeffs = plane.real_coeffs(px_coeffs, source.intrinsics())
            occ = occu.occ_grid(lane_mask, real_coeffs,
                                source.intrinsics(), self.grid_conf, position)

            self.guesses[index] = px_coeffs
            self.masks[index] = lane_mask
            self.grids[index] = occ
            index += 1

    def reduce(self, strategy=np.maximum):
        if len(self.grids) == 0:
            return np.array([])

        return strategy.reduce(self.grids)

    def merged_grid(self):
        seen = np.where(self.reduce(np.maximum) == 255, 255, 127)
        blocked = np.logical_and(self.reduce(np.minimum) == 0, seen != 255)
        return np.where(blocked, 0, seen)
