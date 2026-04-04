# depth data source abstractions

from .common import *
from . import plane, occu

import numpy as np
import h5py
import cv2

from multiprocessing import Pool
from typing import cast
import random

using_zed = True
try:
    import pyzed.sl as sl
    import threading
except ImportError:
    print("[warn] pyzed not found, LiveSource will not work")
    using_zed = False


class DepthSource:
    """Generic data source for for depth segmentation"""

    def timestamp(self) -> int:
        return 0

    def update(self) -> bool:
        return False

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

    def __delete__(self):
        self.file.close()

    def update(self) -> bool:
        self._timestamp = int(self.timestamps[self.frame_number][()])
        self._image = np.array(self.images[self.frame_number])
        self._depth_map = np.array(self.depth_maps[self.frame_number])
        return True

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


class LiveSource(DepthSource):
    """Provides depth data from the zed camera.

    `update()` is blocking so should be called in a thread if needing async updates

    example:
    ````
    def grab(source: DepthSource):
        while source.update():
            time.sleep(0.001)

    grab_thread = threading.Thread(target=grab, args=(source,))
    ```
    """

    def __init__(self, params: sl.InitParameters | None = None,
                 max_res: tuple[int, int] | None = None):
        self._timestamp = 0
        self._image = np.empty((1, 1))
        self._depth_map = np.empty((1, 1))

        if using_zed:
            self.cam = sl.Camera()
            self._image_mat = sl.Mat()
            self._depth_map_mat = sl.Mat()

            status = self.cam.open(params)
            if status != sl.ERROR_CODE.SUCCESS:
                print(repr(status))
                self.cam.close()

            cam_conf = self.cam.get_camera_information().camera_configuration
            left_calib = cam_conf.calibration_parameters.left_cam
            self.res = cam_conf.resolution

            # intrinsics as proportions of resolution
            fx = left_calib.fx / self.res.width
            fy = left_calib.fy / self.res.height
            cx = left_calib.cx / self.res.width
            cy = left_calib.cy / self.res.height
            tx = cam_conf.calibration_parameters.stereo_transform \
                .get_translation().get()[0]

            if max_res:
                self.res = sl.Resolution(min(max_res[0], self.res.width),
                                         min(max_res[1], self.res.height))

            # scale intrinsics back up
            self._intrinsics = Intrinsics(cx=cx * self.res.width,
                                          cy=cy * self.res.height,
                                          fx=fx * self.res.width,
                                          fy=fy * self.res.height,
                                          tx=tx)
        else:
            print("[warn] LiveSource should not be initialised without the Zed SDK")
            pass  # ! TODO: fill this out?

    def __delete__(self):
        if using_zed:
            self.cam.close()

    def update(self) -> bool:
        if not using_zed:
            return False

        runtime = sl.RuntimeParameters()
        err = self.cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            self.cam.retrieve_image(
                self._image_mat, sl.VIEW.LEFT, sl.MEM.CPU, self.res)
            self.cam.retrieve_measure(
                self._depth_map_mat, sl.MEASURE.DEPTH, sl.MEM.CPU, self.res)
            self._timestamp = \
                self.cam.get_timestamp(sl.TIME_REFERENCE.CURRENT).data_ns
            return True

        print("[error] while grabbing frame:", err)
        return False

    def timestamp(self) -> int:
        return self._timestamp

    def image(self) -> np.ndarray:
        return self._image_mat.get_data()

    def depth_map(self) -> np.ndarray:
        return self._depth_map_mat.get_data()

    def intrinsics(self) -> Intrinsics:
        return self._intrinsics

    def about(self) -> str:
        return f"live depth source ({'active' if using_zed else 'inactive'})"


class BasicHSV:
    def __init__(self, lower=np.array([0, 0, 180], dtype=np.uint8),
                 upper=np.array([255, 50, 255], dtype=np.uint8)):
        self.lower = lower
        self.upper = upper

    def __call__(self, image: np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(image, self.lower, self.upper)
        return 255 * mask.astype(np.uint8)


class DepthSegementation:
    """Handles all depth segmentation processing. Requires sources to be `.update()`d before calling `process()`"""

    def __init__(self, sources: list[tuple[DepthSource, CameraPosition]],
                 grid_conf: GridConfiguration, processes=4, *args, mask_method=BasicHSV(), **kwargs):
        self.grid_conf = grid_conf
        self.mask_method = mask_method # keyword-only

        self._sources = sources
        self._guesses = [np.array([0.0, 0.0, 0.0], dtype=float)
                         for _ in sources]

        self.timestamps = [0 for _ in sources]
        self.masks = [np.array([]) for _ in sources]
        self.grids = [np.array([]) for _ in sources]

        self._processes = processes
        self._pool = Pool(processes) if processes > 0 else None

    def process(self) -> bool:
        updated = False

        index = -1
        for source, position in self._sources:
            index += 1
            if self.timestamps[index] != source.timestamp():
                self.timestamps[index] = source.timestamp()
                updated = True
            else:
                continue
            hsv_mask = self.mask_method(source.image())
            depth_map = plane.clean_depths(source.depth_map())
            ground_mask, px_coeffs = plane.ground_plane(
                depth_map, 200, (1, 16), 0.12, self._guesses[index],
                self._pool, self._processes)
            lane_mask = plane.merge_masks(ground_mask, hsv_mask)

            real_coeffs = plane.real_coeffs(px_coeffs, source.intrinsics())
            occ = occu.occ_grid(lane_mask, real_coeffs,
                                source.intrinsics(), self.grid_conf, position)

            self._guesses[index] = px_coeffs
            self.masks[index] = lane_mask
            self.grids[index] = occ

        return updated

    def merge_simple(self, strategy=np.maximum):
        if len(self.grids) == 0:
            return np.array([])

        return strategy.reduce(self.grids)

    def merge_grids(self):
        seen = np.where(self.merge_simple(np.maximum) == 255, 255, 127)
        blocked = np.logical_and(
            self.merge_simple(np.minimum) == 0, seen != 255)
        return np.where(blocked, 0, seen)
