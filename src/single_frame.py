import ransac as rsc

import matplotlib.pyplot as plt
import h5py
import cv2

from multiprocessing import Pool
from typing import cast
import time
import random
import math


def get_frame(filename="res/perspective_test.svo2.hdf5",  frame_number=-1):
    f = h5py.File(filename, "r")

    images = cast(h5py.Dataset, f["images"])
    depth_maps = cast(h5py.Dataset, f["depth_maps"])
    frames = len(depth_maps)

    if frame_number < 0:
        frame_number = random.randint(1, frames - 2)
    elif frame_number >= frames:
        frame_number = frames - 1

    depths = depth_maps[frame_number]
    image = images[frame_number]
    image = image[:, 0: int(image.shape[1] / 2)]

    print()
    print("testing on:", filename)
    print(".hdf5 keys:", list(f.keys()))
    print(f"using frame number: {frame_number}")
    print("\n----- start -----\n")

    f.close()
    return image, depths


def do_ransac(image, depths, pool=None, procs=4):

    # guess camera intrinsics
    h, w = depths.shape
    fx = 360

    # generate the mask
    hsv_mask = rsc.hsv_mask(image)
    depths = rsc.clean_depths(depths)
    ground_mask, px_coeffs = rsc.ground_plane(
        depths, thread_pool=pool, processes=procs)
    lane_mask = rsc.merge_masks(ground_mask, hsv_mask)

    # find real plane equation
    intr = rsc.Intrinsics(w / 2, h / 2, fx, fx)
    real_coeffs = rsc.real_coeffs(px_coeffs, intr)

    # create the occupancy grid for one camera
    conf = rsc.GridConfiguration(5000.0, 5000.0, 50.0)
    pos = rsc.CameraPosition(0, 0, math.radians(0))
    occ = rsc.occ_grid(lane_mask, real_coeffs, intr, conf, pos)

    return lane_mask, occ


def display(lane_mask, occupancy_grid):
    lane_img = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
    occ_img = cv2.cvtColor(occupancy_grid, cv2.COLOR_GRAY2BGR)

    f, ax = plt.subplots(2, 1)
    ax[0].set_title("ransac mask")
    ax[0].imshow(lane_img)
    ax[1].set_title("occupancy grid")
    ax[1].imshow(occ_img)

    plt.show()


def main():
    image, depths = get_frame()
    procs = 4
    pool = Pool(procs)

    start = time.perf_counter_ns()
    lane_mask, occupancy_grid = do_ransac(image, depths, pool, procs)
    end = time.perf_counter_ns()

    print(f"\n----- {(end - start) / 1e6:.0f} ms -----\n")
    display(lane_mask, occupancy_grid)


if __name__ == "__main__":
    main()
