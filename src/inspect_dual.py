import ransac as rsc

import matplotlib.pyplot as plt
import numpy as np
import h5py

import math


def main():
    file = h5py.File("res/dual_camera_calibration.hdf5", "r")
    left = rsc.HDF5Source(file, 0)
    right = rsc.HDF5Source(file, 1)

    conf = rsc.GridConfiguration(5000.0, 5000.0, 50.0)
    depseg = rsc.DepthSegementation(
        [(left, rsc.CameraPosition(0, 0, math.radians(20))),
         (right, rsc.CameraPosition(250, 200, 0))], conf)

    print(f"using frame number: {left.use_frame(-1)}")
    right.use_frame(left.frame_number)
    depseg.process()

    f, axs = plt.subplot_mosaic(
        [
            ['lmask', 'rmask', 'diff'],
            ['lgrid', 'rgrid', 'final']
        ])
    axs['lmask'].set_title("left mask")
    axs['lmask'].imshow(depseg.masks[0], cmap='gray')
    axs['lgrid'].set_title("left grid")
    axs['lgrid'].imshow(depseg.grids[0], cmap='gray')

    axs['rmask'].set_title("right mask")
    axs['rmask'].imshow(depseg.masks[1], cmap='gray')
    axs['rgrid'].set_title("right grid")
    axs['rgrid'].imshow(depseg.grids[1], cmap='gray')

    axs['diff'].set_title('difference grid')
    axs['diff'].imshow(depseg.reduce(np.logical_xor), cmap='gray')
    axs['final'].set_title('combined grid')
    axs['final'].imshow(depseg.merged_grid(), cmap='gray')

    f.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
