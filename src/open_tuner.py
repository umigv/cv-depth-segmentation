import h5py

import math

import numpy as np

from calibrate.ui import CameraUI

import ransac as rsc




def main():

    file = h5py.File("res/dual_camera_calibration.hdf5", "r")
    left = rsc.HDF5Source(file, 0)
    right = rsc.HDF5Source(file, 1)
    
    conf = rsc.GridConfiguration(5000.0, 5000.0, 50.0)
    depseg = rsc.DepthSegementation([
        (left, rsc.CameraPosition(0, 0, 0)),
        (right, rsc.CameraPosition(0, 0, 0))
    ], conf)

    ui = CameraUI()

    frame = 0
    while frame < left.frame_count:

        if not ui.paused:
            print(f"using frame number: {frame}")
            left.use_frame(frame)
            right.use_frame(frame)

            frame += 1

        depseg.process()

        depseg.sources = [
            (depseg.sources[0][0], rsc.CameraPosition(
                10 * ui.params['left']['x_offset'],
                10 * ui.params['left']['z_offset'],
                math.radians(ui.params['left']['angle'])
            )),
            (depseg.sources[1][0], rsc.CameraPosition(
                10 * ui.params['right']['x_offset'],
                10 * ui.params['right']['z_offset'],
                math.radians(ui.params['right']['angle'])
            ))
        ]
        
        close = ui.render(
            grids=depseg.grids,
            merged=depseg.reduce(np.logical_xor) * 255
        )

        if close:
            break
    
    
if __name__ == '__main__':
    main()