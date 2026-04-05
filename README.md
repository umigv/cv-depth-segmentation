# cv-depth-segmentation

Provides the depth segmentation pipeline. Runs RANSAC on depth footage, performs perspective transformations, and outputs an occupancy grid. Depth maps may be sourced from a live feed or from a pre-recorded `hdf5` file.

- `record_multi.py` allows recording from any number of Zed depth cameras
- `inspect_dual.py` provides an example of working on a pre-recorded file
- `live_single.py` shows depth segmentation live on one camera
- `tuner.py` allows tuning of the camera positions from pre-recorded or live footage

## requirements

- Python 3.14 (deferred type annotation evaluation)
- `opencv-python`
- `numpy>=2`
- `h5py`

## tuning camera positions

`DepthSegmentation`, the depth processing abstraction, requires that every `DepthSource` be accompanied with a `CameraPosition`.

The occupancy grid needs to be set up such that the centre of the wheelbase is centred at the bottom of the occupancy grid. The starting tuning parameters should be based on the physical measurements of the camera's positions.

```txt
CameraPosition(x, z, rad)
x: right-shift of the camera on the grid (mm)
z: forward-shift of the camera on the grid (mm)
rad: rotation of the camera to the left (rad)
```

Then, you can fine-tune the calibration to get the grids to match. For two cameras, you can use `tuner.py`. For a single camera, the tuner is not needed.

The merged occupancy grid (on the right) shows the logical XOR of the grids generated from the left and right cameras; you should aim for it to be as black as possible. Once done, press `X` to exit the tuning program. The calibrated `CameraPosition`s will be displayed, and you can copy these directly into the code that needs it.

For `cv-autonav`, this should be replacing the indicated section of code below.

```python
if __name__ == "__main__":
    # replace these values
    left_pos = rsc.CameraPosition(0, 0, math.radians(30))
    right_pos = rsc.CameraPosition(0, 0, math.radians(-30))
    # ...
```
