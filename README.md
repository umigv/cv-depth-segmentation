# cv-depth-segmentation

Provides the depth segmentation pipeline. Runs RANSAC on depth footage, performs perspective transformations, and outputs an occupancy grid. Depth maps may be sourced from a live feed or from a pre-recorded `hdf5` file.

- `record_multi.py` allows recording from any number of Zed depth cameras
- `inspect_dual.py` provides an example of working on a pre-recorded file
- `live_single.py` shows depth segmentation live on one camera
- `tuner.py` allows tuning of the camera positions from pre-recorded or live footage

## tuning camera positions

`DepthSegmentation`, the depth processing abstraction, requires that every `DepthSource` be accompanied with a `CameraPosition`. To calibrate these parameters, you can use `tuner.py`.

The merged occupancy grid (on the right) shows the logical XOR of the grids generated from the left and right cameras; you should aim for it to be as black as possible. Once done, press `X` to exit the tuning program. The calibrated `CameraPosition`s will be displayed, and you can copy these directly into the code that needs it.
