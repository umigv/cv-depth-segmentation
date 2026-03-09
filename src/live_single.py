import sys
import pyzed.sl as sl
from signal import signal, SIGINT
import argparse
import os
import cv2
import numpy as np
import math

from multiprocessing import Pool

import ransac as rsc


def print_params(calibration_params: sl.CalibrationParameters):
    # LEFT CAMERA intrinsics
    fx_left = calibration_params.left_cam.fx
    fy_left = calibration_params.left_cam.fy
    cx_left = calibration_params.left_cam.cx
    cy_left = calibration_params.left_cam.cy

    # RIGHT CAMERA intrinsics
    fx_right = calibration_params.right_cam.fx
    fy_right = calibration_params.right_cam.fy
    cx_right = calibration_params.right_cam.cx
    cy_right = calibration_params.right_cam.cy

    # Translation (baseline) between left and right camera
    tx = calibration_params.stereo_transform.get_translation().get()[0]

    # Print results
    print("\n--- ZED Camera Calibration Parameters ---")
    print("Left Camera Intrinsics:")
    print(f"  fx = {fx_left:.3f}")
    print(f"  fy = {fy_left:.3f}")
    print(f"  cx = {cx_left:.3f}")
    print(f"  cy = {cy_left:.3f}\n")

    print("Right Camera Intrinsics:")
    print(f"  fx = {fx_right:.3f}")
    print(f"  fy = {fy_right:.3f}")
    print(f"  cx = {cx_right:.3f}")
    print(f"  cy = {cy_right:.3f}\n")

    print(f"Stereo Baseline (tx): {tx:.6f} meters")


def intrinsics_from_params(params: sl.CalibrationParameters, sx, sy):
    return rsc.Intrinsics(params.left_cam.cx * sx, params.left_cam.cy * sy,
                          params.left_cam.fx * sx, params.left_cam.fy * sy,
                          params.stereo_transform.get_translation().get()[0])


def run_ransac_on_zed(cam_pos=rsc.CameraPosition(), serial_number=None):
    cam = sl.Camera()

    init = sl.InitParameters()
    if serial_number is not None:
        init.set_from_serial_number(serial_number)
    init.camera_resolution = sl.RESOLUTION.VGA
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.async_image_retrieval = True  # maybe change to False if stuff breaks

    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open", status, "Exit program.")
        exit(1)
    runtime = sl.RuntimeParameters()

    cam_conf = cam.get_camera_information().camera_configuration
    resolution = cam_conf.resolution
    w = min(720, resolution.width)
    h = min(404, resolution.height)
    low_res = sl.Resolution(w, h)
    calibration_params = cam_conf.calibration_parameters
    print_params(calibration_params)

    intr = intrinsics_from_params(calibration_params,
                                  w / float(resolution.width),
                                  h / float(resolution.height))
    grid_conf = rsc.GridConfiguration(5000.0, 5000.0, 50.0)

    thread_pool_size = 4
    thread_pool = Pool(thread_pool_size)
    px_c = np.array([0.0, 0.0, 0.0])

    image_mat = sl.Mat()
    depth_mat = sl.Mat()

    key = 0
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err > sl.ERROR_CODE.SUCCESS:  # good to go
            print("Grab ZED : ", err)
            break

        # FIXME pointing camera at only the ground causing a crash
        cam.retrieve_image(image_mat, sl.VIEW.LEFT, sl.MEM.CPU, low_res)
        cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH, sl.MEM.CPU, low_res)

        image = image_mat.get_data()
        depths = rsc.clean_depths(depth_mat.get_data())

        # generate occupancy grid
        hsv_mask = rsc.hsv_mask(image)
        ground_mask, px_c = rsc.ground_plane(
            depths, 100, (1, 16), 0.13, px_c, thread_pool, thread_pool_size
        )
        lane_mask = rsc.merge_masks(ground_mask, hsv_mask)
        real_c = rsc.real_coeffs(px_c, intr)
        rad = rsc.real_angle(real_c)
        occ = rsc.occ_grid(lane_mask, real_c, intr, grid_conf, cam_pos)

        # convert occupancy grid to image format
        occ_img = cv2.cvtColor(occ, cv2.COLOR_GRAY2BGR)
        occ_img = cv2.resize(
            occ_img, (600, 600), interpolation=cv2.INTER_NEAREST_EXACT
        )
        cv2.imshow("occupancy grid", occ_img)
        print(f"angle: {math.degrees(rad): .3f} deg")

        key = cv2.waitKey(1)

    cv2.destroyAllWindows()
    cam.close()


if __name__ == "__main__":
    run_ransac_on_zed()
