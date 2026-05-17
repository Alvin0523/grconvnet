import numpy as np
import pyrealsense2 as rs
import cv2
import sys
import matplotlib.pyplot as plt

from GraspDetectorPose import GraspDetectorPose

sys.path.append("posecnn/")

from utils import show_crop_region


def run():
    cv2.namedWindow('RealSense')
    detector = GraspDetectorPose('grasp_data.json', 'posecnn', device='cuda')

    # start the camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    # get essential parameters
    profile = pipeline.get_active_profile()

    pc = rs.pointcloud()

    original_resolution = (1280, 720)
    target_resolution = (640, 480)
    
    original_fx = 900
    original_fy = 900
    original_cx = 651
    original_cy = 370

    target_fx = 1067
    target_fy = 1067
    target_cx = 313
    target_cy = 241

    try:
        counter = 0
        while True:
            counter += 1
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            if counter >= 50:
                _, img_cropped = show_crop_region(
                        color_image, original_fx, original_fy, original_cx, original_cy,
                        target_fx, target_fy, target_cx, target_cy,
                        original_resolution, target_resolution
                    )
                cv2.imshow('RealSense', img_cropped)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('i'):  # Press 'i' to save RGB image and predict the pose
                    # Create point cloud
                    pc.map_to(color_frame)
                    points = pc.calculate(depth_frame)
                    points.export_to_ply('./out.ply', color_frame)

                    grasp3d = detector.predict_grasp_3d(color_image, depth_image)
                    print(grasp3d)
                    
    finally:
        pipeline.stop()

if __name__ == "__main__":
    run()