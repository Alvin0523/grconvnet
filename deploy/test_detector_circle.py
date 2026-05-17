import numpy as np
import pyrealsense2 as rs
import cv2

from GraspDetector import GraspDetectorCircle
from grasp_utils import intrinsics_to_numpy

def run():
    # start the camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    # get essential parameters
    profile = pipeline.get_active_profile()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depth_min = 0.11
    depth_max = 4
    # intrinsics
    depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    # extrinsics
    depth_to_color_extrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.color))
    color_to_depth_extrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.depth))

    detector = GraspDetectorCircle(profile=profile, mat=intrinsics_to_numpy(depth_intrin))
    try:
        counter = 0
        while True:
            counter += 1
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            print("counter: ", counter)
            if counter >= 50:
                depth_image = (depth_image.astype(np.float64)/1000.0).astype(np.float32)
                # Example usage:
                image_name = "captured_image"
                image_file = "captured_image.jpg"
                cv2.imwrite(image_file, color_image)
                output_file = image_name + "_with_circle.jpg"
                output_file_grasp = image_name + "_with_grasp.jpg"

                circles = detector.detect_circles(image_file)

                if circles:
                    print(f"Found {len(circles)} circles:")
                    for i, (x, y, r) in enumerate(circles):
                        print(f"  Circle {i+1}: Center=({x}, {y}), Radius={r}")
                    print("Print Any Key to close the window.")
                    detector.draw_circles(image_file, circles ,output_file)
                    grasp_circle = circles[0]
                    grasp = detector.circle_to_2d_grasp(grasp_circle)
                    detector.draw_grasp(image_file, grasp, output_file_grasp)
                else:
                    print("No circles detected.")
                break
                
    finally:
        pipeline.stop()

if __name__ == "__main__":
    run()