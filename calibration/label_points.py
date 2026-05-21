import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)
profile = pipeline.get_active_profile()

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
depth_min = 0.11
depth_max = 4

depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

depth_to_color_extrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.color))
color_to_depth_extrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.depth))

color_points = []
depth_point_3ds = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        color_points.append([x, y])
        print(f"Point added: {x}, {y}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if color_points:
            # Find the nearest point
            distances = [np.linalg.norm(np.array([x, y]) - np.array(point)) for point in color_points]
            nearest_point_index = np.argmin(distances)
            removed_point = color_points.pop(nearest_point_index)
            print(f"Point removed: {removed_point}")

def save_points_to_json(filename):
    data = {"color_points": color_points, "points_3d": depth_point_3ds}
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Points saved to {filename}")

def load_points_from_json(filename):
    global color_points, points_3d
    with open(filename, 'r') as f:
        data = json.load(f)
        color_points = data["color_points"]
        depth_point_3ds = data["points_3d"]
    print(f"Points loaded from {filename}")

def save_rgb_image(filename, image):
    cv2.imwrite(filename, image)
    print(f"RGB image saved to {filename}")

cv2.namedWindow('RealSense')
cv2.setMouseCallback('RealSense', mouse_callback)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # color_points.sort(key=lambda point: (point[1], point[0]))
        depth_point_3ds = []
        for color_point in color_points:
            depth_point_ = rs.rs2_project_color_pixel_to_depth_pixel(depth_frame.get_data(), depth_scale,
                depth_min, depth_max,
                depth_intrin, color_intrin, color_to_depth_extrin, depth_to_color_extrin, color_point)

            # Get the depth value at the projected depth point
            dx = int(np.clip(depth_point_[0], 0, depth_intrin.width - 1))
            dy = int(np.clip(depth_point_[1], 0, depth_intrin.height - 1))
            depth_value = depth_frame.get_distance(dx, dy)

            # Deproject the depth point to 3D space
            depth_pixel = [depth_point_[0], depth_point_[1]]
            depth_point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_value)
            depth_point_3ds.append(depth_point_3d)

            # Draw the points on the color image
            cv2.circle(color_image, (int(color_point[0]), int(color_point[1])), 1, (0, 255, 0), -1)

            # Draw the points on the depth image
            cv2.circle(depth_colormap, (int(depth_point_[0]), int(depth_point_[1])), 1, (0, 255, 0), -1)

            # print(f"Color point: {color_point} -> Depth point: {depth_point_} -> 3D point: {depth_point_3d}")

        images = np.hstack((color_image, depth_colormap))

        # Display the images
        cv2.imshow('RealSense', images)
        # Press 'Esc' to exit
        key = cv2.waitKey(1) 
        if key & 0xFF == 27:  
            break
        elif key & 0xFF == ord('s'):  # Press 's' to save points
            print("save points!")
            save_points_to_json(os.path.join(_HERE, 'points.json'))
        elif key & 0xFF == ord('l'):  # Press 'l' to load points
            print("load points!")
            load_points_from_json(os.path.join(_HERE, 'points.json'))
        elif key & 0xFF == ord('i'):  # Press 'i' to save RGB image
            print("save RGB image!")
            save_rgb_image(os.path.join(_HERE, 'color_image.png'), color_image)
        

finally:
    pipeline.stop()
    cv2.destroyAllWindows()