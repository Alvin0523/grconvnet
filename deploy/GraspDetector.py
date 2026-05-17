import pyrealsense2 as rs
import numpy as np
import sys
import cv2


class GraspDetector(object):
    """
    A base class for grasp detection.
    """
    def __init__(self, mat=None):
        super(GraspDetector, self).__init__()
        if mat is None:
            # Default Camera: Realsense D435i
            self.camera_mat = np.array([[617.79730003, 0, 320], [0, 617.79730003, 240], [0, 0, 1]])
        else:
            self.camera_mat = np.array(mat)
        print("Camera Intrisic: ", self.camera_mat)
        self.depth_raw = None
        self.color_raw = None
    
    def update_raw(self, color_raw, depth_raw):
        '''
        Update the raw data from the camera
        '''
        self.color_raw = color_raw
        self.depth_raw = depth_raw
    
    def predict_grasp(self, img, id=999):
        """
        Predicts a grasp from an input image.
        This method should be implemented by subclasses to provide specific grasp detection logic.
        """
        return None
    
    def grasp_2d_to_3d(self, grasp, dep):
        """
        Converts a 2D grasp to a 3D grasp.

        Args:
            grasp (list): A list containing the 2D grasp center (tuple), angle (float), and width (float).
            dep (np.ndarray): The depth image.

        Returns:
            list: A list containing the 3D grasp rotation matrix (np.ndarray), center (tuple), width (float), and depth (float).
        """
        center, angle, width = grasp # 2D Grasp
        angle = angle + np.pi/2 # rotation bias introduced by defination
        z = dep[center] # query the depth of the center
        f = self.camera_mat[0,0] # the focus in pixel

        # Reproject the grasp point from image space (2D) to camera frame (3D)
        center_homo = (center[1], center[0], 1)
        center_3d = z * np.linalg.inv(self.camera_mat) @ center_homo
        center_3d = (center_3d[0], center_3d[1], center_3d[2])
        
        # Transfer the in-plane rotation to 3D rotation
        mat_3d = np.array([[np.cos(-angle), -np.sin(-angle), 0], [np.sin(-angle), np.cos(-angle), 0], [0, 0, 1]])
        mat_bias = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        mat_3d = mat_3d @ mat_bias # rotation bias introduced by the robot defination
        width_3d = width * z / f # width in pixel to with in meter
        return [mat_3d, center_3d, width_3d, 0.02]

    def predict_grasp_3d(self, img, dep, id=999):
        """
        Predicts a 3D grasp from an input image and depth map.
        This method should be implemented by subclasses to provide specific 3D grasp detection logic.
        """
        return None


class GraspDetectorCircle(GraspDetector):
    """
    A subclass of GraspDetector that detects grasps based on circle detection.
    This class uses OpenCV's Hough Circle Transform to detect circles in an image and
    then infers grasps from the detected circles.
    """
    def __init__(self, profile, mat=None):
        super(GraspDetectorCircle, self).__init__(mat)
        self.profile = profile
    
    def detect_circles(self, image_path):
        """
        Detects circles in an image using OpenCV's Hough Circle Transform.

        Args:
            image_path: The path to the image file.

        Returns:
            A list of circles found in the image, where each circle is represented as a tuple (x, y, radius).
            Returns an empty list if no circles are found.
        """
        # Load the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Error: Could not load image at {image_path}")
            return []

        # Blur the image to reduce noise
        img_blurred = cv2.medianBlur(img, 5)

        # Apply Hough Circle Transform
        # Parameters:
        # - img_blurred: The input image (grayscale).
        # - cv2.HOUGH_GRADIENT: The detection method.
        # - dp: Inverse ratio of the accumulator resolution to the image resolution (e.g., 1 means the same resolution).
        # - minDist: Minimum distance between the 
        # - centers of detected circles.
        # - param1: Upper threshold for the internal Canny edge detector.
        # - param2: Threshold for center detection.
        # - minRadius: Minimum radius of the circle.
        # - maxRadius: Maximum radius of the circle.
        circles = cv2.HoughCircles(img_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                param1=50, param2=50, minRadius=10, maxRadius=40)

        detected_circles = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                detected_circles.append((i[0], i[1], i[2]))  # x, y, radius

        return detected_circles
    
    def draw_circles(self, image_path, circles, output_path=None):
        """
        Draws the detected circles on the original image.

        Args:
            image_path: The path to the original image file.
            circles: A list of circles, where each circle is a tuple (x, y, radius).
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image at {image_path}")
            return

        for x, y, r in circles:
            # Draw the circle
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

        # Save the image with circles
        if output_path is not None:
            cv2.imwrite(output_path, img)
            print(f"Image with detected circles saved to {output_path}")

        # Display the image with circles
        cv2.imshow('Detected Circles', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def draw_grasp(self, image_path, grasp, output_path=None):
        """
        Draws the detected grasp on the original image.

        Args:
            image_path (str): The path to the original image file.
            grasp (list): A list representing a 2D grasp in the format [center, angle, width].
            output_path (str, optional): The path to save the image with the grasp drawn.
                                         If None, the image is only displayed. Defaults to None.
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image at {image_path}")
            return

        center, angle, width = grasp
        y, x  = center

        # Calculate the endpoints of the grasp rectangle
        l = width / 2
        x1 = int(x - l * np.sin(angle+np.pi/2))
        y1 = int(y + l * np.cos(angle+np.pi/2))
        x2 = int(x + l * np.sin(angle+np.pi/2))
        y2 = int(y - l * np.cos(angle+np.pi/2))

        # Draw the grasp line
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw the center of the grasp
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

        # Save the image with the grasp
        if output_path is not None:
            cv2.imwrite(output_path, img)
            print(f"Image with detected grasp saved to {output_path}")

        # Display the image with the grasp
        cv2.imshow('Detected Grasp', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def circle_to_2d_grasp(self, circle):
        """
        Converts a circle representation to a 2D grasp representation.

        Args:
            circle (tuple): A tuple representing a circle in the format (x, y, radius),
                            where (x, y) is the center and radius is the circle's radius.

        Returns:
            grasp2d: A list representing a 2D grasp in the format [center, angle, width],
                where center is a tuple (x, y), angle is in radians, and width is the grasp width.
                Returns None if the input is invalid.
        """
        if not isinstance(circle, tuple) or len(circle) != 3:
            print("Error: Invalid circle format. Expected (x, y, radius).")
            return None

        x, y, radius = circle

        # The center of the grasp is the same as the center of the circle.
        center = (int(y), int(x))

        # For a circle, we can assume the grasp angle is 0 (horizontal).
        angle = 0.0

        # The width of the grasp can be related to the diameter of the circle.
        # Here, we'll use the diameter as the grasp width.
        width = radius.astype(np.float32) * 2
        grasp2d = [center, angle, width]
        return grasp2d

    def predict_grasp(self, img, id=999):
        """
        Predicts a grasp from an input image.
        Args:
            img (np.ndarray): The input image (depth or RGB).
        """
        image_name = "captured_image"
        image_file = image_name + ".jpg"
        cv2.imwrite(image_file, img)
        detected_circles = self.detect_circles(image_file)
        if detected_circles:
            print(f"Found {len(detected_circles)} circles:")
            grasp_circle = detected_circles[0]
            output_file = image_name + "_with_circle.jpg"
            output_file_grasp = image_name + "_with_grasp.jpg"
            self.draw_circles(image_file, detected_circles ,output_file)
            grasp = self.circle_to_2d_grasp(grasp_circle)
            self.draw_grasp(image_file, grasp, output_file_grasp)
            return grasp
        else:
            print("No circles detected.")
            return None
    
    def grasp_from_color_to_depth(self, grasp2d):
        # Move the grasp center from color image space to depth image space.
    
        # Args:
        #     grasp2d (tuple): A tuple containing ((y, x), angle, width) in color image space
        
        # Returns:
        #     tuple: Converted grasp in depth image space ((y, x), angle, width)
        grasp_center = [grasp2d[0][1], grasp2d[0][0]]
        depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        depth_min = 0.11
        depth_max = 4

        depth_intrin = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        color_intrin = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        depth_to_color_extrin = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(self.profile.get_stream(rs.stream.color))
        color_to_depth_extrin = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to(self.profile.get_stream(rs.stream.depth))

        grasp_center = rs.rs2_project_color_pixel_to_depth_pixel(self.depth_raw, depth_scale,
                depth_min, depth_max,
                depth_intrin, color_intrin, color_to_depth_extrin, depth_to_color_extrin, grasp_center)
        grasp2d_depth = [(int(grasp_center[1]), int(grasp_center[0])), grasp2d[1], grasp2d[2]]
        return grasp2d_depth

    
    def predict_grasp_3d(self, img, dep, id=999):
        """
        Predicts a 3D grasp from an input image and depth map.

        This method first predicts a 2D grasp using predict_grasp, then
        converts it to a 3D grasp using grasp_2d_to_3d.

        Args:
            img (str): The input image path.
            dep (np.ndarray): The depth map.
            id (int, optional): An identifier for the grasp prediction. Defaults to 999.

        Returns:
            list: A list containing the 3D grasp rotation matrix (np.ndarray), center (tuple), width (float), and depth (float).
                  Returns None if no 2D grasp is detected.
        """
        grasp2d = self.predict_grasp(img, id=id)
        print(grasp2d)
        if grasp2d is None:
            print("No circle detected, please try again!")
            return None
        grasp2d = self.grasp_from_color_to_depth(grasp2d)
        print("2D Grasp: ", grasp2d)
        grasp3d = self.grasp_2d_to_3d(grasp2d, dep)
        print("3D Grasp: ", grasp3d)
        return grasp3d



