import torch
import numpy as np
import sys

from GraspDetector import GraspDetector

sys.path.append("graspcnn/")

from utils.dataset_processing.grasp import detect_grasps
from utils.dataset_processing import evaluation
from models.common import post_process_output


class GraspDetectorNN(GraspDetector):
    """
    A class for detecting grasps using a neural network
    """
    def __init__(self, model_path, device, mat=None):
        """
        Initializes the GraspDetector.

        Args:
            model_path: the path of the trained model
            device (str): The device to use for computation (e.g., 'cpu', 'cuda').
            mat (np.ndarray, optional): The camera intrinsic matrix. If None, a default matrix is used. Defaults to None.
        """
        super(GraspDetectorNN, self).__init__(mat)
        super().__init__(mat=mat)
        model_name = model_path
        self.model = torch.load(model_name, weights_only=False).to(device)
        self.device = device
        self.input_size = 300
    
    def predict_grasp(self, img, id=999):
        """
        Predicts a grasp from an input image.

        Args:
            img (np.ndarray): The input image (depth or RGB).
            id (int, optional): An identifier for the grasp prediction. Defaults to 999.

        Returns:
            Grasp2d: A list containing the grasp center (tuple), angle (float), and width (float).
        """
        top = (img.shape[0] - self.input_size)//2
        left = (img.shape[1] - self.input_size)//2
        img = img[top:top+self.input_size,left:left+self.input_size]
        if len(img.shape) == 2:
            # depth input
            img = np.clip((img - img.mean()), -1, 1)
            x = img[None,None,...]
            x = torch.tensor(x).to(self.device).float()
        else:
            # RGB input/RGB-D input
            x = img.transpose((2,0,1))[None,...]
            x = torch.tensor(x).to(self.device).float()
        with torch.no_grad():
            pos_img, angle_img, width_img = post_process_output(*self.model(x))
            evaluation.plot_output(img, img, pos_img, angle_img, no_grasps=1, grasp_width_img=width_img, id=id)
            grasps = detect_grasps(pos_img, angle_img, width_img=width_img, no_grasps=1)
            grasp = grasps[0]
        center = (top+grasp.center[0], left+grasp.center[1])
        angle = grasp.angle
        width = grasp.length
        grasp2d = [center, angle, width]
        return grasp2d


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
        grasp2d = self.predict_grasp(dep, id=id)
        print("2D Grasp: ", grasp2d)
        grasp3d = self.grasp_2d_to_3d(grasp2d, dep)
        print("3D Grasp: ", grasp3d)
        return grasp3d