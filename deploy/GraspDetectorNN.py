import torch
import numpy as np

from GraspDetector import GraspDetector
from inference.post_process import post_process_output


MODEL_PATH = "grconvnet/trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/model.pt"
INPUT_SIZE  = 300


class GraspDetectorNN(GraspDetector):
    def __init__(self, model_path=MODEL_PATH, device="cuda", mat=None):
        super().__init__(mat=mat)
        self.model  = torch.load(model_path, weights_only=False).to(device)
        self.device = device

    def predict_grasp(self, rgb, dep, id=999):
        """Run RGBD GR-ConvNet inference and return 2D grasp [center, angle, width]."""
        h, w = dep.shape[:2]
        top  = (h - INPUT_SIZE) // 2
        left = (w - INPUT_SIZE) // 2

        dep_crop = dep[top:top+INPUT_SIZE, left:left+INPUT_SIZE]
        rgb_crop = rgb[top:top+INPUT_SIZE, left:left+INPUT_SIZE]

        # camera streams as BGR — convert to RGB to match training data
        rgb_crop = rgb_crop[:, :, ::-1].copy()

        dep_norm = np.clip((dep_crop - dep_crop.mean()), -1, 1).astype(np.float32)

        rgb_norm = rgb_crop.astype(np.float32) / 255.0
        rgb_norm -= rgb_norm.mean()
        rgb_norm = rgb_norm.transpose((2, 0, 1))

        x = np.concatenate([dep_norm[None, ...], rgb_norm], axis=0)
        x = torch.tensor(x[None, ...]).to(self.device).float()

        with torch.no_grad():
            pos_img, angle_img, width_img = post_process_output(*self.model(x))
            # ensure 2D (H, W) regardless of how squeeze worked
            q   = np.squeeze(pos_img).reshape(INPUT_SIZE, -1)
            ang = np.squeeze(angle_img).reshape(INPUT_SIZE, -1)
            wid = np.squeeze(width_img).reshape(INPUT_SIZE, -1)
            print(f"[GraspDetectorNN] Q max={q.max():.3f} q.shape={q.shape}")
            peak     = np.unravel_index(np.argmax(q), q.shape)
            row, col = int(peak[0]), int(peak[1])
            angle    = float(ang[peak])
            width    = float(wid[peak])

        center = (top + row, left + col)
        return [center, angle, width]

    def predict_grasp_3d(self, rgb, dep, id=999):
        """Predict 2D grasp then reproject to 3D camera frame."""
        grasp2d = self.predict_grasp(rgb, dep, id=id)
        print("2D Grasp:", grasp2d)
        grasp3d = self.grasp_2d_to_3d(grasp2d, dep)
        print("3D Grasp:", grasp3d)
        return grasp3d
