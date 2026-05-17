import numpy as np
import copy
import open3d as o3d

from grasp_utils import get_transformation, save_grasp, create_parallel_gripper
from scipy.spatial.transform import Rotation as R


def visualize_gripper_and_object(obj_path, grasp):
    """
    Visualize parallel gripper and object
    :param gripper_width: distance between gripper fingers
    :param obj_type: type of object to visualize
    :param obj_size: size of the object
    """
    r = grasp[0]
    t = grasp[1]
    w = grasp[2]

    gripper = create_parallel_gripper(r, t, width=w)
    obj = o3d.io.read_triangle_mesh(obj_path, enable_post_processing=True)
    
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # Add all geometries to visualizer
<<<<<<< HEAD
    # geometries = gripper + [obj, coordinate_frame]
    geometries = [obj]
=======
    geometries = gripper + [obj, coordinate_frame]
>>>>>>> 9fbb58826b7e51d9b04e519dd02449cb511dd91f
    
    # Visualize
    o3d.visualization.draw_geometries(geometries, 
                                     window_name="Parallel Gripper and Object",
                                     width=800,
                                     height=600,
                                     left=50,
                                     top=50)

if __name__ == "__main__":
    grasp_dict = {}
    # Example usage
    grasp = [R.from_euler('xyz', [-np.pi/2,0,np.pi/2]).as_matrix(), np.array([0.02,0, -0.02]), 0.08, 0.02]
<<<<<<< HEAD
    obj_path = "models/010_potted_meat_can/textured_simple.obj"
=======
    obj_path = "posecnn/PROPS-Pose-Dataset/model/9_potted_meat_can/textured_simple.obj"
>>>>>>> 9fbb58826b7e51d9b04e519dd02449cb511dd91f
    visualize_gripper_and_object(obj_path, grasp)
    grasp_dict['potted_meat_can'] = [grasp]
    ######
    grasp = [R.from_euler('xyz', [0,np.pi/2,np.pi/2]).as_matrix(), np.array([0.0,0,0.00]), 0.06, 0.02]
<<<<<<< HEAD
    obj_path = "models/007_tuna_fish_can/textured_simple.obj"
=======
    obj_path = "posecnn/PROPS-Pose-Dataset/model/6_tuna_fish_can/textured_simple.obj"
>>>>>>> 9fbb58826b7e51d9b04e519dd02449cb511dd91f
    visualize_gripper_and_object(obj_path, grasp)
    grasp_dict['tuna_fish_can'] = [grasp]
    # ######
    grasp = [R.from_euler('xyz', [0,0,np.pi/3]).as_matrix(), np.array([-0.015,0.01,-0.02]), 0.08, 0.02]
<<<<<<< HEAD
    obj_path = "models/006_mustard_bottle/textured_simple.obj"
=======
    obj_path = "posecnn/PROPS-Pose-Dataset/model/5_mustard_bottle/textured_simple.obj"
>>>>>>> 9fbb58826b7e51d9b04e519dd02449cb511dd91f
    visualize_gripper_and_object(obj_path, grasp)
    grasp_dict['mustard_bottle'] = [grasp]
    save_grasp(grasp_dict)