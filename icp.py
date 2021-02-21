import argparse
import os

import cv2
import numpy as np
import trimesh

import image
from transforms import depth_to_point_cloud, transform_point3s
from camera import Camera, cam_view2pose
import segmentation_helper as sh

parser = argparse.ArgumentParser()
parser.add_argument('--val', action='store_true', help='pose estimation for validation set')
parser.add_argument('--test', action='store_true', help='pose estimation for test set')

LIST_OBJ_FOLDERNAME = [
        "004_sugar_box",  # obj_id == 1
        "005_tomato_soup_can",  # obj_id == 2
        "007_tuna_fish_can",  # obj_id == 3
        "011_banana",  # obj_id == 4
        "024_bowl",  # obj_id == 5
    ]


def obj_mesh2pts(obj_id, point_num, transform=None):
    """
    In:
        obj_id: int, indicating an object in LIST_OBJ_FOLDERNAME.
        point_num: int, number of points to sample.
        transform: Numpy array [4, 4] of float64.
    Out:
        pts: Numpy array [n, 3], sampled point cloud.
    Purpose:
         Sample a point cloud from the mesh of the specific object. If transform is not None, apply it.
    """
    mesh_path = './YCB_subsubset/' + LIST_OBJ_FOLDERNAME[obj_id - 1] + '/model_com.obj'  # objects ID start from 1
    mesh = trimesh.load(mesh_path)
    if transform is not None:
        mesh = mesh.apply_transform(transform)
    pts, _ = trimesh.sample.sample_surface(mesh, count=point_num)
    return pts


def gen_obj_depth(obj_id, depth, mask):
    """
    In:
        obj_id: int, indicating an object in LIST_OBJ_FOLDERNAME.
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
    Out:
        obj_depth: Numpy array [height, width] of float64, where depth value of all the pixels that don't belong to the object is 0.
    Purpose:
        Generate depth image for a specific object.
        Generate depth for all objects when obj_id == -1.
    """
    obj_depth = np.zeros(mask.shape)
    # TODO
    for i in range(0, mask.shape[0]):
        for j in range(0, mask.shape[1]):
            if obj_id != -1:
                if mask[i][j] == obj_id:
                    obj_depth[i][j] = depth[i][j]
            else:
                for k in range(1, 1+len(LIST_OBJ_FOLDERNAME)):
                    if mask[i][j] == k:
                        obj_depth[i][j] = depth[i][j]      
    return obj_depth


def obj_depth2pts(obj_id, depth, mask, camera, view_matrix):
    """
    In:
        obj_id: int, indicating an object in LIST_OBJ_FOLDERNAME.
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
        camera: Camera instance.
        view_matrix: Numpy array [16,] of float64, representing a 4x4 matrix.
    Out:
        world_pts: Numpy array [n, 3], 3D points in the world frame of reference.
    Purpose:
        Generate point cloud projected from depth of the specific object(s) in the world frame of reference.
    Hint:
        The imported depth_to_point_cloud(), cam_view2pose() and transform_point3s() can be useful here.
        The view matrices are provided in the /dataset/val/view_matrix or /dataset/test/view_matrix folder.
    """
    # TODO
    obj_depth = gen_obj_depth(obj_id, depth, mask)
    obj_pc = depth_to_point_cloud(camera.intrinsic_matrix, obj_depth)
    pose_matrix = cam_view2pose(view_matrix)
    world_pts = transform_point3s(pose_matrix, obj_pc)
    return world_pts


def align_pts(pts_a, pts_b, max_iterations=100, threshold=1e-04):
    """
    In:
        pts_a: Numpy array [n, 3].
        pts_b: Numpy array [n, 3].
        max_iterations: int, tunable parameter of trimesh.registration.icp().
        threshold: float，tunable parameter of trimesh.registration.icp().
    Out:
        matrix: Numpy array [4, 4], the transformation matrix sending pts_a to pts_b.
    Purpose:
        Apply the iterative closest point algorithm to estimate a transformation that aligns one point cloud with another.
    Hint:
        Use trimesh.registration.procrustes() and trimesh.registration.icp().
    """
    # TODO
    
    try:
        p_matrix, procrustes_t_pts, p_cost = trimesh.registration.procrustes(pts_a, pts_b)
        matrix, t_pts, cost = trimesh.registration.icp(pts_a, pts_b, initial=p_matrix, threshold=threshold, max_iterations=max_iterations)
    except np.linalg.LinAlgError:
        return None
    return matrix


def estimate_pose(depth, mask, camera, view_matrix):
    """
    In:
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
        camera: Camera instance.
        view_matrix: Numpy array [16,] of float64, representing a 4x4 matrix.
    Out:
        list_obj_pose: a list of transformation matrices (Numpy array [4, 4] of float64).
                       The order is the same as in LIST_OBJ_FOLDERNAME,
                       so list_obj_pose[i] is the pose of the object with obj_id == i+1.
                       If the pose of an object is missing, the item should be None.
    Purpose:
        Perform pose estimation on each object in the given image.
    """
    # TODO
    list_obj_pose = list()
    for obj_id in range(1, 6):
        depth_pts = obj_depth2pts(obj_id, depth, mask, camera, view_matrix) #pts_b
        mesh_pts = obj_mesh2pts(obj_id, len(depth_pts)) #pts_a
        obj_pose_matrix = align_pts(mesh_pts, depth_pts)
        list_obj_pose.append(obj_pose_matrix)
    
    return list_obj_pose


def save_pose(dataset_dir, folder, scene_id, list_obj_pose):
    """
    In:
        dataset_dir: string, path of the val or test folder.
        folder: string, the folder to save the pose.
                "gtmask" -- for pose estimated using ground truth mask
                "predmask" -- for pose estimated using predicted mask
        scene_id: int, ID of the scene.
        list_obj_pose: a list of transformation matrices (Numpy array [4, 4] of float64).
                       The order is the same as in LIST_OBJ_FOLDERNAME,
                       so list_obj_pose[i] is the pose of the object with obj_id == i+1.
                       If the pose of an object is missing, the item would be None.
    Out:
        None.
    Purpose:
        Save the pose of each object in a scene.
    """
    pose_dir = dataset_dir + "pred_pose/" + folder + "/"
    print(f"Save poses as .npy files to {pose_dir}")
    for i in range(len(list_obj_pose)):
        pose = list_obj_pose[i]
        if pose is not None:
            np.save(pose_dir + str(scene_id) + "_" + str(i + 1), pose)


def export_gt_ply(scene_id, depth, gt_mask, camera, view_matrix):
    """
    In:
        scene_id: int, ID of the scene.
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
        camera: Camera instance.
        view_matrix: Numpy array [16,] of float64, representing a 4x4 matrix.
    Out:
        None.
    Purpose:
        Export a point cloud of the ground truth scene -- projected from depth using ground truth mask-- with the color green.
    """
    print("Export gt point cloud as .ply file to ./dataset/val/exported_ply/")
    file_path = "./dataset/val/exported_ply/" + str(scene_id) + "_gtmask.ply"
    pts = obj_depth2pts(-1, depth, gt_mask, camera, view_matrix)
    if len(pts) == 0:
        print("Empty point cloud!")
    else:
        ptcloud = trimesh.points.PointCloud(vertices=pts, colors=[0, 255, 0])  # Green
        ptcloud.export(file_path)


def export_pred_ply(dataset_dir, scene_id, suffix, list_obj_pose):
    """
    In:
        dataset_dir: string, path of the val or test folder.
        scene_id: int, ID of the scene.
        suffix: string, indicating which kind of point cloud is going to be exported.
                "gtmask_transformed" -- transformed with pose estimated using ground truth mask
                "predmask_transformed" -- transformed with pose estimated using prediction mask
        list_obj_pose: a list of transformation matrices (Numpy array [4, 4] of float64).
                       The order is the same as in LIST_OBJ_FOLDERNAME,
                       so list_obj_pose[i] is the pose of the object with obj_id == i+1.
                       If the pose of an object is missing, the item would be None.
    Out:
        None.
    Purpose:
        Export a point cloud of the predicted scene with single color.
    """
    ply_dir = dataset_dir + "exported_ply/"
    print(f"Export predicted point cloud as .ply files to {ply_dir}")
    file_path = ply_dir + str(scene_id) + "_" + suffix + ".ply"
    color_switcher = {
        "gtmask_transformed": [0, 0, 255],  # Blue
        "predmask_transformed": [255, 0, 0],  # Red
    }
    pts = np.empty([0, 3])  # Numpy array [n, 3], the point cloud to be exported.
    for obj_id in range(1, 6):  # obj_id indicates an object in LIST_OBJ_FOLDERNAME
        pose = list_obj_pose[obj_id - 1]
        if pose is not None:
            obj_pts = obj_mesh2pts(obj_id, point_num=1000, transform=pose)
            pts = np.concatenate((pts, obj_pts), axis=0)
    if len(pts) == 0:
        print("Empty point cloud!")
    else:
        ptcloud = trimesh.points.PointCloud(vertices=pts, colors=color_switcher[suffix])
        ptcloud.export(file_path)


def denoise_mask(mask):
    """
    In:
        mask: Numpy array [height, width], where each value is an obj_id.
    Out:
        mask: Numpy array [height, width], where each value is an obj_id, denoised.
    Purpose:
        Denoise the mask.
    Hint:
        Use dilate() and erode() method of OpenCV.
    """
    # TODO
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)
    dilation = cv2.dilate(mask,kernel,iterations = 1)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def main():
    args = parser.parse_args()
    if args.val:
        dataset_dir = "./dataset/val/"
        print("Pose estimation for validation set")
    elif args.test:
        dataset_dir = "./dataset/test/"
        print("Pose estimation for test set")
    else:
        print("Missing argument --val or --test")
        return

    # Setup camera -- to recover coordinate, keep consistency with that in gen_dataset.py
    my_camera = Camera(
        image_size=(240, 320),
        near=0.01,
        far=10.0,
        fov_w=69.40
    )

    if not os.path.exists(dataset_dir + "exported_ply/"):
        os.makedirs(dataset_dir + "exported_ply/")
    if not os.path.exists(dataset_dir + "pred_pose/"):
        os.makedirs(dataset_dir + "pred_pose/")
        os.makedirs(dataset_dir + "pred_pose/predmask/")
        if args.val:
            os.makedirs(dataset_dir + "pred_pose/gtmask/")

    # TODO:
    #  Use the implemented estimate_pose() to estimate the pose of the objects in each scene of the validation set and test set.
    #  For the validation set, use both ground truth mask and predicted mask.
    #  For the test set, use the predicted mask.
    #  Use save_pose(), export_gt_ply() and export_pred_ply() to generate files to be submitted.


    #np.load()
    for scene_id in range(5):
        print("Estimating scene", scene_id)
        depth = image.read_depth(dataset_dir+'depth/'+str(scene_id)+'_depth.png')
        pred_mask = image.read_mask(dataset_dir+'pred/'+str(scene_id)+'_pred.png')
        view_matrix = np.load(dataset_dir+'view_matrix/'+str(scene_id)+'.npy')
        pred_pose = estimate_pose(depth, pred_mask, my_camera, view_matrix)

        if args.val:
            gt_mask = image.read_mask(dataset_dir+'gt/'+str(scene_id)+'_gt.png')
            val_gt_pose = estimate_pose(depth, gt_mask, my_camera, view_matrix)
            save_pose(dataset_dir, 'gtmask', scene_id, val_gt_pose)
            export_gt_ply(scene_id, depth, gt_mask, my_camera, view_matrix)
            export_pred_ply(dataset_dir, scene_id, 'gtmask_transformed', val_gt_pose)

        save_pose(dataset_dir, 'predmask', scene_id, pred_pose)
        export_pred_ply(dataset_dir, scene_id, 'predmask_transformed', pred_pose)
            
        


if __name__ == '__main__':
    main()
