import json
import os

from typing import List, Dict, Any

import tqdm

import torch

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

import pyarrow as pa
import pyarrow.parquet as pq

from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.kitti import KittiDB
from nuscenes.utils.splits import create_splits_logs

class KittiConverter:
    def __init__(self,
                 nusc_data_root: str = '/data/sets/nuscenes',
                 nusc_kitti_dir: str = '~/nusc_kitti',
                 cam_name: str = 'CAM_FRONT',
                 lidar_name: str = 'LIDAR_TOP',
                 image_count: int = 404,
                 nusc_version: str = 'v1.0-mini',
                 split: str = 'mini_train'):
        """
        :param nusc_kitti_dir: Where to write the KITTI-style annotations.
        :param cam_name: Name of the camera to export. Note that only one camera is allowed in KITTI.
        :param lidar_name: Name of the lidar sensor.
        :param image_count: Number of images to convert.
        :param nusc_version: nuScenes version to use.
        :param split: Dataset split to use.
        """
        self.data_path = nusc_data_root
        self.nusc_kitti_dir = os.path.expanduser(nusc_kitti_dir)
        self.cam_name = cam_name
        self.lidar_name = lidar_name
        self.image_count = image_count
        self.nusc_version = nusc_version
        self.split = split

        # Create nusc_kitti_dir.
        if not os.path.isdir(self.nusc_kitti_dir):
            os.makedirs(self.nusc_kitti_dir)

        # Select subset of the data to look at.
        self.nusc = NuScenes(version=nusc_version, dataroot=nusc_data_root)

    def process_depth_sample(self, token, idx, depth_path):
        index_t = token
        rec = self.nusc.get(
            'sample', index_t)

        lidar_sample = self.nusc.get(
            'sample_data', rec['data']['LIDAR_TOP'])
        lidar_pose = self.nusc.get(
            'ego_pose', lidar_sample['ego_pose_token'])
        # yaw = Quaternion(lidar_pose['rotation']).yaw_pitch_roll[0]
        # lidar_rotation = Quaternion(scalar=np.cos(
        #    yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
        lidar_rotation = Quaternion(lidar_pose['rotation'])
        lidar_translation = np.array(lidar_pose['translation'])[:, None]
        lidar_to_world = np.vstack([
            np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
            np.array([0, 0, 0, 1])
        ])

        # get lidar points
        lidar_file = os.path.join(
            self.data_path, lidar_sample['filename'])
        lidar_points = np.fromfile(lidar_file, dtype=np.float32)
        # lidar data is stored as (x, y, z, intensity, ring index).
        lidar_points = lidar_points.reshape(-1, 5)[:, :4]

        # lidar points ==> ego frame
        sensor_sample = self.nusc.get(
            'calibrated_sensor', lidar_sample['calibrated_sensor_token'])
        lidar_to_ego_lidar_rot = Quaternion(
            sensor_sample['rotation']).rotation_matrix
        lidar_to_ego_lidar_trans = np.array(
            sensor_sample['translation']).reshape(1, 3)

        ego_lidar_points = np.dot(
            lidar_points[:, :3], lidar_to_ego_lidar_rot.T)
        ego_lidar_points += lidar_to_ego_lidar_trans

        homo_ego_lidar_points = np.concatenate(
            (ego_lidar_points, np.ones((ego_lidar_points.shape[0], 1))), axis=1)

        homo_ego_lidar_points = torch.from_numpy(
            homo_ego_lidar_points).float()

        camera_sample = self.nusc.get(
            'sample_data', rec['data'][self.cam_name])

        car_egopose = self.nusc.get(
            'ego_pose', camera_sample['ego_pose_token'])
        egopose_rotation = Quaternion(car_egopose['rotation']).inverse
        egopose_translation = - \
            np.array(car_egopose['translation'])[:, None]
        world_to_car_egopose = np.vstack([
            np.hstack((egopose_rotation.rotation_matrix,
                       egopose_rotation.rotation_matrix @ egopose_translation)),
            np.array([0, 0, 0, 1])
        ])

        # From egopose to sensor
        sensor_sample = self.nusc.get(
            'calibrated_sensor', camera_sample['calibrated_sensor_token'])
        intrinsic = torch.Tensor(sensor_sample['camera_intrinsic'])
        sensor_rotation = Quaternion(sensor_sample['rotation'])
        sensor_translation = np.array(
            sensor_sample['translation'])[:, None]
        car_egopose_to_sensor = np.vstack([
            np.hstack(
                (sensor_rotation.rotation_matrix, sensor_translation)),
            np.array([0, 0, 0, 1])
        ])
        car_egopose_to_sensor = np.linalg.inv(car_egopose_to_sensor)

        # Combine all the transformation.
        # From sensor to lidar.
        lidar_to_sensor = car_egopose_to_sensor @ world_to_car_egopose @ lidar_to_world
        lidar_to_sensor = torch.from_numpy(lidar_to_sensor).float()

        # load image for debugging
        image_filename = os.path.join(
            self.data_path, camera_sample['filename'])
        img = Image.open(image_filename)
        img = np.array(img)

        sparse_depth = torch.zeros((img.shape[:2]))

        # Ego(lidar) ==> Camera
        camera_points = torch.mm(
            homo_ego_lidar_points, lidar_to_sensor.t())
        # depth > 0
        depth_mask = camera_points[:, 2] > 0
        camera_points = camera_points[depth_mask]
        # Camera ==> Pixel
        viewpad = torch.eye(4)
        viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
        pixel_points = torch.mm(camera_points, viewpad.t())[:, :3]
        pixel_points[:, :2] = pixel_points[:, :2] / \
                              pixel_points[:, 2:3]

        pixel_uv = pixel_points[:, :2].round().long()
        height, width = sparse_depth.shape
        valid_mask = (pixel_uv[:, 0] >= 0) & (
                pixel_uv[:, 0] <= width - 1) & (pixel_uv[:, 1] >= 0) & (pixel_uv[:, 1] <= height - 1)

        valid_pixel_uv = pixel_uv[valid_mask]
        valid_depth = camera_points[..., 2][valid_mask]

        sparse_depth[valid_pixel_uv[:, 1], valid_pixel_uv[:, 0]
        ] = valid_depth
        sparse_depth = sparse_depth.numpy()

        # np.save(os.path.join(depth_path, idx + token + '.npy'), sparse_depth)
        pq.write_table(pa.table({'depth': sparse_depth.flatten()}), os.path.join(depth_path, idx + token + '.parquet'),
                       compression='BROTLI')

        # print(f'finish depth processing token = {token}')

    def nuscenes_gt_to_kitti(self) -> None:
        """
        Converts nuScenes GT annotations to KITTI format.
        """
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
        imsize = (1600, 900)

        token_idx = 0  # Start tokens from 0.

        # Get assignment of scenes to splits.
        split_logs = create_splits_logs(self.split, self.nusc)

        # Create output folders.
        label_folder = os.path.join(self.nusc_kitti_dir, self.split, self.cam_name, 'label_2')
        calib_folder = os.path.join(self.nusc_kitti_dir, self.split, self.cam_name, 'calib')
        image_folder = os.path.join(self.nusc_kitti_dir, self.split, self.cam_name, 'image_2')
        lidar_folder = os.path.join(self.nusc_kitti_dir, self.split, self.cam_name, 'velodyne')
        pose_folder = os.path.join(self.nusc_kitti_dir, self.split, self.cam_name, 'pose')
        depth_folder = os.path.join(self.nusc_kitti_dir, self.split, self.cam_name, 'depth')
        for folder in [label_folder, calib_folder, image_folder, lidar_folder, pose_folder, depth_folder]:
            if not os.path.isdir(folder):
                os.makedirs(folder)

        # Use only the samples from the current split.
        sample_tokens = self._split_to_samples(split_logs)
        sample_tokens = sample_tokens[:self.image_count]

        # write to list file
        token_list_filepath = os.path.join(self.nusc_kitti_dir, self.split, 'token_list.txt')
        token_list_file = open(token_list_filepath, 'w')

        tokens = []
        for sample_token in tqdm.tqdm(sample_tokens):

            # Get sample data.
            sample = self.nusc.get('sample', sample_token)
            sample_annotation_tokens = sample['anns']
            cam_token = sample['data'][self.cam_name]
            lidar_token = sample['data'][self.lidar_name]

            # Retrieve sensor records.
            sd_record_cam = self.nusc.get('sample_data', cam_token)
            sd_record_lid = self.nusc.get('sample_data', lidar_token)
            cs_record_cam = self.nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
            cs_record_lid = self.nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])

            # Combine transformations and convert to KITTI format.
            # Note: cam uses same conventions in KITTI and nuScenes.
            lid_to_ego = transform_matrix(cs_record_lid['translation'], Quaternion(cs_record_lid['rotation']),
                                          inverse=False)
            ego_to_cam = transform_matrix(cs_record_cam['translation'], Quaternion(cs_record_cam['rotation']),
                                          inverse=True)
            velo_to_cam = np.dot(ego_to_cam, lid_to_ego)

            # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
            velo_to_cam_kitti = np.dot(velo_to_cam, kitti_to_nu_lidar.transformation_matrix)

            # Currently not used.
            imu_to_velo_kitti = np.eye(3, 4)  # np.zeros((3, 4))  # Dummy values.
            r0_rect = Quaternion(axis=[1, 0, 0], angle=0)  # Dummy values.

            # Projection matrix.
            p_left_kitti = np.zeros((3, 4))
            p_left_kitti[:3, :3] = cs_record_cam['camera_intrinsic']  # Cameras are always rectified.

            # Create KITTI style transforms.
            velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
            velo_to_cam_trans = velo_to_cam_kitti[:3, 3]

            # Check that the rotation has the same format as in KITTI.
            # assert (velo_to_cam_rot.round(0) == np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])).all()
            # assert (velo_to_cam_trans[1:3] < 0).all()

            # Retrieve the token from the lidar.
            # Note that this may be confusing as the filename of the camera will include the timestamp of the lidar,
            # not the camera.
            filename_cam_full = sd_record_cam['filename']
            filename_lid_full = sd_record_lid['filename']
            token = '%06d_' % token_idx  # Alternative to use KITTI names.
            token_idx += 1

            # save token to token list file
            token_list_file.write(token + sample_token + '\n')

            # save ego pose (we found that lid time stamp is more accurate) jako
            ego_pose = self.nusc.get('ego_pose', sd_record_lid['ego_pose_token'])
            ego_pose_matrix = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']),
                                               inverse=False)
            pose_file = os.path.join(pose_folder, token + sample_token + '.npy')
            np.save(pose_file, ego_pose_matrix)

            # save depth
            self.process_depth_sample(sample_token, token, depth_folder)

            # Convert image (jpg to png).
            src_im_path = os.path.join(self.nusc.dataroot, filename_cam_full)
            dst_im_path = os.path.join(image_folder, token + sample_token + '.png')
            if not os.path.exists(dst_im_path):
                im = Image.open(src_im_path)
                im.save(dst_im_path, "PNG")

            # Convert lidar.
            # Note that we are only using a single sweep, instead of the commonly used n sweeps.
            src_lid_path = os.path.join(self.nusc.dataroot, filename_lid_full)
            dst_lid_path = os.path.join(lidar_folder, token + sample_token + '.bin')
            assert not dst_lid_path.endswith('.pcd.bin')
            pcl = LidarPointCloud.from_file(src_lid_path)
            pcl.rotate(kitti_to_nu_lidar_inv.rotation_matrix)  # In KITTI lidar frame.
            with open(dst_lid_path, "w") as lid_file:
                pcl.points.T.tofile(lid_file)

            # Add to tokens.
            tokens.append(sample_token)

            # Create calibration file.
            kitti_transforms = dict()
            kitti_transforms['P0'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['P1'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['P2'] = p_left_kitti  # Left camera transform.
            kitti_transforms['P3'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['R0_rect'] = r0_rect.rotation_matrix  # Cameras are already rectified.
            kitti_transforms['Tr_velo_to_cam'] = np.hstack((velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
            kitti_transforms['Tr_imu_to_velo'] = imu_to_velo_kitti
            calib_path = os.path.join(calib_folder, token + sample_token + '.txt')
            with open(calib_path, "w") as calib_file:
                for (key, val) in kitti_transforms.items():
                    val = val.flatten()
                    val_str = '%.12e' % val[0]
                    for v in val[1:]:
                        val_str += ' %.12e' % v
                    calib_file.write('%s: %s\n' % (key, val_str))

            # Write label file.
            label_path = os.path.join(label_folder, token + sample_token + '.txt')
            if os.path.exists(label_path):
                # print('Skipping existing file: %s' % label_path)
                continue
            # else:
            # print('Writing file: %s' % label_path)
            with open(label_path, "w") as label_file:
                for sample_annotation_token in sample_annotation_tokens:
                    sample_annotation = self.nusc.get('sample_annotation', sample_annotation_token)

                    # Get box in LIDAR frame.
                    _, box_lidar_nusc, _ = self.nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE,
                                                                     selected_anntokens=[sample_annotation_token])
                    box_lidar_nusc = box_lidar_nusc[0]

                    # Truncated: Set all objects to 0 which means untruncated.
                    truncated = 0.0

                    # Occluded: Set all objects to full visibility as this information is not available in nuScenes.
                    occluded = 0

                    # Convert nuScenes category to nuScenes detection challenge category.
                    detection_name = category_to_detection_name(sample_annotation['category_name'])

                    # Skip categories that are not part of the nuScenes detection challenge.
                    if detection_name is None:
                        continue

                    # Convert from nuScenes to KITTI box format.
                    box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
                        box_lidar_nusc, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect)

                    # Project 3d box to 2d box in image, ignore box if it does not fall inside.
                    bbox_2d = KittiDB.project_kitti_box_to_image(box_cam_kitti, p_left_kitti, imsize=imsize)
                    if bbox_2d is None:
                        continue

                    # Set dummy score so we can use this file as result.
                    box_cam_kitti.score = 0

                    # Convert box to output string format.
                    output = KittiDB.box_to_string(name=detection_name, box=box_cam_kitti, bbox_2d=bbox_2d,
                                                   truncation=truncated, occlusion=occluded)

                    # Write to disk.
                    label_file.write(output + '\n')

    def render_kitti(self, render_2d: bool) -> None:
        """
        Renders the annotations in the KITTI dataset from a lidar and a camera view.
        :param render_2d: Whether to render 2d boxes (only works for camera data).
        """
        if render_2d:
            print('Rendering 2d boxes from KITTI format')
        else:
            print('Rendering 3d boxes projected from 3d KITTI format')

        # Load the KITTI dataset.
        kitti = KittiDB(root=self.nusc_kitti_dir, splits=(self.split,))

        # Create output folder.
        render_dir = os.path.join(self.nusc_kitti_dir, 'render')
        if not os.path.isdir(render_dir):
            os.mkdir(render_dir)

        # Render each image.
        for token in kitti.tokens[:self.image_count]:
            for sensor in ['lidar', 'camera']:
                out_path = os.path.join(render_dir, '%s_%s.png' % (token, sensor))
                print('Rendering file to disk: %s' % out_path)
                kitti.render_sample_data(token, sensor_modality=sensor, out_path=out_path, render_2d=render_2d)
                plt.close()  # Close the windows to avoid a warning of too many open windows.

    def kitti_res_to_nuscenes(self, meta: Dict[str, bool] = None) -> None:
        """
        Converts a KITTI detection result to the nuScenes detection results format.
        :param meta: Meta data describing the method used to generate the result. See nuscenes.org/object-detection.
        """
        # Dummy meta data, please adjust accordingly.
        if meta is None:
            meta = {
                'use_camera': False,
                'use_lidar': True,
                'use_radar': False,
                'use_map': False,
                'use_external': False,
            }

        # Init.
        results = {}

        # Load the KITTI dataset.
        kitti = KittiDB(root=self.nusc_kitti_dir, splits=(self.split,))

        # Get assignment of scenes to splits.
        split_logs = create_splits_logs(self.split, self.nusc)

        # Use only the samples from the current split.
        sample_tokens = self._split_to_samples(split_logs)
        sample_tokens = sample_tokens[:self.image_count]

        for sample_token in sample_tokens:
            # Get the KITTI boxes we just generated in LIDAR frame.
            kitti_token = '%s_%s' % (self.split, sample_token)
            boxes = kitti.get_boxes(token=kitti_token)

            # Convert KITTI boxes to nuScenes detection challenge result format.
            sample_results = [self._box_to_sample_result(sample_token, box) for box in boxes]

            # Store all results for this image.
            results[sample_token] = sample_results

        # Store submission file to disk.
        submission = {
            'meta': meta,
            'results': results
        }
        submission_path = os.path.join(self.nusc_kitti_dir, 'submission.json')
        print('Writing submission to: %s' % submission_path)
        with open(submission_path, 'w') as f:
            json.dump(submission, f, indent=2)

    def _box_to_sample_result(self, sample_token: str, box: Box, attribute_name: str = '') -> Dict[str, Any]:
        # Prepare data
        translation = box.center
        size = box.wlh
        rotation = box.orientation.q
        velocity = box.velocity
        detection_name = box.name
        detection_score = box.score

        # Create result dict
        sample_result = dict()
        sample_result['sample_token'] = sample_token
        sample_result['translation'] = translation.tolist()
        sample_result['size'] = size.tolist()
        sample_result['rotation'] = rotation.tolist()
        sample_result['velocity'] = velocity.tolist()[:2]  # Only need vx, vy.
        sample_result['detection_name'] = detection_name
        sample_result['detection_score'] = detection_score
        sample_result['attribute_name'] = attribute_name

        return sample_result

    def _split_to_samples(self, split_logs: List[str]) -> List[str]:
        """
        Convenience function to get the samples in a particular split.
        :param split_logs: A list of the log names in this split.
        :return: The list of samples.
        """
        samples = []
        for sample in self.nusc.sample:
            scene = self.nusc.get('scene', sample['scene_token'])
            log = self.nusc.get('log', scene['log_token'])
            logfile = log['logfile']
            if logfile in split_logs:
                samples.append(sample['token'])
        return samples


if __name__ == '__main__':
    import argparse
    from concurrent import futures

    parser = argparse.ArgumentParser(description='nusc converter')

    # parser.add_argument('--nusc_data_root', type=str, default='/nuscenes/')
    # parser.add_argument('--nusc_kitti_dir', type=str, default='/nusc_kitti/')
    parser.add_argument('--nusc_data_root', type=str, default='/root/autodl-tmp/zhouxiaoyu/nuscenes')
    parser.add_argument('--nusc_kitti_dir', type=str, default='/root/autodl-tmp/zhouzhexian/nusc_kitti/')
    parser.add_argument('--multi', type=bool, default=False)

    args = parser.parse_args()

    camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

    with futures.ThreadPoolExecutor() as executor:
        for camera_name in tqdm.tqdm(camera_names):
            for split in ['mini_train', 'mini_val']:
                converter = KittiConverter(nusc_data_root=args.nusc_data_root,
                                           nusc_kitti_dir=args.nusc_kitti_dir,
                                           split=split,
                                           cam_name=camera_name)
                if args.multi:
                    executor.submit(converter.nuscenes_gt_to_kitti)
                else:
                    converter.nuscenes_gt_to_kitti()
        # fire.Fire(KittiConverter(cam_name=camera_name))
