import os.path
from argparse import Namespace
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import configargparse
import numpy as np
import torch
from smart_open import open
from tqdm import tqdm

from metadata_utils import get_frame_range, get_bounds_from_depth, normalize_timestamp, scale_bounds, get_neighbor, \
    write_metadata, get_val_frames, OPENCV_TO_OPENGL
from suds.data.image_metadata import ImageMetadata
from suds.stream_utils import image_from_stream, get_filesystem

from nuscenes.nuscenes import NuScenes


def get_nusc_items(nusc_kitti_root: str,
                   nusc_cameras: List[str],
                   nusc, ) -> \
        Tuple[List[ImageMetadata], List[str], torch.Tensor, float, torch.Tensor]:
    metadata_items: List[ImageMetadata] = []
    static_masks = []
    min_bounds = None
    max_bounds = None

    for is_val, sub_dir in [(False, 'mini_train'), (True, 'mini_val')]:
        nusc_sequence_list = []
        with open(os.path.join(nusc_kitti_root, sub_dir, 'token_list.txt'), 'r') as f:
            nusc_sequence_list = [line.strip() for line in f.readlines()]

        for nusc_sequence in tqdm(nusc_sequence_list):
            nusc_token = nusc_sequence[7:]
            timestamp = nusc.get('sample', nusc_token)['timestamp']
            for nusc_camera in nusc_cameras:
                cam_dir = os.path.join(nusc_kitti_root, sub_dir, nusc_camera)

                calib: Dict[str, torch.Tensor] = {}
                with open(os.path.join(cam_dir, 'calib', f'{nusc_sequence}.txt'), 'r') as f:
                    for line in f:
                        tokens = line.strip().split()
                        calib[tokens[0]] = torch.DoubleTensor([float(x) for x in tokens[1:]])

                # print(calib)

                imu2velo = torch.eye(4, dtype=torch.float64)
                imu2velo[:3] = calib['Tr_imu_to_velo:'].view(3, 4)

                velo2cam_base = torch.eye(4, dtype=torch.float64)
                velo2cam_base[:3] = calib['Tr_velo_to_cam:'].view(3, 4)

                cam_base2rect = torch.eye(4, dtype=torch.float64)
                cam_base2rect[:3, :3] = calib['R0_rect:'].view(3, 3)

                P2 = calib['P2:'].view(3, 4)
                K_inv = torch.inverse(P2[:, :3])
                R_t = P2[:, 3]
                rect2P2 = torch.eye(4, dtype=torch.float64)
                rect2P2[:3, 3] = torch.matmul(K_inv, R_t)
                P22imu = torch.inverse(rect2P2 @ cam_base2rect @ velo2cam_base @ imu2velo)

                # P3 = calib['P3:'].view(3, 4)
                # K_inv = torch.inverse(P3[:, :3])
                # R_t = P3[:, 3]
                # rect2P3 = torch.eye(4, dtype=torch.float64)
                # rect2P3[:3, 3] = torch.matmul(K_inv, R_t)
                # P32imu = torch.inverse(rect2P3 @ cam_base2rect @ velo2cam_base @ imu2velo)

                # val_frames = get_val_frames(num_frames, test_every, train_every)
                # item_frame_ranges: List[Tuple[int]] = []

                # use_masks = True
                # min_frame = None
                # max_frame = None
                # scale = None

                imu_pose = torch.asarray(np.load(os.path.join(cam_dir, 'pose', f'{nusc_sequence}.npy')))

                for kitti_camera, transformation, intrinsics in [('2', P22imu, P2)]:
                    image_index = len(metadata_items)
                    c2w = ((imu_pose @ transformation) @ OPENCV_TO_OPENGL)[:3]

                    # is_val = image_index // 2 in val_frames

                    # if is_val:
                    #     backward_neighbor = image_index - 2
                    #     forward_neighbor = image_index + 2
                    # else:
                    #     backward_neighbor = get_neighbor(image_index, val_frames, -2)
                    #     forward_neighbor = get_neighbor(image_index, val_frames, 2)
                    #
                    # backward_suffix = '' if (image_index - backward_neighbor) // 2 == 1 else '-{}'.format(
                    #     (image_index - backward_neighbor) // 2)
                    # forward_suffix = '' if (forward_neighbor - image_index) // 2 == 1 else '-{}'.format(
                    #     (forward_neighbor - image_index) // 2)

                    # backward_flow_path = '{0}/dino_correspondences_0{1}{2}/{3}/{4:06d}.parquet'.format(nusc_kitti_root,
                    #                                                                                    camera,
                    #                                                                                    backward_suffix,
                    #                                                                                    nusc_sequence,
                    #                                                                                    frame - (
                    #                                                                                            image_index - backward_neighbor) // 2)
                    # forward_flow_path = '{0}/dino_correspondences_0{1}{2}/{3}/{4:06d}.parquet'.format(nusc_kitti_root,
                    #                                                                                   camera,
                    #                                                                                   forward_suffix,
                    #                                                                                   nusc_sequence,
                    #                                                                                   frame)

                    image_path = os.path.join(cam_dir, 'image_2', f'{nusc_sequence}.png')
                    image = image_from_stream(image_path)

                    # sky_mask_path = '{0}/sky_0{1}/{2}/{3:06d}.png'.format(nusc_kitti_root, kitti_camera, nusc_sequence, frame) \
                    #     if (kitti_camera == '2' and use_masks) else None
                    # if sky_mask_path is not None and use_masks:
                    #     fs = get_filesystem(sky_mask_path)
                    #     if (fs is None and (not Path(sky_mask_path).exists())) or \
                    #             (fs is not None and (not fs.exists(sky_mask_path))):
                    #         print('Did not find sky mask at {} - not including static or sky masks in metadata'.format(
                    #             sky_mask_path))
                    #         use_masks = False
                    #         sky_mask_path = None

                    item = ImageMetadata(
                        image_path,
                        c2w,
                        image.size[0],
                        image.size[1],
                        torch.DoubleTensor(
                            [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]]),
                        image_index,
                        timestamp,
                        0,
                        os.path.join(cam_dir, 'depth', f'{nusc_sequence}.parquet'),
                        None,
                        None,  # sky_mask_path,
                        None,
                        # '{0}/dino_0{1}/{2}/{3:06d}.parquet'.format(nusc_kitti_root, camera, nusc_sequence, frame),
                        None,  # backward_flow_path,
                        None,  # forward_flow_path,
                        None,  # backward_neighbor,
                        None,  # forward_neighbor,
                        is_val,
                        1,
                        None
                    )

                    metadata_items.append(item)
                    # item_frame_ranges.append(frame_range)
                    #
                    # if use_masks:
                    #     static_mask_path = '{0}/static_02/{1}/{2:06d}.png'.format(nusc_kitti_root, nusc_sequence, frame) \
                    #         if kitti_camera == '2' else '{0}/all-false.png'.format(nusc_kitti_root)
                    #     static_masks.append(static_mask_path)
                    #
                    min_bounds, max_bounds = get_bounds_from_depth(item, min_bounds, max_bounds)

    # for item in metadata_items:
    #     normalize_timestamp(item, min_frame, max_frame)

    # for item in metadata_items:
    #     if item.backward_neighbor_index < 0 \
    #             or item_frame_ranges[item.image_index] != item_frame_ranges[item.backward_neighbor_index]:
    #         item.backward_flow_path = None
    #         item.backward_neighbor_index = None
    #
    #     if item.forward_neighbor_index >= len(metadata_items) \
    #             or item_frame_ranges[item.image_index] != item_frame_ranges[item.forward_neighbor_index]:
    #         item.forward_flow_path = None
    #         item.forward_neighbor_index = None

    origin, pose_scale_factor, scene_bounds = scale_bounds(metadata_items, min_bounds, max_bounds)

    return metadata_items, static_masks, origin, pose_scale_factor, scene_bounds


def _get_opts() -> Namespace:
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    # parser.add_argument('--config_file', is_config_file=True)

    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--nusc_version', type=str, default='v1.0-mini')
    parser.add_argument('--nusc_root', type=str, default=None, required=True)
    parser.add_argument('--nusc_kitti_root', type=str, default=None, required=True)
    # parser.add_argument('--kitti_sequence', type=str, required=True)
    # parser.add_argument('--frame_ranges', type=int, nargs='+', default=None)
    # parser.add_argument('--train_every', type=int, default=None)
    # parser.add_argument('--test_every', type=int, default=None)

    return parser.parse_args()


def main(hparams: Namespace) -> None:
    # assert hparams.train_every is not None or hparams.test_every is not None, \
    #     'Exactly one of train_every or test_every must be specified'
    #
    # assert hparams.train_every is None or hparams.test_every is None, \
    #     'Only one of train_every or test_every must be specified'

    # if hparams.frame_ranges is not None:
    #     frame_ranges = []
    #     for i in range(0, len(hparams.frame_ranges), 2):
    #         frame_ranges.append([hparams.frame_ranges[i], hparams.frame_ranges[i + 1]])
    # else:
    #     frame_ranges = None

    nusc = NuScenes(version='v1.0-mini', dataroot=hparams.nusc_root)

    metadata_items, static_masks, origin, pose_scale_factor, scene_bounds = get_nusc_items(hparams.nusc_kitti_root,
                                                                                           ['CAM_FRONT',
                                                                                            'CAM_FRONT_RIGHT',
                                                                                            'CAM_BACK_RIGHT',
                                                                                            'CAM_BACK', 'CAM_BACK_LEFT',
                                                                                            'CAM_FRONT_LEFT'], nusc)

    write_metadata(hparams.output_path, metadata_items, static_masks, origin, pose_scale_factor, scene_bounds)


if __name__ == '__main__':
    main(_get_opts())
