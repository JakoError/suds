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
                   nusc, 
                   ) -> \
        Tuple[List[ImageMetadata], List[str], torch.Tensor, float, torch.Tensor]:
    metadata_items: List[ImageMetadata] = []
    static_masks = []
    min_bounds = None
    max_bounds = None

    train_val_idx = 0

    for is_val, sub_dir in [(False, 'mini_train'), (True, 'mini_val')]:
        nusc_sequence_list = []
        with open(os.path.join(nusc_kitti_root, sub_dir, 'token_list.txt'), 'r') as f:
            nusc_sequence_list = [line.strip() for line in f.readlines()]

        for nusc_idx, nusc_sequence in enumerate(tqdm(nusc_sequence_list)):
            nusc_token = nusc_sequence[7:]
            if nusc is not None:
                timestamp = nusc.get('sample', nusc_token)['timestamp']
            else:
                timestamp = train_val_idx
            
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

                imu_pose = torch.asarray(np.load(os.path.join(cam_dir, 'pose', f'{nusc_sequence}.npy')))

                for kitti_camera, transformation, intrinsics in [('2', P22imu, P2)]:
                    image_index = len(metadata_items)
                    c2w = ((imu_pose @ transformation) @ OPENCV_TO_OPENGL)[:3]

                    # is_val = image_index // 2 in val_frames

                    backward_neighbor = image_index - 6
                    forward_neighbor = image_index + 6

                    if nusc_idx == 0:
                        backward_neighbor = None
                    if nusc_idx + 1 >= len(nusc_sequence_list):
                        forward_neighbor = None

                    backward_flow_path = f'{cam_dir}/dino_correspondences/{nusc_sequence_list[nusc_idx - 1]}.parquet' if nusc_idx != 0 else None
                    forward_flow_path = f'{cam_dir}/dino_correspondences/{nusc_sequence_list[nusc_idx + 1]}.parquet' if nusc_idx+1 < len(nusc_sequence_list) else None

                    image_path = os.path.join(cam_dir, 'image_2', f'{nusc_sequence}.png')
                    image = image_from_stream(image_path)

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
                        os.path.join(cam_dir, 'dino', f'{nusc_sequence}.parquet'),
                        backward_flow_path,
                        forward_flow_path,
                        backward_neighbor,
                        forward_neighbor,
                        is_val,
                        1,
                        None
                    )

                    metadata_items.append(item)

                    min_bounds, max_bounds = get_bounds_from_depth(item, min_bounds, max_bounds)

            train_val_idx += 1

    origin, pose_scale_factor, scene_bounds = scale_bounds(metadata_items, min_bounds, max_bounds)

    return metadata_items, static_masks, origin, pose_scale_factor, scene_bounds


def _get_opts() -> Namespace:
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    # parser.add_argument('--config_file', is_config_file=True)

    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--nusc_version', type=str, default='v1.0-mini')
    parser.add_argument('--nusc_root', type=str, default=None, required=True)
    parser.add_argument('--nusc_kitti_root', type=str, default=None, required=True)
    parser.add_argument('--nusc_timestamp', type=bool, default=False)
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

    nusc = NuScenes(version='v1.0-mini', dataroot=hparams.nusc_root) if hparams.nusc_timestamp else None

    metadata_items, static_masks, origin, pose_scale_factor, scene_bounds = get_nusc_items(hparams.nusc_kitti_root,
                                                                                           ['CAM_FRONT',
                                                                                            'CAM_FRONT_RIGHT',
                                                                                            'CAM_BACK_RIGHT',
                                                                                            'CAM_BACK', 'CAM_BACK_LEFT',
                                                                                            'CAM_FRONT_LEFT'], nusc )

    write_metadata(hparams.output_path, metadata_items, static_masks, origin, pose_scale_factor, scene_bounds)


if __name__ == '__main__':
    main(_get_opts())
