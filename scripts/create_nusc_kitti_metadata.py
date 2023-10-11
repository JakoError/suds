import os.path
from argparse import Namespace
from typing import List, Tuple, Dict, Optional

import configargparse
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from smart_open import open
from tqdm import tqdm

from metadata_utils import get_bounds_from_depth, normalize_timestamp, scale_bounds, write_metadata, OPENCV_TO_OPENGL, \
    get_val_frames, get_neighbor, get_frame_range
from suds.data.image_metadata import ImageMetadata
from suds.stream_utils import image_from_stream


def get_nusc_items(nusc_kitti_root: str,
                   nusc_cameras: List[str],
                   frame_ranges: Optional[List[Tuple[int]]],
                   train_every: Optional[int],
                   test_every: Optional[int],
                   train_num,
                   val_num,
                   ) -> \
        Tuple[List[ImageMetadata], List[str], torch.Tensor, float, torch.Tensor]:
    metadata_items: List[ImageMetadata] = []
    item_frame_ranges: List[Tuple[int]] = []
    static_masks = []
    min_bounds = None
    max_bounds = None

    nusc_sequence_list = []
    full_sequence_list = []
    train_val_idx = 0
    for is_val, sub_dir in [(False, 'mini_train'), (True, 'mini_val')]:
        with open(os.path.join(nusc_kitti_root, sub_dir, 'token_list.txt'), 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            sequences = []
            for line in lines:
                sequences.append((line, sub_dir, is_val, train_val_idx))
                train_val_idx += 1
            full_sequence_list += sequences
            # cut before append
            if not is_val and train_num is not None:
                sequences = sequences[:train_num]
            if is_val and val_num is not None:
                sequences = sequences[:val_num]
            nusc_sequence_list += sequences

    num_frames = len(nusc_sequence_list)
    val_frames = get_val_frames(num_frames, test_every, train_every)

    for nusc_idx, (nusc_sequence, sub_dir, is_val, timestamp) in enumerate(tqdm(nusc_sequence_list)):
        frame_range = get_frame_range(frame_ranges, nusc_idx) if frame_ranges is not None else None
        nusc_token = nusc_sequence[7:]

        is_val = nusc_idx in val_frames

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

                backward_neighbor = image_index - len(nusc_cameras)
                forward_neighbor = image_index + len(nusc_cameras)

                if nusc_idx == 0:
                    backward_neighbor = None
                if nusc_idx + 1 >= len(nusc_sequence_list):
                    forward_neighbor = None

                backward_flow_path = f'{cam_dir}/dino_correspondences/{nusc_sequence_list[nusc_idx - 1][0]}.parquet' if backward_neighbor is not None else None
                forward_flow_path = f'{cam_dir}/dino_correspondences/{nusc_sequence_list[nusc_idx + 1][0]}.parquet' if forward_neighbor is not None else None

                # backward_flow_path = f'{cam_dir}/dino_correspondences/{nusc_sequence_list[nusc_idx - 1][0]}.parquet'
                # forward_flow_path = f'{cam_dir}/dino_correspondences/{nusc_sequence_list[nusc_idx + 1][0]}.parquet'

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
                item_frame_ranges.append(frame_range)

                min_bounds, max_bounds = get_bounds_from_depth(item, min_bounds, max_bounds)

    for item in tqdm(metadata_items, desc="normalize_timestamp"):
        normalize_timestamp(item, 0, len(full_sequence_list))

    for item in metadata_items:
        if item.backward_neighbor_index is None:
            pass
        elif item.backward_neighbor_index < 0 \
                or item_frame_ranges[item.image_index] != item_frame_ranges[item.backward_neighbor_index]:
            item.backward_flow_path = None
            item.backward_neighbor_index = None

        if item.forward_neighbor_index is None:
            pass
        elif item.forward_neighbor_index >= len(metadata_items) \
                or item_frame_ranges[item.image_index] != item_frame_ranges[item.forward_neighbor_index]:
            item.forward_flow_path = None
            item.forward_neighbor_index = None

    origin, pose_scale_factor, scene_bounds = scale_bounds(metadata_items, min_bounds, max_bounds)

    return metadata_items, static_masks, origin, pose_scale_factor, scene_bounds


def _get_opts() -> Namespace:
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--config_file', is_config_file=True)

    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--nusc_version', type=str, default='v1.0-mini')
    parser.add_argument('--nusc_root', type=str, default=None)
    parser.add_argument('--nusc_kitti_root', type=str, default=None, required=True)
    # parser.add_argument('--nusc_timestamp', type=bool, default=False)
    parser.add_argument('--nusc_cameras', type=str, nargs='+', default=None)
    parser.add_argument('--frame_ranges', type=int, nargs='+', default=None)
    parser.add_argument('--train_every', type=int, default=None)
    parser.add_argument('--test_every', type=int, default=None)
    parser.add_argument('--train_num', type=int, default=None)
    parser.add_argument('--val_num', type=int, default=None)
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

    # nusc = NuScenes(version='v1.0-mini', dataroot=hparams.nusc_root) if hparams.nusc_timestamp else None

    if hparams.frame_ranges is not None:
        frame_ranges = []
        for i in range(0, len(hparams.frame_ranges), 2):
            frame_ranges.append([hparams.frame_ranges[i], hparams.frame_ranges[i + 1]])
    else:
        frame_ranges = None

    camera_set = {'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'}
    if hparams.nusc_cameras is not None:
        for camera in hparams.nusc_cameras:
            if camera not in camera_set:
                raise ValueError(f'camera {camera} is illegal must in {camera_set}')
        nusc_cameras = hparams.nusc_cameras
    else:
        nusc_cameras = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT']

    print(f'camera: {nusc_cameras}')

    metadata_items, static_masks, origin, pose_scale_factor, scene_bounds = get_nusc_items(hparams.nusc_kitti_root,
                                                                                           nusc_cameras,
                                                                                           # nusc,
                                                                                           frame_ranges=frame_ranges,
                                                                                           train_every=hparams.train_every,
                                                                                           test_every=hparams.test_every,
                                                                                           train_num=hparams.train_num,
                                                                                           val_num=hparams.val_num,
                                                                                           )

    write_metadata(hparams.output_path, metadata_items, static_masks, origin, pose_scale_factor, scene_bounds)
    print(f'write to metadata file: {hparams.output_path}')

if __name__ == '__main__':
    main(_get_opts())
