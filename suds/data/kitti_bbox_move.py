import glob
import os
from pathlib import Path
from argparse import Namespace
from typing import List, Tuple

import configargparse
import numpy as np
import seaborn as sns
import torch
import tqdm
from PIL import Image, ImageDraw

from collections import defaultdict

colors = sns.color_palette('Paired', 9 * 2)
names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']

TYPE_INDEX = 0
ALPHA_INDEX = 3
X_INDEX = 11
Y_INDEX = 12
Z_INDEX = 13
RY_INDEX = 14

BBOX_IMAGE_DIR_NAME = 'bbox_image'

# TYPE_INDEX = 2
# ALPHA_INDEX = 5
# X_INDEX = 13
# Y_INDEX = 14
# Z_INDEX = 15
# RY_INDEX = 16

img_filename_pattern = '*.*'


# From https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/devkits/convertOxtsPose/python/utils.py
def _lat_lon_to_mercator(lat: float, lon: float, scale: float) -> Tuple[float, float]:
    """ converts lat/lon coordinates to mercator coordinates using mercator scale """
    er = 6378137.  # average earth radius at the equator

    mx = scale * lon * np.pi * er / 180
    my = scale * er * np.log(np.tan((90 + lat) * np.pi / 360))

    return mx, my


# From https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/devkits/convertOxtsPose/python/utils.py
def _lat_to_scale(lat: float) -> float:
    """ compute mercator scale from latitude """
    scale = np.cos(lat * np.pi / 180.0)
    return scale


# From https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/devkits/convertOxtsPose/python/utils.py
def mercatorToLatlon(mx, my, scale):
    """ converts mercator coordinates using mercator scale to lat/lon coordinates """
    er = 6378137.  # average earth radius at the equator

    lon = mx * 180. / (scale * np.pi * er)
    lat = 360. / np.pi * np.arctan(np.exp(my / (scale * er))) - 90.
    return lat, lon


def get_kitti_label_files(kitti_root):
    return glob.glob(os.path.join(kitti_root, 'label_2', '*.txt'))


def get_kitti_rendered_image_paths(render_path: str, video_id='0', camera_index=0):
    if video_id is not None:
        img_path = os.path.join(render_path, video_id, f'camera_{camera_index}', img_filename_pattern)
    else:
        img_path = os.path.join(render_path, f'camera_{camera_index}', img_filename_pattern)
    paths = glob.glob(img_path)
    return paths


def get_img_index(filename):
    return int(Path(filename).stem.split('-')[1])


def read_label_by_frame(label_filepath: str):
    frame_labels_map: defaultdict[int, list] = defaultdict(list)
    with open(label_filepath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    for line in labels:
        info = line.split(' ')
        assert len(info) == 17, 'info option number must be 17'
        frame = int(info[0])
        frame_labels_map[frame].append(info[2:])
    return frame_labels_map


def read_calib_P2(calib_filepath):
    with open(calib_filepath, 'r') as f:
        lines = f.readlines()
        P2 = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
    return P2


def read_calib_P3(calib_filepath):
    with open(calib_filepath, 'r') as f:
        lines = f.readlines()
        P3 = np.array(lines[3].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
    return P3


def bbox_to_image(image: Image.Image, labels, P2):
    # deep copy
    image = image.copy()

    # load labels
    # with open(label_filepath, 'r') as f:
    #     labels = f.readlines()

    # load calibration file
    # with open(calib_filepath, 'r') as f:
    #     lines = f.readlines()
    #     P2 = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)

    draw = ImageDraw.Draw(image)

    for line in labels:
        # line = line.split()
        lab, _, _, _, _, _, _, _, h, w, l, x, y, z, rot = line
        h, w, l, x, y, z, rot = map(float, [h, w, l, x, y, z, rot])
        if lab != 'DontCare':
            x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
            y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
            z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
            corners_3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

            # transform the 3d bbox from object coordiante to camera_0 coordinate
            R = np.array([[np.cos(rot), 0, np.sin(rot)],
                          [0, 1, 0],
                          [-np.sin(rot), 0, np.cos(rot)]])
            corners_3d = np.dot(R, corners_3d).T + np.array([x, y, z])

            # transform the 3d bbox from camera_0 coordinate to camera_x image
            corners_3d_hom = np.concatenate((corners_3d, np.ones((8, 1))), axis=1)
            corners_img = np.matmul(corners_3d_hom, P2.T)
            corners_img = corners_img[:, :2] / corners_img[:, 2][:, None]

            def line(p1, p2, front=1):
                draw.line([p1[0], p1[1], p2[0], p2[1]],
                          fill=tuple((np.asarray(colors[names.index(lab) * 2 + front]) * 255).astype(np.uint8)),
                          width=4)

            # draw the upper 4 horizontal lines
            line(corners_img[0], corners_img[1], 0)  # front = 0 for the front lines
            line(corners_img[1], corners_img[2])
            line(corners_img[2], corners_img[3])
            line(corners_img[3], corners_img[0])

            # draw the lower 4 horizontal lines
            line(corners_img[4], corners_img[5], 0)
            line(corners_img[5], corners_img[6])
            line(corners_img[6], corners_img[7])
            line(corners_img[7], corners_img[4])

            # draw the 4 vertical lines
            line(corners_img[4], corners_img[0], 0)
            line(corners_img[5], corners_img[1], 0)
            line(corners_img[6], corners_img[2])
            line(corners_img[7], corners_img[3])

    return image


def move_label_bbox(label_lines: list[list[str]], pos_shift, rotation):
    label_info = []
    for info in label_lines:
        assert len(info) == 15
        if info[TYPE_INDEX] != 'DontCare':
            alpha = float(info[ALPHA_INDEX])
            x, y, z = float(info[X_INDEX]), float(info[Y_INDEX]), float(info[Z_INDEX])
            rotation_y = float(info[RY_INDEX])

            # position
            x -= pos_shift[0]
            y += pos_shift[2]
            z -= pos_shift[1]

            # rotation
            if np.any(np.array(rotation) != 0):
                rx = rotation[0]  # roll
                ry = rotation[1]  # pitch
                rz = rotation[2]  # heading
                Rx = torch.FloatTensor([[1, 0, 0],
                                        [0, np.cos(rx), -np.sin(rx)],
                                        [0, np.sin(rx), np.cos(rx)]])
                Ry = torch.FloatTensor([[np.cos(ry), 0, np.sin(ry)],
                                        [0, 1, 0],
                                        [-np.sin(ry), 0, np.cos(ry)]])
                Rz = torch.FloatTensor([[np.cos(rz), -np.sin(rz), 0],
                                        [np.sin(rz), np.cos(rz), 0],
                                        [0, 0, 1]])
                rot_mat = torch.matmul(torch.matmul(Rz, Ry), Rx)
                point = torch.FloatTensor([x, y, z])
                point = torch.matmul(rot_mat, point)
                point = point.cpu().numpy()
                x, y, z = point[0], point[1], point[2]

            # beta = rotation_y - alpha
            if rotation[1] != 0:
                # beta = np.arctan2(x, z) + rotation[1]
                rotation_y += rotation[1]
                # alpha = rotation_y + beta不变

            info[RY_INDEX] = str(rotation_y)
            info[X_INDEX], info[Y_INDEX], info[Z_INDEX] = str(x), str(y), str(z)
        label_info.append(info)
    # if target_label_path is not None:
    #     with open(target_label_path, 'w') as f:
    #         f.writelines('\n'.join([' '.join(info) for info in label_info]))
    return label_info


def save_integrate_label_file(label_filepath, moved_label_filepath, pos_shift, rotation):
    with open(label_filepath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    head_infos = []
    frame_infos = []
    for line in labels:
        info = line.split(' ')
        assert len(info) == 17, 'info option number must be 17'
        head_infos.append(info[:2])
        frame_infos.append(info[2:])
    frame_infos = move_label_bbox(frame_infos, pos_shift, rotation)
    infos = [head_infos[i] + frame_infos[i] for i in range(len(labels))]
    with open(moved_label_filepath, 'w') as f:
        f.writelines('\n'.join([' '.join(info) for info in infos]))


def save_split_label_file(label_filepath, moved_label_dir, pos_shift, rotation):
    frame_labels_map_origin = read_label_by_frame(label_filepath)
    frame_labels_map_moved: defaultdict[int, list] = defaultdict(list)

    for key, value in frame_labels_map_origin.items():
        frame_labels_map_moved[key] = move_label_bbox(value, pos_shift, rotation)

    for key, value in frame_labels_map_moved.items():
        with open(os.path.join(moved_label_dir, f'{key:06}.txt'), 'w') as f:
            f.writelines('\n'.join([' '.join(info) for info in value]))


# def oxt_to_pose(oxts):
#     scale = _lat_to_scale(oxts[0])
#
#     imu_pose = torch.eye(4, dtype=torch.float64)
#     imu_pose[0, 3], imu_pose[1, 3] = _lat_lon_to_mercator(oxts[0], oxts[1], scale)
#     imu_pose[2, 3] = oxts[2]
#
#     # From https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/devkits/convertOxtsPose/python/convertOxtsToPose.py
#     rx = oxts[3]  # roll
#     ry = oxts[4]  # pitch
#     rz = oxts[5]  # heading
#     Rx = torch.DoubleTensor([[1, 0, 0],
#                              [0, np.cos(rx), -np.sin(rx)],
#                              [0, np.sin(rx), np.cos(rx)]])  # base => nav  (level oxts => rotated oxts)
#     Ry = torch.DoubleTensor([[np.cos(ry), 0, np.sin(ry)],
#                              [0, 1, 0],
#                              [-np.sin(ry), 0, np.cos(ry)]])  # base => nav  (level oxts => rotated oxts)
#     Rz = torch.DoubleTensor([[np.cos(rz), -np.sin(rz), 0],
#                              [np.sin(rz), np.cos(rz), 0],
#                              [0, 0, 1]])  # base => nav  (level oxts => rotated oxts)
#     imu_pose[:3, :3] = torch.matmul(torch.matmul(Rz, Ry), Rx)
#     return imu_pose
#
#
# def move_pose(pose, pos_shift, rotation, pose_scale_factor):
#     if pos_shift is not None:
#         pose[..., 3] += pos_shift.to(pose) / pose_scale_factor
#     if rotation is not None:
#         rx = rotation[0]  # roll
#         ry = rotation[1]  # pitch
#         rz = rotation[2]  # heading
#         Rx = torch.FloatTensor([[1, 0, 0],
#                                 [0, np.cos(rx), -np.sin(rx)],
#                                 [0, np.sin(rx), np.cos(rx)]])  # base => nav  (level oxts => rotated oxts)
#         Ry = torch.FloatTensor([[np.cos(ry), 0, np.sin(ry)],
#                                 [0, 1, 0],
#                                 [-np.sin(ry), 0, np.cos(ry)]])  # base => nav  (level oxts => rotated oxts)
#         Rz = torch.FloatTensor([[np.cos(rz), -np.sin(rz), 0],
#                                 [np.sin(rz), np.cos(rz), 0],
#                                 [0, 0, 1]])  # base => nav  (level oxts => rotated oxts)
#         pose[..., :3, :3] = pose[..., :3, :3] @ torch.matmul(torch.matmul(Rz, Ry), Rx)
#
#
# def pose_to_oxts(imu_pose):
#     scale = _lat_to_scale(origin_oxts[0])
#
#     ox, oy = _lat_lon_to_mercator(origin_oxts[0], origin_oxts[1], scale)
#     origin = np.array([ox, oy, 0])
#
#     # rotation and translation
#     R = imu_pose[0:3, 0:3]
#     t = imu_pose[0:3, 3]
#
#     # unnormalize translation
#     t = t + origin
#
#     # translation vector
#     lat, lon = mercatorToLatlon(t[0], t[1], scale)
#     alt = t[2]
#
#     # rotation matrix (OXTS RT3000 user manual, page 71/92)
#     yaw = np.arctan2(R[1, 0], R[0, 0])
#     pitch = np.arctan2(- R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
#     roll = np.arctan2(R[2, 1], R[2, 2])
#     return lat, lon, alt, roll, pitch, yaw


def save_integrate_oxts_file(oxts_filepath, moved_oxts_filepath, pos_shift, rotation):
    with open(oxts_filepath, 'r') as f:
        oxts = [line.strip() for line in f.readlines()]
    infos = []
    for oxt in oxts:
        info = [float(param) for param in oxt.split(' ')]
        assert len(info) == 30, 'info option number must be 30'
        lat, lon, alt, roll, pitch, yaw = info[0], info[1], info[2], info[3], info[4], info[5]

        lat += pos_shift[0]
        lon += pos_shift[1]
        alt += pos_shift[2]
        roll += rotation[0]
        pitch += rotation[1]
        yaw += rotation[2]

        info[0], info[1], info[2], info[3], info[4], info[5] = lat, lon, alt, roll, pitch, yaw
        infos.append([str(param) for param in info])
    with open(moved_oxts_filepath, 'w') as f:
        f.writelines('\n'.join([' '.join(info) for info in infos]))


def save_split_oxts_file(oxts_filepath, moved_oxts_dir, pos_shift, rotation):
    with open(oxts_filepath, 'r') as f:
        oxts = [line.strip() for line in f.readlines()]
    infos = []
    for oxt in oxts:
        info = [float(param) for param in oxt.split(' ')]
        assert len(info) == 30, 'info option number must be 30'
        lat, lon, alt, roll, pitch, yaw = info[0], info[1], info[2], info[3], info[4], info[5]

        lat += pos_shift[0]
        lon += pos_shift[1]
        alt += pos_shift[2]
        roll += rotation[0]
        pitch += rotation[1]
        yaw += rotation[2]

        info[0], info[1], info[2], info[3], info[4], info[5] = lat, lon, alt, roll, pitch, yaw
        infos.append([str(param) for param in info])
    for index, info in enumerate(infos):
        with open(os.path.join(moved_oxts_dir, f'{index:06}.txt'), 'w') as f:
            f.write(' '.join(info))


def bbox_image_transform(hparams):
    frame_labels_map_origin = read_label_by_frame(hparams.render_label_file)
    frame_labels_map_moved: defaultdict[int, list] = defaultdict(list)

    for key, value in frame_labels_map_origin.items():
        frame_labels_map_moved[key] = move_label_bbox(value, hparams.pos_shift, hparams.rotation)

    P2 = read_calib_P2(hparams.calib_file)
    P3 = read_calib_P3(hparams.calib_file)

    for camera_index in range(hparams.camera_num):
        if hparams.read_from_subdir:
            render_path = os.path.join(hparams.render_path, pos_rot_str(hparams.pos_shift, hparams.rotation))
        else:
            render_path = os.path.join(hparams.render_path)
        print(f'render_path {render_path}')
        for img_path in tqdm.tqdm(get_kitti_rendered_image_paths(render_path, camera_index=camera_index)):
            frame = int(get_img_index(img_path) / hparams.camera_num)
            img = Image.open(img_path)
            labels = frame_labels_map_moved[frame]

            if camera_index == 0:
                img_with_bbox = bbox_to_image(img, labels, P2)
            elif camera_index == 1:
                img_with_bbox = bbox_to_image(img, labels, P3)
            else:
                raise ValueError(f'unsupported camera {camera_index}')

            save_path = os.path.join(hparams.target_path, BBOX_IMAGE_DIR_NAME, f'camera_{camera_index}',
                                     os.path.basename(img_path))
            img_with_bbox.save(save_path)


def get_opts() -> Namespace:
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--config_file', is_config_file=True)

    parser.add_argument('--kitti_root', type=str, default='data/kitti/training/')
    parser.add_argument('--calib_file', type=str, default=None, help='Set None if you dont want to render')
    parser.add_argument('--render_path', type=str, default=None, help='Set None if you dont want to render')
    parser.add_argument('--render_label_file', type=str, default=None, help='Set None if you dont want to render')

    parser.add_argument('--target_path', type=str, default=None)
    parser.add_argument('--save_to_subdir', type=bool, default=True)
    parser.add_argument('--read_from_subdir', type=bool, default=True)

    parser.add_argument('--camera_num', type=int, default=2)

    parser.add_argument('--pos_shift', type=float, default=[0, 0, 0], nargs='+')
    parser.add_argument('--rotation', type=float, default=[0, 0, 0], nargs='+')
    # parser.add_argument('--pose_scale_factor', type=float, default=87.76204257170876)

    return parser.parse_args()


def pos_rot_str(pos_shift, rotation):
    def array_to_str(arr, seg=','):
        s = ''
        for idx, ele in enumerate(arr):
            if idx == 0:
                s += str(ele)
            else:
                s += seg + str(ele)
        return s.replace('.', '_')

    return f'pos_{array_to_str(pos_shift)}_rot_{array_to_str(rotation)}'


def main(hparams: Namespace) -> None:
    # pose_scale_factor = hparams.pose_scale_factor
    # for f in hparams.pos_shift:
    #     f /= hparams.pose_scale_factor

    # prepare paths
    if hparams.save_to_subdir:
        hparams.target_path = os.path.join(hparams.target_path, pos_rot_str(hparams.pos_shift, hparams.rotation))

    kitti_path = hparams.kitti_root
    kitti_oxts_path = os.path.join(kitti_path, 'oxts')
    kitti_label_path = os.path.join(kitti_path, 'label_02')

    target_path = hparams.target_path
    target_bbox_image_path = os.path.join(target_path, BBOX_IMAGE_DIR_NAME)
    target_oxts_path = os.path.join(target_path, 'oxts')
    target_label_path = os.path.join(target_path, 'label_02')

    os.makedirs(target_path, exist_ok=True)
    os.makedirs(target_bbox_image_path, exist_ok=True)
    for camera_index in range(hparams.camera_num):
        os.makedirs(os.path.join(target_path, BBOX_IMAGE_DIR_NAME, f'camera_{camera_index}'), exist_ok=True)
    os.makedirs(target_oxts_path, exist_ok=True)
    os.makedirs(target_label_path, exist_ok=True)

    # bbox_to_images
    if hparams.render_path is not None and hparams.render_label_file is not None:
        bbox_image_transform(hparams)
    else:
        print('skip render image with bbox')

    # save moved label file to target path

    if os.path.isdir(kitti_label_path):
        for label_file in tqdm.tqdm(glob.glob(os.path.join(kitti_label_path, '*.txt')), desc='label_files'):
            save_integrate_label_file(label_file,
                                      os.path.join(target_label_path, os.path.basename(label_file)),
                                      hparams.pos_shift, hparams.rotation)
            split_label_dir = os.path.join(target_label_path, Path(label_file).stem)
            os.makedirs(split_label_dir, exist_ok=True)
            save_split_label_file(label_file, split_label_dir, hparams.pos_shift, hparams.rotation)
    else:
        print(f'label_files({kitti_label_path}) is not a directory, skip label file transformation')

    # save moved calibration file to target path
    if os.path.isdir(kitti_oxts_path):
        for oxts_file in tqdm.tqdm(glob.glob(os.path.join(kitti_oxts_path, '*.txt')), desc='oxts_files'):
            save_integrate_oxts_file(oxts_file,
                                     os.path.join(target_oxts_path, os.path.basename(oxts_file)),
                                     hparams.pos_shift, hparams.rotation)
            split_oxts_dir = os.path.join(target_oxts_path, Path(oxts_file).stem)
            os.makedirs(split_oxts_dir, exist_ok=True)
            save_split_oxts_file(oxts_file, split_oxts_dir, hparams.pos_shift, hparams.rotation)
    else:
        print(f'kitti_oxts_path({kitti_oxts_path}) is not a directory, skip oxts file transformation')

    print(f'target path: {hparams.target_path}')


if __name__ == '__main__':
    main(get_opts())
