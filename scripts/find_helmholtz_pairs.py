import os
import json
import argparse
import numpy as np


def load_params(obj_dir):
    path = os.path.join(obj_dir, 'params.json')
    with open(path, 'r') as f:
        para = json.load(f)
    return para


def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1
    return v / n


def get_cam_dirs(poses):
    # camera viewing direction is the -z axis of rotation matrix
    dirs = -np.asarray(poses, dtype=np.float32)[:, :3, 2]
    return normalize(dirs)


def get_light_dirs(para):
    if para['light_is_same']:
        dirs = np.asarray(para['light_direction'], dtype=np.float32)
        dirs = np.broadcast_to(dirs[None, ...], (para['n_view'],) + dirs.shape)
    else:
        dirs = np.asarray(para['light_direction'], dtype=np.float32)
    return normalize(dirs)


def angle_between(a, b):
    a = normalize(a)
    b = normalize(b)
    dot = np.clip((a * b).sum(-1), -1.0, 1.0)
    return np.rad2deg(np.arccos(dot))


def find_pairs(para, thresh=15.5):
    poses = np.asarray(para['pose_c2w'], dtype=np.float32)
    cam_dirs = get_cam_dirs(poses)
    light_dirs = get_light_dirs(para)

    n_view, n_light, _ = light_dirs.shape
    pairs = []
    for i in range(n_view):
        for j in range(i + 1, n_view):
            for li in range(n_light):
                for lj in range(n_light):
                    ang1 = angle_between(light_dirs[i, li], cam_dirs[j])
                    ang2 = angle_between(light_dirs[j, lj], cam_dirs[i])
                    if ang1 <= thresh and ang2 <= thresh:
                        pairs.append(((i, li), (j, lj)))
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Find Helmholtz pairs in DiLiGenT-MV style dataset')
    parser.add_argument('obj_dir', type=str, help='Path to object directory containing params.json')
    parser.add_argument('--thresh', type=float, default=15.5, help='Angle threshold in degrees')
    args = parser.parse_args()

    para = load_params(args.obj_dir)
    pairs = find_pairs(para, args.thresh)

    print('# Found %d Helmholtz pairs' % len(pairs))
    for (c1, l1), (c2, l2) in pairs:
        print('View %02d - Light %03d <-> View %02d - Light %03d' % (c1 + 1, l1 + 1, c2 + 1, l2 + 1))


if __name__ == '__main__':
    main()
