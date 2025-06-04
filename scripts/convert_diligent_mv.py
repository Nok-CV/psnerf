import os
import json
import argparse
from shutil import copy2
import numpy as np
from PIL import Image
from scipy.io import loadmat


def _search_file(calib_dir, keywords, exts):
    """Return the first file in ``calib_dir`` whose name contains any of
    ``keywords`` and ends with one of ``exts`` (case-insensitive)."""
    for f in os.listdir(calib_dir):
        name = f.lower()
        if any(k in name for k in keywords) and any(name.endswith(e) for e in exts):
            return os.path.join(calib_dir, f)
    return None


def _load_mat_var(path, keys):
    data = loadmat(path)
    for k in keys:
        if k in data:
            return data[k]
    for k, v in data.items():
        if not k.startswith("__"):
            return v
    raise KeyError(f"No usable variable found in {path}")


def _load_matrix(calib_dir, def_txt, def_mat, keywords, mat_keys):
    if def_txt:
        p = os.path.join(calib_dir, def_txt)
        if os.path.exists(p):
            return np.loadtxt(p)
    if def_mat:
        p = os.path.join(calib_dir, def_mat)
        if os.path.exists(p):
            return _load_mat_var(p, mat_keys)

    p = _search_file(calib_dir, keywords, ['.txt'])
    if p is not None:
        return np.loadtxt(p)
    p = _search_file(calib_dir, keywords, ['.mat'])
    if p is not None:
        return _load_mat_var(p, mat_keys)
    raise FileNotFoundError(f"Cannot find {keywords[0]} file")


def load_calibration(calib_dir):
    """Load intrinsics, extrinsics and light directions from DiLiGenT-MV calib files.
    The function searches for common file names if the default ones are not
    present."""

    K = _load_matrix(calib_dir, 'intrinsics.txt', 'intrinsics.mat',
                     ['intrin', 'k'], ['K'])
    K = K.reshape(3, 3)

    poses = _load_matrix(calib_dir, 'extrinsics.txt', 'extrinsics.mat',
                         ['extrin', 'pose', 'c2w'], ['pose_c2w'])
    poses = np.asarray(poses)
    if poses.shape[-2:] == (4, 4):
        poses = poses.reshape(-1, 4, 4)
    else:
        poses = poses.reshape(4, 4, -1).transpose(2, 0, 1)

    light_dirs = _load_matrix(calib_dir, 'light_directions.txt',
                              'light_directions.mat',
                              ['light', 'dir'], ['light_direction'])
    light_dirs = np.asarray(light_dirs)
    if light_dirs.ndim == 2 and light_dirs.shape[0] == 3:
        light_dirs = light_dirs.T
    return K, poses, light_dirs


def copy_images(src_dir, dst_dir):
    view_dirs = sorted([d for d in os.listdir(src_dir) if d.lower().startswith('view')])
    if not view_dirs:
        raise RuntimeError('No view directories found')
    os.makedirs(os.path.join(dst_dir, 'img'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'mask'), exist_ok=True)

    im_hw = None
    for v in view_dirs:
        src_view = os.path.join(src_dir, v)
        dst_view = os.path.join(dst_dir, 'img', v)
        os.makedirs(dst_view, exist_ok=True)
        for img_name in sorted(os.listdir(src_view)):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            copy2(os.path.join(src_view, img_name), os.path.join(dst_view, img_name))
            if im_hw is None:
                with Image.open(os.path.join(src_view, img_name)) as img:
                    im_hw = img.size[::-1]

        mask_path = os.path.join(src_dir, 'mask', f'{v}.png')
        if os.path.exists(mask_path):
            copy2(mask_path, os.path.join(dst_dir, 'mask', f'{v}.png'))
    return len(view_dirs), im_hw


def main():
    parser = argparse.ArgumentParser(description='Convert DiLiGenT-MV data to PS-NeRF format')
    parser.add_argument('src', type=str, help='Path to DiLiGenT-MV object directory')
    parser.add_argument('dst', type=str, help='Destination directory')
    parser.add_argument('--name', type=str, default=None, help='Object name')
    args = parser.parse_args()

    obj_name = args.name or os.path.basename(os.path.normpath(args.dst))

    K, poses, light_dirs = load_calibration(os.path.join(args.src, 'calib'))
    n_light = light_dirs.shape[0]

    n_view, imhw = copy_images(os.path.join(args.src, 'images'), args.dst)

    para = {
        'obj_name': obj_name,
        'n_view': int(n_view),
        'imhw': list(map(int, imhw)),
        'gt_normal_world': False,
        'view_train': list(range(n_view)),
        'view_test': [],
        'K': K.tolist(),
        'pose_c2w': poses.tolist(),
        'light_is_same': True,
        'light_direction': light_dirs.tolist(),
    }

    with open(os.path.join(args.dst, 'params.json'), 'w') as f:
        json.dump(para, f, indent=2)
    print('Converted object saved to', args.dst)


if __name__ == '__main__':
    main()
