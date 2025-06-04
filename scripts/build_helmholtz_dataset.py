import os
import json
import argparse
from convert_diligent_mv import load_calibration, copy_images
from find_helmholtz_pairs import find_pairs


def main():
    parser = argparse.ArgumentParser(
        description='Convert DiLiGenT-MV object and compute Helmholtz pairs')
    parser.add_argument('src', type=str, help='Path to DiLiGenT-MV object directory')
    parser.add_argument('dst', type=str, help='Destination directory')
    parser.add_argument('--name', type=str, default=None, help='Object name')
    parser.add_argument('--thresh', type=float, default=15.5,
                        help='Angle threshold for Helmholtz pairing in degrees')
    args = parser.parse_args()

    obj_name = args.name or os.path.basename(os.path.normpath(args.dst))

    K, poses, light_dirs = load_calibration(os.path.join(args.src, 'calib'))
    n_view, im_hw = copy_images(os.path.join(args.src, 'images'), args.dst)

    para = {
        'obj_name': obj_name,
        'n_view': int(n_view),
        'imhw': list(map(int, im_hw)),
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

    pairs = find_pairs(para, args.thresh)
    with open(os.path.join(args.dst, 'helmholtz_pairs.json'), 'w') as f:
        json.dump(pairs, f, indent=2)

    print('Converted object saved to', args.dst)
    print('Found %d Helmholtz pairs' % len(pairs))
    print('Pairs written to helmholtz_pairs.json')


if __name__ == '__main__':
    main()
