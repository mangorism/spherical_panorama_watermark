import os
import cv2
import numpy as np
from DWT_wm_code import DWT_wm_extract
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_pattern_path', type=str, default=r'Baldoni_watermarking\sample_ref_pattern.npy')
    parser.add_argument('--target_folder', type=str, default=r'sample_data\Baldoni_watermarked\watermarked_img')
    parser.add_argument('--out_folder', type=str, default=r'sample_data\Baldoni_watermarked\result')
    args = parser.parse_args()

    if not os.path.isdir(args.out_folder):
        os.mkdir(args.out_folder)

    ref_pattern = np.load(args.ref_pattern_path, allow_pickle=True)
    
    corr_list = list()

    for img_idx, img_name in enumerate(os.listdir(args.target_folder)):
        recov_equi_img = cv2.imread(os.path.join(args.target_folder, img_name))
                
        block_corr_max = -1
        block_counter = 0
        for y in range(4):
            for x in range(8):
                block = recov_equi_img[256 * y:256 * (y + 1), 256 * x:256 * (x + 1), :]
                block_corr = DWT_wm_extract(block, ref_pattern)
                if block_corr_max < block_corr:
                    block_corr_max = block_corr

                block_counter += 1
        corr_list.append(block_corr_max)
    mean_corr = np.mean(corr_list)
    print('mean corr : {}'.format(mean_corr))
    np.save(os.path.join(args.out_folder, 'baldoni_wm_corr_{}.npy').format(mean_corr), corr_list)
    