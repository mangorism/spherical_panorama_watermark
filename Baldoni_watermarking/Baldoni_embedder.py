import os
import cv2
import numpy as np
from DWT_wm_code import DWT_wm_embed, DWT_wm_extract
from skimage.metrics import structural_similarity as ssim
import argparse
import math

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_pattern_path', type=str, default=r'Baldoni_watermarking\sample_ref_pattern.npy', help='If the file exists, use it; if file is not exist, create and save it.')
    parser.add_argument('--src_folder', type=str, default=r'sample_data\original_equi', help='If you want to use saved reference pattern. Default makes new reference pattern.')
    parser.add_argument('--out_folder', type=str, default=r'sample_data\Baldoni_watermarked', help='If you want to use saved reference pattern. Default makes new reference pattern.')

    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)
    
    src_equi_img_name_list = os.listdir(args.src_folder)
    score_sum = 0

    if os.path.isfile(args.ref_pattern_path):
        ref_pattern = np.load(args.ref_pattern_path, allow_pickle=True)
    else:
        ## 256x256 패치에 DWT level 2에서 삽입 -> 64x64
        ## 해상도가 이미지마다 다르면 ref_pattern 코드 위치변경 및 수정이 필요함.
        ref_pattern = np.random.randn(64,64)
        np.save(args.ref_pattern_path, ref_pattern)

    for equi_idx, equi_name in enumerate(src_equi_img_name_list):
        print('Watermarking started : {}'.format(equi_name))
        src_equi = cv2.imread(os.path.join(args.src_folder, equi_name))
        assert src_equi is not None, 'cv2.imread error, check image path'

        wm_equi = src_equi.copy()

        for y in range(4):
            for x in range(8):
                block = src_equi[256 * y:256 * (y + 1), 256 * x:256 * (x + 1), :]
                wm_block, _ = DWT_wm_embed(block, ref_pattern)
                wm_equi[256 * y:256 * (y + 1), 256 * x:256 * (x + 1), :] = wm_block
        print('PSNR between source({}) and watermarked : {}'.format(equi_name, psnr(src_equi, wm_equi)))
        cv2.imwrite(os.path.join(args.out_folder, equi_name), wm_equi)
    print('Watermark embedding is done.')

