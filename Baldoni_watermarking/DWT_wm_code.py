import numpy as np
import math
import cv2
import pywt

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def normalize(input):
    return (input-input.min()) / (input.max() - input.min())

def xcorr(x1, x2):
    return np.mean((x1-np.mean(x1))*(x2-np.mean(x2))) / (np.std(x1)*np.std(x2))

def get_1d_corr(a, b):
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    up = np.sum(np.multiply((a - a_mean), (b - b_mean)))
    down = math.sqrt(np.sum(np.power(a - a_mean, 2)) * np.sum(np.power(b - b_mean, 2)))
    return up / down

def nonblind(org, wm, alpha):
    output = (wm-org)
    output = output/alpha
    output = normalize(output)
    return output

def xcorrDWT(input, LL, LH, HL, HH):
    return [xcorr(input, LL), xcorr(input, LH), xcorr(input, HL), xcorr(input, HH)]

# Watermark Embedding ##################################################################################################
def DWT_wm_embed(img, wm):
    # convert color space: BGR to YCrCb
    imgYcrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    imgY = imgYcrcb[:, :, 0]

    # DWT2
    coeffs1 = pywt.dwt2(imgY, 'haar')   # 1-level
    LL1, (LH1, HL1, HH1) = coeffs1
    coeffs2 = pywt.dwt2(LL1, 'haar')    # 2-level
    LL2, (LH2, HL2, HH2) = coeffs2

    # generate watermark
    alpha = 21

    wm = normalize(wm)
    wm = alpha * (wm*2-1)

    # embed watermark
    LL2 = LL2 + wm
    LH2 = LH2 + wm
    HL2 = HL2 + wm
    HH2 = HH2 + wm

    # IDWT2
    coeffs2 = LL2, (LH2, HL2, HH2)
    coeffs1LL = pywt.idwt2(coeffs2, 'haar')     # 2-level
    coeffs1rec = coeffs1LL, (LH1, HL1, HH1)
    imgYrec = pywt.idwt2(coeffs1rec, 'haar')    # 1-level
    imgYrec[imgYrec > 255] = 255
    imgYrec[imgYrec < 0] = 0

    # convert color space: YCrCb to BGR
    imgYcrcb[:, :, 0] = imgYrec
    imgWM = cv2.cvtColor(imgYcrcb, cv2.COLOR_YCrCb2BGR)

    return imgWM, wm



# Watermark Extraction #################################################################################################
def DWT_wm_extract(tgt_img, wm_ref):
    # convet color space: BGR to YCrCb
    tgt_imgYcrcb = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2YCrCb)
    
    tgt_imgY = tgt_imgYcrcb[:, :, 0]
    

    # DWT2
    tgt_coeffs1 = pywt.dwt2(tgt_imgY, 'haar')   # 1-level
    tgt_LL1, (tgt_LH1, tgt_HL1, tgt_HH1) = tgt_coeffs1
    tgt_coeffs2 = pywt.dwt2(tgt_LL1, 'haar')    # 2-level
    tgt_LL2, (tgt_LH2, tgt_HL2, tgt_HH2) = tgt_coeffs2

    
    tgt_Mean = (tgt_LL2 + tgt_LH2 + tgt_HL2 + tgt_HH2)/4

    # cross correlation: mean
    corr = xcorr(wm_ref, tgt_Mean)
    
    return corr