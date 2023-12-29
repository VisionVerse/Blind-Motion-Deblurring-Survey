import os
from skimage import io
import cv2
import numpy as np
from skimage.metrics import structural_similarity
import concurrent.futures

def image_align(deblurred, gt):
    # this function is based on kohler evaluation code
    z = deblurred
    c = np.ones_like(z)
    x = gt

    zs = (np.sum(x * z) / np.sum(z * z)) * z  # simple intensity matching

    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrix = np.eye(3, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 100

    termination_eps = 0

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY),
                                             warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5)

    target_shape = x.shape
    shift = warp_matrix

    zr = cv2.warpPerspective(
        zs,
        warp_matrix,
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT)

    cr = cv2.warpPerspective(
        np.ones_like(zs, dtype='float32'),
        warp_matrix,
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0)

    zr = zr * cr
    xr = x * cr

    return zr, xr, cr, shift

def compute_psnr(image_true, image_test, image_mask, data_range=None):
    # this function is based on skimage.metrics.peak_signal_noise_ratio
    err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
    return 10 * np.log10((data_range ** 2) / err)


def compute_ssim(tar_img, prd_img, cr1):
    ssim_pre, ssim_map = structural_similarity(tar_img, prd_img, multichannel=True, gaussian_weights=True,
                                               use_sample_covariance=False, data_range=1.0, full=True)
    ssim_map = ssim_map * cr1
    r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2
    ssim = ssim_map[pad:-pad, pad:-pad, :]
    crop_cr1 = cr1[pad:-pad, pad:-pad, :]
    ssim = ssim.sum(axis=0).sum(axis=0) / crop_cr1.sum(axis=0).sum(axis=0)
    ssim = np.mean(ssim)
    return ssim

total_psnr = 0.
total_ssim = 0.
count = 0
img_path = './out/Our_realblur_J_results'
gt_path = './datasets/Realblur_J/test/sharp'
print(img_path)
for file in os.listdir(img_path):
    for img_name in os.listdir(img_path + '/' + file):
        count += 1
        number = img_name.split('_')[1]
        gt_name = 'gt_' + number
        img_dir = img_path + '/' + file + '/' + img_name
        gt_dir = gt_path + '/' + file + '/' + gt_name
        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            tar_img = io.imread(gt_dir)
            prd_img = io.imread(img_dir)
            tar_img = tar_img.astype(np.float32) / 255.0
            prd_img = prd_img.astype(np.float32) / 255.0
            prd_img, tar_img, cr1, shift = image_align(prd_img, tar_img)
            PSNR = compute_psnr(tar_img, prd_img, cr1, data_range=1)
            SSIM = compute_ssim(tar_img, prd_img, cr1)
            total_psnr += PSNR
            total_ssim += SSIM
            print(count, PSNR)

print('PSNR:', total_psnr / count)
print('SSIM:', total_ssim / count)
print(img_path)

