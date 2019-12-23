from skimage.external.tifffile import imread
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os, fnmatch
import time

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def transform(img):
    img = cv2.resize(img, (300,300), interpolation=cv2.INTER_LINEAR)
    img = cv2.flip(img, -1)
    h, w, _ = img.shape
    point = (w/2, h/2)
    M = cv2.getRotationMatrix2D(point, angle=-45, scale=1)
    dst = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return dst

base_path = 'data/'

mpm16bit_path = base_path+'MPM4C_16bit'
sixteen_bit_max = 65535

mpm_16_file_list = find('*.tif', path=base_path+'MPM4C_16bit')
start_time = time.time()
count = 0
img_list = []
for img_file in mpm_16_file_list:
    img_single_c = cv2.imread(img_file, cv2.IMREAD_ANYDEPTH).astype(float)
    count += 1
    if count < 4:
        img_list.append(img_single_c)
    else:
        count = 0
        temp = transform(np.stack(img_list, axis=-1))
        img_list = []




print('16bit total io time:', time.time()-start_time)

mpm_rgn_file_list = find('*.tif', path=base_path+'MPM')
start_time = time.time()
for img_file in mpm_16_file_list:
    img_single_c = transform(cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB))
print('rgb total io time:', time.time()-start_time)
    # img_scaled_down = cv2.resize(np.stack((img_single_c,)*2,axis=-1), (300, 300), interpolation=cv2.INTER_LINEAR)
    # print(img_scaled_down.shape)
    # cv2.normalize(src=img_single_c, dst=img_single_c, alpha=0, beta=255,
    #               norm_type=cv2.NORM_MINMAX)
    # img_single_c = img_single_c.astype('uint8')
    # # print(img_single_c.dtype)
    # # print(np.max(img_single_c))
    #
    # # img_single_c = (img_single_c/256).astype('uint8')
    # cv2.imwrite(filename=base_path+'MPM4C_8bit/'+img_file.split('/')[-1],
    #             img=img_single_c)

# img_rgb = cv2.imread(base_path+'MPM/MPM 001.tif')
#
# # print(img.shape)
# plt.figure(1)
# plt.imshow(img_rgb[...,0], cmap='gray')
# plt.figure(2)
# plt.imshow(img_single_c, cmap='gray')
# plt.show()