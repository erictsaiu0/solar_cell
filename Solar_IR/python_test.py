import os
import random
import shutil
import cv2
import numpy as np

num_sample = 30

# path = r'F:\solar_IR\SAMPLED\label_img\label_C'
# tgt_A = r'F:\solar_IR\SAMPLED\label_img\validation'
# tgt_B = r'F:\solar_IR\SAMPLED\label_img\train'
#
# for dir in os.listdir(path):
#     if 'Zoon' in dir:
#         dir_path = os.path.join(path, dir)
#         tgt_path = os.path.join(tgt, dir)
#         if not os.path.exists(tgt_path):
#             os.mkdir(tgt_path)
#
#         for file in random.sample(os.listdir(dir_path), num_sample):
#             print('copying', file)
#             shutil.copy(os.path.join(dir_path, file), os.path.join(tgt_path, file))

src_path = r'F:\solar_IR'
tgt_path = r'F:\solar_IR\Masked'
if not os.path.exists(tgt_path):
    os.mkdir(tgt_path)
move_path = []
move_path = []
for dp, dn, fn in os.walk(src_path):
    if 'Zoon' in dp and 'SAMPLED' not in dp:
        for f in fn:
            if 'masked' in f:
                # print(os.path.basename(dp), f)
                tgt_dir_path = os.path.join(tgt_path, os.path.basename(dp))
                if not os.path.exists(tgt_dir_path):
                    os.mkdir(tgt_dir_path)
                # print(os.path.join(tgt_dir_path, f))
                move_path.append([os.path.join(dp, f), os.path.join(tgt_dir_path, f)])

for p in move_path:
    print(p[0], p[1])
    shutil.move(p[0], p[1])
