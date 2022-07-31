import os
import matplotlib.image as image
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

img_path = r'F:\202204_IR_jpg'


def read_file(file_path):

    def rgb2gray(rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray

    # im = rgb2gray(image.imread(file_path)
    im = image.imread(file_path, 1)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im = np.array(im)
    return im


def preprocess(img):
    img = img

    # img = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    th, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    return img

sample_num = 1
image_ls = []
processed_ls = []
for dir in os.listdir(img_path):
    dir_path = os.path.join(img_path, dir)
    file_ls = random.sample(os.listdir(dir_path), sample_num)
    for f in file_ls:
        f_np = read_file(os.path.join(dir_path, f))
        image_ls.append(f_np)
        processed_img = preprocess(f_np)
        processed_ls.append(processed_img)

image_ls = np.array(image_ls)

final_ls = []
for i in range(len(image_ls)):
    img = cv2.vconcat([image_ls[i], processed_ls[i]])
    final_ls.append(img)


plt.figure(figsize=(10*len(os.listdir(img_path)), 10*sample_num))
for i, img in enumerate(final_ls):
    if sample_num > 1:
        plt.subplot(len(os.listdir(img_path)), sample_num, i+1)
    else:
        plt.subplot(sample_num, len(os.listdir(img_path)), i + 1)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.axis('off')

plt.tight_layout()
plt.savefig('hsv_compare.jpg', dpi=300)
