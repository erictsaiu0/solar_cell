import shutil

import matplotlib.pyplot as plt
import os
import numpy as np
import random
import cv2

src_path = r'F:\solar_IR\Masked'
tgt_path = r'F:\solar_IR\Masked\thresholding'
sample_num = 10


# random sample images to tgt_path
def random_sample():
    for dp, dn, fn in os.walk(src_path):
        if 'Zoon' in dp and 'thresholding' not in dp:
            tgt_dir_path = os.path.join(tgt_path, os.path.basename(dp))
            if os.path.exists(tgt_dir_path):
                shutil.rmtree(tgt_dir_path)
            os.mkdir(tgt_dir_path)
            # print(dp, fn)
            sampled = random.sample(fn, sample_num)
            for s in sampled:
                src_file_path = os.path.join(dp, s)
                tgt_file_path = os.path.join(tgt_dir_path, s)
                # print(src_file_path, tgt_file_path)
                shutil.copyfile(src_file_path, tgt_file_path)


def remove_big_objects(im):
    # find all of the connected components (white blobs in your image).
    # im_with_separated_blobs is an image where each detected blob has a different pixel value ranging from 0 to nm_components.
    nb_components, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(im)
    # stats (and the silenced output centroids) gives a lot of information about the components. See the docs for more information.
    # Here, we're interested only in the size and width of the components.
    sizes = stats[:, -1]
    length = stats[:, -2]
    width = stats[:, -3]
    # taking out the background which is also considered a component, which I find for most applications to not be the expected output.
    sizes = sizes[1:]
    length = length[1:]
    width = width[1:]

    nb_components -= 1

    # max size and max width of an object
    max_size = 50
    min_size = 0.2*max_size
    max_width, max_length = int(np.sqrt(max_size)), int(np.sqrt(max_size))

    # output image with only the kept components
    im_result = np.zeros((im.shape))
    # for every component in the image, keep it only if it's above min_size
    for comp in range(nb_components):
        if max_size >= sizes[comp] >= min_size and width[comp] <= max_width and length[comp] <= max_length:
            # see description of im_with_separated_blobs above
            im_result[im_with_separated_blobs == comp + 1] = 255
    return im_result


# thresholding
def img_threholding(img_path):
    img = cv2.imread(img_path)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pcsd_img = cv2.equalizeHist(img)
    ret2, pcsd_img = cv2.threshold(pcsd_img, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    pcsd_img = cv2.morphologyEx(pcsd_img, cv2.MORPH_OPEN, kernel)
    pcsd_img = cv2.morphologyEx(pcsd_img, cv2.MORPH_CLOSE, kernel)

    pcsd_img = remove_big_objects(pcsd_img)


    return img, pcsd_img


# plot all thresholded image
def thesholding_show(path):
    dir_ls = []
    for dp, dn, fn in os.walk(path):
        if 'Zoon' in dp:
            # print(dp)
            img_ls = []
            for f in fn:
                img_path = os.path.join(dp, f)
                [img, pcsd_img] = img_threholding(img_path)
                img_ls.append([img, pcsd_img])
            dir_ls.append(img_ls)

    fig = plt.figure(figsize=(20, 20), dpi=300)
    for dir_idx in range(len(dir_ls)):
        for i in range(sample_num):
            fig.add_subplot(len(dir_ls), sample_num, dir_idx*sample_num+i+1)
            plt.axis("off")
            plt.imshow(dir_ls[dir_idx][i][0], cmap='gray')
            plt.imshow(dir_ls[dir_idx][i][1], cmap='copper', alpha=0.5)
            plt.tight_layout()
    # plt.savefig('thresholding_open.png')
    plt.savefig('thresholding_rm_big.png')
    # plt.savefig('original.png')
    # plt.savefig('thresholding.png')


thesholding_show(tgt_path)
