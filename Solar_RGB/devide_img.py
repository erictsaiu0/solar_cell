import random
import shutil

from PIL import Image
from itertools import product
import os


def tile(filename, dir_in, dir_out, d=1024):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size

    grid = product(range(0, h - h % d, d), range(0, w - w % d, d))
    for i, j in grid:
        box = (j, i, j + d, i + d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)


def sampling(src_dir, sample_dir, sample_num):
    # Sample images from FTASK folders and copy to src_dir\FTASK_dir\sample_dir
    if not os.path.exists(os.path.join(src_dir, sample_dir)):
        os.mkdir(os.path.join(src_dir, sample_dir))
    for dp, dn, fn in os.walk(src_dir):
        if sample_dir not in dp and 'FTASK' in dp:
            if not os.path.exists(os.path.join(src_dir, sample_dir, os.path.basename(dp))):
                os.mkdir(os.path.join(src_dir, sample_dir, os.path.basename(dp)))
            sample_ls = random.sample(fn, sample_num)
            for f in sample_ls:
                if 'JPG' in f:
                    src_path = os.path.join(dp, f)
                    tgt_path = os.path.join(src_dir, sample_dir, os.path.basename(dp), f)
                    print('Copying', src_path, 'to', tgt_path)
                    shutil.copyfile(src_path, tgt_path)



def divide(sample_dir, output_dir):
    # divide all images from sample_dir and
    if not os.path.exists(os.path.join(sample_dir, output_dir)):
        os.mkdir(os.path.join(sample_dir, output_dir))
    for dp, dn, fn in os.walk(sample_dir):
        if output_dir not in dp and 'FTASK' in dp and 'divided' not in dp:
            if not os.path.exists(os.path.join(sample_dir, output_dir, os.path.basename(dp))):
                os.mkdir(os.path.join(sample_dir, output_dir, os.path.basename(dp)))
            for f in fn:
                if 'JPG' in f:
                    print('Dividing', os.path.join(dp, f))
                    tile(f, dp, os.path.join(sample_dir, output_dir, os.path.basename(dp)))


src_dir = r'F:\Solar_RGB'
# sample
sample_save_dir = r'Sampled'
sample_num = 3

# dividing
sampled_dir = r'F:\Solar_RGB\Sampled'
output_dir = r'unlabeled'

sampling(src_dir, sample_save_dir, sample_num)
divide(sampled_dir, output_dir)