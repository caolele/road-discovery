import shutil
import os
import glob
import numpy as np
import cv2 as cv
from os.path import basename
import sys


# Example CLI:
# python gen_lmdb_deeplab.py ../../data/dataset/images ../../data/dataset/labels ../../data/lmdb/dlcrf_data ../../data/lmdb/dlcrf_label ../../data/lmdb/dlcrf_index.txt
if __name__ == '__main__':
    num_args = len(sys.argv) - 1
    if num_args != 5:
        print('Usage: python gen_lmdb_deeplab.py ori_dir lbl_dir ori_db_dir lbl_db_dir list_file')
        exit(0)

    keys = np.arange(15000000)
    np.random.shuffle(keys)

    # input folders
    ori_dir = sys.argv[1]
    lbl_dir = sys.argv[2]

    # output folders
    ori_db_dir = sys.argv[3]
    lbl_db_dir = sys.argv[4]

    # list file
    list_file = sys.argv[5]
    fw = open(list_file, 'w')

    # get file names and reshuffle
    fns = np.asarray(sorted(glob.glob('%s/*.png*' % lbl_dir)))
    index = np.arange(len(fns))
    np.random.shuffle(index)
    fns = fns[index]
    n_all_files = len(fns)
    print('total images: ', n_all_files)

    # create file list for ori and lbl
    ori_fns = []
    lbl_fns = []
    for line in fns:
        bname = basename(line).split(".")[0]
        ori_fns.append(os.path.join(ori_dir, bname) + ".jpg")
        lbl_fns.append(os.path.join(lbl_dir, bname) + ".png")

    # init output folders
    if os.path.exists(ori_db_dir):
        shutil.rmtree(ori_db_dir)
    if os.path.exists(lbl_db_dir):
        shutil.rmtree(lbl_db_dir)
    os.makedirs(ori_db_dir)
    os.makedirs(lbl_db_dir)

    # set cut params
    patch_size = 609
    stride = 256
    print('patch size: ', patch_size)
    print('stride: ', stride)

    # start process one file at a time
    n_patches = 0
    for file_i, (ori_fn, lbl_fn) in enumerate(zip(ori_fns, lbl_fns)):
        ori_im = cv.imread(ori_fn, cv.IMREAD_COLOR)
        ori_im = cv.bilateralFilter(ori_im, 5, 5*2, 5/2)
        lbl_im = cv.imread(lbl_fn, cv.IMREAD_GRAYSCALE)

        if ori_im is None or lbl_im is None:
            continue

        for y in range(0, ori_im.shape[0] + stride, stride):
            for x in range(0, ori_im.shape[1] + stride, stride):
                if (y + patch_size) > ori_im.shape[0]:
                    y = ori_im.shape[0] - patch_size
                if (x + patch_size) > ori_im.shape[1]:
                    x = ori_im.shape[1] - patch_size

                ori_patch = np.copy(ori_im[y:y + patch_size, x:x + patch_size])
                lbl_patch = np.copy(lbl_im[y:y + patch_size, x:x + patch_size])

                # exclude patch including big white/black region
                tmpsum = np.sum(ori_patch, axis=2)
                if np.sum(tmpsum == (255 * 3)) > 64 or np.sum(tmpsum == 0) > 16:
                    print("ori_patch: skip!")
                    continue

                # exclude patch that does not have significant label information
                if np.sum(lbl_patch == 1) < 128:
                    print("lbl_patch: skip!")
                    continue

                # get a random key of 10 digits
                key = '%010d' % keys[n_patches]

                # write patch and corresponding label
                tmp_ori = os.path.join(ori_db_dir, key+'.jpg')
                cv.imwrite(tmp_ori, ori_patch)
                tmp_lbl = os.path.join(lbl_db_dir, key+'.png')
                cv.imwrite(tmp_lbl, lbl_patch)
                fw.write(tmp_ori+" "+tmp_lbl+"\n")

                n_patches += 1

        print(file_i, '/', n_all_files, ': n_patches=', n_patches)

    print('patches:\t', n_patches)

    fw.close()
