import shutil
import os
import glob
import lmdb
import numpy as np
import cv2 as cv
from os.path import basename
import sys

sys.path.append('/Users/caolele/Documents/caffe/python/')
import caffe

if __name__ == '__main__':

    # Example CLI:
    # python gen_lmdb_googlenet.py
    #   ../../data/dataset/images ../../data/dataset/labels ../../data/lmdb/gneti_data ../../data/lmdb/gneti_label 10

    num_args = len(sys.argv) - 1
    if num_args < 4:
        print('Usage: python gen_lmdb_googlenet.py ori_dir lbl_dir ori_lmdb_dir lbl_lmdb_dir max_files')
        exit(0)

    max_files = 0
    if num_args == 5:
        max_files = int(sys.argv[5])

    # input folders
    ori_dir = sys.argv[1]
    lbl_dir = sys.argv[2]

    # output folders
    ori_lmdb_dir = sys.argv[3]
    lbl_lmdb_dir = sys.argv[4]

    # get file names and reshuffle
    fns = np.asarray(sorted(glob.glob('%s/*.png*' % lbl_dir)))
    index = np.arange(len(fns))
    np.random.shuffle(index)
    fns = fns[index]

    if max_files > 0:
        n_all_files = max_files
    else:
        n_all_files = len(fns)
    print('total images: ', n_all_files)

    # create file list for ori and lbl
    ori_fns = []
    lbl_fns = []
    for line in fns:
        bname = basename(line)
        ori_fns.append(os.path.join(ori_dir, os.path.splitext(bname)[0] + ".jpg"))
        lbl_fns.append(os.path.join(lbl_dir, bname))

    # init output folders
    if os.path.exists(ori_lmdb_dir):
        shutil.rmtree(ori_lmdb_dir)
    if os.path.exists(lbl_lmdb_dir):
        shutil.rmtree(lbl_lmdb_dir)
    os.makedirs(ori_lmdb_dir)
    os.makedirs(lbl_lmdb_dir)

    # config lmdb
    sat_env = lmdb.Environment(ori_lmdb_dir, map_size=1099511627776)
    sat_txn = sat_env.begin(write=True, buffers=True)
    map_env = lmdb.Environment(lbl_lmdb_dir, map_size=1099511627776)
    map_txn = map_env.begin(write=True, buffers=True)
    keys = np.arange(15000000)
    np.random.shuffle(keys)

    # set cut params
    patch_size = 512
    stride = 256
    print('patch size: ', patch_size)
    print('stride: ', stride)

    # start process one file at a time
    n_patches = 0
    for file_i, (ori_fn, lbl_fn) in enumerate(zip(ori_fns, lbl_fns)):
        ori_im = cv.imread(ori_fn, cv.IMREAD_COLOR)
        ori_im = cv.bilateralFilter(ori_im, 5, 5 * 2, 5 / 2)
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
                    continue

                # get a random key of 10 digits
                key = '%010d' % keys[n_patches]
                cv.imwrite('./tmp/' + key + '.jpg', ori_patch)

                # ori db
                ori_patch = ori_patch.swapaxes(0, 2).swapaxes(1, 2)
                datum = caffe.io.array_to_datum(ori_patch, 0)
                value = datum.SerializeToString()
                sat_txn.put(key, value)

                # map db
                lbl_patch = lbl_patch.reshape((1, lbl_patch.shape[0],
                                               lbl_patch.shape[1]))
                datum = caffe.io.array_to_datum(lbl_patch, 0)
                value = datum.SerializeToString()
                map_txn.put(key, value)

                n_patches += 1

                if n_patches % 10000 == 0:
                    sat_txn.commit()
                    sat_txn = sat_env.begin(write=True, buffers=True)
                    map_txn.commit()
                    map_txn = map_env.begin(write=True, buffers=True)

        print(file_i + 1, '/', n_all_files, ': n_patches=', n_patches)

        if max_files > 0 and max_files - file_i == 1:
            break

    sat_txn.commit()
    sat_env.close()
    map_txn.commit()
    map_env.close()

    print("LMDB generation done!\n Total patches: ", n_patches)
