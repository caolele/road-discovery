import os
import sys
import warnings
import numpy as np
import pandas as pd
from shapely.wkt import loads
from skimage.draw import polygon
import progressbar
import skimage.io as io
from skimage.io import imread, imsave
from util.stretch_8bit import stretch_8bit


io.use_plugin('matplotlib', 'imread')


# Example CLI:
# python process_label.py ../../data/raw/3band_rgb ../../data/raw ../../data/dataset/labels 0
if len(sys.argv) < 4 or len(sys.argv) > 5:
    print('Usage: python process_label.py raw_data label_folder output_folder [mode(0/1/2)]')
    print("mode=0: generate label for training")
    print("mode=1: paint the label in a visible manner on map")
    print("mode=2: paint the label in a visible manner on blank canvas")
    exit(0)

raw_data = sys.argv[1]
label_folder = sys.argv[2]
out_folder = sys.argv[3]
wmode = 0
if len(sys.argv) == 5:
    wmode = int(sys.argv[4])


def fill_mask(_mask, polygon_array, value, _W, _H, _Xmax, _Ymin):

    po = np.array(polygon_array)
    po[:, 0] = po[:, 0] / _Xmax * _W * _W / (_W + 1)
    po[:, 1] = po[:, 1] / _Ymin * _H * _H / (_H + 1)
    rr, cc = polygon(po[:, 1], po[:, 0], shape=_mask.shape)
    if len(_mask.shape) == 3:
        _mask[rr, cc, 0] = value
    else:
        _mask[rr, cc] = value
    return _mask


def fill_polygon(_poly, _mask, _W, _H, _Xmax, _Ymin, _wmode):

    if _wmode == 0:
        value = 1
    else:
        value = 255

    polygon_array = _poly.exterior.coords
    _mask = fill_mask(_mask, polygon_array, value, _W, _H, _Xmax, _Ymin)
    if hasattr(_poly, "interiors"):
        for interior in _poly.interiors:
            polygon_array = interior.coords
            _mask = fill_mask(_mask, polygon_array, 0, _W, _H, _Xmax, _Ymin)
    return _mask


def main():

    train_wkt = pd.read_csv(os.path.join(label_folder, 'train_wkt_v4.csv'))
    train_images = train_wkt.ImageId.unique()

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    print("Processing labels for all raw images ...")

    pb = progressbar.ProgressBar(max_value=len(train_images))

    # For each raw image
    for i, train_image in enumerate(train_images):

        pb.update(i + 1)

        target_wkt = train_wkt.ix[train_wkt.ImageId == train_image, :]
        target_wkt.reset_index(inplace=True, drop=True)

        grid_sizes = pd.read_csv(os.path.join(label_folder, 'grid_sizes.csv'))
        grid_size = grid_sizes.ix[grid_sizes.iloc[:, 0] == train_image, :]
        Xmax = float(grid_size.Xmax)
        Ymin = float(grid_size.Ymin)

        img = imread(os.path.join(raw_data, train_image + '.tif'))
        W = img.shape[1]
        H = img.shape[0]

        if wmode == 1:
            img = img.astype("float")
            img = stretch_8bit(img)
            mask = img
        else:
            mask = np.zeros((H, W), dtype='int')

        # only extract road marks
        for cls in range(3, 5):
            lbl = target_wkt.MultipolygonWKT[target_wkt.ClassType == cls].tolist()[0]
            if lbl != 'MULTIPOLYGON EMPTY':
                for poly in loads(lbl):
                    mask = fill_polygon(poly, mask, W, H, Xmax, Ymin, wmode)

        output_file_name = train_image + '.png'

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(os.path.join(out_folder, output_file_name), mask)

    pb.finish()


if __name__ == '__main__':
    main()
