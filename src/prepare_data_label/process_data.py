import os
import sys
import pandas as pd
import progressbar
from skimage.io import imread, imsave
from util.stretch_8bit import stretch_8bit

# Example CLI:
# python process_data.py ../../data/raw/3band_rgb ../../data/raw/train_wkt_v4.csv ../../data/dataset/images
if len(sys.argv) != 4:
    print('Usage: python process_data.py raw_data index_file output_folder')
    print("- raw_data: the folder containing the raw data files of 3band_rgb images")
    print("- index_file: location of train_wkt_v4.csv file")
    print("- output_folder: folder that images are saved to")
    print("Download raw data from https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data")
    exit(0)

raw_data = sys.argv[1]
index_file = sys.argv[2]
out_folder = sys.argv[3]


def select_and_save(file_names, output_dir_path):

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    p = progressbar.ProgressBar(max_value=len(file_names))

    for i, name in enumerate(file_names):
        p.update(i+1)
        file_name = name + '.tif'
        img = imread(os.path.join(raw_data, file_name))
        img = img.astype("float")
        img = stretch_8bit(img)
        output_file_name = name + '.jpg'
        imsave(os.path.join(output_dir_path, output_file_name), img)


train_wkt = pd.read_csv(index_file)
train_names = train_wkt.ImageId.unique().tolist()

print('Pre-processing raw data files ...')
select_and_save(train_names, out_folder)
