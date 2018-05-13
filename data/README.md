## Note: how to obtain raw data

This folder only contains a backbone of folders structures and a few example files.
You have to download the raw data files into "raw" folder following instructions 
[here](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data). 
The major steps are:

* Install Kaggle API: https://github.com/Kaggle/kaggle-api
* Download all Dstl data: ```kaggle competitions download -c dstl-satellite-imagery-feature-detection```
* Unpack the three_band data; and copy the unpacked data, grid_sizes.csv, and train_wkt_v4 
to corresponding locations according to the example files under this folder.