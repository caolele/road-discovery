# Road-Discovery: Tooling & Models for Fine-Grained Road Detection from Aerial/Satellite Images

**STATEMENT:** This is ONLY and nothing more than a TOY EXAMPLE for understanding the basic possible approach of detecting objects from aerial images. The models included in this repository is under-trained with some public datasets, hence is not suitable for practical use cases. In case of using any source code and/or model, please add reference of this repository to your work.

This repo contains two complete examples of aerial detection using [DeepLab+CRF](https://arxiv.org/abs/1606.00915) and [GoogleNet](https://arxiv.org/abs/1409.4842) models respectively.

This toolbox contains functions of
- data-preprocessing (of [Kaggle dstl dataset](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data)): src/prepare_data_label/
- training models: train_dev/
- training metrics visualization: src/visualize_training_log/
- compression and de-compression of exported models (c.f. [Deep Compression Paper](https://arxiv.org/pdf/1510.00149v5.pdf)): model_compression/
- model deployment and test: deploy_test/
- a demo in docker: demo_docker/

**Prerequisite**: Make sure you have [Docker](https://www.docker.com/) installed on your computer.

## To Start Demo
Run in terminal:
```
git clone https://github.com/caolele/road-discovery.git

cd ./demo_docker

./demo.sh
```
Now, you need to wait until it finishes with a Unix prompt: `root@86d8e0a049b9:/workspace#`; and you are good to go.

B.T.W. To cleanup the demo environment, use 
```
./cleanup.sh
```

## Models in Action
If you want to change the prediction source code, you may rebuild the binary with ```make -j4``` in the corresponding folders (i.e. deploy_test/road_finder_dlcrf/ and deploy_test/road_finder_gneti/).  

The DeepLab prediction has 2 modes: fast (non-overlap prediction) and slow (overlapped prediction); the input patch has size of 449x449. To try out in docker container:
```
cd /workspace/road-discovery/deploy_test/road_finder_dlcrf/bin

./dlcrfRF ../../test_resource/img1.jpg /workspace/mount/dlcrf_img1_out
```

The GoogleNet model is generally slower because it merely predicts the central 16x16 small area for each input patch (224x224). To try out (the output file is in the same folder by default):
```
cd /workspace/road-discovery/deploy_test/road_finder_gneti/bin

./gnetiRF ../../test_resource/img2.jpg
```
Feel free to try out your own images by copying them to the mounted folder "road-discovery/demo_docker/mount/"


## Demonstration: A Few Examples
You might get slightly worse predictions than the examples illustrated below, which is due to the [model compression operation](#compression-and-de-compression-of-models) carried out on the original models.
### DeepLab+CRF:
upper-left: input image; upper-right: prediction overlay; bottom-left: label; bottom-right: prediction  
![alt text](./demo_docker/mount/dlcrf1.jpg)

upper-left: input image; upper-right: prediction overlay; bottom-left: label; bottom-right: prediction  
![alt text](./demo_docker/mount/dlcrf2.jpg)

### GoogleNet:
upper: prediction image; bottom: input raw image  
![alt text](./demo_docker/mount/gneti1.jpg)

upper: prediction image; bottom: input raw image  
![alt text](./demo_docker/mount/gneti2.jpg)

------

## [OPTIONAL for DEMO] - Retrain/Finetune Your Own Models
### Download Raw Data
The dstl raw data+label has to be downloaded to data/raw folder following detailed instructions 
[here](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data). 
The major steps are:
1. Install Kaggle API: https://github.com/Kaggle/kaggle-api
1. Download all Dstl data: ```kaggle competitions download -c dstl-satellite-imagery-feature-detection```
1. Unpack the three_band data; and copy the unpacked data, grid_sizes.csv, and train_wkt_v4 
to corresponding locations according to the example files under folder data/raw.

### Preprocess Data and Label
This step aims for generating a dataset from raw data. Make sure Python 2.7 is installed; then navigate to folder src/prepare_data_label and Run
```
python process_data.py ../../data/raw/3band_rgb ../../data/raw/train_wkt_v4.csv ../../data/dataset/images

python process_label.py ../../data/raw/3band_rgb ../../data/raw ../../data/dataset/labels 0
```

### Generate DB for Caffe Training
Python 2.7 with cv2, glob, and lmdb installed is required. You need to navigate to folder src/prepare_data_label.
To generate DB for training a DeepLab+CRF model:
```
python gen_db_deeplab.py ../../data/dataset/images ../../data/dataset/labels ../../data/imgdb/dlcrf_data ../../data/imgdb/dlcrf_label ../../data/imgdb/dlcrf_index.txt
```
To generate DB for training a GoogleNet model:
```
python gen_lmdb_googlenet.py ../../data/dataset/images ../../data/dataset/labels ../../data/lmdb/gneti_data ../../data/lmdb/gneti_label 10
```

### Training Models
This requires all previous steps finishes. **In docker container**, navigate to folder train_dev/. Before running any training, you may want to change the hyperparameters in solver.prototxt according to [this page](https://github.com/BVLC/caffe/wiki/Solver-Prototxt).

To train a DeepLab+CRF model:
```
cd /workspace/road-discovery/train_dev/dlcrf_deeplab_crf/

nohup caffe train -solver solver.prototxt -weights ./pretrained.caffemodel > log_dlcrf.txt &
```
To train a GoogleNet model:
```
cd /workspace/road-discovery/train_dev/gneti_googlenet_inception/

nohup caffe train -solver solver.prototxt -weights ./pretrained.caffemodel > log_gneti.txt &
```

You can visualize and analyse the training performance by doing:
```
cd /workspace/road-discovery/src/visualize_training_log

# visualize training log of DeepLab+CRF
python plot_loss_from_log_dlcrf.py ./log_example_dlcrf.txt

# visualize training log of GoogleNet
python plot_loss_from_log_dlcrf.py ./log_example_gneti.txt
``` 


### Compression and De-compression of Models
**In docker container**, navigate to folder /workspace/road-discovery/model_compression/, and you can do the following things:

Compression of DeepLab+CRF model:
```
# DeepLab+CRF model
python caffemodel_compressor.py compress ../deploy_test/road_finder_dlcrf/bin/deploy_finder_cpu.prototxt ../train_dev/dlcrf_deeplab_crf/snapshots/dl2crf_iter_xxxxx.caffemodel

# GoogleNet model
python caffemodel_compressor.py compress ../deploy_test/road_finder_gneti/bin/deploy_finder_cpu.prototxt ../train_dev/gneti_googlenet_inception/snapshots/goneti_iter_xxxxx.caffemodel
```

De-compression examples:
```
# DeepLab+CRF model
python caffemodel_compressor.py decompress ../deploy_test/road_finder_dlcrf/bin/deploy_finder_cpu.prototxt ./dl2crf_iter_xxxxx.npz

# GoogleNet model
python caffemodel_compressor.py decompress ../deploy_test/road_finder_gneti/bin/deploy_finder_cpu.prototxt ./goneti_iter_xxxxx.npz
```

## Stay Tuned for Road-Vectorization and Change-Detection 
:see_no_evil: :hear_no_evil:  :speak_no_evil: