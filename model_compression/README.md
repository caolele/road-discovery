Caffe Model Compression
===================


This is an adapted python tool used to compress the trained caffe weights.
For Deeplab2 + CRF, it manages to reduce the model size from 145MB to about 13MB.
For GoogleNet inception model, it reduce from 24MB to about 3.7MB
The idea comes from [Deep Compression](http://arxiv.org/pdf/1510.00149v5.pdf) . 
This work does not implement purning and Huffman coding, but implement the Kmeans 
quantization to compress the weights of convolution and full-connected layer.
Some minor changes were made to fit in the network architecture of Deeplap model.


#### Dependency

> - Python/Numpy
> - Caffe

#### How to Build:
```
cd quantz_kit 
 ./build.sh
```
#### How to use it:
Compression example:
```
python caffemodel_compressor.py compress ../deploy_test/road_finder_dlcrf/deploy_finder_cpu.prototxt ../train_dev/dlcrf_deeplab_crf/snapshots/dl2crf_iter_xxxxx.caffemodelz")
```

De-compression example:
```
python caffemodel_compressor.py decompress ../deploy_test/road_finder_dlcrf/deploy_finder_cpu.prototxt ./dl2crf_iter_xxxxx.npz")
```