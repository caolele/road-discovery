## The Tooling and Models for Fine-Grained Road Detection from Aerial/Satellite Images

This repo contains two complete examples of aerial detection using [DeepLab+CRF](https://arxiv.org/abs/1606.00915) and [GoogleNet](https://arxiv.org/abs/1409.4842) models respectively.

This toolbox contains functions of
- data-preprocessing (of [Kaggle dstl dataset](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data)): src/prepare_data_label/
- training models: train_dev/
- training metrics visualization: src/visualize_training_log/
- compression and de-compression of exported models (c.f. [Deep Compression Paper](https://arxiv.org/pdf/1510.00149v5.pdf)): model_compression/
- model deployment and test: deploy_test/

```
nohup caffe train -solver solver.prototxt -weights ./pretrained.caffemodel > log_dlcrf.txt &
```

```
nohup caffe train -solver solver.prototxt -weights ./pretrained.caffemodel > log_gneti.txt &
```

./dlcrfRF /dvol/road-discovery/deploy_test/road_finder_dlcrf/test/img3.jpg ../test/img3_road
./gnetiRF ../../test_resource/img3.jpg

docker build -t road-discovery .

