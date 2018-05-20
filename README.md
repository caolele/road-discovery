## The Tooling and Models for Fine-Grained Road Detection from Aerial/Satellite Images

This toolbox contains

```
nohup caffe train -solver solver.prototxt -weights ./pretrained.caffemodel > log_dlcrf.txt &
```

```
nohup caffe train -solver solver.prototxt -weights ./pretrained.caffemodel > log_gneti.txt &
```

./dlcrfRF /dvol/road-discovery/deploy_test/road_finder_dlcrf/test/img3.jpg ../test/img3_road
./gnetiRF ../../test_resource/img3.jpg