FROM bvlc/caffe:cpu

LABEL maintainer "caolele@gmail.com"
LABEL description "A demo docker for Road Discovery"

USER root

RUN cd ~ && \ 
    apt-get update && \
    apt-get install -y vim && \
    apt-get install -y curl && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    wget https://curl.haxx.se/download/curl-7.59.0.tar.gz && \
    tar -xvf curl-7.59.0.tar.gz && \
    cd ~/curl-7.59.0 && \
    mkdir build && \
    ./configure --prefix=/root/curl-7.59.0/build && \
    make -j4 && \
    make install && \
    cd /workspace && \
    git clone --recursive https://github.com/caolele/road-discovery.git && \
    cd /opt/caffe && \ 
    rm -rf examples include python src tools build && \
    cp -rf /workspace/road-discovery/caffe/* . && \  
    apt-get install -y libmatio-dev && \
    cd /opt/caffe/build && \
    cmake .. && \
    make all -j4 && \
    cd /workspace && \
    mkdir mount && \
    echo 'export LD_LIBRARY_PATH=/opt/caffe/build/lib:$LD_LIBRARY_PATH' >> ~/.bashrc && \
    export LD_LIBRARY_PATH=/opt/caffe/build/lib:$LD_LIBRARY_PATH && \
    cd /workspace/road-discovery/model_compression && \
    python caffemodel_compressor.py decompress ../deploy_test/road_finder_dlcrf/bin/deploy_finder_cpu.prototxt ./dl2crf_iter_xxxxx.npz && \
    mv ./dl2crf_iter_xxxxx.caffemodel ../train_dev/dlcrf_deeplab_crf/snapshots/ &&\
    python caffemodel_compressor.py decompress ../deploy_test/road_finder_gneti/bin/deploy_finder_cpu.prototxt ./goneti_iter_xxxxx.npz && \
    mv ./goneti_iter_xxxxx.caffemodel ../train_dev/gneti_googlenet_inception/snapshots/

# WORKDIR ~/road-discovery