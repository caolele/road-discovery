FROM bvlc/caffe:cpu

LABEL maintainer "caolele@gmail.com"
LABEL description "A demo docker for Road Discovery"

USER root

RUN cd ~ && \ 
    wget https://curl.haxx.se/download/curl-7.59.0.tar.gz && \
    tar -xvf curl-7.59.0.tar.gz && \
    cd ~/curl-7.59.0 && \
    mkdir build && \
    ./configure --prefix=/root/curl-7.59.0/build && \
    make -j4 && \
    make install && \
    cd ~ && \
    git clone --recursive https://github.com/caolele/road-discovery.git && \
    cd /opt/caffe && \ 
    rm -rf examples include python src tools build && \
    cp -rf ~/road-discovery/caffe/* . && \ 
    apt-get update && \ 
    apt-get install -y libmatio-dev && \
    cd /opt/caffe/build && \
    make all -j4

WORKDIR ~/road-discovery