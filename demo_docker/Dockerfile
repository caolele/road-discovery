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
    mkdir mount

# WORKDIR ~/road-discovery