#
#  Makefile
#  road-discovery: deploy_test:gneti
#
#  Created by Larry Cao on 18/4/8.
#

# Determine the platform
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
  CC := clang++ -arch x86_64
else
  CC := g++
endif

SRCDIR := src
BUILDDIR := build
TARGETDIR := bin
TARGET := bin/gnetiRF

# Normal CPP files
SRCEXT := cpp
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))


CFLAGS := -c
ifeq ($(UNAME_S),Linux)
  CFLAGS += -std=c++0x -O2 # -fPIC
else
  CFLAGS += -std=c++11 -stdlib=libc++ -O2
endif


LIB_BOOST := -L /dvol/packages/boost_1_63_0/build/lib -lboost_thread -lboost_system
LIB_OPENCV := -L /dvol/packages/opencv-2.4.13/build/lib -lopencv_imgproc -lopencv_core -lopencv_highgui
#LIB_CAFFE := -L /opt/caffe/build/lib -Wl,--whole-archive -lcaffe -Wl,--no-whole-archive
LIB_CAFFE := -L /opt/caffe/build/lib -lcaffe
LIB_GLOG := -L /dvol/packages/glog-0.3.4/build/lib -lglog
LIB_CURL := -L /dvol/packages/curl-7.59.0/build/lib -lcurl
LIB_PROTOBUF := -L /dvol/packages/protobuf-2.6.0/build/lib -lprotobuf
LIB_Z := -L /dvol/packages/zlib-1.2.10/build/lib -lz
LIB_JASPER := -L /dvol/packages/opencv-2.4.13/build/3rdparty/lib -llibjasper
LIB_ILMIMF := -L /dvol/packages/openexr-2.2.0/build/lib -lIlmImf
LIB_SSL := -L /dvol/packages/openssl-1.1.0c/build/lib -lcrypto -lssl
LIB_PNG := -L /dvol/packages/libpng-1.6.28/build/lib -lpng
LIB_JPEG_TIFF := -L /dvol/packages/opencv-2.4.13/build/3rdparty/lib -llibjpeg -llibtiff
LIB_HDF := -L /dvol/packages/hdf5-1.8.18/build/lib -lhdf5_hl -lhdf5
LIB_BLAS := -L /dvol/packages/openblas-0.2.19/build/lib -lopenblas 
LIB_IDN := -L /dvol/packages/libidn-1.3/build/lib -lidn
LIB_DL := -L /usr/lib64 -ldl
LIB_GFLAGS := -L /dvol/packages/gflags-2.1.2/build/lib -lgflags
LIB_LMDB := -L /dvol/packages/lmdb-0.9.19/libraries/liblmdb -llmdb
LIB_LEVELDB := -L /dvol/packages/leveldb-1.19/out-static -lleveldb

INC := -I include
INC_CURL := -I /dvol/packages/curl-7.59.0/build/include
INC_OPENCV := -I /dvol/packages/opencv-2.4.13/include
INC_CAFFE := -I /opt/caffe/include -I /opt/caffe/build/include
INC_BOOST := -I /dvol/packages/boost_1_63_0/build/include
INC_GFLAGS := -I /dvol/packages/gflags-2.1.2/build/include
INC_GLOG := -I /dvol/packages/glog-0.3.4/build/include
INC_PROTOBUF := -I /dvol/packages/protobuf-2.6.0/build/include
INC_BLAS := -I /dvol/packages/openblas-0.2.19/build/include


# $(TARGET): $(OBJECTS)
# 	@mkdir -p $(TARGETDIR)
# 	@echo " Linking..."
# 	$(CC) $^ -o $(TARGET) $(LIB_LMDB) $(LIB_LEVELDB) $(LIB_CAFFE) $(LIB_BOOST) $(LIB_OPENCV) $(LIB_GLOG) $(LIB_PROTOBUF) $(LIB_Z) $(LIB_JASPER) $(LIB_ILMIMFI) $(LIB_IDN) $(LIB_SSL) $(LIB_PNG) $(LIB_JPEG_TIFF) $(LIB_DL) $(LIB_HDF) $(LIB_BLAS) $(LIB_CURL) $(LIB_GFLAGS) -llmdb -lleveldb -lboost_thread -lz -ldl -lidn -lrt -lcrypto -lssl ./lib/libcurl.a -pthread -static

# $(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
# 	@mkdir -p $(BUILDDIR)
# 	$(CC) $(CFLAGS) $(INC) $(INC_CURL) $(INC_OPENCV) $(INC_CAFFE) $(INC_BOOST) $(INC_GFLAGS) $(INC_GLOG) $(INC_PROTOBUF) $(INC_BLAS) -c -o $@ $<

$(TARGET): $(OBJECTS)
	@mkdir -p $(TARGETDIR)
	@echo " Linking..."
	$(CC) $^ -o $(TARGET) $(LIB_BOOST) $(LIB_GLOG) $(LIB_CAFFE) $(LIB_OPENCV) $(LIB_CURL)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INC) $(INC_CURL) $(INC_OPENCV) $(INC_CAFFE) $(INC_BOOST) $(INC_GFLAGS) $(INC_GLOG) $(INC_PROTOBUF) $(INC_BLAS) -c -o $@ $<

clean:
	@echo " Cleaning...";
	@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; $(RM) -r $(BUILDDIR) $(TARGET)


install:
	@echo "Nothing to do."

uninstall:
	@echo "Nothing to do."


.PHONY: clean install uninstall
