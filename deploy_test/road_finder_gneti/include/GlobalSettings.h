//
//  GlobalSettings.h
//  road-discovery: deploy_test:gneti
//
//  Created by Larry Cao on 18/4/8.
//

#ifndef GlobalSettings_h
#define GlobalSettings_h

#include <string>
#include <iostream>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

#include "curl/curl.h"

#define DELALL delete IS_DEBUG;delete SHIFT_TOLERANCE;delete BINARY_THRESHOLD

typedef unsigned int UInt;

typedef pair<double, double> DoublePair;

typedef pair<UInt, UInt> UIntPair;

#define CPU_ONLY // comment this line when using GPU
#include "caffe/caffe.hpp"
#ifdef CPU_ONLY
const string CAFFE_MODE = "cpu";
// the upper limit of numPredTile, link that violate this limit will output -2 as CONFIDENCE
const UInt LIMIT_PREDTILE = 2000; // tiles
const UInt LIMIT_PREDTIME = 3000; // seconds
#else
const string CAFFE_MODE = "gpu";
const UInt LIMIT_PREDTILE = 1e5; // tiles
const UInt LIMIT_PREDTIME = 1e3; // seconds
#endif
using namespace caffe;

using namespace std;


struct LonLatBbox{
    double MinLon;
    double MinLat;
    double MaxLon;
    double MaxLat;
    
    LonLatBbox(){
        MinLon = -1;
        MinLat = -1;
        MaxLon = -1;
        MaxLat = -1;
    }
};


const UInt NUM_PRECISION = 32;

const UInt ZOOM = 18;

const UInt TILE_SIZE = 256;

// the maximun road-shift tolerance (in pixel) of either left or right
extern UInt* const SHIFT_TOLERANCE;

// this value is used to threshold the probability map
extern double* const BINARY_THRESHOLD;

// true: store all intermediate maps/images under the same folder
extern bool* const IS_DEBUG;

// the control var for show time stampe while executing the program
const bool SHOW_TIMESTAMP = true;

// the location of caffe model structure for RoadFinder
const string MODEL_FINDER = "./deploy_finder_cpu.prototxt";

// the location of caffe model weights for RoadFinder
const string WEIGHT_FINDER = "../../../train_dev/gneti_googlenet_inception/snapshots/goneti_iter_xxxxx.caffemodel";

// the location of mean file
const string MEAN_DIFFER = "./mean.binaryproto";

// the name of the reshapre layer in the model definition
const string RESHAPE_LAYER_NAME = "loss3/reshape";

// need the fusion (legacy) diff-score??
const double NEED_FUSION_DIFFSCORE = true;

// the weight of probimg when calculating the fusion diff score
const double WEIGHT_PROBIMG = 0.7;

// the scale-up ratio of probimg when used in caculating diff score
const UInt PROBIMG_SCALEUP = 1e3;

// the limit of any image size/dimension
const UInt LIMIT_IMG_DIM = 1e5;

// the added expand-factor for road searching
const double EXPFAC_RDSEARCH = 1.5;

// ensemble all scores??
const double ENSEMBLE_ALL_SCORE = true;

// the timeout limit for downloading one tile
const UInt DOWNLOAD_TIMEOUT = 25;

#endif /* GlobalSettings_h */
