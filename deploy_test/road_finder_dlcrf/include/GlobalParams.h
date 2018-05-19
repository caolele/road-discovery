//
//  GlobalParams.h
//  road-discovery: deploy_test
//
//  Created by Larry Cao on 18/3/8.
//

#ifndef GlobalParams_h
#define GlobalParams_h

#include <string>
#include <iostream>
#include "time.h"
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

#include "curl/curl.h"
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fstream>


typedef unsigned int UInt;

typedef pair<double, double> DoublePair;

typedef pair<UInt, UInt> UIntPair;

typedef pair<string, unsigned int> StrUIntPair;

#define CPU_ONLY // comment this line when using GPU
#include "caffe/caffe.hpp"
#ifdef CPU_ONLY
const string CAFFE_MODE = "cpu";
#else
const string CAFFE_MODE = "gpu";
#endif
const UInt DEVICE_ID = 0; // which GPU to use?

using namespace caffe;

// available work mode for compile time
typedef enum {
    NORMAL,
    DEBUG,
    DOWNLOAD_ONLY,
    FINDER_ONLY
} COMPILE_MODE;

const COMPILE_MODE MODE = FINDER_ONLY;

/* When calculating roi, if original width/height < ROI_LIMIT (pixels),
 * the width/heght will be expand to ROI_LIMIT to cover larger neighbouring area.
 * This value is only valid in FINDER_ONLY mode  */
const UInt ROI_LIMIT = 128 * 3;

// the upper limit of execution time for finding one rd.
const UInt LIMIT_PREDTIME = 3000; // seconds


const UInt NUM_PRECISION = 32; // MUST >= 16
const UInt TILE_SIZE = 256;

// the fidelity of aerial image
const UInt ZOOM = 18;

/* bilaterial filter size, recommendation:
 * use 5 for ZOOM=18;
 * use 10 for ZOOM=16; */
const UInt BFILTER_SIZE = 5;

// the maximun road-shift tolerance (in pixel) of either side of rd.
const UInt SHIFT_TOLERANCE = 15;

// the limit of any image size/dimension
const UInt LIMIT_IMG_DIM = 1e5;

// this value is used to threshold the probability map
double const BINARY_THRESHOLD = 0.15 * 255;

// the weight of probimg when calculating the fusion diff score
const double WEIGHT_PROBIMG = 0.7;

// the location of caffe model structure for RoadFinder
const string MODEL_FINDER = "./deploy_finder_cpu.prototxt";

// the location of caffe model weights for RoadFinder
const string WEIGHT_FINDER = "../../../train_dev/dlcrf_deeplab_crf/snapshots/dl2crf_iter_xxxxx.caffemodel";

// the name of the debug folder for storing intermediate images
const string DEBUG_FOLDER = "Debug";

// the name of the folder where output shapefiles will be placed
const string SHAPEFILE_FOLDER = "ShapeFile";

// the name of the added confidence field in shapefile
const string CONFIDENCE_FIELD = "CONFIDENCE";

// the control var for show time stampe while executing the program
const bool SHOW_TIMESTAMP = true;

// ensemble all scores??
const double ENSEMBLE_ALL_SCORE = true;

// the timeout limit for downloading one tile
const UInt DOWNLOAD_TIMEOUT = 25;

#endif /* GlobalParams_h */
