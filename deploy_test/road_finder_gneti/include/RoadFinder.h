//
//  RoadFinder.h
//  road-discovery: deploy_test:gneti
//
//  Created by Larry Cao on 18/4/8.
//

#ifndef RoadFinder_h
#define RoadFinder_h

#include "GlobalSettings.h"
#include "AerialMap.hpp"

class RoadFinder{
    
private:
    
    // a reference to an AerialImage object
    AerialMap &aerialMap;
    
    // the pointer of a caffe network entity
    boost::shared_ptr<Net<float> > net;
    
    // the ptr to input later in the model
    Blob<float>* inputLayer;
    
    // size of the 0~255 prediction map: 1/16 size of rawmap
    Size predmapSize;
    
    // the points that needs to be predicted (non-zero points)
    Mat predmap;
    vector<Point> needPredList;
    
    // batch size for prediction
    UInt batchSize;
    
    // number of channels in the input image
    UInt channels;
    
    // number of pixels of the input SQUARE image (inDim x inDim)
    UInt inDim;
    
    // number of pixels of the output SQUARE image (outDim x outDim)
    UInt outDim;
    
    // probability map of road according to aerialMap
    Mat probimg;
    
    // batch-warp the input layer for finder model
    void WrapBatchInputLayer(vector<vector<Mat>> *inputBatch);
    
    
public:
    
    // constructor: initialize the caffe model and build mandatory maps
    RoadFinder(AerialMap &am, string model, string weights);
    
    ~RoadFinder();
    
    // get the index of a blob from the given name
    int getBlobIndByName(string blobName);
    
    // get a copy of probimg
    Mat getProbImg();
    
    // get a copy of needPredList
    vector<Point> getPredList();
    
    // create and set predmap according to rawmap in road obj.
    // set needPredList; and return the expanded predmap
    Mat setPredList();
    
    // try to predict the probability of road existance on rawmap
    // if successful, it will set probimg as the recognition result
    int tryFindRoad(bool needFilter = true);
    
    
    UInt getOutDim();
};


#endif /* RoadFinder_h */
