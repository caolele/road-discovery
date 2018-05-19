//
//  DlcrfCore.hpp
//  road-discovery: deploy_test
//
//  Created by Larry Cao on 18/3/8.
//

#ifndef DlcrfCore_hpp
#define DlcrfCore_hpp

#include "GlobalParams.h"


class DlcrfCore{
    
private:
    
    //the pointer of a caffe network entity
    boost::shared_ptr<Net<float> > net;
    
    // the ptr to input layer in the model
    Blob<float>* inputLayer;
    
    // batch size for prediction
    UInt batchSize;
    
    // number of pixels of the input/output SQUARE image
    UInt dim;
    
    // number of total classes
    UInt cls;
    
    // input batch binded to input blob
    vector<vector<Mat>> inputBatch;
    
    // number of channels in the input image; must be 3
    static const UInt channels;
    
    // the mean value for each channel
    static const float meanval_ch1;
    static const float meanval_ch2;
    static const float meanval_ch3;
    
    // the mean matrix that shoud be subtracted from input
    Mat _mean;
    
    // batch-warp the input layer for the model
    bool wrapBatchInputLayer(vector<vector<Mat>> *inputBatch);
    
    // get the index of a blob from the given name
    int getBlobIndByName(string blobName);
    
    
public:
    
    // constructor
    DlcrfCore(const string& model, const string& weights);
    
    ~DlcrfCore();
    
    /* predict a batch of images; 
     * fs: filter size;
     * mode=A : deeplab only
     * mode=B : deeplab+crf
     * mode=A+B (default) : A+B */
    vector<Mat> predictBatch(const vector<Mat>& imgs,
                             const UInt fs, const string mode = "A+B");
    
    // get the input/output dimension
    UInt getDim();
    
    // get batchSize
    UInt getBatchSize();
    
};

#endif /* DlcrfCore_hpp */
