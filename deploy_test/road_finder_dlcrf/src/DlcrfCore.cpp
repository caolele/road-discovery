//
//  DlcrfCore.cpp
//  road-discovery: deploy_test
//
//  Created by Larry Cao on 18/3/8.
//

#include "DlcrfCore.hpp"

const UInt DlcrfCore::channels = 3;

const float DlcrfCore::meanval_ch1 = 104.008;
const float DlcrfCore::meanval_ch2 = 116.669;
const float DlcrfCore::meanval_ch3 = 122.675;

DlcrfCore::DlcrfCore(const string& model, const string& weights){
    
    // Init. model
    net.reset(new Net<float>(model, TEST));
    net->CopyTrainedLayersFrom(weights);
    inputLayer = net->input_blobs()[0];
    batchSize = inputLayer->num();
    dim = inputLayer->width();
    
    // check if channels == 3
    if(channels != inputLayer->channels())
        throw "illegal number of channels";
    
    // check if the input image is a square
    if(inputLayer->width() != inputLayer->height())
        throw "illegal input size";
    
    // get the total number of classes
    int upsampleLayerId = getBlobIndByName("upscore");
    if(upsampleLayerId < 0) throw "illegal model definition";
    boost::shared_ptr<Blob<float>> upsampleLayer = net->blobs()[upsampleLayerId];
    cls = upsampleLayer->channels();
    
    // initialize mean
    _mean = Mat(1, channels, CV_64F);
    _mean.row(0).col(0) = meanval_ch1;
    _mean.row(0).col(1) = meanval_ch2;
    _mean.row(0).col(2) = meanval_ch3;
    
    // batch wrap input layer
    inputLayer->Reshape(batchSize, channels, dim, dim);
    if(!wrapBatchInputLayer(&inputBatch))
        throw "DlcrfCore wrap error";
}

DlcrfCore::~DlcrfCore(){}


UInt DlcrfCore::getDim(){
    return dim;
}


UInt DlcrfCore::getBatchSize(){
    return batchSize;
}


int DlcrfCore::getBlobIndByName(string blobName){
    vector<string> const & blob_names = net->blob_names();
    for(int i = 0; i < blob_names.size(); i++){
        if(blobName == blob_names[i]) return i;
    }
    return -1;
}


bool DlcrfCore::wrapBatchInputLayer(vector<vector<Mat>> *_inputBatch){
    // health check
    if(inputBatch.size() > 0) return false;
    
    float* input_data = inputLayer->mutable_cpu_data();
    for(int j = 0; j < batchSize; j++){
        vector<Mat> input_channels;
        for (int i = 0; i < channels; ++i){
            Mat channel(dim, dim, CV_32FC1, input_data);
            input_channels.push_back(channel);
            input_data += dim * dim;
        }
        _inputBatch->push_back(vector<Mat>(input_channels));
    }
    
    return true;
}


vector<Mat> DlcrfCore::predictBatch(const vector<Mat>& imgs,
                                    const UInt fs, const string mode){
    
    vector<Mat> probimgs;

    // health check
    if(imgs.size() != batchSize || inputLayer == NULL ||
       net->output_blobs().size() != 2 || cls < 2)
        return probimgs;
    
    // In GPU mode, InputLayer needs to be wrapped for every batch
    if(Caffe::mode() == Caffe::GPU){
        inputBatch.clear();
        if(!wrapBatchInputLayer(&inputBatch))
            throw "DlcrfCore wrap error";
    }
    
    // predict each image patch in a batch
    for(int i = 0; i < batchSize; i++){
        // health check per image
        if(imgs[i].empty() ||
           imgs[i].cols != dim ||
           imgs[i].rows != dim) return probimgs;
        
        // filter operation
        Mat patch = imgs[i].clone();
        if(fs > 0)
            bilateralFilter(imgs[i], patch, fs, fs*2, fs/2);
        
        // convert input to float
        Mat patchFloat;
        patch.convertTo(patchFloat, CV_32FC3);

        // subtract the means from sample value
        Mat patchFloatNorm;
        subtract(patchFloat, _mean, patchFloatNorm);
        
        // write data to input layer
        vector<Mat> *input_channels = &(inputBatch.at(i));
        /* This operation will write the separate BGR planes directly to the
         * input layer of the network because it is wrapped by the cv::Mat
         * objects in input_channels. */
        split(patchFloatNorm, *input_channels);
        
        if(i == 0){ // this check step may be dummy
            CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
                  == inputLayer->cpu_data())
            << "Input channels are not wrapping the input layer";
        }
    }//end of for
    
    // predict this batch
    net->ForwardFrom(0);
    
    // cast output values to probimgs
    const float* headPtrA = net->output_blobs()[0]->cpu_data();
    const float* headPtrB = net->output_blobs()[1]->cpu_data();
    Mat ctm;
    for(int i = 1; i < batchSize * cls; i += cls){
        if(mode == "A"){
            ctm = Mat(dim, dim, CV_32FC1,
                      const_cast<float *>(headPtrA + i*dim*dim));
        }else if(mode == "B"){
            ctm = Mat(dim, dim, CV_32FC1,
                      const_cast<float *>(headPtrB + i*dim*dim));
        }else{
            add(Mat(dim, dim, CV_32FC1,
                    const_cast<float *>(headPtrA + i*dim*dim)),
                Mat(dim, dim, CV_32FC1,
                    const_cast<float *>(headPtrB + i*dim*dim)),
                ctm);
        }
        
        probimgs.push_back(ctm);
        ctm.release();
    }
    
    return probimgs;
}
