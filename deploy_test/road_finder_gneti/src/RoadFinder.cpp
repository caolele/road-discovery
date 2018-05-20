//
//  RoadFinder.cpp
//  road-discovery: deploy_test:gneti
//
//  Created by Larry Cao on 18/4/8.
//

#include "RoadFinder.h"


RoadFinder::RoadFinder(AerialMap &am, string model, string weights):aerialMap(am){
    // Init caffe model
    // Logging before InitGoogleLogging() is written to STDERR
    net.reset(new Net<float>(model, TEST));
    net->CopyTrainedLayersFrom(weights);
    
    // init. batch size and number of channels
    inputLayer = net->input_blobs()[0];
    batchSize = inputLayer->num();
    channels = inputLayer->channels();
    
    // check if the input image is a square
    if(inputLayer->width() != inputLayer->height())
        throw "illegal input size";
        
    inDim = inputLayer->width();
    
    // check if reshape layer is properly defined
    int reshapeLayerId = getBlobIndByName(RESHAPE_LAYER_NAME);
    if(reshapeLayerId < 0) throw "illegal model definition";
    
    boost::shared_ptr<Blob<float>> shapeLayer = net->blobs()[reshapeLayerId];
    
    // check if reshapre layer is a square
    if(shapeLayer->width() != shapeLayer->height())
        throw "illegal definition of reshape layer";
    
    outDim = shapeLayer->width();
}


RoadFinder::~RoadFinder(){}


int RoadFinder::getBlobIndByName(string blobName){
    vector<string> const & blob_names = net->blob_names();
    for(int i = 0; i < blob_names.size(); i++){
        if(blobName == blob_names[i]) return i;
    }
    return -1;
}


Mat RoadFinder::setPredList(){
    Mat resmap;
    
    // HERE: just predict all positions as a toy example
    // Mat rawmap = road.getRawMap();
    Mat rawmap = Mat::ones(aerialMap.getImgSize(), CV_8UC1) * 255;
    if(rawmap.empty()) return resmap;
    
    // check if size of aerialMap is set or not
    Size igs = aerialMap.getImgSize();
    if(igs.width <= 0 || igs.height <= 0) return resmap;
    
    // INTER_AREA = 3
    predmapSize = Size(igs.width/outDim, igs.height/outDim);
    resize(rawmap, predmap, predmapSize, 0, 0, INTER_AREA);
    
    //DEBUG
    if(*IS_DEBUG)
        if(!imwrite("./predmap.jpg", predmap)) return resmap;
    
    // find the non-zero points in predmap
    Mat nziPredMap;
    findNonZero(predmap, nziPredMap);
    
    // construct needPredList and expanded predmap
    UInt tmpx, tmpy;
    resmap = predmap.clone();
    for (int i = 0; i < nziPredMap.total(); i++){
        tmpx = nziPredMap.at<Point>(i).x;
        tmpy = nziPredMap.at<Point>(i).y;
        // record the top-left anchor of (outDim x outDim) output square
        needPredList.push_back(Point(tmpx*outDim, tmpy*outDim));
        
        // need to crop a (outDim x outDim) rect centered by (px,py);
        // here we use loose boundary of 2*outDim instead of inDim/outDim
        rectangle(resmap, Point(tmpx - outDim, tmpy - outDim),
                  Point(tmpx + outDim, tmpy + outDim), 255, CV_FILLED);
    }
    
    //DEBUG
    if(*IS_DEBUG)
        if(!imwrite("./predmap_ext.jpg", resmap)) return resmap;

    return resmap;
}


UInt RoadFinder::getOutDim(){
    return outDim;
}


Mat RoadFinder::getProbImg(){
    return probimg;
}


vector<Point> RoadFinder::getPredList(){
    return needPredList;
}


void RoadFinder::WrapBatchInputLayer(vector<vector<Mat>> *_inputBatch){
    float* input_data = inputLayer->mutable_cpu_data();
    for(int j = 0; j < batchSize; j++){
        vector<Mat> input_channels;
        for (int i = 0; i < channels; ++i){
            Mat channel(inDim, inDim, CV_32FC1, input_data);
            input_channels.push_back(channel);
            input_data += inDim * inDim;
        }
        _inputBatch->push_back(vector<Mat>(input_channels));
    }
}


int RoadFinder::tryFindRoad(bool needFilter){
    // rawimg should not be empty; needPredList should not be empty
    // the inputLayer should have a valid pointer
    Mat rawimg = aerialMap.getRawImage();
    UInt numPredTile = UInt(needPredList.size());
    if(*IS_DEBUG) {
        cout << rawimg.empty() << "," << numPredTile << "," << inputLayer << endl;
    }
    if(rawimg.empty() || numPredTile < 1 || inputLayer == NULL)
        return -1;
    
    UInt div_ioDim = int((inDim - outDim) / 2);
    
    inputLayer->Reshape(batchSize, channels, inDim, inDim);
    
    // batch wrap input layer
    vector<vector<Mat>> inputBatch;
    WrapBatchInputLayer(&inputBatch);
    
    probimg = Mat::zeros(aerialMap.getImgSize(), CV_32FC1);
    
    // batch prediction
    UInt tmpx, tmpy, j;
    vector<UIntPair> pAnchors;
    time_t start, end;
    time(&start); // record the time when prediction started
    for(int i = 0; i < numPredTile; ++i){
        tmpx = needPredList[i].x;
        tmpy = needPredList[i].y;
        if (tmpx < div_ioDim || tmpy < div_ioDim
            || tmpx - div_ioDim + inDim > rawimg.cols
            || tmpy - div_ioDim + inDim > rawimg.rows) {
            continue;
        }
        pAnchors.push_back(make_pair(tmpx, tmpy));
        Mat patch;
        if(needFilter){
            if(*IS_DEBUG){
                cout<<"patch - "<<tmpx - div_ioDim <<","<<tmpy - div_ioDim <<","<<inDim<<","<<inDim<<endl;
            }
            bilateralFilter(rawimg(Rect(tmpx - div_ioDim,
                                        tmpy - div_ioDim,
                                        inDim, inDim)),
                            patch, 5, 5 * 2, 5 / 2);
        }
        Mat patchFloat;
        patch.convertTo(patchFloat, CV_32FC3);
        
        j = i % batchSize;
        vector<Mat> *input_channels = &(inputBatch.at(j));
        
        /* This operation will write the separate BGR planes directly to the
         * input layer of the network because it is wrapped by the cv::Mat
         * objects in input_channels. */
        split(patchFloat, *input_channels);
        
        if(j == 0){ // this check step may be dummy
            CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
                  == inputLayer->cpu_data())
            << "Input channels are not wrapping the input layer";
        }
        
        // batch full or end-of-data: start predict this batch.
        if(batchSize == j+1 || i == numPredTile-1){
            // predict the current batch
            net->ForwardFrom(0);
            
            // get the current time and calculate the total time elapsed
            time(&end);
            double predtime_elapse = difftime(end, start);
            cout << "i=" << i+1 << "; time=" << predtime_elapse << endl;
            if(predtime_elapse > LIMIT_PREDTIME) return -2;
            
            // cast output values to probimg
            const float* headPtr = net->output_blobs()[0]->cpu_data();
            for(int k = 0; k < pAnchors.size(); k++){
                Mat ctm = Mat(outDim, outDim, CV_32FC1,
                              const_cast<float *>(headPtr + k*outDim*outDim));
                ctm.copyTo(probimg(Rect(pAnchors[k].first,
                                        pAnchors[k].second,
                                        outDim, outDim)));
            }
            
            pAnchors.clear();
        }
    }
    
    //DEBUG
    if(*IS_DEBUG){
        if(!imwrite("./probimg.jpg", probimg*255)) return -1;
    }
    
    return 1;
}
