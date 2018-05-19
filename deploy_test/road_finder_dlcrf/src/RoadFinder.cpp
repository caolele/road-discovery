//
//  RoadFinder.cpp
//  road-discovery: deploy_test
//
//  Created by Larry Cao on 18/3/8.
//

#include "RoadFinder.hpp"


RoadFinder::RoadFinder(DlcrfCore &_dlc):dlc(_dlc){}


RoadFinder::~RoadFinder(){}


Mat RoadFinder::postImgProcess(const Mat &img, const UInt gks,
                               const UInt bks, const UInt btl){
    // health check
    if(img.empty() || gks % 2 != 1) return img;
    
    // gaussian blur
    GaussianBlur(img, img, Size(gks, gks), 0, 0);
    
    // bilateral filter
    Mat oimg;
    bilateralFilter(img, oimg, bks, bks*2, bks/2);
    
    // hrad limit binary thresholding
    threshold(oimg, oimg, btl, 255, THRESH_BINARY);
    
    return oimg;
}


Mat RoadFinder::findRoadsRigid(const Mat &aimg){
    UInt td = dlc.getDim(); //dimension of prediction tiles
    UInt bs = dlc.getBatchSize(); //batchSize of Dlcrf model
    UInt rows = aimg.rows;
    UInt cols = aimg.cols;
    
    //health check
    if(aimg.empty()) throw "illegal aerial image";
    if(td < 1 || td > LIMIT_IMG_DIM || bs < 1)
        throw "illegal DLcrfCore model";
    
    // init. output
    Mat pimg = Mat::zeros(rows, cols, CV_32FC1);
    
    // input and output vector of patches and ther top-left anchors
    vector<Mat> ipatches;
    vector<UIntPair> anchors;
    
    // max number of predictions in both directions
    UInt maxi = rows/td;
    UInt maxj = cols/td;
    
    // predict batch-by-batch
    UInt cidx = 0, tmpi, tmpj; // current index of prediction tile
    for(UInt i = 0; i < maxi; i++){
        for(UInt j = 0; j < maxj; j++){
            cidx = (i * maxj + j + 1);
            tmpi = i * td;
            tmpj = j * td;
            anchors.push_back(make_pair(tmpj, tmpi));
            ipatches.push_back(aimg(Rect(tmpj, tmpi, td, td)));
            
            // batch full or end-of-data: start predict this batch.
            if(cidx % bs == 0 || cidx == maxi * maxj){
                cout << cidx << endl;
                if(bs > anchors.size()){ // fillin dummy patches
                    for(UInt k = 0; k < bs - anchors.size(); k++){
                        ipatches.push_back(Mat::zeros(td, td, CV_8UC3));
                    }
                }
                
                // predict a batch
                vector<Mat> opatches(dlc.predictBatch(ipatches, BFILTER_SIZE));
                if(opatches.size() < bs) throw "batch prediction error";
                
                // write to output
                for(UInt k = 0; k < anchors.size(); k++){
                    opatches[k].copyTo(pimg(Rect(anchors[k].first,
                                                 anchors[k].second,
                                                 td, td)));
                }
                
                // clear cache for next batch
                anchors.clear();
                ipatches.clear();
            }
        }
    }
    
    return pimg * 255;
}


Mat RoadFinder::findRoadsOverlap(const Mat &aimg, const UInt cps){
    UInt td = dlc.getDim(); //dimension of prediction tiles
    UInt bs = dlc.getBatchSize(); //batchSize of Dlcrf model
    UInt rows = aimg.rows;
    UInt cols = aimg.cols;
    
    //health check
    if(aimg.empty() || cols < cps || rows < cps || cps >= td)
        throw "illegal aerial image";
    if(td < 1 || td > LIMIT_IMG_DIM || bs < 1)
        throw "illegal DLcrfCore model";
    
    // init. output
    Mat pimg = Mat::zeros(rows, cols, CV_32FC1);
    
    // input and output vector of patches and ther top-left anchors
    vector<Mat> ipatches;
    vector<UIntPair> anchors;
    
    // predict batch-by-batch
    UInt cidx = 0; // total patch count
    UInt offset = (td - cps) / 2;
    for(UInt i = 0; i <= rows - td; i += cps){
        for(UInt j = 0; j <= cols - td; j += cps){
            anchors.push_back(make_pair(j, i));
            ipatches.push_back(aimg(Rect(j, i, td, td)));
            ++cidx;
            
            // batch full or end-of-data: start predict this batch.
            if(cidx % bs == 0 || (i > rows - td && j > cols - td)){
                cout << cidx << endl;
                if(bs > anchors.size()){ // fillin dummy patches
                    for(UInt k = 0; k < bs - anchors.size(); k++){
                        ipatches.push_back(Mat::zeros(td, td, CV_8UC3));
                    }
                }
                
                // predict a batch
                vector<Mat> opatches(dlc.predictBatch(ipatches, BFILTER_SIZE));
                if(opatches.size() < bs) throw "batch prediction error";
                
                // write to output
                for(UInt k = 0; k < anchors.size(); k++){
                    opatches[k](Rect(offset, offset, cps, cps)).\
                        copyTo(pimg(Rect(anchors[k].first + offset,
                                         anchors[k].second + offset,
                                         cps, cps)));
                }
                
                // clear cache for next batch
                anchors.clear();
                ipatches.clear();
            }
        }
    }
    
    return pimg * 255;
}


Mat RoadFinder::findRefmapRoadsOverlap(const Mat &aimg, const Mat &rmap,
                                       void (*pfunc)(const string, const UInt, const string),
                                       const UInt cps){
    UInt td = dlc.getDim(); //dimension of prediction tiles
    UInt bs = dlc.getBatchSize(); //batchSize of Dlcrf model
    UInt rows = aimg.rows;
    UInt cols = aimg.cols;
    
    //health check
    if(aimg.empty() || cols < cps || rows < cps || cps >= td)
        throw "illegal aerial image";
    if(td < 1 || td > LIMIT_IMG_DIM || bs < 1)
        throw "illegal DLcrfCore model";
    if(rows != rmap.rows || cols != rmap.cols)
        throw "illegal reference map";
    if(rows % cps != 0 || cols % cps != 0)
        throw "illegal cps/image size";
    
    // record the time when prediction started
    time_t start, end;
    time(&start);
    
    // construct prediction map
    Mat prdmap;
    resize(rmap, prdmap, Size(cols / cps, rows / cps), 0, 0, INTER_AREA);
    
    // init. output
    Mat pimg = Mat::zeros(rows, cols, CV_32FC1);
    
    // input and output vector of patches and ther top-left anchors
    vector<Mat> ipatches;
    vector<UIntPair> anchors;

    // find the non-zero points in prdmap
    Mat nziPrdmap;
    findNonZero(prdmap, nziPrdmap);
    size_t nziTotal = nziPrdmap.total();
    int i, j, offset = (td - cps) / 2;
    UInt nvp = 0; // total number of valid patches
    
    // show total tiles
    if(SHOW_TIMESTAMP){
        pfunc("Tiles", (UInt)nziTotal, "=");
    }
    
    for(int k = 0; k < nziTotal; k++){
        j = cps * nziPrdmap.at<Point>(k).x - offset;
        i = cps * nziPrdmap.at<Point>(k).y - offset;
        
        // border check
        if(i < 0 || j < 0 || i + td > rows || j + td > cols){
            if(SHOW_TIMESTAMP && MODE != NORMAL && pfunc != NULL)
                pfunc("Skip tile no.", k + 1, " ");
            continue;
        }
        
        anchors.push_back(make_pair(j, i));
        ipatches.push_back(aimg(Rect(j, i, td, td)));
        nvp++;
        
        // batch full or end-of-data: start predict this batch.
        if(nvp % bs == 0 || k + 1 == nziTotal){
            if(bs > anchors.size()){ // fillin dummy patches
                for(UInt v = 0; v < bs - anchors.size(); v++){
                    ipatches.push_back(Mat::zeros(td, td, CV_8UC3));
                }
            }
            
            // predict a batch
            vector<Mat> opatches(dlc.predictBatch(ipatches, BFILTER_SIZE));
            if(opatches.size() < bs) throw "batch prediction error";
            
            // write to output
            for(UInt v = 0; v < anchors.size(); v++){
                opatches[v](Rect(offset, offset, cps, cps)).\
                copyTo(pimg(Rect(anchors[v].first + offset,
                                 anchors[v].second + offset,
                                 cps, cps)));
            }
            
            // clear cache for next batch
            anchors.clear();
            ipatches.clear();
            
            // get the current time and calculate the total time elapsed
            time(&end);
            double predtime_elapse = difftime(end, start);
            if(predtime_elapse > LIMIT_PREDTIME) throw "violate time limit";
            
            // print progress when allowed
            if(SHOW_TIMESTAMP && MODE != NORMAL && pfunc != NULL){
                stringstream ss;
                ss << "tile:" << k + 1 << "/" << nziTotal << "\ttime(sec.)-->";
                pfunc(ss.str(), (UInt)predtime_elapse, "");
            }
        }
    }
    
    return pimg * 255;
}

