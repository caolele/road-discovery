//
//  main.cpp
//  road-discovery: deploy_test
//
//  Created by Larry Cao on 18/3/8.
//

#include "GlobalParams.h"
#include "AerialMap.hpp"
#include "RoadFinder.hpp"

// initialize caffe with respect to cpu/gpu mode setting
void initCaffe(const string& caffe_mode){
    // init for CPU mode
    if(caffe_mode == "gpu"){
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(DEVICE_ID);
    }
    /* default = cpu
     else if(caffe_mode != "cpu"){
     throw "wrong value of CAFFE_MODE";
     }*/
}

int main(int argc, const char * argv[]) {
    
    // prohibit caffe logging
    google::InitGoogleLogging("VR");
    FLAGS_stderrthreshold = google::ERROR;
    
    // probability precision
    cout.precision(NUM_PRECISION);
    
    /* dummy is used to ensure that ODPS does not
     * schedule two many tasks for one node */
    string src_img, dst_img;
    vector<double> probs;
    
    try{// try to catch any FATAL error
        
        initCaffe(CAFFE_MODE);
        
        if(argc == 3) {// execute ONCE via parameter-transcending
            src_img = argv[1];
            dst_img = argv[2];

            cout << "Reading aerial image ..." << endl;
            AerialMap *_am = new AerialMap(src_img);
            Mat rawimg = _am->readImg();
            //cout << rawimg << endl;

            cout << "Loading model ..." << endl;
            DlcrfCore *_dlc = new DlcrfCore(MODEL_FINDER, WEIGHT_FINDER);
            RoadFinder *_rf = new RoadFinder(*_dlc);

            cout << "Detecting roads (fast mode) ..." << endl;
            stringstream ss;
            ss << dst_img << "_fast.jpg";
            Mat prbimg = _rf->findRoadsRigid(rawimg);
            if(!imwrite(ss.str(), prbimg)) {
                cout << "Error when writting detection results to " << ss.str() << endl;
            }

            if (rawimg.cols > 512 && rawimg.rows > 512) {
                cout << "Detecting roads (slow mode) for big input ..." << endl;
                prbimg = _rf->findRoadsOverlap(rawimg);
                ss.str("");
                ss << dst_img << "_slow.jpg";
                if(!imwrite(ss.str(), prbimg)) {
                    cout << "Error when writting detection results to " << ss.str() << endl;
                }
            }
            

        } else {
            cout << "Usage: ..." << endl;
        }
            
        
        
    }catch(char const* errMsg){
        cout << "Fatal Error: " << errMsg << endl;
    }
    
    
    google::ShutdownGoogleLogging();
    
    return 0;
    
}
