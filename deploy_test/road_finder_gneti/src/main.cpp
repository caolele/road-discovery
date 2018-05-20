//
//  main.cpp
//  road-discovery: deploy_test:gneti
//
//  Created by Larry Cao on 18/4/8.
//

#include "time.h"
#include "GlobalSettings.h"
#include "AerialMap.hpp"
#include "RoadFinder.h"

UInt* const SHIFT_TOLERANCE = new UInt(10);
bool* const IS_DEBUG = new bool(true);
double* const BINARY_THRESHOLD = new double(0.15);


//the print function of date and time
void printDateTime(string _id){
    struct tm *now;
    char now_str[80];
    time_t lt_now;
    
    lt_now = time(NULL);
    now = localtime(&lt_now);
    strftime(now_str, 100, "%G-%m-%d %H:%M:%S", now);
    cerr << now_str << "\t" << _id <<endl;
}


// initialize caffe with respect to cpu/gpu mode setting
void initCaffe(const string& caffe_mode){
    // init for CPU mode
    if(caffe_mode == "gpu"){
        Caffe::set_mode(Caffe::GPU);
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
    
    string src_img, dst_img;
    vector<double> probs;
    
    try{// try to catch any FATAL error
        
        initCaffe(CAFFE_MODE);
        
        if(argc == 2) {// execute ONCE via parameter-transcending
            src_img = argv[1];
            
            cout << "Reading aerial image ..." << endl;
            AerialMap *_am = new AerialMap(src_img);
            Mat rawimg = _am->readImg();
            
            //cout << rawimg << endl;
            
            cout << "Loading model ..." << endl;
            RoadFinder *_rf = new RoadFinder(*_am, MODEL_FINDER, WEIGHT_FINDER);
            
            // recognize road on aerial map according to predmap in road finder
            cout << "Try to find road on map ..." << endl;
            _rf->setPredList();
            int rf_rt_code = _rf->tryFindRoad();
            if(rf_rt_code == -1) throw "failed predicting road";
            if(rf_rt_code == -2) throw "road prediction timeout";
            
            delete _am;
            delete _rf;
            
        } else {
            cout << "Usage: gnetiRF input_file.jpg" << endl;
        }
        
    }catch(char const* errMsg){
        cout << "Fatal Error: " << errMsg << endl;
    }
    
    DELALL;
    google::ShutdownGoogleLogging();
    return 0;
}
