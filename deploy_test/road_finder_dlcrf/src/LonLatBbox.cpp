//
//  LonLatBbox.cpp
//  road-discovery: deploy_test
//
//  Created by Larry Cao on 18/3/8.
//

#include "LonLatBbox.hpp"


LonLatBbox::LonLatBbox(){
    MinLon = -1;
    MinLat = -1;
    MaxLon = -1;
    MaxLat = -1;
}


LonLatBbox::LonLatBbox(double minLon, double minLat,
                       double maxLon, double maxLat){
    MinLon = minLon;
    MinLat = minLat;
    MaxLon = maxLon;
    MaxLat = maxLat;
}


LonLatBbox::~LonLatBbox(){}
