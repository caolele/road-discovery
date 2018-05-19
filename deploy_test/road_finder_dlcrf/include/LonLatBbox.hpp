//
//  LonLatBbox.hpp
//  road-discovery: deploy_test
//
//  Created by Larry Cao on 18/3/8.
//

#ifndef LonLatBbox_hpp
#define LonLatBbox_hpp


class LonLatBbox{
    
public:
    
    double MinLon;
    double MinLat;
    double MaxLon;
    double MaxLat;
    
    LonLatBbox();
    LonLatBbox(double minLon, double minLat,
               double maxLon, double maxLat);
    
    ~LonLatBbox();
    
};

#endif /* LonLatBbox_hpp */
