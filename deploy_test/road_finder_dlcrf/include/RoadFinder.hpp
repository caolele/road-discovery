//
//  RoadFinder.hpp
//  road-discovery: deploy_test
//
//  Created by Larry Cao on 18/3/8.
//

#ifndef RoadFinder_hpp
#define RoadFinder_hpp

#include "DlcrfCore.hpp"


class RoadFinder{
    
private:
    
    // a reference to a DlcrfCore object
    DlcrfCore &dlc;
    
    
public:
    
    // constructor: DlcrfCore object as parameter
    RoadFinder(DlcrfCore &_dlc);
    
    ~RoadFinder();
    
    /* find all roads in an Aerial Image in non-overlap manner;
     * the prediction is done on e.g. 449x449 basis;
     * there might be blackout areas on the border,
     * when 449x449 doesn't fit.
     * This function returns a same sized image */
    Mat findRoadsRigid(const Mat &aimg);
    
    /* find all roads in an Aerial Image in an overlap manner;
     * cps: prediction stride;
     * the prediction is still done on e.g. 449x449 basis;
     * but only the center cps*cps prediction result will be used
     * there might be blackout areas on the border,
     * when 449x449 doesn't fit.
     * This function returns a same sized image */
    Mat findRoadsOverlap(const Mat &aimg, const UInt cps = TILE_SIZE);
    
    /* find roads in an overlap manner according to refmap;
     * cps: prediction stride, i.e. inner crop size
     * the prediction is still done on e.g. 449x449 basis;
     * but only the center cps*cps prediction result will be used
     * This function returns a same sized image */
    Mat findRefmapRoadsOverlap(const Mat &aimg, const Mat &rmap,
                               void (*pfunc)(const string, const UInt, const string) = NULL,
                               const UInt cps = TILE_SIZE);
    
    /* post process probability image:
     * (1) gaussian blur: gks - kernel size
     * (2) bilateral filter: bks - kernel size
     * (3) binary threshod: btl - thresholding limit */
    Mat postImgProcess(const Mat &img, const UInt gks = 9,
                       const UInt bks = 10, const UInt btl = 100);
    
};


#endif /* RoadFinder_hpp */
