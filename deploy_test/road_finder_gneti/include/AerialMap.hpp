//
//  AerialMap.hpp
//  road-discovery: deploy_test:gneti
//
//  Created by Larry Cao on 18/4/8.
//

#ifndef AerialMap_hpp
#define AerialMap_hpp

#include "GlobalSettings.h"

class AerialMap{
    
private:
    
    // location of the input image
    string srcImg;

    // the top-left pixel coordinates (expanded)
    UIntPair tl_xy;
    
    // the bottom-right pixel coordinates (expanded)
    UIntPair br_xy;
    
    // the pixel dimension
    Size imgSize;
    
    // the dimension based on number of tiles
    Size tileDim;
    
    // the Mat that stores the actual downloaded aerial map
    Mat rawimg;
    
    // the data write function served for curlImg
    static size_t writeData(char *ptr, size_t size,
                            size_t nmemb, void *userdata);
    
    
public:
    
    // the top-left lonlat coordinates (cooresponding to tl_xy)
    DoublePair tl_lonlat;
    
    // the bottom-right lonlat coordinates (cooresponding to br_xy)
    DoublePair br_lonlat;
    
    // simple constructor
    AerialMap(string _srcImg);
    
    // constructor: initialize tl_xy, br_xy, tl_lonlat, br_lonlat, etc.
    AerialMap(string _srcImg, LonLatBbox &_llbx);
    
    ~AerialMap();
    
    // get a copy of imgSize
    Size getImgSize();
    
    // get a copy of rawimg
    Mat getRawImage();

    // set image source
    void setImgSrc(string _srcImg);

    // download image using libcurl
    Mat curlImg(const char *img_url,
                int timeout = DOWNLOAD_TIMEOUT);
    
    // read an image from a file
    Mat readImg();
    
};


#endif /* AerialMap_hpp */
