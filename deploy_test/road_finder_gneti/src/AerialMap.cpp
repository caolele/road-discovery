//
//  AerialMap.cpp
//  road-discovery: deploy_test:gneti
//
//  Created by Larry Cao on 18/3/8.
//

#include "AerialMap.hpp"


AerialMap::AerialMap(string _srcImg):srcImg(_srcImg){}


AerialMap::AerialMap(string _srcImg, LonLatBbox &_llbx):srcImg(_srcImg){
    //if(!init(_llbx)) throw "illegal LonLatBbox";
}


AerialMap::~AerialMap(){}


void AerialMap::setImgSrc(string _srcImg) {
    srcImg = _srcImg;
}


size_t AerialMap::writeData(char *ptr, size_t size,
                            size_t nmemb, void *userdata){
    vector<uchar> *stream = (vector<uchar>*)userdata;
    size_t count = size * nmemb;
    stream->insert(stream->end(), ptr, ptr + count);
    return count;
}


Mat AerialMap::curlImg(const char *img_url, int timeout){
    vector<uchar> stream;
    CURL *curl = curl_easy_init();
    // the img url
    curl_easy_setopt(curl, CURLOPT_URL, img_url);
    // pass the writefunction
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeData);
    // pass the stream ptr to the writefunction
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream);
    // timeout if curl_easy hangs
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout);
    // start curl
    CURLcode res = curl_easy_perform(curl);
    // cleanup
    curl_easy_cleanup(curl);
    
    if(res == CURLE_OK){
        return imdecode(stream, -1);
    }else{
        Mat tmp;
        return tmp;
    }
}

Mat AerialMap::readImg() {
    Mat image = imread(srcImg, CV_LOAD_IMAGE_COLOR); 
    if(! image.data ) {
        cout <<  "Could not open or find the image" << std::endl;
        Mat tmp;
        return tmp;
    } else {
        imgSize = Size(image.rows, image.cols); //(resulting image size: hxw)
        rawimg = image.clone();
        return image;
    }
}

Size AerialMap::getImgSize(){
    return imgSize;
}

Mat AerialMap::getRawImage(){
    return rawimg;
}
