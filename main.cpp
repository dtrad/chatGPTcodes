#include "opencv2/opencv.hpp"
using namespace cv;
Mat matchFilter(Mat img, Mat templ) {
    Mat result;
    matchTemplate(img, templ, result, TM_CCOEFF_NORMED);
    return result;
}


void main(){}

    Mat img = imread("image.jpg", IMREAD_GRAYSCALE);
    Mat templ = imread("template.jpg", IMREAD_GRAYSCALE);
    Mat result = matchFilter(img, templ);
    double minVal, maxVal;
    Point minLoc, maxLoc;

    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);



return 0;



}