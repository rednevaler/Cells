#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void contrast(Mat &img, int n, int m) {
    long int sum = 0, avrg = 0;
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < n; ++i) {
            sum += img.at<uchar>(j, i);
        }
    }

    avrg = sum / (n * m);
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < n; ++i) {
            if ((img.at<uchar>(j, i) + (img.at<uchar>(j, i) - avrg) * 5 <
                 256) &&
                (img.at<uchar>(j, i) + (img.at<uchar>(j, i) - avrg) * 5 >=
                 0)) {  // ususally5
                img.at<uchar>(j, i) =
                    img.at<uchar>(j, i) + (img.at<uchar>(j, i) - avrg) * 5;
            } else if (img.at<uchar>(j, i) + (img.at<uchar>(j, i) - avrg) * 5 >=
                       256) {
                img.at<uchar>(j, i) = 255;
            } else {
                img.at<uchar>(j, i) = 0;
            }
        }
    }
    return;
}
