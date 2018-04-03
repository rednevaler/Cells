#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "project.h"

int main() {
    Mat img = imread(
        "/Users/polinakomissarova/CLionProjects/full_project/cells03.png",
        IMREAD_GRAYSCALE);
    Mat img_res = imread(
        "/Users/polinakomissarova/CLionProjects/full_project/cells03.png",
        IMREAD_GRAYSCALE);
    Mat img_res2 = imread(
        "/Users/polinakomissarova/CLionProjects/full_project/cells03.png");

    int n = img.size[1], m = img.size[0];

    contrast(img, n, m);
    for (int j = 0; j != m; ++j) {
        for (int i = 0; i != n; ++i) {
            img_res.at<uchar>(j, i) = img.at<uchar>(j, i);
        }
    }
    alignment(img, img_res, n, m);
    laplacianZeroCrossings(img, n, m);
    for (int j = 0; j != m; ++j) {
        for (int i = 0; i != n; ++i) {
            img_res2.at<Vec3b>(j, i).val[0] = img.at<uchar>(j, i);
            img_res2.at<Vec3b>(j, i).val[1] = img.at<uchar>(j, i);
            img_res2.at<Vec3b>(j, i).val[2] = img.at<uchar>(j, i);
        }
    }
    full_redraw(img, img_res2, n, m);
    imwrite("cells03_res.png", img_res2);

    return 0;
}