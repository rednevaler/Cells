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

void alignment(Mat &img, Mat &img_res, int n, int m) {
    for (int j = 1; j < m - 1; ++j) {
        for (int i = 1; i < n - 1; ++i) {
            int clr = (img.at<uchar>(j - 1, i - 1) + img.at<uchar>(j - 1, i) +
                       img.at<uchar>(j - 1, i + 1) + img.at<uchar>(j, i - 1) +
                       img.at<uchar>(j, i) + img.at<uchar>(j, i + 1) +
                       img.at<uchar>(j + 1, i - 1) + img.at<uchar>(j + 1, i) +
                       img.at<uchar>(j + 1, i + 1)) /
                      9;
            img_res.at<uchar>(j, i) = clr;
        }
    }
    for (int j = 0; j != m; ++j) {
        for (int i = 0; i != n; ++i) {
            img.at<uchar>(j, i) = img_res.at<uchar>(j, i);
        }
    }
}
