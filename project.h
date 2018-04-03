#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

void contrast(Mat &img, int n, int m);
void alignment(Mat &img, Mat &img_res, int n, int m);
void laplacianZeroCrossings(Mat &img, int n, int m);
void full_redraw(Mat &img, Mat &img_res2, int n, int m);
