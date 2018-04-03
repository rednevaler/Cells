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

void redraw(const Mat &img, Mat img_res, int startI, int startJ, int **index,
            int &sz);
void preredraw(Mat img, int startI, int startJ, int **index);
bool inside_picture(long int i, long int j, long int n, long int m);

void full_redraw(Mat &img, Mat &img_res2, int n, int m) {
    int i, j;
    int num_of_cells = 0;
    int **index;
    int sz;

    index = new int *[n];
    for (i = 0; i < n; i++) {
        index[i] = new int[m];
    }

    for (i = 0; i < n; ++i) {
        for (j = 0; j < m; ++j) {
            index[i][j] = 0;
        }
    }

    for (j = 4; j < m - 4; ++j) {
        for (i = 4; i < n - 4; ++i) {
            if (img.at<uchar>(j, i) == 0 && index[i][j] == 0) {
                preredraw(img, i, j, index);
            }
        }
    }

    for (i = 0; i < n; ++i) {
        for (j = 0; j < m; ++j) {
            index[i][j] = 0;
        }
    }

    for (j = 4; j < m - 4; ++j) {
        for (i = 4; i < n - 4; ++i) {
            if (img.at<uchar>(j, i) == 0 && index[i][j] == 0) {
                redraw(img, img_res2, i, j, index, sz);
                if (sz >= 20) {
                    ++num_of_cells;
                }
            }
        }
    }
    printf("Number of cells = %d\n", num_of_cells);
    return;
}

void redraw(const Mat &img, Mat img_res, int startI, int startJ, int **index,
            int &sz) {
    int n = img.size[1], m = img.size[0];
    int x[2000], y[2000];
    sz = 0;
    int k, x_center = 0, y_center = 0, count = 0;

    int center_inside_cell = 0;

    for (k = 0; k != 2000; ++k) {
        x[k] = n + 1;
        y[k] = m + 1;
    }

    std::vector<std::pair<int, int> > stack;
    stack.emplace_back(startI, startJ);

    while (!stack.empty()) {
        auto coords = stack.back();
        int i = coords.first, j = coords.second;

        index[i][j] = 1;
        x[sz] = i, y[sz] = j;
        ++sz;

        // move right
        if (i + 1 < n - 4 && i + 1 >= 4) {
            if (img.at<uchar>(j, i + 1) == 0 && (index[i + 1][j] == 0)) {
                stack.emplace_back(i + 1, j);
                continue;
            }
        }

        // move left
        if (i - 1 >= 4 && i - 1 < n - 4) {
            if (img.at<uchar>(j, i - 1) == 0 && (index[i - 1][j] == 0)) {
                stack.emplace_back(i - 1, j);
                continue;
            }
        }

        // move down
        if (j + 1 < m - 4 && j + 1 >= 4) {
            if (img.at<uchar>(j + 1, i) == 0 && (index[i][j + 1] == 0)) {
                stack.emplace_back(i, j + 1);
                continue;
            }
        }

        // move up
        if (j - 1 >= 4 && j - 1 < m - 4) {
            if (img.at<uchar>(j - 1, i) == 0 && (index[i][j - 1] == 0)) {
                stack.emplace_back(i, j - 1);
                continue;
            }
        }

        stack.pop_back();
    }
    for (k = 0; k != 1000; ++k) {
        if (x[k] != n + 1 && y[k] != m + 1) {
            x_center += x[k];
            y_center += y[k];
            ++count;
        }
    }
    x_center = x_center / count;
    y_center = y_center / count;

    for (k = 0; k != 1000; ++k) {
        if (x_center == x[k] && y_center == y[k]) {
            ++center_inside_cell;
            break;
        }
    }

    if (sz >= 20 && center_inside_cell > 0) {
        for (k = 0; k != 1000; ++k) {
            if (x[k] != n + 1 && y[k] != m + 1) {
                img_res.at<Vec3b>(y[k], x[k]).val[0] = 0;
                img_res.at<Vec3b>(y[k], x[k]).val[1] = 0;
                img_res.at<Vec3b>(y[k], x[k]).val[2] = 255;
            }
        }
    }
    if (center_inside_cell == 0) {
        sz = 0;
    }
}

bool inside_picture(long int i, long int j, long int n, long int m) {
    if (i < n && i >= 0 && j < m && j >= 0) {
        return 1;
    } else {
        return 0;
    }
}

void preredraw(Mat img, int startI, int startJ, int **index) {
    int n = img.size[1], m = img.size[0], k, l, sz = 0;
    int x[100000], y[100000];
    for (k = 0; k != 100000; ++k) {
        x[k] = n + 1;
        y[k] = m + 1;
    }
    std::vector<std::pair<int, int> > stack;
    stack.emplace_back(startI, startJ);

    while (!stack.empty()) {
        auto coords = stack.back();
        int i = coords.first, j = coords.second;

        index[i][j] = 1;
        x[sz] = i, y[sz] = j;
        ++sz;

        // move right
        if (i + 1 < n - 4 && i + 1 >= 4) {
            if (img.at<uchar>(j, i + 1) == 0 && (index[i + 1][j] == 0)) {
                stack.emplace_back(i + 1, j);
                continue;
            }
        }

        // move left
        if (i - 1 >= 4 && i - 1 < n - 4) {
            if (img.at<uchar>(j, i - 1) == 0 && (index[i - 1][j] == 0)) {
                stack.emplace_back(i - 1, j);
                continue;
            }
        }

        // move down
        if (j + 1 < m - 4 && j + 1 >= 4) {
            if (img.at<uchar>(j + 1, i) == 0 && (index[i][j + 1] == 0)) {
                stack.emplace_back(i, j + 1);
                continue;
            }
        }

        // move up
        if (j - 1 >= 4 && j - 1 < m - 4) {
            if (img.at<uchar>(j - 1, i) == 0 && (index[i][j - 1] == 0)) {
                stack.emplace_back(i, j - 1);
                continue;
            }
        }

        stack.pop_back();
    }

    for (k = 0; k != sz - 1; ++k) {
        for (l = 0; l != sz - k - 1; ++l) {
            if (x[l] > x[l + 1]) {
                int xx = x[l + 1];
                x[l + 1] = x[l];
                x[l] = xx;
                int yy = y[l + 1];
                y[l + 1] = y[l];
                y[l] = yy;
            } else if (x[l] == x[l + 1]) {
                if (y[l] > y[l + 1]) {
                    int yy = y[l + 1];
                    y[l + 1] = y[l];
                    y[l] = yy;
                }
            }
        }
    }
    for (k = 1; k != sz; ++k) {
        if (x[k - 1] == x[k]) {
            if (y[k - 1] < y[k] - 1) {
                for (int l = y[k - 1] + 1; l != y[k]; ++l) {
                    img.at<uchar>(l, x[k]) = 0;
                }
            }
        }
    }

    for (k = 0; k != sz - 1; ++k) {
        for (l = 0; l != sz - k - 1; ++l) {
            if (y[l] > y[l + 1]) {
                int xx = x[l + 1];
                x[l + 1] = x[l];
                x[l] = xx;
                int yy = y[l + 1];
                y[l + 1] = y[l];
                y[l] = yy;
            } else if (y[l] == y[l + 1]) {
                if (x[l] > x[l + 1]) {
                    int xx = x[l + 1];
                    x[l + 1] = x[l];
                    x[l] = xx;
                }
            }
        }
    }
    for (k = 1; k != sz; ++k) {
        if (y[k - 1] == y[k]) {
            if (x[k - 1] < x[k] - 1) {
                for (int l = x[k - 1] + 1; l != x[k]; ++l) {
                    img.at<uchar>(y[k], l) = 0;
                }
            }
        }
    }
}
