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

double Laplacian(int i, int j, const Mat& img);
bool zero_neighbour(int i, int j, int n, int m, double** log);
bool black_neighbour(int i, int j, const Mat& img);
bool one_black_neighbour(int i, int j, const Mat& img);
int SumOfLaplacianSigns(int i, int j, int n, int m, double** lapl);
bool inside_picture(int i, int j, int n, int m);
int NumSign(double x);
void laplacianZeroCrossings(Mat& img, int n, int m) {
    int i, j;
    int num_of_neighbours;
    double** lapl;

    lapl = new double*[n];
    for (i = 0; i < n; i++) {
        lapl[i] = new double[m];
    }
    //    int index[n][m];
    //    int red[n][m];
    for (i = 0; i < n; ++i) {
        for (j = 0; j < m; ++j) {
            lapl[i][j] = 0.;
        }
    }

    for (j = 0; j < m; ++j) {
        for (i = 0; i < n; ++i) {
            lapl[i][j] = Laplacian(i, j, img);
        }
    }

    for (j = 0; j < m; ++j) {
        for (i = 0; i < n; ++i) {
            if (fabs(lapl[i][j]) <= 2 /*&& zero_neighbour(i,j,n,m,lapl)==1*/) {
                if (i == 0 || j == 0 || i == n - 1 || j == m - 1) {
                    if ((i == 0 && j == 0) || (i == 0 && j == m - 1) ||
                        (i == n - 1 && j == 0) || (i == n - 1 && j == m - 1)) {
                        num_of_neighbours = 3;
                        if (fabs(SumOfLaplacianSigns(i, j, n, m, lapl)) !=
                            num_of_neighbours) {
                            img.at<uchar>(j, i) = 255;
                        } else {
                            img.at<uchar>(j, i) = 0;
                        }
                    } else {
                        num_of_neighbours = 5;
                        if (fabs(SumOfLaplacianSigns(i, j, n, m, lapl)) !=
                            num_of_neighbours) {
                            img.at<uchar>(j, i) = 255;
                        } else {
                            img.at<uchar>(j, i) = 0;
                        }
                    }
                } else {
                    num_of_neighbours = 8;
                    if (fabs(SumOfLaplacianSigns(i, j, n, m, lapl)) !=
                        num_of_neighbours) {
                        img.at<uchar>(j, i) = 255;
                    } else {
                        img.at<uchar>(j, i) = 0;
                    }
                }

            } else {
                img.at<uchar>(j, i) = 0;
            }
        }
    }

    for (j = 0; j < m; ++j) {
        for (i = 0; i < n; ++i) {
            if (img.at<uchar>(j, i) == 0 && black_neighbour(i, j, img) == 0) {
                img.at<uchar>(j, i) = 255;
            }
        }
    }

    return;
}

int SumOfLaplacianSigns(int i, int j, int n, int m, double** lapl) {
    double l1, l2, l3, l4, l5, l6, l7, l8;
    /*
     1 2 3
     4   5
     6 7 8
     */
    if (i == 0) {
        if (j == 0) {
            l5 = lapl[i][j + 1];
            l7 = lapl[i + 1][j];
            l8 = lapl[i + 1][j + 1];
            return NumSign(l5) + NumSign(l7) + NumSign(l8);
        }
        if (j == m - 1) {
            l4 = lapl[i][j - 1];
            l6 = lapl[i + 1][j - 1];
            l7 = lapl[i + 1][j];
            return NumSign(l4) + NumSign(l6) + NumSign(l7);
        } else {
            l4 = lapl[i][j - 1];
            l5 = lapl[i][j + 1];
            l6 = lapl[i + 1][j - 1];
            l7 = lapl[i + 1][j];
            l8 = lapl[i + 1][j + 1];
            return NumSign(l4) + NumSign(l5) + NumSign(l6) + NumSign(l7) +
                   NumSign(l8);
        }
    }
    if (j == 0) {
        if (i == n - 1) {
            l2 = lapl[i - 1][j];
            l3 = lapl[i - 1][j + 1];
            l5 = lapl[i][j + 1];
            return NumSign(l2) + NumSign(l3) + NumSign(l5);
        } else {
            l2 = lapl[i - 1][j];
            l3 = lapl[i - 1][j + 1];
            l5 = lapl[i][j + 1];
            l7 = lapl[i + 1][j];
            l8 = lapl[i + 1][j + 1];
            return NumSign(l2) + NumSign(l3) + NumSign(l5) + NumSign(l7) +
                   NumSign(l8);
        }
    }
    if (i == n - 1) {
        if (j == m - 1) {
            l1 = lapl[i - 1][j - 1];
            l2 = lapl[i - 1][j];
            l4 = lapl[i][j - 1];
            return NumSign(l1) + NumSign(l2) + NumSign(l4);
        } else {
            l1 = lapl[i - 1][j - 1];
            l2 = lapl[i - 1][j];
            l3 = lapl[i - 1][j + 1];
            l4 = lapl[i][j - 1];
            l5 = lapl[i][j + 1];
            return NumSign(l1) + NumSign(l2) + NumSign(l3) + NumSign(l4) +
                   NumSign(l5);
        }
    }
    if (j == m - 1) {
        l1 = lapl[i - 1][j - 1];
        l2 = lapl[i - 1][j];
        l4 = lapl[i][j - 1];
        l6 = lapl[i + 1][j - 1];
        l7 = lapl[i + 1][j];
        return NumSign(l1) + NumSign(l2) + NumSign(l4) + NumSign(l6) +
               NumSign(l7);
    } else {
        l1 = lapl[i - 1][j - 1];
        l2 = lapl[i - 1][j];
        l3 = lapl[i - 1][j + 1];
        l4 = lapl[i][j - 1];
        l5 = lapl[i][j + 1];
        l6 = lapl[i + 1][j - 1];
        l7 = lapl[i + 1][j];
        l8 = lapl[i + 1][j + 1];
        return NumSign(l1) + NumSign(l2) + NumSign(l3) + NumSign(l4) +
               NumSign(l5) + NumSign(l6) + NumSign(l7) + NumSign(l8);
    }
}

int NumSign(double x) {
    if (x > 0) {
        return 1;
    }
    if (x < 0) {
        return -1;
    } else {
        return 0;
    }
}

double Laplacian(int i, int j, const Mat& img) {
    int n = img.size[1], m = img.size[0];
    if (j + 1 < m) {
        if (j - 1 >= 0) {
            if (i + 1 < n) {
                if (i - 1 >= 0) {
                    return (img.at<uchar>(j + 1, i) + img.at<uchar>(j - 1, i) +
                            img.at<uchar>(j, i + 1) + img.at<uchar>(j, i - 1) -
                            4 * img.at<uchar>(j, i));
                } else {
                    return (img.at<uchar>(j + 1, i) + img.at<uchar>(j - 1, i) +
                            img.at<uchar>(j, i + 1) - 3 * img.at<uchar>(j, i));
                }
            } else {
                if (i - 1 >= 0) {
                    return (img.at<uchar>(j + 1, i) + img.at<uchar>(j - 1, i) +
                            img.at<uchar>(j, i - 1) - 3 * img.at<uchar>(j, i));
                } else {
                    return (img.at<uchar>(j + 1, i) + img.at<uchar>(j - 1, i) -
                            2 * img.at<uchar>(j, i));
                }
            }
        } else {
            if (i + 1 < n) {
                if (i - 1 >= 0) {
                    return (img.at<uchar>(j + 1, i) + img.at<uchar>(j, i + 1) +
                            img.at<uchar>(j, i - 1) - 3 * img.at<uchar>(j, i));
                } else {
                    return (img.at<uchar>(j + 1, i) + img.at<uchar>(j, i + 1) -
                            2 * img.at<uchar>(j, i));
                }
            } else {
                if (i - 1 >= 0) {
                    return (img.at<uchar>(j + 1, i) + img.at<uchar>(j, i - 1) -
                            2 * img.at<uchar>(j, i));
                } else {
                    return (img.at<uchar>(j + 1, i) - img.at<uchar>(j, i));
                }
            }
        }
    } else {
        if (j - 1 >= 0) {
            if (i + 1 < n) {
                if (i - 1 >= 0) {
                    return (img.at<uchar>(j - 1, i) + img.at<uchar>(j, i + 1) +
                            img.at<uchar>(j, i - 1) - 3 * img.at<uchar>(j, i));
                } else {
                    return (img.at<uchar>(j - 1, i) + img.at<uchar>(j, i + 1) -
                            2 * img.at<uchar>(j, i));
                }
            } else {
                if (i - 1 >= 0) {
                    return (img.at<uchar>(j - 1, i) + img.at<uchar>(j, i - 1) -
                            2 * img.at<uchar>(j, i));
                } else {
                    return (img.at<uchar>(j - 1, i) - img.at<uchar>(j, i));
                }
            }
        } else {
            if (i + 1 < n) {
                if (i - 1 >= 0) {
                    return (img.at<uchar>(j, i + 1) + img.at<uchar>(j, i - 1) -
                            2 * img.at<uchar>(j, i));
                } else {
                    return (img.at<uchar>(j, i + 1) - img.at<uchar>(j, i));
                }
            } else {
                if (i - 1 >= 0) {
                    return (img.at<uchar>(j, i - 1) - img.at<uchar>(j, i));
                } else {
                    return (img.at<uchar>(j, i));
                }
            }
        }
    }
}

bool zero_neighbour(int i, int j, int n, int m, double** log) {
    int k, l;
    int num_of_zero_neighbours = 0;
    for (k = -1; k <= 1; ++k) {
        for (l = -1; l <= 1; ++l) {
            if (inside_picture(i + k, j + l, n, m) == 1) {
                if (fabs(log[i + k][j + l]) <= 2 && (k != 0 || l != 0)) {
                    ++num_of_zero_neighbours;
                    if (num_of_zero_neighbours >= 2) {
                        return 1;
                    }
                }
            }
        }
    }
    return 0;
}

bool inside_picture(int i, int j, int n, int m) {
    if (i < n && i >= 0 && j < m && j >= 0) {
        return 1;
    } else {
        return 0;
    }
}

bool black_neighbour(int i, int j, const Mat& img) {
    int k, l, num_of_black_neighbours;
    int n = img.size[1], m = img.size[0];
    num_of_black_neighbours = 0;
    for (k = -1; k <= 1; ++k) {
        for (l = -1; l <= 1; ++l) {
            if (inside_picture(i + k, j + l, n, m) == 1 && (k != 0 || l != 0)) {
                if (img.at<uchar>(j + l, i + k) == 0) {
                    ++num_of_black_neighbours;
                    if (num_of_black_neighbours >= 2) {
                        return 1;
                    }
                }
            }
        }
    }
    return 0;
}

bool one_black_neighbour(int i, int j, const Mat& img) {
    int k, l, num_of_black_neighbours;
    int n = img.size[1], m = img.size[0];
    num_of_black_neighbours = 0;
    k = -1;
    l = 0;

    if (inside_picture(i + k, j + l, n, m) == 1 && (k = 0 || l != 0)) {
        if (img.at<uchar>(j + l, i + k) == 0) {
            ++num_of_black_neighbours;
            if (num_of_black_neighbours >= 1) {
                return 1;
            }
        }
    }

    k = 1;
    l = 0;

    if (inside_picture(i + k, j + l, n, m) == 1 && (k = 0 || l != 0)) {
        if (img.at<uchar>(j + l, i + k) == 0) {
            ++num_of_black_neighbours;
            if (num_of_black_neighbours >= 1) {
                return 1;
            }
        }
    }

    k = 0;
    l = -1;

    if (inside_picture(i + k, j + l, n, m) == 1 && (k = 0 || l != 0)) {
        if (img.at<uchar>(j + l, i + k) == 0) {
            ++num_of_black_neighbours;
            if (num_of_black_neighbours >= 1) {
                return 1;
            }
        }
    }

    k = 0;
    l = 1;

    if (inside_picture(i + k, j + l, n, m) == 1 && (k = 0 || l != 0)) {
        if (img.at<uchar>(j + l, i + k) == 0) {
            ++num_of_black_neighbours;
            if (num_of_black_neighbours >= 1) {
                return 1;
            }
        }
    }

    return 0;
}
