#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

double Laplacian (long int i, long int j, const Mat& img);
bool black_neighbour(long int i, long int j, const Mat& img);
int SumOfLaplacianSigns (long int i, long int j, long int n, long int m, double **lapl);
bool inside_picture(long int i, long int j, long int n, long int m);
int NumSign(double x);
void redraw(const Mat& img, Mat img_res, int startI, int startJ, int **index, int &sz);
void preredraw(Mat img, int startI, int startJ, int **index);

int main() {
    Mat img = imread("cells.png", IMREAD_GRAYSCALE);
    Mat img_res = imread("cells.png", IMREAD_GRAYSCALE);
    Mat img_res2 = imread("cells.png");

    //CONTRAST

    int sum, avrg;
    int n = img.size[1], m = img.size[0];
    for(int j=0; j<m; ++j) {
        for (int i = 0; i < n; ++i) {
            sum+=img.at<uchar>(j,i);
        }
    }

    avrg = sum/(n*m);
    for(int j=0; j<m; ++j) {
        for (int i = 0; i < n; ++i) {
            if((img.at<uchar>(j,i) + (img.at<uchar>(j,i)-avrg)*5<256) && (img.at<uchar>(j,i) + (img.at<uchar>(j,i)-avrg)*5>=0)){ //ususally5
                img.at<uchar>(j,i) = img.at<uchar>(j,i) + (img.at<uchar>(j,i)-avrg)*5;
                img_res.at<uchar>(j,i) = img_res.at<uchar>(j,i) + (img_res.at<uchar>(j,i)-avrg)*5;
            }
            else if(img.at<uchar>(j,i) + (img.at<uchar>(j,i)-avrg)*5>=256){
                img.at<uchar>(j,i) = 255;
                img_res.at<uchar>(j,i) = 255;
            }
            else{
                img.at<uchar>(j,i) = 0;
                img_res.at<uchar>(j,i) = 0;
            }
        }
    }

    //5 times ALIGNMENT

    for(int k=0; k!=5; ++k){
        for(int j=1; j<m-1; ++j) {
            for (int i = 1; i < n - 1; ++i) {
                int clr = (img.at<uchar>(j-1,i-1)+
                           img.at<uchar>(j-1,i)+
                           img.at<uchar>(j-1,i+1)+
                           img.at<uchar>(j,i-1)+
                           img.at<uchar>(j,i)+
                           img.at<uchar>(j,i+1)+
                           img.at<uchar>(j+1,i-1)+
                           img.at<uchar>(j+1,i)+
                           img.at<uchar>(j+1,i+1))/9;
                img_res.at<uchar>(j,i) = clr;
            }
        }
        for(int j=0; j!=m; ++j){
            for (int i=0; i!=n; ++i){
                img.at<uchar>(j,i)=img_res.at<uchar>(j,i);
            }
        }
    }

    //LAPLACIAN_ZERO_CROSSINGS EDGES

    int num_of_neighbours;
    double **lapl;


    lapl = new double * [n];
    for (int i = 0; i < n; i++) {
        lapl[i] = new double [m];
    }
    for(int i=0; i<n;++i){
        for(int j=0; j<m;++j){
            lapl[i][j]=0.;
        }
    }

    for(int j=0; j<m; ++j) {
        for(int i=0; i<n; ++i) {
            lapl[i][j]=Laplacian(i,j,img);
        }
    }


    for(int j=0; j<m; ++j) {
        for(int i=0; i<n; ++i) {
            if(fabs(lapl[i][j])<=2){
                if(i==0 || j==0 || i==n-1 || j==m-1){
                    if ((i==0 && j==0) || (i==0 && j==m-1) || (i==n-1 && j==0) || (i==n-1 && j==m-1)){
                        num_of_neighbours = 3;
                        if (fabs(SumOfLaplacianSigns(i,j,n,m,lapl)) != num_of_neighbours){
                            img.at<uchar>(j, i) = 255;
                        }
                        else{
                            img.at<uchar>(j, i) = 0;
                        }
                    }
                    else{
                        num_of_neighbours = 5;
                        if (fabs(SumOfLaplacianSigns(i,j,n,m,lapl)) != num_of_neighbours){
                            img.at<uchar>(j, i) = 255;
                        }
                        else{
                            img.at<uchar>(j, i) = 0;
                        }
                    }
                }
                else{
                    num_of_neighbours = 8;
                    if (fabs(SumOfLaplacianSigns(i,j,n,m,lapl)) != num_of_neighbours){
                        img.at<uchar>(j, i) = 255;
                    }
                    else{
                        img.at<uchar>(j, i) = 0;
                    }
                }
            }
            else {
                img.at<uchar>(j, i) = 0;
            }
        }
    }

    for(int j = 0; j < m; ++j) {
        for (int i = 0; i < n; ++i) {
            if(img.at<uchar>(j,i)==0 && black_neighbour(i,j,img)==0){
                img.at<uchar>(j,i)=255;
            }
        }
    }

    //REDRAW

    int num_of_cells = 0;
    int **index;
    int sz;

    index = new int * [n];
    for (int i = 0; i < n; i++) {
        index[i] = new int [m];
    }
    for(int i=0; i<n;++i){
        for(int j=0; j<m;++j){
            index[i][j]=0;
        }
    }

    for(int j=0; j!=m; ++j){
        for (int i=0; i!=n; ++i){
            img_res2.at<Vec3b>(j,i).val[0]=img.at<uchar>(j,i);
            img_res2.at<Vec3b>(j,i).val[1]=img.at<uchar>(j,i);
            img_res2.at<Vec3b>(j,i).val[2]=img.at<uchar>(j,i);
        }
    }

    for(int j = 4; j < m-4; ++j) {
        for (int i = 4; i < n-4; ++i) {
            if(img.at<uchar>(j,i)==0 && index[i][j]==0){
                preredraw(img, i, j, index);
            }
        }
    }


    for(int i=0; i<n;++i){
        for(int j=0; j<m;++j){
            index[i][j]=0;
        }
    }


    for(int j = 4; j < m-4; ++j) {
        for (int i = 4; i < n-4; ++i) {
            if(img.at<uchar>(j,i)==0 && index[i][j]==0){
                redraw(img, img_res2, i, j, index, sz);
                if(sz>=20){
                    ++num_of_cells;
                }

            }

        }
    }
    imwrite("cells_res.png", img_res2);
    printf("Number of cells = %d\n", num_of_cells);
    return 0;

}

int SumOfLaplacianSigns (long int i, long int j, long int n, long int m, double **lapl){
    double l1,l2,l3,l4,l5,l6,l7,l8;
    /*
     1 2 3
     4   5
     6 7 8
     */
    if(i==0){
        if(j==0){
            l5 = lapl[i][j+1];
            l7 = lapl[i+1][j];
            l8 = lapl[i+1][j+1];
            return NumSign(l5)+NumSign(l7)+NumSign(l8);
        }
        if(j==m-1){
            l4 = lapl[i][j-1];
            l6 = lapl[i+1][j-1];
            l7 = lapl[i+1][j];
            return NumSign(l4)+NumSign(l6)+NumSign(l7);
        }
        else{
            l4 = lapl[i][j-1];
            l5 = lapl[i][j+1];
            l6 = lapl[i+1][j-1];
            l7 = lapl[i+1][j];
            l8 = lapl[i+1][j+1];
            return NumSign(l4)+NumSign(l5)+NumSign(l6)+NumSign(l7)+NumSign(l8);
        }
    }
    if(j==0){
        if(i==n-1){
            l2 = lapl[i-1][j];
            l3 = lapl[i-1][j+1];
            l5 = lapl[i][j+1];
            return NumSign(l2)+NumSign(l3)+NumSign(l5);
        }
        else{
            l2 = lapl[i-1][j];
            l3 = lapl[i-1][j+1];
            l5 = lapl[i][j+1];
            l7 = lapl[i+1][j];
            l8 = lapl[i+1][j+1];
            return NumSign(l2)+NumSign(l3)+NumSign(l5)+NumSign(l7)+NumSign(l8);
        }
    }
    if(i==n-1){
        if(j==m-1){
            l1 = lapl[i-1][j-1];
            l2 = lapl[i-1][j];
            l4 = lapl[i][j-1];
            return NumSign(l1)+NumSign(l2)+NumSign(l4);
        }
        else{
            l1 = lapl[i-1][j-1];
            l2 = lapl[i-1][j];
            l3 = lapl[i-1][j+1];
            l4 = lapl[i][j-1];
            l5 = lapl[i][j+1];
            return NumSign(l1)+NumSign(l2)+NumSign(l3)+NumSign(l4)+NumSign(l5);
        }
    }
    if(j==m-1){
        l1 = lapl[i-1][j-1];
        l2 = lapl[i-1][j];
        l4 = lapl[i][j-1];
        l6 = lapl[i+1][j-1];
        l7 = lapl[i+1][j];
        return NumSign(l1)+NumSign(l2)+NumSign(l4)+NumSign(l6)+NumSign(l7);
    }
    else{
        l1 = lapl[i-1][j-1];
        l2 = lapl[i-1][j];
        l3 = lapl[i-1][j+1];
        l4 = lapl[i][j-1];
        l5 = lapl[i][j+1];
        l6 = lapl[i+1][j-1];
        l7 = lapl[i+1][j];
        l8 = lapl[i+1][j+1];
        return NumSign(l1)+NumSign(l2)+NumSign(l3)+NumSign(l4)+NumSign(l5)+NumSign(l6)+NumSign(l7)+NumSign(l8);
    }
}



int NumSign(double x){
    if(x>0){
        return 1;
    }
    if(x<0){
        return -1;
    }
    else{
        return 0;
    }
}



double Laplacian (long int i, long int j, const Mat& img){
    long int n = img.size[1], m = img.size[0];
    if (j+1<m){
        if(j-1>=0){
            if(i+1<n){
                if(i-1>=0){
                    return(img.at<uchar>(j+1,i)+img.at<uchar>(j-1,i)+img.at<uchar>(j,i+1)+img.at<uchar>(j,i-1)-4*img.at<uchar>(j,i));
                }
                else{
                    return(img.at<uchar>(j+1,i)+img.at<uchar>(j-1,i)+img.at<uchar>(j,i+1)-3*img.at<uchar>(j,i));
                }
            }
            else{
                if(i-1>=0){
                    return(img.at<uchar>(j+1,i)+img.at<uchar>(j-1,i)+img.at<uchar>(j,i-1)-3*img.at<uchar>(j,i));
                }
                else{
                    return(img.at<uchar>(j+1,i)+img.at<uchar>(j-1,i)-2*img.at<uchar>(j,i));
                }
            }
        }
        else{
            if(i+1<n){
                if(i-1>=0){
                    return(img.at<uchar>(j+1,i)+img.at<uchar>(j,i+1)+img.at<uchar>(j,i-1)-3*img.at<uchar>(j,i));
                }
                else{
                    return(img.at<uchar>(j+1,i)+img.at<uchar>(j,i+1)-2*img.at<uchar>(j,i));
                }
            }
            else{
                if(i-1>=0){
                    return(img.at<uchar>(j+1,i)+img.at<uchar>(j,i-1)-2*img.at<uchar>(j,i));
                }
                else{
                    return(img.at<uchar>(j+1,i)-img.at<uchar>(j,i));
                }
            }
        }
    }
    else{
        if(j-1>=0){
            if(i+1<n){
                if(i-1>=0){
                    return(img.at<uchar>(j-1,i)+img.at<uchar>(j,i+1)+img.at<uchar>(j,i-1)-3*img.at<uchar>(j,i));
                }
                else{
                    return(img.at<uchar>(j-1,i)+img.at<uchar>(j,i+1)-2*img.at<uchar>(j,i));
                }
            }
            else{
                if(i-1>=0){
                    return(img.at<uchar>(j-1,i)+img.at<uchar>(j,i-1)-2*img.at<uchar>(j,i));
                }
                else{
                    return(img.at<uchar>(j-1,i)-img.at<uchar>(j,i));
                }
            }
        }
        else{
            if(i+1<n){
                if(i-1>=0){
                    return(img.at<uchar>(j,i+1)+img.at<uchar>(j,i-1)-2*img.at<uchar>(j,i));
                }
                else{
                    return(img.at<uchar>(j,i+1)-img.at<uchar>(j,i));
                }
            }
            else{
                if(i-1>=0){
                    return(img.at<uchar>(j,i-1)-img.at<uchar>(j,i));
                }
                else{
                    return(img.at<uchar>(j,i));
                }
            }
        }
    }
}

bool inside_picture(long int i, long int j, long int n, long int m){
    if(i<n && i>=0 && j<m && j>=0){
        return 1;
    }
    else{
        return 0;
    }
}

bool black_neighbour(long int i, long int j, const Mat& img){
    int k, l, num_of_black_neighbours;
    long int n = img.size[1], m = img.size[0];
    num_of_black_neighbours = 0;
    for (k=-1; k<=1; ++k){
        for(l=-1; l<=1; ++l){
            if(inside_picture(i+k,j+l,n,m)==1 && (k!=0 || l!=0)){
                if(img.at<uchar>(j+l, i+k)==0){
                    ++num_of_black_neighbours;
                    if(num_of_black_neighbours>=2){
                        return 1;
                    }
                }
            }
        }
    }
    return 0;
}

void redraw(const Mat& img, Mat img_res, int startI, int startJ, int **index, int &sz) {
    int n = img.size[1], m = img.size[0];
    int x[2000], y[2000];
    sz = 0;
    int k, x_center = 0, y_center = 0, count = 0;

    int center_inside_cell = 0;

    for(k=0; k!=2000; ++k) {
        x[k] = n+1; y[k] = m+1;
    }

    std::vector<std::pair<int, int> > stack;
    stack.emplace_back(startI, startJ);

    while (!stack.empty()) {
        auto coords = stack.back();
        int i = coords.first, j = coords.second;

        index[i][j] = 1;
        x[sz]=i, y[sz]=j;
        ++sz;

        // move right
        if (i+1<n-4 && i+1>=4){
            if(img.at<uchar>(j,i+1)==0
               && (index[i + 1][j] == 0)){
                stack.emplace_back(i + 1, j);
                continue;
            }
        }




        // move left
        if (i-1>=4 && i-1<n-4){
            if(img.at<uchar>(j,i-1)==0
               && (index[i - 1][j] == 0)){
                stack.emplace_back(i - 1, j);
                continue;
            }
        }


        // move down
        if (j+1<m-4 && j+1>=4){
            if(img.at<uchar>(j+1,i)==0
               && (index[i][j+1] == 0)){
                stack.emplace_back(i, j+1);
                continue;
            }
        }


        // move up
        if (j-1>=4 && j-1<m-4){
            if(img.at<uchar>(j-1,i)==0
               && (index[i][j-1] == 0)){
                stack.emplace_back(i, j-1);
                continue;
            }
        }


        stack.pop_back();
    }
    for(k=0; k!=1000; ++k){
        if(x[k]!=n+1 && y[k]!=m+1){
            x_center += x[k];
            y_center += y[k];
            ++count;
        }
    }
    x_center = x_center/count;
    y_center = y_center/count;


    for(k=0; k!=1000; ++k){
        if(x_center == x[k] && y_center == y[k]){
            ++center_inside_cell;
            break;
        }
    }

    if(sz>=20 && center_inside_cell>0){
        for(k=0; k!=1000; ++k){
            if(x[k]!=n+1 && y[k]!=m+1){
                img_res.at<Vec3b>(y[k],x[k]).val[0] = 0;
                img_res.at<Vec3b>(y[k],x[k]).val[1] = 0;
                img_res.at<Vec3b>(y[k],x[k]).val[2] = 255;
            }
        }
    }
    if(center_inside_cell == 0){
        sz=0;
    }
}

void preredraw(Mat img, int startI, int startJ, int **index){
    int n = img.size[1], m = img.size[0], k,l, sz = 0;
    int x[1000], y[1000];
    for(k=0; k!=1000; ++k) {
        x[k] = n+1; y[k] = m+1;
    }
    std::vector<std::pair<int, int> > stack;
    stack.emplace_back(startI, startJ);

    while (!stack.empty()) {
        auto coords = stack.back();
        int i = coords.first, j = coords.second;

        index[i][j] = 1;
        x[sz]=i, y[sz]=j;
        ++sz;

        // move right
        if (i+1<n-4 && i+1>=4){
            if(img.at<uchar>(j,i+1)==0
               && (index[i + 1][j] == 0)){
                stack.emplace_back(i + 1, j);
                continue;
            }
        }


        // move left
        if (i-1>=4 && i-1<n-4){
            if(img.at<uchar>(j,i-1)==0
               && (index[i - 1][j] == 0)){
                stack.emplace_back(i - 1, j);
                continue;
            }
        }


        // move down
        if (j+1<m-4 && j+1>=4){
            if(img.at<uchar>(j+1,i)==0
               && (index[i][j+1] == 0)){
                stack.emplace_back(i, j+1);
                continue;
            }
        }


        // move up
        if (j-1>=4 && j-1<m-4){
            if(img.at<uchar>(j-1,i)==0
               && (index[i][j-1] == 0)){
                stack.emplace_back(i, j-1);
                continue;
            }
        }


        stack.pop_back();
    }

    for(k=0; k!=sz-1; ++k){
        for(l=0; l!=sz-k-1; ++l){
            if(x[l]>x[l+1]){
                int xx = x[l+1];
                x[l+1] = x[l];
                x[l] = xx;
                int yy = y[l+1];
                y[l+1] = y[l];
                y[l] = yy;
            }
            else if(x[l]==x[l+1]){
                if(y[l]>y[l+1]){
                    int yy = y[l+1];
                    y[l+1] = y[l];
                    y[l] = yy;
                }
            }
        }
    }
    int x_min = x[0], x_max = x[sz-1];
    for(k=1; k!= sz; ++k){
        if(x[k-1]==x[k]){
            if(y[k-1]<y[k]-1){
                for(int l=y[k-1]+1; l!=y[k]; ++l){
                    img.at<uchar>(l,x[k]) = 0;
                }
            }
        }
    }

    for(k=0; k!=sz-1; ++k){
        for(l=0; l!=sz-k-1; ++l){
            if(y[l]>y[l+1]){
                int xx = x[l+1];
                x[l+1] = x[l];
                x[l] = xx;
                int yy = y[l+1];
                y[l+1] = y[l];
                y[l] = yy;
            }
            else if(y[l]==y[l+1]){
                if(x[l]>x[l+1]){
                    int xx = x[l+1];
                    x[l+1] = x[l];
                    x[l] = xx;
                }
            }
        }
    }
    int y_min = y[0], y_max = y[sz-1];
    for(k=1; k!= sz; ++k){
        if(y[k-1]==y[k]){
            if(x[k-1]<x[k]-1){
                for(int l=x[k-1]+1; l!=x[k]; ++l){
                    img.at<uchar>(y[k],l) = 0;
                }
            }
        }
    }
}
