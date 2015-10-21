#include<opencv/cv.h>
#include<opencv/cxcore.h>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc,char * argv[])
{
    if (argv[1] == NULL || argv[2] == NULL){
	fprintf(stderr,"Usage: %s [filename1] [filename2]\n",argv[0]);
	return -1;
    }
    
    //read image;
    Mat img1 = imread(argv[1],CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(argv[2],CV_LOAD_IMAGE_COLOR);
    
    if (!img1.data || !img2.data){
	printf("Error: No image data\n");
	return -1;
    }
    
    //split image1
    vector <Mat> planes1;
    split(img1,planes1);
    
    //split image2
    vector <Mat> planes2;
    split(img2,planes2);
    
    //compute difference images
    Mat red_diff;
    Mat green_diff;
    Mat blue_diff;
    
    absdiff(planes1[0],planes2[0],blue_diff);
    absdiff(planes1[1],planes2[1],green_diff);
    absdiff(planes1[2],planes2[2],red_diff);
    
    //normalize on range [0 - 255]
    normalize(blue_diff,blue_diff,0,255,NORM_MINMAX,CV_8UC1);
    normalize(green_diff,green_diff,0,255,NORM_MINMAX,CV_8UC1);
    normalize(red_diff,red_diff,0,255,NORM_MINMAX,CV_8UC1);
    
    //image output
    imwrite("redDiff.jpg",red_diff);
    imwrite("greenDiff.jpg",green_diff);
    imwrite("blueDiff.jpg",blue_diff);
}