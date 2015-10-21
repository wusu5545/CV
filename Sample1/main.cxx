// main.cpp : Defines the entry point for the console application.

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char* argv[])
{
		cv::Mat img = cv::imread("lena.jpg");

		cv::namedWindow("Lena", 1);
		cv::imshow("Lena", img);

		cv::waitKey();
       
        return 0;
}