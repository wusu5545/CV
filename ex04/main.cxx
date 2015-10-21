#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

/* Gauss
*********/
void gauss(Mat1b &src, Mat1b &smooth, int kernelWidth, float stdDev){
	smooth = src.clone();

	Mat1f kernel = Mat::ones(kernelWidth, kernelWidth, CV_32F);
	float kernelsum = 0;
	int a = (int) kernelWidth/2;
	for(int i = -a; i <= a; ++i){
		for(int j = -a; j <= a; ++j){
			kernel(i+a,j+a) = exp(-(i*i + j*j)/(2*stdDev*stdDev));
			kernelsum += kernel(i+a,j+a);
		}
	}

	for(int y = a; y < src.rows - a; ++y){
		for(int x = a; x < src.cols - a; ++x){
			float sum = 0;
			for(int i = -a; i <= a; ++i){
				for(int j = -a; j <= a; ++j){
					sum += src(y+i, x+j) * kernel(a-i, a-j);
				}
			}
			smooth(y,x) = uchar(sum/kernelsum);
		}
	}

	imwrite("Gauss.png", smooth);
}

/* Median
*********/
void median(Mat1b &src, Mat1b &smooth, int kernelWidth){
	smooth = src.clone();
	vector<int> tmp;
	int a = (int) kernelWidth/2;
	for(int y = a; y < src.rows - a; ++y){
		for(int x = a; x < src.cols - a; ++x){
			for(int i = -a; i <= a; ++i){
				for(int j = -a; j <= a; ++j){
					tmp.push_back(src(y+i, x+j));
				}
			}
			sort(tmp.begin(), tmp.end());
			smooth(y,x) = tmp.at((kernelWidth*kernelWidth)/2);
			tmp.clear();
		}
	}
	imwrite("Median.png", smooth);
}

/* Sobel
*******/
void sobel(Mat1b &src, Mat1f &sobelX, Mat1f &sobelY, Mat1f &sobelOrientation, Mat1f &sobelStrength){
	sobelX = src.clone();
	sobelY = src.clone();
	sobelStrength = Mat::zeros(src.rows, src.cols, CV_32F);
	sobelOrientation = Mat::zeros(src.rows, src.cols, CV_32F);

	//filter weights
	float mX[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
	float mY[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
	//create filter masks
	Mat1f kernelX = Mat(3, 3, CV_32F, mX);
	Mat1f kernelY = Mat(3, 3, CV_32F, mY);

	// a = kernelwidth
	//here: kernel width 3 = 2a + 1 => a = 1
	int a = 1;
	for(int y = a; y < src.rows - a; ++y){
		for(int x = a; x < src.cols - a; ++x){
			float sumX = 0;
			float sumY = 0;
			for(int i = -a; i <= a; ++i){
				for(int j = -a; j <= a; ++j){
					sumX += src(y+i, x+j) * kernelX(a-i, a-j);
					sumY += src(y+i, x+j) * kernelY(a-i, a-j);
				}
			}
			sobelX(y,x) = sumX;
			sobelY(y,x) = sumY;
		}
	}

	float tmp = 0;
	for(int y = 0; y < src.rows; ++y){
		for(int x = 0; x < src.cols; ++x){
			sobelStrength(y,x) = sqrt(sobelX(y,x)*sobelX(y,x) + sobelY(y,x)*sobelY(y,x));
			tmp = atan2(sobelY(y,x), sobelX(y,x))*180/3.1415; //atan2 calculates in radians
			if(tmp < 0){
				sobelOrientation(y,x) = tmp + 180;
			}
			else{
				sobelOrientation(y,x) = tmp;
			}
		}
	}
	//careful! the actual values in EdgeStrength_sobel exceed 255! Therefore image just an indication!
	imwrite("SobelX.png", sobelX);
	imwrite("SobelY.png", sobelY);
	imwrite("EdgeStrength_sobel.png", sobelStrength);
	imwrite("EdgeOrientation.png", sobelOrientation);
}

/* Nonmax Edge Enhancement
**************************/
void nonmax(Mat1b &src, Mat1f &sobelOrientation, Mat1f &sobelStrength, Mat1b &sobelOrientation2, Mat1f &enhancedEdges){
	sobelOrientation2 = sobelOrientation.clone();
	enhancedEdges = sobelStrength.clone();

	//assign either 0°, 45°, 90°, or 135° to all edges
	for(int y = 0; y < src.rows; ++y){
		for(int x = 0; x < src.cols; ++x){
			if(sobelOrientation(y,x) < 45/2 || sobelOrientation(y,x) >= 135+45/2){
				sobelOrientation2(y,x) = 0;
			}
			else if(sobelOrientation(y,x) >= 45/2 && sobelOrientation(y,x) < 90-45/2){
				sobelOrientation2(y,x) = 45;
			}

			else if(sobelOrientation(y,x) >= 90-45/2 && sobelOrientation(y,x) < 90+45/2){
				sobelOrientation2(y,x) = 90;
			}
			else{
				sobelOrientation2(y,x) = 135;
			}
		}
	}

	// look in the directions perpendicular to the edge!
	for(int y = 1; y < src.rows - 1; ++y){
		for(int x = 1; x < src.cols - 1; ++x){
			if(sobelOrientation2(y,x) == 0){
				if(sobelStrength(y,x) < sobelStrength(y-1,x) || sobelStrength(y,x) < sobelStrength(y+1,x)){
					enhancedEdges(y,x) = 0;
				}
			}
			if(sobelOrientation2(y,x) == 45){
				if(sobelStrength(y,x) < sobelStrength(y-1,x-1) || sobelStrength(y,x) < sobelStrength(y+1,x+1)){
					enhancedEdges(y,x) = 0;
				}
			}
			if(sobelOrientation2(y,x) == 90){
				if(sobelStrength(y,x) < sobelStrength(y,x-1) || sobelStrength(y,x) < sobelStrength(y,x+1)){
					enhancedEdges(y,x) = 0;
				}
			}
			if(sobelOrientation2(y,x) == 135){
			int el = sobelStrength(y,x);
				if(sobelStrength(y,x) < sobelStrength(y-1,x+1) || sobelStrength(y,x) < sobelStrength(y+1,x-1)){
					enhancedEdges(y,x) = 0;
				}
			}
		}
	}
	imwrite("EdgeOrientation_quantized.png", sobelOrientation2);
	imwrite("EdgeStrength_nonmax.png", enhancedEdges);
}

/* Recursive Tracker
*******************/
// We first come here from a pixel with t_h
// We walk along the edge, and as long as the edge strength stays above t_1 (that is, they are directly connected to a t_h pixel along 
// the edge), they are considered a valid edge

void rec(Mat1b &src, Mat1f &enhancedEdges, Mat1b sobelOrientation2, Mat1b &visited, float t_1, Mat1f &refinedImage, int y, int x){
	if(y > 0 &&  y < src.rows - 1 && x > 0 && x < src.cols - 1){
		if( visited(y,x) == 1 ){
		}
		else{
			if(enhancedEdges(y,x) > t_1){
				visited(y,x) = 1;
				refinedImage(y,x) = enhancedEdges(y,x);

				if(sobelOrientation2(y,x) == 0){
					rec(src, enhancedEdges, sobelOrientation2, visited, t_1, refinedImage, y,x-1);
					rec(src, enhancedEdges, sobelOrientation2, visited, t_1, refinedImage, y,x+1);
				}
				if(sobelOrientation2(y,x) == 45){
					rec(src, enhancedEdges, sobelOrientation2, visited, t_1, refinedImage, y+1,x-1);
					rec(src, enhancedEdges, sobelOrientation2, visited, t_1, refinedImage, y-1,x+1);
				}
				if(sobelOrientation2(y,x) == 90){
					rec(src, enhancedEdges, sobelOrientation2, visited, t_1, refinedImage, y-1,x);
					rec(src, enhancedEdges, sobelOrientation2, visited, t_1, refinedImage, y+1,x);
				}
				if(sobelOrientation2(y,x) == 135){
					rec(src, enhancedEdges, sobelOrientation2, visited, t_1, refinedImage, y-1, x-1);
					rec(src, enhancedEdges, sobelOrientation2, visited, t_1, refinedImage, y+1, x+1);
				}
			}
		}
	}
}

/* Hysteresis Thresholding
**************************/
void hyst(Mat1b &src, Mat1f &enhancedEdges, Mat1b &sobelOrientation2, float t_h, float t_1, Mat1b &visited, Mat1f &refinedImage){
	visited = Mat::zeros(src.rows, src.cols, CV_8UC1);
	refinedImage = Mat::zeros(src.rows, src.cols, CV_8UC1);

	for(int y = 1; y < src.rows-1; ++y){
		for(int x = 1; x < src.cols-1; ++x){
			if( visited(y,x) != 1 ){
				if(enhancedEdges(y,x) > t_h){
					visited(y,x) = 1;
					refinedImage(y,x) = enhancedEdges(y,x);

					if(sobelOrientation2(y,x) == 0){
						rec(src, enhancedEdges, sobelOrientation2, visited, t_1, refinedImage, y,x-1);
						rec(src, enhancedEdges, sobelOrientation2, visited, t_1, refinedImage, y,x+1);
					}
					if(sobelOrientation2(y,x) == 45){
						rec(src, enhancedEdges, sobelOrientation2, visited, t_1, refinedImage, y+1,x-1);
						rec(src, enhancedEdges, sobelOrientation2, visited, t_1, refinedImage, y-1,x+1);
					}
					if(sobelOrientation2(y,x) == 90){
						rec(src, enhancedEdges, sobelOrientation2, visited, t_1, refinedImage, y-1,x);
						rec(src, enhancedEdges, sobelOrientation2, visited, t_1, refinedImage, y+1,x);
					}
					if(sobelOrientation2(y,x) == 135){
						rec(src, enhancedEdges, sobelOrientation2, visited, t_1, refinedImage, y-1, x-1);
						rec(src, enhancedEdges, sobelOrientation2, visited, t_1, refinedImage, y+1, x+1);
					}
					}
			}
		}
	}
	//Mat1b res = Mat(size
	//normalize(refinedImage,refinedImage,0,255);
	imwrite("EdgeStrength_hysteresis.png", refinedImage);
}

/* Error-MsobelStrengthsage
****************/
int error_msg(){
	cout << "Usage: ./OpenCV_Solution4 [Image] [Filter Width (odd)] [Filter Type (Gaussian or Median)] [Standard Deviation if (Gaussian)] ([Threshold t_1] [Threshold t_h > t_1])" << endl;
	return 1;
}

/* Main
*******/
int main(int argc, char** argv)
{

	/* Important VariablsobelStrength and MatricsobelStrength
	********************************/
	Mat1b src;		// Source Image
	int kernelWidth;		// Filter Width
	char* filter;		// Filter Type
	float stdDev;		// Standard Deviation
	float t_1 = 100;	// ThrsobelStrengthhold t_1
	float t_h = 200;	// ThrsobelStrengthhold t_h
	Mat1b smooth;		// Smoothed Image
	Mat1f sobelX;		// Sobel X Matrix
	Mat1f sobelY;		// Sobel Y Matrix
	Mat1f sobelStrength;		// Edge Strength
	Mat1f sobelOrientation;		// Edge Orientation
	Mat1b sobelOrientation2;		// sobelStrength fitting orientation: 0, 45, 90, 135
	Mat1f enhancedEdges;		// enhanced edge image
	Mat1f refinedImage;		// refined Image
	Mat1b visited;		// Matrix of visited Points


	if(argc > 3)
	{

		/* Read in Variables
		********************/

		src = imread(argv[1],0);
		if(src.empty()){
			error_msg();
			return 1;
		}
		cout << "Picture Size: " << "\t" << src.cols << " x " << src.rows << endl;

		kernelWidth = atoi(argv[2]);
		cout << "Filter Size: " << "\t" << kernelWidth << endl;

		filter = argv[3];
		if(strcmp(filter,"Median") != 0 && strcmp(filter,"Gaussian") != 0){
			error_msg();
			return 1;
		}
		cout << "Filter Type: " << "\t" << filter << endl;

		if(strcmp(filter,"Gaussian") == 0){
			stdDev = atof(argv[4]);
			cout << "Standard Dev.: " << "\t" << stdDev << endl;
			if(argc == 5){}
			else if(argc == 7){
				t_1 = atof(argv[5]);
				t_h = atof(argv[6]);

				if(t_1 > t_h){
					error_msg();
					return 1;
				}
				cout << "Threshold t_1: " << "\t" << t_1 << endl;
				cout << "Threshold t_h: " << "\t" << t_h << endl;
			}
			else{
				error_msg();
				return 1;
			}
		}

		if(strcmp(filter,"Median") == 0){
			if(argc == 4){}
			else if(argc == 6){
				t_1 = atof(argv[4]);
				t_h = atof(argv[5]);

				if(t_1 > t_h){
					error_msg();
					return 1;
				}
				cout << "Threshold t_1: " << "\t" << t_1 << endl;
				cout << "Threshold t_h: " << "\t" << t_h << endl;
			}
			else{
				error_msg();
				return 1;
			}
		}

		/* Smoothing
		************/
		std::string filename = std::string(argv[1]);
		std::string base = filename.substr(0,filename.size()-4);
		
		char tmp[512];
		if(strcmp(filter,"Median") == 0){
			median(src, smooth, kernelWidth);
			snprintf(tmp,256,"_Median_w%d_t1%.02f_th%.02f.png",kernelWidth,t_1, t_h);

		}
		else if(strcmp(filter,"Gaussian") == 0){
			gauss(src, smooth, kernelWidth, stdDev);
			snprintf(tmp,256,"_Gauss_w%d_sig%.01f_t1%.02f_th%.02f.png",kernelWidth,stdDev,t_1, t_h);
		}

		filename = base + std::string(tmp);

		/* filtering with Sobel
		**********************/
		sobel(smooth, sobelX, sobelY, sobelOrientation, sobelStrength);

		/* Nonmax Edge Enhancement
		**************************/
		nonmax(src, sobelOrientation, sobelStrength, sobelOrientation2, enhancedEdges);

		/* Hysteresis Thresholding
		**************************/
		hyst(src, enhancedEdges, sobelOrientation2, t_h, t_1, visited, refinedImage);

		
		imwrite(filename,refinedImage);
	}
	else{
		error_msg();
		return 1;
	}
	return 0;
}