// main.cpp : Defines the entry point for the console application.

#include <opencv/cv.h>
#include <opencv/cxcore.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
		Mat img = cv::imread("lena.jpg");
		namedWindow("Lena", 1);
		imshow("Lena", img);

		// -------------------------------------------------------------
		// -------------------    NEW SOURCE CODE    -------------------
		// -------------------------------------------------------------

		Mat img_yuv;
		// convert image to YUV color space.
		// The output image will be allocated automatically
		cvtColor(img, img_yuv, CV_RGB2YCrCb);

		// split the image into separate color planes
		vector<Mat> planes;
		split(img_yuv, planes);

		// another Mat constructor; allocates a matrix of the specified size and type;
		Mat noise(img.size(), CV_8U);

		// fills the matrix with normally distributed random values;
		randn(noise, Scalar::all(128), Scalar::all(20));

		// blur the noise a bit, kernel size is 3x3 and both sigma's are set to 0.5
		GaussianBlur(noise, noise, Size(3, 3), 0.5, 0.5);

		const double brightness_gain = 0;
		const double contrast_gain = 1.7;

		addWeighted(planes[0], contrast_gain, noise, 1, -128 + brightness_gain, planes[0]);

		const double color_scale = 0.5;
		// Scale and add values to plane[1];
		planes[1].convertTo(planes[1], planes[1].type(), color_scale, 128*(1-color_scale));

		// alternative form of convertTo if we know the datatype
		// at compile time ("uchar" here).
		// This expression will not create any temporary arrays
		// and should be almost as fast as the above variant
		planes[2] = Mat_<uchar>(planes[2]*color_scale + 128*(1-color_scale));

		planes[0] = planes[0].mul(planes[0], 1./255);

		// now merge the results back
		merge(planes, img_yuv);
		// and produce the output RGB image
		cvtColor(img_yuv, img, CV_YCrCb2RGB);

		// this is counterpart for cvNamedWindow
		namedWindow("image with grain", CV_WINDOW_AUTOSIZE);

		imshow("image with grain", img);

		// -------------------------------------------------------------
		// -------------------  / NEW SOURCE CODE    -------------------
		// -------------------------------------------------------------



		waitKey();
       
        return 0;
}