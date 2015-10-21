#include<opencv/cv.h>
#include<opencv/cxcore.h>
#include<opencv2/highgui/highgui.hpp>
#include<iostream>
#include<fstream>
#include<sstream>

using namespace std;
using namespace cv;

int main(int argc,char* argv[])
{
  if (argv[1]==NULL){
    fprintf(stderr,"Usage: %s [filename]\n",argv[0]);
    return -1;
  }
  
  //read image path
  string path = argv[1];
  
  //initialize image parameters
  int rows = 0;
  int cols = 0;
  int type = 0;
  
  ifstream myfile (path, ios::binary);
  string line;
  if (myfile.is_open()){
    ///read image type
    getline(myfile,line);
    if (line.compare("P5") == 0)
      type = CV_8UC1;
    else if (line.compare("P6") == 0)
      type = CV_8UC3;
    else
      return 1;
    
    //skip all comments
    getline(myfile,line);
    while (line.at(0) == '#')
      getline(myfile,line);
    
    //read dimension
    string colStr = line.substr(0,line.find(' '));
    string rowStr = line.substr(line.find(' '),line.size()-line.find(' '));
    int cols = atoi(colStr.c_str());
    int rows = atoi(rowStr.c_str());
    
    //read max value
    getline(myfile,line);
    int maxVal = atoi(line.c_str());
    
    Mat imageGray(rows,cols,CV_8UC1);
    Mat imageRGB(rows,cols,CV_8UC3);
    
    //read all data and set matrix values
    for (int row = 0;row<rows; ++row){
      for (int col = 0;col<cols; ++col){
	if (type == CV_8UC1)//read only 1 byte if grayscale
	{
	  char intensity;
	  myfile.get(intensity);
	  //invert the gray value
	  unsigned char newIntensity = maxVal - (unsigned char)intensity;
	  imageGray.at<uchar>(row,col) = newIntensity;
	}
	else //read 3 bytes if color image
	{
	  char r,g,b;
	  myfile.get(r);
	  myfile.get(g);
	  myfile.get(b);
	  int r1 = (uchar)r;
	  int g1 = (uchar)g;
	  int b1 = (uchar)b;
	  //switch the values
	  imageRGB.at<Vec3b>(row,col)[0] = r1;
	  imageRGB.at<Vec3b>(row,col)[1] = g1;
	  imageRGB.at<Vec3b>(row,col)[2] = b1;
	  //average the values
	  imageGray.at<uchar>(row,col) = (uchar)((r1 + g1 + b1) / 3.0);
	}
      }
    }
    
    if (type == CV_8UC1)
    {
      imwrite("lenaNew.pgm",imageGray);
    }
    else
    {
      normalize(imageGray,imageGray,0,255,NORM_MINMAX);
      imwrite("lenaNew.pgm",imageGray);
      imwrite("lenaNew.ppm",imageRGB);
    }
    
    myfile.close();
  }
}