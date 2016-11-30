#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>

using namespace cv;
using std::string;
using std::ifstream;
using std::ofstream;

class SpoonsCounter {
private:
	size_t CountWeight(Mat& img) {
		size_t ret = 0;
		for (size_t i = 0; i < img.rows; ++i)
			for (size_t j = 0; j < img.cols; ++j) 
				if (img.at<Vec3b>(i, j)[2] > 
				    0.5*img.at<Vec3b>(i,j)[0] +
				    img.at<Vec3b>(i,j)[1])
					ret++;
		return ret;
	}

	size_t barrier01_;
	size_t barrier12_;

public: 
	SpoonsCounter() : barrier01_ (0), barrier12_ (0) {}
	~SpoonsCounter() {};

	bool Train(string train_file) {
		size_t sum_weight[3] = {0};
		size_t count[3] = {0};
	
		ifstream fin(train_file.c_str()); 
		if (!fin.is_open())
			return false;
		while (!fin.eof()) {
			int cnt = -1;
			string file;
			fin >> cnt >> file;
			if (cnt == -1)
				break;
			
			Mat img = imread(file.c_str(), CV_LOAD_IMAGE_COLOR);
			if (!img.data)
				return false;
			sum_weight[cnt] += CountWeight(img);
			count[cnt]++;
		}
		
		if (count[0] == 0 || count[1] == 0 || count[2] == 0)
			return false;
		
		barrier01_ = (sum_weight[0] / count[0] + sum_weight[1] / count[1]) / 2;
		barrier12_ = (sum_weight[1] / count[1] + sum_weight[2] / count[2]) / 2;
		return true;
	}

	bool Test(string test_file, string output_file) {
		ifstream fin(test_file.c_str()); 
		if (!fin.is_open())
			return false;
		ofstream fout(output_file.c_str()); 
		if (!fout.is_open())
			return false;
		while (!fin.eof()) {
			string file;
			fin >> file;
			if (file.length() == 0)
				return true;

			Mat img = imread(file.c_str(), CV_LOAD_IMAGE_COLOR);
			if (!img.data)
				return false;

			size_t weight = CountWeight(img);
			size_t spoons_cnt = 0;
			if (weight >= barrier01_)
				spoons_cnt = 1;
			if (weight >= barrier12_)
				spoons_cnt = 2;
			fout << spoons_cnt << std::endl;
		}
		return true;
	}

};

int main(int argc, char** argv) {
	SpoonsCounter counter;
	if (!counter.Train("train") || !counter.Test("test", "test_res"))
		return -1;
		
	return 0;
}
