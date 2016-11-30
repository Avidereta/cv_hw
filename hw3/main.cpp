#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <vector>
#include <set>
#include <assert.h>

using std::ifstream;
using std::ofstream;
using namespace cv;

void SignPicturePreprocessing(Mat *psign_picture) {
	cvtColor(*psign_picture, *psign_picture, CV_BGR2GRAY);
	GaussianBlur(*psign_picture, *psign_picture, Size(5, 5), 0);
	Canny(*psign_picture, *psign_picture, 60, 100);
}


void ShowPicture(Mat img, float score = 0) {
	static int i = 0;
	char win_name[100] = {0};
	sprintf(win_name, "%d %f", i++, score);

	namedWindow(win_name, 0);
	imshow(win_name, img);
}

bool ProcessSign(Mat unknown_sign, const vector<Mat> &known_signs,
    const vector<string> &sign_names, vector<string> *presults) {
	assert(presults);
	Mat unknown_sign_tmp = unknown_sign;

	const int MAX_MATCHES = 10, PAD_X = 1, PAD_Y = 1, SCALES = 10;
	const float TEMPL_SCALE = 1.0, MIN_MATCH_DISTANCE = 0.1, MIN_SCALE = 0.9,
		    MAX_SCALE = 1.3, ORIENTATION_WEIGHT = 0.9, TRUNCATE = 1000;

	SignPicturePreprocessing(&unknown_sign_tmp);
//	ShowPicture(unknown_sign_tmp);

	vector<Point> best_points;
	float best_score = 1 << 30; // some very big number
	size_t best_idx = 0;	

	for (size_t i = 0; i < known_signs.size(); ++i) {
		Mat scaled_candidate;
		vector<vector<Point>> results;
		vector<float> costs;
		// recise sign to the size of candidate and extract edges
		resize(known_signs[i], scaled_candidate,
		    unknown_sign_tmp.size());
		SignPicturePreprocessing(&scaled_candidate);

		int found = chamerMatching(unknown_sign_tmp, scaled_candidate,
		    results, costs, TEMPL_SCALE, MAX_MATCHES, 
		    MIN_MATCH_DISTANCE, PAD_X, PAD_Y, SCALES, MIN_SCALE, 
		    MAX_SCALE, ORIENTATION_WEIGHT, TRUNCATE);
		if (found == -1)
			continue;
		if (costs[found] >= best_score) 
			continue;
		
		best_score = costs[found];		
		best_points = results[found];
		best_idx = i;
	}

	presults->push_back(sign_names[best_idx]);

	Mat tmp_img = unknown_sign;
	for (Point &pt : best_points) 
		if (pt.inside(Rect(0, 0, tmp_img.cols, tmp_img.rows)))
			tmp_img.at<Vec3b>(pt) = Vec3b(0, 255, 0);
	ShowPicture(tmp_img, best_score);

	return true;
}


bool ProcessSignComposite(Mat &sign_composite, const vector<Mat> &known_signs,
    const vector<string> &sign_names, vector<string> *presults) {
	assert(presults);
	Mat tmp_sign_composite;
	cvtColor(sign_composite, tmp_sign_composite, CV_BGR2GRAY);

//	namedWindow(window_name, CV_WINDOW_AUTOSIZE );

	threshold(tmp_sign_composite, tmp_sign_composite, 235,
	    255, THRESH_TOZERO_INV);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(tmp_sign_composite, contours, hierarchy, CV_RETR_EXTERNAL, 
	    CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<vector<Point>> contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());

	int signs_count = 0;
	for (int i = 0; i < contours.size(); i++) {
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		Rect tmp_rect = boundingRect(Mat(contours_poly[i]));
		if (tmp_rect.area() > 1800) 
			boundRect[signs_count++] = tmp_rect;
	}

	boundRect.resize(signs_count);
	for (Rect &rect : boundRect) 
		if (!ProcessSign(sign_composite(rect), known_signs, sign_names, presults))
		    return false;

//	Mat tmp = src.clone();
//	for( int i = 0; i< contours.size(); i++ )
//	       drawContours( tmp, contours_poly, i, Scalar(0, 255, 0), 1, 8, vector<Vec4i>(), 0, Point() );
//	for( int i = 0; i< signs_count; i++ )
//	       rectangle( tmp, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 255), 2, 8, 0 );

//	imshow(window_name, tmp);
	
	return true;

}

bool ReadTestFile(const string &file_name, vector<Mat> *ppictures, 
    vector<string> *psign_names) {
	assert(ppictures);
	assert(psign_names);
	ppictures->clear();
	psign_names->clear();

	ifstream fin(file_name.c_str()); 
	if (!fin.is_open())
		return false;

	while (!fin.eof()) {
		string tmp_file_name;
		fin >> tmp_file_name;
		if (tmp_file_name.length() == 0)
			return true;

		Mat img = imread(tmp_file_name.c_str(), CV_LOAD_IMAGE_COLOR);
		if (!img.data)
			return false;

		ppictures->push_back(img);
		size_t sign_candidates_count = 0;
		fin >> sign_candidates_count;
		for (size_t i = 0; i < sign_candidates_count; i++) {
			string tmp_sign_name;
			fin >> tmp_sign_name;
			psign_names->push_back(tmp_sign_name);
		}

	}
	return true;
}

bool ReadLearningPictures(const string &file_name, vector<Mat> *psign_pictures,
    vector<string> *psign_names) {
	assert(psign_names);
	assert(psign_pictures);
	psign_names->clear();
	psign_pictures->clear();

	ifstream fin(file_name.c_str()); 
	if (!fin.is_open())
		return false;

	while (!fin.eof()) {
		string tmp_file_name;
		fin >> tmp_file_name;
		if (tmp_file_name.length() == 0)
			return true;
		Mat img = imread(tmp_file_name.c_str(), CV_LOAD_IMAGE_COLOR);
		img = img(Rect(5, 0, img.size().width - 10, img.size().height));
		if (!img.data)
			return false;
		psign_pictures->push_back(img);

		string sign_name = tmp_file_name.substr(0,
		    tmp_file_name.size() - 4);
		psign_names->push_back(sign_name);
	}
	return true;
}

void ComputePerformanceMetrics(const vector<string> &answers,
    const vector<string> &results, const vector<string> &known_signs) {
	int all_classified = 0;
	int not_all_classified = 0;
	int fp = 0, fn = 0, tp = 0, tn = 0;

	for(size_t i = 0; i < known_signs.size(); i++) 
		for (size_t j = 0; j < answers.size(); j++) 
			if (answers[j] == known_signs[i]) {
				all_classified++;
				if (results[j] == answers[j]) 
				    tp++;
				else
				    fn++;
			}
			else 
				not_all_classified++;

	fp = all_classified - tp;
	tn = not_all_classified - fn;

	float precision = (float) tp / (float) (tp + fp);
	float recall = (float) tp / (float) (tp + fn);
	float accuracy = (float) (tp + tn) / (float) (tp + fp + tn + fn);

	printf("Average quality metrics: accuracy = %f, precision = %f,"
	    "recall = %f\n", accuracy, precision, recall);
}

void WaitUntilExit() {
	while (true) {
		int c = waitKey( 20 );
		if ((char) c == 27) 
			break; 
	}
}

int main(int argc, char** argv) {
	vector<Mat> known_signs;
	vector<string> sign_names;
	ReadLearningPictures("learning_signs.txt", &known_signs, 
	    &sign_names);

	vector<Mat> sign_composites;
	vector<string> correct_names;
	vector<string> predicted_names;
	ReadTestFile("test_sample.txt", &sign_composites, &correct_names);

	for (Mat& sign_composite : sign_composites) 
		ProcessSignComposite(sign_composite, known_signs, sign_names,
		    &predicted_names); 
	
	ComputePerformanceMetrics(correct_names, predicted_names, sign_names);

	WaitUntilExit();
}

