#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <vector>
#include <set>

using std::string;
using std::multiset;
using std::vector;
using std::ifstream;
using std::ofstream;
using namespace cv;

const size_t BOTTLES_COUNT = 5;
const int MAX_LINE_WIDTH = 5;
const int MAX_LABEL_MARGIN = 20;
const int MIN_LABEL_MARGIN = 1;
const int DISTANCE_EPS = 4;
const float ANGLE_EPS = 3;


struct test_result_t {
        bool operator ==(const test_result_t& tr) {
		return is_labeled == tr.is_labeled && 
		    is_centered == tr.is_centered && 
		    is_straight == tr.is_straight;
	}
	bool is_labeled;
	bool is_centered;
	bool is_straight;
};


// structure to store for corner points of some rectangle object
struct object_corners_t {
	object_corners_t():
	    top_left(Point(0,0)),
	    bottom_left(Point(0,0)),
	    top_right(Point(0,0)),
	    bottom_right(Point(0,0)) {}

	object_corners_t(Point _tl, Point _bl, Point _tr, Point _br):
	    top_left(_tl),
	    bottom_left(_bl),
	    top_right(_tr),
	    bottom_right(_br) {}

	Point top_left;
	Point bottom_left;
	Point top_right;
	Point bottom_right;
};

bool ComparePointsByXCoord(const Point& p1, const Point& p2) {
	return p1.x < p2.x;
}

bool ComparePointsByYCoord(const Point& p1, const Point& p2) {
	return p1.y < p2.y;
}

typedef bool (*ComparePointsT) (const Point& p1, const Point& p2);

// This function performs Canny algorithm to detect edges aand next uses
// findContours to find all contours on the picture
void FindContourPoints(const Mat& mat,
    multiset<Point, ComparePointsT>* pcontour_points) {
	assert(pcontour_points);

	Mat tmp_img;
	cvtColor(mat, tmp_img, CV_BGR2GRAY);
	blur(tmp_img, tmp_img, Size(4,4));
	Canny(tmp_img, tmp_img, 60, 100, 3);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(tmp_img, contours, hierarchy, CV_RETR_TREE, 
	    CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	for (int i = 0; i < contours.size(); i++) 
		for (Point p : contours[i]) 
			pcontour_points->insert(p);

}

object_corners_t FindTubeCorners(
    multiset<Point, ComparePointsT>* pcontour_points) {

	multiset<Point, ComparePointsT> left_tube_side(ComparePointsByYCoord);
	multiset<Point, ComparePointsT> right_tube_side(ComparePointsByYCoord);

	// Find left tube side as top left points
	int min_x = pcontour_points->begin()->x;
	while (pcontour_points->begin()->x - min_x < MAX_LINE_WIDTH) {		
		left_tube_side.insert(*(pcontour_points->begin()));
		pcontour_points->erase(pcontour_points->begin());
	}

	// Find right tube side as top left points
	int max_x = (--pcontour_points->end())->x;
	while (max_x - (--pcontour_points->end())->x < MAX_LINE_WIDTH) {		
		right_tube_side.insert(*(--pcontour_points->end()));
		pcontour_points->erase(--pcontour_points->end());
	} 

	// As multisets sort points by the Y coord, the top point is set::begin 
	// bottom point is set::end()
	return object_corners_t(
	    *left_tube_side.begin(),  *(--left_tube_side.end()),
	    *right_tube_side.begin(), *(--right_tube_side.end()));
}

object_corners_t FindLableCorners(
    const multiset<Point, ComparePointsT> &contour_points, 
    const object_corners_t &tube) {
	// find label using approach that distance to lable from each side is
	// bigger than MIN_LABEL_MARGIN and smaller than MAX_LABEL_MARGIN

	object_corners_t lable;

	// Find top left
	for (auto i = contour_points.begin(); i != contour_points.end(); ++i) {
		if ((i->y - tube.top_left.y < MIN_LABEL_MARGIN) ||
		    (i->y - tube.top_left.y > MAX_LABEL_MARGIN))
			continue;
		if (i->x - tube.top_left.x < MIN_LABEL_MARGIN) 
			continue;
		if (i->x - tube.top_left.x > MAX_LABEL_MARGIN)
			break;
	
		lable.top_left = *i;	
		break;

	}

	// Find bottom left
	for (auto i = contour_points.begin(); i != contour_points.end(); ++i) {
		if ((tube.bottom_left.y - i->y < MIN_LABEL_MARGIN) ||
		    (tube.bottom_left.y - i->y > MAX_LABEL_MARGIN))
			continue;
		if (i->x - tube.bottom_left.x < MIN_LABEL_MARGIN) 
			continue;
		if (i->x - tube.bottom_left.x > MAX_LABEL_MARGIN)
			break;
		lable.bottom_left = *i;	
		break;
	}

	// Find top right 
	for (auto i = contour_points.rbegin(); i != contour_points.rend(); ++i){
		if ((i->y - tube.top_right.y < MIN_LABEL_MARGIN) ||
		    (i->y - tube.top_right.y > MAX_LABEL_MARGIN))
			continue;
		if (tube.top_right.x - i->x < MIN_LABEL_MARGIN) 
			continue;
		if (tube.top_right.x - i->x > MAX_LABEL_MARGIN)
			break;

		lable.top_right = *i;	
		break;
	}

	// Find bottom right 
	for (auto i = contour_points.rbegin(); i != contour_points.rend(); ++i){
		if ((tube.bottom_right.y - i->y < MIN_LABEL_MARGIN) ||
		    (tube.bottom_right.y - i->y > MAX_LABEL_MARGIN))
			continue;
		if (tube.bottom_right.x - i->x < MIN_LABEL_MARGIN) 
			continue;
		if (tube.bottom_right.x - i->x > MAX_LABEL_MARGIN)
			break;
	
		lable.bottom_right = *i;	
		break;
	}

	return lable;
}

// simple draw corner points to illustrate an algorithm
void DrawFoundPoints(Mat mat, const object_corners_t &tube, 
    const object_corners_t &lable) {
	static int i = 0;
	circle(mat, tube.top_left, 1, Scalar(0,255,0));
	circle(mat, tube.bottom_left, 1, Scalar(0,255,0));
	circle(mat, lable.top_left, 1, Scalar(0,255,0));
	circle(mat, lable.bottom_left, 1, Scalar(0,255,0));
	circle(mat, tube.top_right, 1, Scalar(0,255,0));
	circle(mat, tube.bottom_right, 1, Scalar(0,255,0));
	circle(mat, lable.top_right, 1, Scalar(0,255,0));
	circle(mat, lable.bottom_right, 1, Scalar(0,255,0));

	char win_name[100] = {0};
	sprintf(win_name, "%d", i++);
	namedWindow(win_name, 0);
	imshow(win_name, mat);
}

test_result_t TestSingleBottle(Mat mat) {
	test_result_t result = {};
	result.is_labeled = result.is_straight = result.is_centered = false;
	
	multiset<Point, ComparePointsT> contour_points(ComparePointsByXCoord);
	FindContourPoints(mat, &contour_points);
	object_corners_t tube = FindTubeCorners(&contour_points);
	object_corners_t lable = FindLableCorners(contour_points, tube);
	//DrawFoundPoints(mat, tube, lable);

	// lable exists if at least one corner point found
	if ((lable.top_left.x    != 0 && lable.top_left.y     != 0) ||
	    (lable.bottom_left.x != 0 && lable.bottom_left.y  != 0) ||
	    (lable.top_right.x   != 0 && lable.top_right.y    != 0) ||
	    (lable.bottom_left.x != 0 && lable.bottom_right.y != 0)) 
		result.is_labeled = true;


	// test if label is straight

	// firsty if it has only left side
	if ((lable.top_left.x     != 0 && lable.top_left.y      != 0) && 
	    (lable.bottom_left.x  != 0 && lable.bottom_left.y   != 0) &&
	    (lable.top_right.x    == 0 && lable.top_right.y    == 0) &&
	    (lable.bottom_right.x == 0 && lable.bottom_right.y == 0)) {
		float lable_angle = (float)
		    (lable.top_left.x - lable.bottom_left.x) /
		    (lable.top_left.y - lable.bottom_left.y);
		float tube_angle = (float)
		    (tube.top_left.x - tube.bottom_left.x) /
		    (tube.top_left.y - tube.bottom_left.y);
		if (fabs(lable_angle - tube_angle) < ANGLE_EPS) 
			result.is_straight = true;
	}
	// secondly if it has only right side
	if ((lable.top_right.x    != 0 && lable.top_right.y     != 0) && 
	    (lable.bottom_right.x != 0 && lable.bottom_right.y  != 0) &&
	    (lable.top_left.x     == 0 && lable.top_left.y      == 0) &&
	    (lable.bottom_left.x  == 0 && lable.bottom_left.y   == 0)) {
		float lable_angle = (float)
		    (lable.top_right.x - lable.bottom_right.x) /
		    (lable.top_right.y - lable.bottom_right.y);
		float tube_angle = (float)
		    (tube.top_right.x - tube.bottom_right.x) /
		    (tube.top_right.y - tube.bottom_right.y);
		if (fabs(lable_angle - tube_angle) < ANGLE_EPS) 
			result.is_straight = true;
	}
	// and finally if it has both sides
	if ((lable.top_right.x    != 0 && lable.top_right.y     != 0) && 
	    (lable.bottom_right.x != 0 && lable.bottom_right.y  != 0) &&
	    (lable.top_left.x     != 0 && lable.top_left.y      != 0) &&
	    (lable.bottom_left.x  != 0 && lable.bottom_left.y   != 0)) {
		float left_lable_angle = (float)
		    (lable.top_left.x - lable.bottom_left.x);
		float left_tube_angle = (float)
		    (tube.top_left.x - tube.bottom_left.x);
		float right_lable_angle = (float)
		    (lable.top_right.x - lable.bottom_right.x);
		float right_tube_angle = (float)
		    (tube.top_right.x - tube.bottom_right.x);

		if (fabs(left_lable_angle) < ANGLE_EPS &&
		    fabs(right_lable_angle) < ANGLE_EPS) 
			result.is_straight = true;
		if (abs(lable.top_left.x - tube.top_left.x - 
		    (tube.top_right.x - lable.top_right.x)) < DISTANCE_EPS && 
		    abs(lable.bottom_left.x - tube.bottom_left.x - 
		    (tube.bottom_right.x - lable.bottom_right.x)) <DISTANCE_EPS)  
			result.is_centered = true;

	}

	return result;
}

bool TestImageWithBottles(const string& file_name, 
    vector<test_result_t> *presult) {
	assert(presult);

	Mat img = imread(file_name.c_str(), CV_LOAD_IMAGE_COLOR);
	size_t width = img.cols / BOTTLES_COUNT;
	if (!img.data)
		return false;
	for (size_t i = 0; i < BOTTLES_COUNT; ++i) 
		presult->push_back(
		    TestSingleBottle(img(Rect(i * width, 0, width, img.rows))));
	return true;	
}

void ComputePerformanceMetrics(const vector<test_result_t> &answers,
    const vector<test_result_t> &result) {
	// Macros here because we need to produce the sae calcuations for each 
	// struct field
#define TEST_PROP(prop)\
	int fp_##prop = 0;\
	int fn_##prop = 0;\
	int tp_##prop = 0;\
	int tn_##prop = 0;\
\
	for (int i = 0; i < answers.size(); i++)\
		/* first process "positive samples" */ \
		if (answers[i].prop) \
			if (result[i].prop)\
				tp_##prop++;\
			else \
				fn_##prop++;\
		/* next process "negative samples" */ \
		else \
			if (!result[i].prop)\
				tn_##prop++;\
			else \
				fp_##prop++;

	TEST_PROP(is_labeled);
	TEST_PROP(is_centered);
	TEST_PROP(is_straight);
#undef test_prop
        float accuracy = (float) (tp_is_labeled + tn_is_labeled +
	    tp_is_straight + tn_is_straight + tp_is_centered + tn_is_centered) 
	    / result.size() / 3;
	float precision =  (float) (tp_is_labeled + tp_is_straight + 
	    tp_is_centered) / (tp_is_labeled + fp_is_labeled +
	    tp_is_straight + fp_is_straight + tp_is_centered + fp_is_centered);
	float recall =  (float) (tp_is_labeled + tp_is_straight + 
	    tp_is_centered) / (tp_is_labeled + fn_is_labeled +
	    tp_is_straight + fn_is_straight + tp_is_centered + fn_is_centered);
	printf("Average quality metrics: accuracy = %f, precision = %f,"
	    "recall = %f", accuracy, precision, recall);
}
 

bool ReadTestFile(const string& file_name, 
   vector<test_result_t> *panswers, vector<string> *ppictures) {
	assert(panswers);
	assert(ppictures);
	ppictures->clear();
	panswers->clear();

	ifstream fin(file_name.c_str()); 
	if (!fin.is_open())
		return false;

	while (!fin.eof()) {
		string tmp_name;
		fin >> tmp_name;
		if (tmp_name.length() == 0)
			return true;
		ppictures->push_back(tmp_name);
		for (int i = 0; i < BOTTLES_COUNT; ++i) {
			test_result_t tmp_ans;
			char c1 = 0, c2 = 0, c3 = 0;
			fin >> c1 >> c2 >> c3;
			if (c1 == 'y')
				tmp_ans.is_labeled = true;
			else {
				if (c1 != 'n')
					return false;
				tmp_ans.is_labeled = false;
			}
			if (c2 == 'y')
				tmp_ans.is_centered = true;
			else {
				if (c2 != 'n')
					return false;
				tmp_ans.is_centered = false;
			}
			if (c3 == 'y')
				tmp_ans.is_straight = true;
			else {
				if (c3 != 'n') 
					return false;
				tmp_ans.is_straight = false;
			}
			panswers->push_back(tmp_ans);
		}
	}
	return true;
}

int main(int argc, char** argv) {
	vector<test_result_t> true_res;
	vector<test_result_t> computed_res;
	vector<string> files;

	if (!ReadTestFile("test.txt", &true_res, &files)) {
		fprintf(stderr, "Invalid test file name or format\n");
		return -1;
	}

	for (string& file_name : files)
		if (!TestImageWithBottles(file_name, &computed_res)) {
			fprintf(stderr, "Invalid image file name in test\n");
			return -1;
		}
	ComputePerformanceMetrics(true_res, computed_res);	
waitKey(0);
		
	return 0;
}

