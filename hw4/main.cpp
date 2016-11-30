// This program detects abondonment objects on the videos provided. Please,
// read report.pdf for more information


#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/video/video.hpp>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>

using namespace cv;
using std::vector;
using std::string;
using std::stringstream;
using std::ifstream;
using std::cerr;
using std::cout;
using std::list;

// Comment next line if you want to run program without visualization
#define VISUALIZATION 1

// Algorithm detects abandonment object as objects which bounding rectangles
// are stay unchanged during MIN_FRAMES frames of video. MAX_SIMILAR_DISTANCE
// is the maximum deviation from the bounding rectangle appeared in the first 
// frame. Deviations may take place because of e.g. changing lighting.
const unsigned int MAX_SIMILAR_DISTANCE = 10;
const unsigned int MIN_FRAMES = 40;
// To remove noise from the foreground we use erosion with the EROSION_SIZE radius
const unsigned int EROSION_SIZE = 2; 
// To connect somehow disconnected parts of one object we use dilation with the
// DILATION_DIZE radius
const unsigned int DILATION_SIZE = 20; 

// structure to store once appeared object information.
struct AccumulatedObject {
	AccumulatedObject(unsigned int appear_frame_,
	    unsigned int frames_count_, unsigned int last_frame_,
	    Rect bounding_rectangle_):
		appear_frame (appear_frame_),
		frames_count (frames_count_),
		last_frame (last_frame_),
		bounding_rectangle (bounding_rectangle_) {}

	unsigned int appear_frame; // first appearence frame
	unsigned int frames_count; // count of continious appearence frames
	unsigned int last_frame;   // last frame in which the object appears
	Rect bounding_rectangle;   // bounding rectangle of object
};

// Main function processing the video file. See report.pdf for the algoright details
bool ProcessVideo(string filename, vector<AccumulatedObject> *paccum);
// Reads inpt sample from the input file
bool ReadTestFile(string filename, vector<string> *ptest_files);
// Tests if two rectangles are similar (See the description
// of MAX_SIMILAR_DISTANCE and report.pdf for detatils)
bool AreAlmostSimilar(Rect& bounding_rectangle1, Rect& bounding_rectangle2);
// Visualize current state of processing with all intermediate steps
void VisualizeVideoProcessing(const Mat& frame, const Mat& foreground_mask_mog,
    const Mat& eroded, const Mat& dilated, VideoCapture& capture,
    const vector<Rect>& bounding_rectangles, 
    const list<AccumulatedObject>& objects_accumulator,
    const vector<AccumulatedObject>& found_objects); 

int main(int argc, char* argv[])
{
	vector<string> test_files;
	if (!ReadTestFile("test_sample.txt", &test_files)) {
		cerr << "Cannot read sample from file!\n";
		return -1;
	}

	for (string& filename : test_files) {
		vector<AccumulatedObject> found_objects;
		if (!ProcessVideo(filename, &found_objects)) {
			cerr << "Error opening video from test sample";
			return -1;
		}
		cout << filename << ": ";
		for (const AccumulatedObject& obj : found_objects) {
			cout << "rectangle: (" <<
			    obj.bounding_rectangle.x << ", " << 
			    obj.bounding_rectangle.y << ", " <<
			    obj.bounding_rectangle.width << ", " << 
			    obj.bounding_rectangle.height << ") - " <<
			    "timespan: (" << obj.appear_frame << ", " <<
			    obj.last_frame << ")";
		}
		cout << std::endl;
	}


	return 0;
}


bool ReadTestFile(string filename, vector<string> *ptest_files) {
	assert(ptest_files);
	ptest_files->clear();
	ifstream fin(filename.c_str()); 
	if (!fin.is_open())
		return false;

	while (!fin.eof()) {
		string tmp_file_name;
		fin >> tmp_file_name;
		if (tmp_file_name.length() == 0)
			return true;

		ptest_files->push_back(tmp_file_name);
	}

	return true;
}

bool AreAlmostSimilar(Rect& bounding_rectangle1, Rect& bounding_rectangle2) {
	if (abs(bounding_rectangle1.x - bounding_rectangle2.x) <
	    MAX_SIMILAR_DISTANCE &&
	    abs(bounding_rectangle1.y - bounding_rectangle2.y) <
	    MAX_SIMILAR_DISTANCE &&
	    abs(bounding_rectangle1.width - bounding_rectangle2.width) <
	    MAX_SIMILAR_DISTANCE &&
	    abs(bounding_rectangle1.height - bounding_rectangle2.height) <
	     MAX_SIMILAR_DISTANCE) 
		return true;
	return false;
}

void VisualizeVideoProcessing(const Mat& frame, const Mat& foreground_mask_mog,
    const Mat& eroded, const Mat& dilated, VideoCapture& capture,
    const vector<Rect>& bounding_rectangles, 
    const list<AccumulatedObject>& objects_accumulator,
    const vector<AccumulatedObject>& found_objects) {
	Mat tmp_frame = frame.clone();

	static bool is_first_call = true;
	if (is_first_call) {
		namedWindow("Frame");
		namedWindow("FG Mask MOG");
		namedWindow("FG Mask MOG eroded");
		namedWindow("FG Mask MOG dilated");
	}
	is_first_call = false;

	stringstream ss;
	rectangle(tmp_frame, cv::Point(10, 2), cv::Point(100,20), 
	    cv::Scalar(255,255,255), -1);
	ss << capture.get(CV_CAP_PROP_POS_FRAMES);
	string frameNumberString = ss.str();
	putText(tmp_frame, frameNumberString.c_str(), cv::Point(15, 15), 
	    FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));

	for (const Rect& bounding_rectangle : bounding_rectangles)
		rectangle(tmp_frame, bounding_rectangle.tl(), 
		    bounding_rectangle.br(), Scalar(0, 0, 255), 2, 8, 0);
	for (AccumulatedObject accum : objects_accumulator) {
		unsigned char luminance = 255;
		if (accum.frames_count < MIN_FRAMES)
			luminance = 255 * accum.frames_count / MIN_FRAMES;
		rectangle(tmp_frame, accum.bounding_rectangle.tl(),
		    accum.bounding_rectangle.br(), Scalar(0, luminance, 0), 2, 
		    8, 0);
	}
	for (AccumulatedObject accum : found_objects) 
		rectangle(tmp_frame, accum.bounding_rectangle.tl(),
		    accum.bounding_rectangle.br(), Scalar(255, 0, 0), 2, 8, 0);


	imshow("Frame", tmp_frame);
	imshow("FG Mask MOG", foreground_mask_mog);
	imshow("FG Mask MOG eroded", eroded);
	imshow("FG Mask MOG dilated", dilated);

	waitKey(30);
}

bool ProcessVideo(string filename, vector<AccumulatedObject> *pfound_objects) {
	assert(pfound_objects);
	pfound_objects->clear();


	BackgroundSubtractorMOG *pmog = new BackgroundSubtractorMOG(); 
	VideoCapture capture(filename);
	if(!capture.isOpened())
		return false;

	Mat frame, foreground_mask_mog;
	list<AccumulatedObject> objects_accumulator;
	for (unsigned int frame_num = 0; capture.read(frame); frame_num++) {
		//update the background model
		pmog->operator()(frame, foreground_mask_mog);

		// erode/dilate
		Mat eroded, dilated;
		Mat erosion_element = getStructuringElement(MORPH_ELLIPSE, 
		    Size(2 * EROSION_SIZE + 1, 2 * EROSION_SIZE + 1),
		    Point(EROSION_SIZE, EROSION_SIZE));
		erode(foreground_mask_mog, eroded, erosion_element);

		Mat dilation_element = getStructuringElement(MORPH_ELLIPSE, 
		    Size(2 * DILATION_SIZE + 1, 2 * DILATION_SIZE + 1),
		    Point(DILATION_SIZE, DILATION_SIZE));
		dilate(eroded, dilated, dilation_element);

		// find contours
		Mat tmp_dilated = dilated.clone();
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(tmp_dilated, contours, hierarchy, CV_RETR_EXTERNAL, 
		    CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		// find bounding rectangles
		vector<vector<Point>> contours_poly(contours.size());
		vector<Rect> bounding_rectangles(contours.size());

		unsigned int objects_count = 0;
		for (unsigned int i = 0; i < contours.size(); i++) {
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3,
			    true);
			Rect tmp_bounding_rectangle = 
			    boundingRect(Mat(contours_poly[i]));
			bounding_rectangles[objects_count++] = 
			    tmp_bounding_rectangle;
		}
		bounding_rectangles.resize(objects_count);

		// for each accumulated rectangle check if it appears in current frame
		for (Rect& bounding_rectangle : bounding_rectangles) {
			bool is_new_object = true;
			for(AccumulatedObject& accum : objects_accumulator) 
				// if yes - update accumulator
				if (AreAlmostSimilar(bounding_rectangle, 
				    accum.bounding_rectangle)) {
					accum.frames_count++;
					accum.last_frame = frame_num;
					is_new_object = false;
					break;
				}
			// if no - it's new object. Create new accumulator for it
			if (is_new_object)
				objects_accumulator.emplace_back(frame_num, 1,
				    frame_num, bounding_rectangle);
		}

		// delete all accumulated objects which are eliminated in this frame
		for (auto iaccum = objects_accumulator.begin(); 
		    iaccum != objects_accumulator.end();) 
			if (iaccum->last_frame != frame_num) {
				// if it has been appeared in more than MIN_FRAMES
				// frames - it's stable object - add it to found objects
				if (iaccum->frames_count >= MIN_FRAMES)
					pfound_objects->push_back(*iaccum);
				iaccum = objects_accumulator.erase(iaccum);
			}
			else
				iaccum++;

#ifdef VISUALIZATION
		VisualizeVideoProcessing(frame, foreground_mask_mog, eroded,
		    dilated, capture, bounding_rectangles, objects_accumulator,
		    *pfound_objects); 
#endif
	}

	capture.release();
	delete pmog;

#ifdef VISUALIZATION
	destroyAllWindows();
#endif
	return true;
}

