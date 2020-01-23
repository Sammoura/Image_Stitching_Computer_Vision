#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ccalib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

class ImageMatcher
{

private:
	Ptr<Feature2D> detector;

	double minDist;			// min distance to epipolar
	bool seamLess;
public:
	ImageMatcher(double d, int keyPtsMaxNum, bool s)
		: minDist(1.0), seamLess(false)
	{
		minDist = d;
		seamLess = s;
		detector = xfeatures2d::SIFT::create();
		//detector = xfeatures2d::SIFT::create(keyPtsMaxNum);
		//extractor = SIFT::create(100);
	}

	Ptr<Feature2D> getDetector() { return detector; }

	void setDetector(Ptr<Feature2D> newDetector) { detector = newDetector; }

	void match(Mat& image1, Mat& image2, vector<DMatch>& matches, 
		vector<KeyPoint>& keyPts1, vector<KeyPoint>& keyPts2);

	// extract good matches by RANSAC
	// return the fundamental matrix
	// disabeled method
	Mat ransacTest(const vector<DMatch>& matches, const vector<KeyPoint>& pts1, const vector<KeyPoint>& pts2,
		vector<DMatch>& outMatches);

	void ImageMatcher::project(float x1, float y1, const Mat& H, float& x2, float& y2);

	void computeInlierCount(Mat H, const vector<DMatch>& matches, const vector<KeyPoint>& keyPts1,
		const vector<KeyPoint>& keyPts2, int& numMatches, float inlierThreshold);

	void RANSAC(vector<DMatch>& matches, const vector<KeyPoint>& keyPts1, const vector<KeyPoint>& keyPts2, int& numMatches, int numIterations, float inlierThreshold, Mat& hom, Mat& homInv, 
		Mat& image1Display, Mat& image2Display);

	void displayInliers(const vector<DMatch>& matches, const vector<Point2f>& points, const vector<uchar>& inliers, Mat& imageDisplay);

	void minLeastSquares(const Mat& points1, const Mat& points2, Mat& hom, float &minErr);

	// [2]
	void stitch(const Mat& image1,const Mat& image2, const Mat& hom, const Mat& homInv, Mat& stitchedImage);

	void stitch2Images(const Mat& img1,const Mat& img2, Mat& result, int iterRansac);

	void computeBlendingWeights(Mat& W);
};

/* 

references:
	[1] openCV 2 computer Vision Application Programming Cookbook, Robert Langaniere, PACKT Publishing, 2011, chapter 9, pages 228-246. 
	[2] https://github.com/eschwabe/computer-vision/blob/master/hw2/src/Project2.cpp

*/