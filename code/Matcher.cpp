#include "Matcher.h"
#include <numeric>      // std::iota

// matches feature points
void ImageMatcher::match(Mat& img1, Mat& img2,
	vector<DMatch>& outMatches, vector<KeyPoint>& keyPts1, vector<KeyPoint>& keyPts2)
{
	// [1] Feature Detection (detect feature points)
	cout << "[1] Feature detection " << endl;
	detector->detect(img1, keyPts1);
	detector->detect(img2, keyPts2);
	//cout << keyPts1.size() << endl;
	//cout << keyPts2.size() << endl;
	//-- Draw keypoints
	Mat img_keypoints_1; Mat img_keypoints_2;
	drawKeypoints(img1, keyPts1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img2, keyPts2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	// Show detected (drawn) keypoints
	cv::imshow("Keypoints 1", img_keypoints_1);
	cv::imshow("Keypoints 2", img_keypoints_2);
	waitKey(0);

	// [2] Feature Extraction
	cout << "[2] Feature extraction " << endl;
	Mat descriptors1, descriptors2;
	detector->compute(img1, keyPts1, descriptors1);
	detector->compute(img2, keyPts2, descriptors2);

	// [3] Match descriptors
	cout << "[3] Match descriptors " << endl;

	BFMatcher matcher(NORM_L2, true);
	matcher.match(descriptors1, descriptors2, outMatches);

	Mat img_matches;
	drawMatches(img1, keyPts1, img2, keyPts2, outMatches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//Show detected matches
	cv::imshow("Initial_matches", img_matches);
	imwrite("output/2.png", img_matches);
	waitKey(0);

	/* keep disabeled */
	//// [4] validate matches using RANSAC test
	//// cout << "[4] RANSAC test validation " << endl;
	//// vector<DMatch> initialMatches = outMatches;
	//// Mat fundamental = ransacTest(initialMatches, keyPts1, keyPts2, outMatches);
}

// extract good matches by fundamental matrix using RANSAC (disabeled)
// return the fundamental matrix
Mat ImageMatcher::ransacTest(const vector<DMatch>& initMatches, const vector<KeyPoint>& keyPts1,
	const vector<KeyPoint>& keyPts2, vector<DMatch>& outMatches)
{
	// convert keyPts into Point2f
	vector<Point2f> pts1, pts2;
	for (vector<DMatch>::const_iterator it = initMatches.begin(); it != initMatches.end(); ++it)
	{
		pts1.push_back(Point2f(keyPts1[it->queryIdx].pt));
		pts2.push_back(Point2f(keyPts2[it->trainIdx].pt));
	}

	// compute F matrix using RANSAC
	vector<uchar> inliers(pts1.size(), 0);

	Mat fundamental = findFundamentalMat(Mat(pts1), Mat(pts2), inliers, CV_FM_RANSAC, minDist, 0.99);

	// extrct the surviving matches
	vector<DMatch>::const_iterator it2 = initMatches.begin();
	for (vector<uchar>::const_iterator it1 = inliers.begin();
		it1 != inliers.end(); ++it1, ++it2)
	{
		if (*it1)
		{
			outMatches.push_back(*it2);
		}
	}

	pts1.clear();
	pts2.clear();
	for (vector<DMatch>::const_iterator it = outMatches.begin();
		it != outMatches.end(); ++it)
	{
		pts1.push_back(Point2f(keyPts1[it->queryIdx].pt));
		pts2.push_back(Point2f(keyPts2[it->trainIdx].pt));
	}

	// compute 8-point F from all survived matches
	if (true)
	{
		fundamental = findFundamentalMat(Mat(pts1), Mat(pts2), CV_FM_8POINT);
	}

	return fundamental;
}

/*
	<< PANORAMA MOSAIC STITCHING >>
	Step 3: Compute the homography between the images using RANSAC
*/

/*
	A. Implement a function project(x1, y1, H, x2, y2). This should project point (x1, y1) using the
	homography “H”. Return the projected point (x2, y2). Hint: See the slides for details on how to
	project using homogeneous coordinates. You can verify your result by comparing it to the result of
	the function cv::perspectiveTransform(...). [Only use this function for verification and then
	comment it out.].
*/

void ImageMatcher::project(float x1, float y1, const Mat& H, float& x2, float& y2)
{
	Mat p1 = (Mat_<double>(3, 1) << x1, y1, 1);
	//for (int i = 0; i < p1.rows; i++) {
	//	for (int j = 0; j < p1.cols; j++) {
	//		cout << p1.at<float>(i, j) << "  ";
	//	}
	//	cout << endl;
	//}
	Mat p2  ;
	//cout << "H size = " << H.cols << " , " << H.rows << endl;
	assert(H.cols == p1.rows );
	
	p2 = H * p1;
	//cout << "p2 size = " << p2.cols << " , " << p2.rows << endl;
	float z2 = p2.at<double>(0, 2);
	x2 = p2.at<double>(0, 0)/z2;
	y2 = p2.at<double>(0, 1)/z2;
	//cout << "(" << x2 << "," << y2 << ")" << endl;
}

/*
	B. Implement the function computeInlierCount(H, matches, numMatches, inlierThreshold).
	computeInlierCount is a helper function for RANSAC that computes the number of inlying points
	given a homography "H". That is, project the first point in each match using the function "project".
	If the projected point is less than the distance "inlierThreshold" from the second point, it is an
	inlier. Return the total number of inliers.
*/
void ImageMatcher::computeInlierCount(Mat H, const vector<DMatch>& matches, const vector<KeyPoint>& keyPts1,
	const vector<KeyPoint>& keyPts2, int& numMatches, float inlierThreshold)
{
	float a, b, x1, y1, x2, y2;
	numMatches = 0;

	for (vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
	{
		a = keyPts1[it->queryIdx].pt.x;
		b = keyPts1[it->queryIdx].pt.y;
		x1 = keyPts2[it->trainIdx].pt.x;
		y1 = keyPts2[it->trainIdx].pt.y;

		project(a, b, H, x2, y2);
		Point2f dif = (x2 - x1, y2 - y1);
		float dist = sqrt(dif.x*dif.x + dif.y*dif.y);

		if (dist < inlierThreshold) {
			numMatches++;
		}
	}
}

/*
	C. Implement the function RANSAC (matches , numMatches, numIterations, inlierThreshold, hom,
	homInv, image1Display, image2Display). This function takes a list of potentially matching points
	between two images and returns the homography transformation that relates them. To do this
	follow these steps:

		a. For "numIterations" iterations do the following:

			i. Randomly select 4 pairs of potentially matching points from "matches".
			ii. Compute the homography relating the four selected matches with the function
			cv::findHomography(...)***. Using the computed homography, compute the
			number of inliers using "computeInlierCount".
			iii. If this homography produces the highest number of inliers, store it as the best
			homography.

		b. Given the highest scoring homography, once again find all the inliers. Compute a new
			refined homography using all of the inliers (not just using four points as you did
			previously. ) Compute an inverse homography as well, and return their values in "hom" and
			"homInv".

		c. Display the inlier matches using cv::drawMatches(...).
*/
void ImageMatcher::RANSAC(vector<DMatch>& matches, const vector<KeyPoint>& keyPts1, const vector<KeyPoint>& keyPts2, int& numMatches, int numIterations, float inlierThreshold, Mat& hom, Mat& homInv,
	Mat& image1Display, Mat& image2Display)
{
	cout << "[4] RANSAC validation " << endl;
	const size_t N = matches.size();
	vector<Point2f> obj, scene;		// from keyPts1 and their matches in keyPts2
	Mat H;
	int maxInliers = 0;

	//// convert keyPts into Point2f
	//vector<Point2f> points1, points2;
	//vector<int> idx1, idx2;
	//KeyPoint::convert(keyPts1, points1, idx1);
	//KeyPoint::convert(keyPts2, points2, idx2);

	// Create a random permutation of the range 0, 1, ..., N - 1 
	vector<int> idx(matches.size());
	iota(idx.begin(), idx.end(), 0);

	// a.
	for (size_t i = 0; i < numIterations; ++i)
	{
		random_shuffle(idx.begin(), idx.end());
		// and pick the first 4 to be used as indices in matches
		DMatch m;

		for (size_t j = 0; j < 4; ++j)
		{
			m = matches[idx[j]];
			//cout << "m = (" << m.queryIdx << " , " << m.trainIdx << ")" << endl;
			obj.push_back(Point2f(keyPts1[m.queryIdx].pt));
			scene.push_back(Point2f(keyPts2[m.trainIdx].pt));
		}

		//Mat objT, sceneT;
		//transpose(obj, objT);
		//transpose(scene, sceneT);

		H = findHomography(Mat(obj), Mat(scene), 0);
		//for (int i = 0; i < H.rows; i++) {
		//	for (int j = 0; j < H.cols; j++) {
		//		cout << H.at<float>(i, j) << "  ";
		//	}
		//	cout << endl;
		//}

		computeInlierCount(H, matches, keyPts1, keyPts2, numMatches, inlierThreshold);
		//cout << i <<" numMatches = " << numMatches << " , maxInliers = " << maxInliers << endl;
		if (numMatches > maxInliers)
		{
			maxInliers = numMatches;
			hom = H;
		}
		obj.clear();
		scene.clear();

	}

	/*
		b. Given the highest scoring homography, once again find all the inliers. Compute a new
		refined homography using all of the inliers (not just using four points as you did
		previously. ) Compute an inverse homography as well, and return their values in "hom" and
		"homInv".
	*/

	float a, b, x1, y1, x2, y2;
	vector<DMatch> bestInliers;
	vector<uchar> bestInliers1;
	vector<uchar> bestInliers2;

	for (vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
	{
		a = keyPts1[it->queryIdx].pt.x;
		b = keyPts1[it->queryIdx].pt.y;
		x1 = keyPts2[it->trainIdx].pt.x;
		y1 = keyPts2[it->trainIdx].pt.y;

		project(a, b, hom, x2, y2);
		Point2f dif = (x2 - x1, y2 - y1);
		float dist = sqrt(dif.x*dif.x + dif.y + dif.y);


		if (dist < inlierThreshold) {
			bestInliers1.push_back(it->queryIdx);
			bestInliers2.push_back(it->trainIdx);
			bestInliers.push_back(*it);
		}
	}
	matches = bestInliers;
	// find homograhy and inverse homography using all points
	obj.clear();
	scene.clear();
	for (vector<DMatch>::const_iterator it = bestInliers.begin(); it != bestInliers.end(); ++it)
	{
		obj.push_back(Point2f(keyPts1[it->queryIdx].pt));
		scene.push_back(Point2f(keyPts2[it->trainIdx].pt));
	}
	hom = findHomography(Mat(obj), Mat(scene), 0);
	homInv = findHomography(Mat(scene), Mat(obj), 0);

	// To Do ...........
	// check 
	//Mat prod = hom*homInv;
	//for (int i = 0; i < prod.rows; i++)
	//{
	//	for (int j = 0; j < prod.cols; j++)
	//	{
	//		cout << prod.at<float>(i, j) << "		";
	//	}
	//	cout << endl;
	//}

	// c.
	// convert keyPts into Point2f
	vector<Point2f> points1, points2;
	for (vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
	{
		points1.push_back(Point2f(keyPts1[it->queryIdx].pt));
		points2.push_back(Point2f(keyPts2[it->trainIdx].pt));
	}
	 Mat image1 = image1Display.clone();
	 Mat image2 = image2Display.clone();
	//displayInliers(matches, points1, bestInliers1, image1Display);
	//displayInliers(matches, points2, bestInliers2, image2Display);

	//cv::imshow("image1_best_inliers", image1Display);
	//cv::imshow("image2_best_inliers", image2Display);
	//waitKey(0);
}

void ImageMatcher::displayInliers(const vector<DMatch>& matches, const vector<Point2f>& points, const vector<uchar>& inliers, Mat& imageDisplay)
{
	vector<Point2f>::const_iterator itPts = points.begin();
	for (vector<uchar>::const_iterator itIn = inliers.begin(); itIn != inliers.end(); ++itIn, ++itPts)
	{
		if (*itIn)
		{
			circle(imageDisplay, *itPts, 3, Scalar(255, 255, 255), 2);
		}
	}
}

// To DO ........
void ImageMatcher::minLeastSquares(const Mat& points1, const Mat& points2, Mat& hom, float &minErr)
{
	hom = Mat::zeros(3, 3, CV_32FC1);

	assert(points1.rows == points2.rows);
	const int N = points1.rows;			// number of matches
	
	Mat points1T, points2T;
	transpose(points1, points1T);
	transpose(points2, points2T);

	Mat A = Mat::zeros(2 * N, 9, CV_32FC1);

	for (int r = 0; r < points2.rows; r++)
	{
		int j = 2 * r;
		A.at<float>(j, 0) = points1.at<float>(r, 0);
		A.at<float>(j, 1) = points1.at<float>(r, 1);
		A.at<float>(j, 2) = 1;
		A.at<float>(j, 6) = -points2.at<float>(r, 0)*points1.at<float>(r, 0);
		A.at<float>(j, 7) = -points2.at<float>(r, 0)*points1.at<float>(r, 1);
		A.at<float>(j, 8) = -points2.at<float>(r, 0);

		A.at<float>(j+1, 3) = points1.at<float>(r, 0);
		A.at<float>(j+1, 4) = points1.at<float>(r, 1);
		A.at<float>(j+1, 5) = 1;
		A.at<float>(j+1, 6) = -points2.at<float>(r, 1)*points1.at<float>(r, 0);
		A.at<float>(j+1, 7) = -points2.at<float>(r, 1)*points1.at<float>(r, 1);
		A.at<float>(j+1, 8) = -points2.at<float>(r, 1);
	}

	Mat B;
	B.create(2 * N, 1, CV_32FC1);
	for (int i = 0; i < N; ++i)
	{
		B.at<float>(2 * i, 0) = points2.at<float>(i, 0);
		B.at<float>(2 * i + 1, 0) = points2.at<float>(i, 1);
	}

	Mat h;
	//solve(A, B, h, DECOMP_QR);
	int ctr = 0;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++){
			hom.at<float>(i, j) = h.at<float>(ctr);
			ctr++;
		}
	}
}

/*
	Step 4: Stitch the images together using the computed homography. Following these steps:
	A. Implement the function stitch(image1, image2, hom, homInv, stitchedImage). Follow these steps:
		a. Compute the size of "stitchedImage. " To do this project the four corners of "image2" onto
		"image1" using project(...) and "homInv". Allocate the image.
		b. Copy "image1" onto the "stitchedImage" at the right location.
		c. For each pixel in "stitchedImage", project the point onto "image2". If it lies within image2's
		boundaries, add or blend the pixel's value to "stitchedImage. " When finding the value of
		image2's pixel use bilinear interpolation [cv::getRectSubPix(...)].
*/

void ImageMatcher::stitch(const Mat& image1,const Mat& image2, const Mat& hom, const Mat& homInv, Mat& stitImg)
{
	cout << "[5] stitching images" << endl << endl;
	//cv::warpPerspective(image2, stitImg, homInv, Size(2 * image2.cols, image2.rows));
	//// copy image 1 on the first half of full image
	//Mat half(stitImg, Rect(0, 0, image1.cols, image1.rows));
	//image1.copyTo(half);
	//return;

	int w = image2.cols;
	int h = image2.rows;
	//cout << w << " , " << h << endl;
	//float b = image2.at<Vec3b>(h - 1, w - 1)[0];
	//float g = image2.at<Vec3b>(h - 1, w - 1)[1];
	//float r = image2.at<Vec3b>(h - 1, w - 1)[2];
	//cout << b << " , " << g << " , " << r << endl;
	// a.
	// get 4 corners of image2  { r , c }
	float TL[2] = { 0, 0 };
	float TR[2] = { 0, w - 1 };
	float BL[2] = { h - 1, 0 }; 
	float BR[2] = { h - 1, w - 1 };

	// project 4 corners of image 2 onto image1 using homInv and get the modified corners
	project(TL[1], TL[0], homInv, TL[1], TL[0]);
	project(TR[1], TR[0], homInv, TR[1], TR[0]);
	project(BL[1], BL[0], homInv, BL[1], BL[0]);
	project(BR[1], BR[0], homInv, BR[1], BR[0]);

	//cout << " image1 => cols, rows " << image1.cols << " , " << image1.rows << endl;
	//cout << "TL => " << TL[0] << " , " << TL[1] << endl;
	//cout << "TR => " << TR[0] << " , " << TR[1] << endl;
	//cout << "BL => " << BL[0] << " , " << BL[1] << endl;
	//cout << "BR => " << BR[0] << " , " << BR[1] << endl;

	// compute the size of stitched image
	int left = min(0, (int) min(TL[1], BL[1]));
	int right = max(image1.cols - 1, (int) max(TR[1], BR[1]));
	int top = min(0, (int) min(TL[0], TR[0]));
	int bottom = max(image1.rows - 1, (int) max(BL[0], BR[0]));

	//cout << left << " " << right << endl;
	//cout << top << " " << bottom << endl;
	int stitCols = right - left + 1;
	int stitRows = bottom - top + 1;
	// cout << "stitCols = " << stitCols << " , stitRows = " << stitRows << endl;
	// allocate stitched image
	stitImg = Mat(stitRows, stitCols, image1.type(), Scalar(0, 0, 0));
	// cout << " stitImg : ( " << stitImg.cols << " , " << stitImg.rows << " )" << endl;

	//imshow("stitImg_0", stitImg);
	//return;
	// b.
	// Copy "image1" onto the "stitchedImage" at the right location
	// imshow("img1", image1);
	image1.copyTo(stitImg(cv::Rect(abs(left), abs(top), image1.cols, image1.rows)));

	//int rs = abs(top);
	//for (int r = 0; r < image1.rows; r++) 
	//{
	//	int cs = abs(left);
	//	for (int c = 0; c < image1.cols; c++)
	//	{

	//		//cout << r << " , " << c << endl;
	//
	//		stitImg.at<Vec3b>(rs, cs) = image1.at<Vec3b>(r, c);
	//		cs++;
	//	}
	//	rs++;
	//	/*if (rs == 457) { break; }*/
	//}

	//imshow("stitImg_1", stitImg);
	//return;

	// c.
	// For each pixel in "stitchedImage", project the point onto "image2".
	// If it lies within image2's boundaries, add or blend the pixel's value to "stitchedImage. 
	// When finding the value of image2's pixel use bilinear interpolation [cv::getRectSubPix(...)].
	// cout << " before " << endl;
	// For each pixel in "stitchedImage"

	// for extra_4
	Mat W1 = Mat::zeros(image1.size(), CV_32FC1);
	Mat W2 = Mat::zeros(image2.size(), CV_32FC1);
	computeBlendingWeights(W1);
	computeBlendingWeights(W2);

	float alpha = 0.5;
	float beta = 0.5;

	for (int c = left; c < right; c++)
	{
		for (int r = top; r < bottom; r++)
		{
			// project the point onto image2

			float x, y;
			project(c, r, hom, x, y);

			// If it lies within image2's boundaries, add or blend the pixel's value to "stitchedImage.
			if (x > 0 && y > 0 && x < image2.cols && y < image2.rows)
			{
				Vec3b pBGR;
				Mat patch;
				getRectSubPix(image2, cv::Size(1, 1), Point2f(x, y), patch);

				pBGR = patch.at<Vec3b>(0, 0);
				//cout << (int)pBGR[0] << endl;

				// image1 col , row
				int cs = c + abs(left);
				int rs = r + abs(top);

				// if overlaps with image 1
				if (c > 0 && c < image1.cols && r > 0 && r < image1.rows)
				{
					if (seamLess) {
						alpha = W2.at<float>(y, x);
						beta = W1.at<float>(r, c);
						float w = alpha + beta;
						alpha /= w;
						beta /= w;
					}
					pBGR = alpha*pBGR + beta*stitImg.at<Vec3b>(rs, cs);
				}

				stitImg.at<Vec3b>(rs, cs) = pBGR;
				//cout << "after" << endl;
			}
		}
	}
}

void ImageMatcher::stitch2Images(const Mat& img1, const Mat& img2, Mat& stitImg, int iterRansac)
{
	Mat image1, image2;
	img1.convertTo(image1, CV_8U);
	img2.convertTo(image2, CV_8U);

	vector<DMatch> matches;
	vector<KeyPoint> keyPoints1, keyPoints2;
	match(image1, image2, matches, keyPoints1, keyPoints2);

	Mat hom, homInv;
	int numMatches = 0;

	// [4] validate matches using RANSAC

	RANSAC(matches, keyPoints1, keyPoints2, numMatches, iterRansac, minDist, hom, homInv, image1, image2);
	
	Mat refined_matches;
	drawMatches(image1, keyPoints1, image2, keyPoints2,
		matches, refined_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imshow("Ransac_refined_matches", refined_matches);
	imwrite("output/3.png", refined_matches);
	waitKey(0);
	stitch(img1, img2, hom, homInv, stitImg);
}

void ImageMatcher::computeBlendingWeights(Mat& wMask)
{
	float maxDist = 0;
	
	int h = wMask.rows;
	int w = wMask.cols;

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			// find the minimum number of pixels (distance) to an edge of the image
			double dist = min(min(i, h - i), min(j, w - j)) + 1;
			wMask.at<float>(i,j) = dist;

			if (dist > maxDist)
			{
				maxDist = dist;
			}
		}
	}

	Mat result;
	divide(wMask, maxDist, result);
	wMask = result;
}