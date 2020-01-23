#include "Matcher.h"

#include <string>
#include <iostream>
#include <vector>

/*
	FEATURE DETECTION AND MATCHING [Steps 1 and 2] - assignment 2
	Step 1: Compute the Harris corner detector using the following steps:
		A. Compute the x and y derivatives on the image
		B. Compute the covariance matrix H of the image derivatives. Typically, when you compute the
		covariance matrix you compute the sum of Ix2, Iy2 and IxIy over a window or small area of the image.
		To obtain a smooth result for better detection of corners, use a Gaussian weighted window.
		C. Compute the Harris response using determinant(H)/trace(H).
		D. Find peaks in the response that are above the threshold, and store the interest point locations.
		
		Required: 
		Open "Boxes.png" and compute the Harris corners. 
		Save an image "1a.png" showing the Harrisresponse on "Boxes.png" 
		(you'll need to scale the response of the Harris detector to lie between 0 and 255. ) When
		you're debugging your code, I would recommend using "Boxes.png" to see if you're getting the right result.
		Compute the Harris corner detector for images "Rainier1.png" and "Rainier2.png". 
		Save the images "1b.png" and "1c.png" of the detected corners (use the drawing function 
		cv::circle(...) or cv::drawKeypoints(...) to draw the interest points overlaid on the images)
*/
static bool seamLess = false;

void task_1()
{
	// a.
	Mat imgBoxes = imread("input/Boxes.png");
	imgBoxes.convertTo(imgBoxes, CV_8U);
	vector<KeyPoint> keyPts;

	cout << "[1] Feature detection " << endl;
	Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
	detector-> detect(imgBoxes, keyPts);

	//-- Draw keypoints
	Mat img_keypoints;
	drawKeypoints(imgBoxes, keyPts, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	// Show detected (drawn) keypoints
	imshow("1a", img_keypoints);
	imwrite("output/1a.png", img_keypoints);

	// b. c.
	Mat Rain1 = imread("input/Rainier1.png");
	Mat Rain2 = imread("input/Rainier2.png");

	if (Rain1.empty() || Rain1.empty())
	{
		printf("Can't read one of the images\n");
		return;
	}
	Mat img1, img2;
	vector<KeyPoint>keyPts1, keyPts2;

	Rain1.convertTo(img1, CV_8U);
	Rain2.convertTo(img2, CV_8U);
	cout << "[1] Feature detection " << endl;
	detector->detect(img1, keyPts1);
	detector->detect(img2, keyPts2);

	//-- Draw keypoints
	Mat img_keypoints_1; Mat img_keypoints_2;
	drawKeypoints(img1, keyPts1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img2, keyPts2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	//-- Show detected (drawn) keypoints
	imwrite("output/1b.png", img_keypoints_1);
	imwrite("output/1c.png", img_keypoints_2);
}

/*
	Step 2: Matching the interest points between two images.
		A. Compute the descriptors for each interest point.
		B. For each interest point in image 1, find its best match in image 2. The best match is defined as the
		interest point with the closest descriptor (SSD or ratio test).
		C. Add the pair of matching points to the list of matches.
		D. Display the matches using cv::drawMatches(...). You should see many correct and incorrect matches.

		Required: 
		Compute the Harris corner detector and find matches for images "Rainier1.png" and "Rainier2.png".
		Save the image "2.png" showing the image with the found matches 
		[use cv::drawMatches(...) to draw the matches on the two image].
*/
void task_2_3_4()
{
	Mat img1 = imread("input/Rainier1.png");
	Mat img2 = imread("input/Rainier2.png");

	if (img1.empty() || img2.empty())
	{
		printf("Can't read one of the images\n");
		return;
	}
	Mat stitImg;
	ImageMatcher imgMatcher(0.1, 200, seamLess);
	imgMatcher.stitch2Images(img1, img2, stitImg, 10);
	imshow("stitched_image", stitImg);
	imwrite("output/4.png", stitImg);
}


/*
	Create a panorama that stitches together the six Mt. Rainier photographs, i.e. , Rainier1.png, ... Painier6.png.
	The final result should look similar to "AllStitched.png".
*/

Mat pano(queue<Mat> imgQ, int itr)
{
	ImageMatcher imgMatcher(0.1, 200, seamLess);
	int ctr = 1;
	Mat stitImg;
	Mat img1 = imgQ.front();
	imgQ.pop();
	while (!imgQ.empty())
	{
		cout << "====< Stitching_all_with_ " << ++ctr << " >====" << endl;
		Mat img2 = imgQ.front();
		imgQ.pop();
		imgMatcher.stitch2Images(img1, img2, stitImg, itr);
		imshow("stitImg_" + to_string(ctr), stitImg);
		img1 = stitImg;
	}
	return stitImg;
}

void extra_1a()
{
	queue<Mat> imgQ;
	for (int i = 1; i <= 6; i++)
	{
		imgQ.push(imread("input/Rainier" + to_string(i) + ".png"));
	}
	Mat stitImg = pano(imgQ, 100);
	imshow("RainierAllStitched", stitImg);
	imwrite("output/RainierAllStitched.png", stitImg);
	cout << "stitch all is done " << endl;
}

void extra_1b()
{
	queue<Mat> imgQ;
	for (int i = 1; i <= 4; i++)
	{
		imgQ.push(imread("input/MelakwaLake" + to_string(i) + ".png"));
	}
	Mat stitImg = pano(imgQ, 2000);
	imshow("MelakwaLakeALLStitched", stitImg);
	imwrite("output/MelakwaLakeALLStitched.png", stitImg);
	cout << "stitch all is done " << endl;
}
void extra_2()
{
	ImageMatcher imgMatcher(0.1, 200, seamLess);

	Mat im1 = imread("input/s1.jpg");
	Mat im2 = imread("input/s2.jpg");
	Mat im3 = imread("input/s3.jpg");

	// find the least size and downsample the other images to have the least size
	int leastCols = min(im1.cols, min(im2.cols, im3.cols));
	int leastRows = min(im1.rows, min(im2.rows, im3.rows));
	// cout << "leastCols = " << leastCols << " , leastRows = " << leastRows << endl;

	Size sz((leastCols+3)/4, (leastRows+3)/4);

	Mat img1, img2, img3;
	Mat stitImg;

	resize(im1, img1, sz, BORDER_DEFAULT);
	resize(im2, img2, sz, BORDER_DEFAULT);
	resize(im3, img3, sz, BORDER_DEFAULT);

	Mat temp;
	imgMatcher.stitch2Images(img1, img2, temp, 2000);
	imgMatcher.stitch2Images(temp, img3, stitImg, 2000);

	imshow("myStithedImages", stitImg);
	imwrite("output/myStithedImages.png", stitImg);
}

void extra_2_auto()
{
	ImageMatcher imgMatcher(0.1, 200, seamLess);

	Mat im1 = imread("input/s1.jpg");
	Mat im2 = imread("input/s2.jpg");
	Mat im3 = imread("input/s3.jpg");

	// find the least size and downsample the other images to have the least size
	int leastCols = min(im1.cols, min(im2.cols, im3.cols));
	int leastRows = min(im1.rows, min(im2.rows, im3.rows));

	// cout << "leastCols = " << leastCols << " , leastRows = " << leastRows << endl;
	Size sz((leastCols + 3) / 4, (leastRows + 3) / 4);

	Mat image1, image2, image3;
	Mat img1, img2, img3;
	Mat stitImg;

	//if (im1.cols != leastCols || im1.rows != leastRows) { resize(im1, img1, sz, BORDER_DEFAULT); }
	//else { img1 = im1; }
	//if (im2.cols != leastCols || im2.rows != leastRows) { resize(im2, img2, sz, BORDER_DEFAULT); }
	//else { img2 = im2; }
	//if (im3.cols != leastCols || im3.rows != leastRows) { resize(im3, img3, sz, BORDER_DEFAULT); }
	//else { img3 = im3; }

	resize(im1, img1, sz, BORDER_DEFAULT);
	resize(im2, img2, sz, BORDER_DEFAULT);
	resize(im3, img3, sz, BORDER_DEFAULT);

	img1.convertTo(image1, CV_8U);
	img2.convertTo(image2, CV_8U);
	img3.convertTo(image3, CV_8U);

	// find the number of pairwise matches and pick the highest
	vector<DMatch> matches12, matches23, matches13;
	vector<KeyPoint> keyPoints1, keyPoints2, keyPoints3;
	Mat hom12, hom23, hom13, homInv12, homInv23, homInv13;
	int numMatches12 = 0, numMatches23 = 0, numMatches13 = 0;
	int itr = 1000;

	imgMatcher.match(image1, image2, matches12, keyPoints1, keyPoints2);
	imgMatcher.match(image2, image3, matches23, keyPoints2, keyPoints3);
	imgMatcher.match(image1, image3, matches13, keyPoints1, keyPoints3);

	imgMatcher.RANSAC(matches12, keyPoints1, keyPoints2, numMatches12, itr, 0.1, hom12, homInv12, image1, image2);
	imgMatcher.RANSAC(matches23, keyPoints2, keyPoints3, numMatches23, itr, 0.1, hom23, homInv23, image2, image3);
	imgMatcher.RANSAC(matches13, keyPoints1, keyPoints3, numMatches13, itr, 0.1, hom13, homInv13, image1, image3);

	cout << "pairwise matches => " << numMatches12 << " , " << numMatches23 << " , " << numMatches13 << endl;
	int maxMatches = max(numMatches12, max(numMatches23, numMatches13));
	Mat temp;

	if (numMatches12 == maxMatches) {
		imgMatcher.stitch(img1, img2, hom12, homInv12, temp);
		imgMatcher.stitch2Images(temp, img3, stitImg, itr);
	}
	else if(numMatches23 == maxMatches){
		imgMatcher.stitch(img2, img3, hom23, homInv23, temp);
		imgMatcher.stitch2Images(temp, img1, stitImg, itr);
	}
	else {
		imgMatcher.stitch(img1, img3, hom13, homInv13, temp);
		imgMatcher.stitch2Images(temp, image2, stitImg, itr);
	}
	imshow("myStithedImages_auto", stitImg);
	imwrite("output/myStithedImages_auto.png", stitImg);
}
/*
	Implement a new image descriptor or detector that can stitch the images "Hanging1.png" and "Hanging2.png".
	Save the Stitched image and the "match" images. In case you're wondering - No, you cannot rotate
	"Hanging2.png" by hand before processing, that's cheating.
*/

void extra_3()	/// use ORB descriptor for it is rotation invariant
{
	Mat img1 = imread("input/Hanging1.png");
	Mat img2 = imread("input/Hanging2.png");

	ImageMatcher imgMatcher(1.0, 200, seamLess);
	
	imgMatcher.setDetector(ORB::create());

	Mat stitImg;
	imgMatcher.stitch2Images(img1, img2, stitImg, 1000);
	imshow("HangingStithed", stitImg);
	imwrite("output/HangingStithed.png", stitImg);
}

// enables seams removals and runs task_2_3_4() , extra_1a(), extra_3() as examples
void extra_4()
{
	seamLess = true;
}

void extra_5()
{	// not working
	Mat img1 = imread("input/ND1.png");
	Mat img2 = imread("input/ND2.png");
	ImageMatcher imgMatcher(1.0, 200, seamLess);
	Mat stitImg;
	imgMatcher.stitch2Images(img1, img2, stitImg, 5000);
	imshow("myStithedImages", stitImg);
	imwrite("output/ND_stitched.png", stitImg);
}

int main() {

	//// de-comment next line to run all methods with seams reduction/removal  
	//extra_4();			// feature is implemented by center weighing in ImageMatcher::stitch

	task_1();
	waitKey(0);

	task_2_3_4();
	waitKey(0);

	extra_1a();
	waitKey(0);

	extra_1b();
	waitKey(0);

	extra_2();
	waitKey(0);

	extra_3();
	waitKey(0);

	//extra_2_auto();	// takes some time to find suitable ordering of stitching 3 images only
	//waitKey(0);

	waitKey(0);
	return 0;
}