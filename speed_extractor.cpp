#include "speed_extractor.h"

using namespace std;
using namespace cv;

/*
 Just makes an background subtraction for 2 pictures
 Returns the mask for the two images
 */
Mat SpeedExtractor::BackgroundSubtraction_image(Image<uchar> image1, Image<uchar>image2) {

	Mat return_mask(image1.size(), CV_8UC1, Scalar::all(0));
	Mat fgMaskMOG2; //forground mask fg mask generated by MOG2 method
	Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
	
	int n_erode_dilate = 1; 

	pMOG2 = createBackgroundSubtractorMOG2(10); //MOG2 

	pMOG2->apply(image1, fgMaskMOG2, 1);
	pMOG2->apply(image2, fgMaskMOG2, 0);

	// Some basic image processing opertions
	blur(fgMaskMOG2, fgMaskMOG2, cv::Size(5, 5));
	erode(fgMaskMOG2, fgMaskMOG2, cv::Mat(), cv::Point(-1, -1), n_erode_dilate);
	dilate(fgMaskMOG2, fgMaskMOG2, cv::Mat(), cv::Point(-1, -1), n_erode_dilate);
	vector< vector<Point> > contours;
	vector<Point> points;

	//Find a contour 
	findContours(fgMaskMOG2, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	
	for (size_t i = 0; i < contours.size(); i++) {
		for (size_t j = 0; j < contours[i].size(); j++) {
			cv::Point p = contours[i][j];
			points.push_back(p);
		}
	}

	// Make an rectangel mask including the whole image
	if (points.size() > 0) {
		Rect brect = cv::boundingRect(cv::Mat(points).reshape(2));
		rectangle(fgMaskMOG2, brect.tl(), brect.br(), cv::Scalar(100, 100, 200), 2, CV_AA);
		//set only the area inside the rectangle(that goes around the contour to 255)
		return_mask(brect).setTo(Scalar::all(255));
	}
	else {
		//If no contour is found, the whole mask is set to 255, so the whole image is then important
		return_mask.setTo(Scalar::all(255));
	}

	return return_mask;
}

SpeedExtractor::ImageQuad SpeedExtractor::BackgroundSubtraction(Image<uchar> L0, Image<uchar>R0, Image<uchar>L1, Image<uchar>R1) {
	// Masked original images
	Image<uchar> masked_image_L0;
	Image<uchar> masked_image_L1;
	Image<uchar> masked_image_R0;
	Image<uchar> masked_image_R1;
	// Mask for the image
	Mat mask_left;
	Mat mask_right;

	// Apply mask to original images
	mask_left = SpeedExtractor::BackgroundSubtraction_image(L0, L1); // Generates the forground mask for this two pictures
	mask_right = SpeedExtractor::BackgroundSubtraction_image(R0, R1); 
	L0.copyTo(masked_image_L0, mask_left);
	L1.copyTo(masked_image_L1, mask_left);
	R0.copyTo(masked_image_R0, mask_right);
	R1.copyTo(masked_image_R1, mask_right);

	SpeedExtractor::ImageQuad imageQuad = { masked_image_L0, masked_image_R0, masked_image_L1, masked_image_R1};
	return imageQuad;
}

SpeedExtractor::ImageQuad SpeedExtractor::loadImages(string l0, string r0, string l1, string r1) {
    Image<uchar> imageL0 = Image<uchar>(imread(l0, CV_LOAD_IMAGE_COLOR));
    Image<uchar> imageR0 = Image<uchar>(imread(r0, CV_LOAD_IMAGE_COLOR));
    Image<uchar> imageL1 = Image<uchar>(imread(l1, CV_LOAD_IMAGE_COLOR));
    Image<uchar> imageR1 = Image<uchar>(imread(r1, CV_LOAD_IMAGE_COLOR));
	//Background subtraction can be activated here 
	bool background_subtraction_on = true;
	SpeedExtractor::ImageQuad imageQuad;
	
	if (background_subtraction_on) {
		//Use the pictures with background subtraction
		imageQuad = SpeedExtractor::BackgroundSubtraction(imageL0, imageR0, imageL1, imageR1);
	} else {
		//Take the original picture
		imageQuad = {imageL0, imageR0, imageL1, imageR1 };
	}
    
    return imageQuad;
}

float SpeedExtractor::estimateSpeed(string l0, string r0, string l1, string r1, int timeDelta) {
    SpeedExtractor::ImageQuad imageQuad = this->loadImages(l0, r0, l1, r1);
    return this->estimateSpeed(imageQuad, timeDelta);
}

// taken from here https://stackoverflow.com/a/45399188
vector<float> SpeedExtractor::filterSpeeds(vector<float> speeds, float epsilon) {
    vector<float> MAD = SpeedExtractor::medianAbsoluteDeviations(speeds, epsilon);
    float medianMAD = this->median(MAD);
    if (medianMAD <= epsilon) {
        return speeds;
    }
    vector<float> inliers;
    for (size_t i = 0; i < MAD.size(); i++) {
        if (MAD[i] < medianMAD) {
            inliers.push_back(speeds[i]);
        }
    }
    return inliers;
}

float SpeedExtractor::estimateSpeed(SpeedExtractor::ImageQuad& imageQuad, int timeDelta) {
    // Key points
    vector<KeyPoint> keyPointsL0, keyPointsR0, keyPointsL1, keyPointsR1;
    // Key point features
    Mat descL0, descR0, descL1, descR1;
    // Detect the keypoints for the different images
    this->descriptor->detectAndCompute(imageQuad.l0, noArray(), keyPointsL0, descL0);
    this->descriptor->detectAndCompute(imageQuad.r0, noArray(), keyPointsR0, descR0);
    this->descriptor->detectAndCompute(imageQuad.l1, noArray(), keyPointsL1, descL1);
    this->descriptor->detectAndCompute(imageQuad.r1, noArray(), keyPointsR1, descR1);
    
    vector<SpeedExtractor::MatchFilterPair> matchesL, matchesR, matches0, matches1;
    
    this->findDumbMatches(imageQuad.l0, imageQuad.l1, descL0, descL1, keyPointsL0, keyPointsL1, matchesL, "matchesL.png");
    this->findDumbMatches(imageQuad.r0, imageQuad.r1, descR0, descR1, keyPointsR0, keyPointsR1, matchesR, "matchesR.png");
    this->findDumbMatches(imageQuad.l0, imageQuad.r0, descL0, descR0, keyPointsL0, keyPointsR0, matches0, "matches0.png");
    this->findDumbMatches(imageQuad.l1, imageQuad.r1, descL1, descR1, keyPointsL1, keyPointsR1, matches1, "matches1.png");
    
    vector<pair<SpeedExtractor::MatchFilterPair, SpeedExtractor::MatchFilterPair> > filteredMatches;
    this->filterMatches(matchesL, matchesR, matches0, matches1, filteredMatches);
    
    vector<float> euclideanNorms;
    
    for (auto filteredMatch : filteredMatches) {
        Mat pointL0(filteredMatch.first.queryPoint);
        Mat pointL1(filteredMatch.first.trainPoint);
        Mat pointR0(filteredMatch.second.queryPoint);
        Mat pointR1(filteredMatch.second.trainPoint);
        
        Mat point0(1, 3, CV_64F);
        Mat point1(1, 3, CV_64F);
        triangulatePoints(this->cameraLeft, this->cameraRight, pointL0, pointR0, point0);
        triangulatePoints(this->cameraLeft, this->cameraRight, pointL1, pointR1, point1);
        
        Point3f difference = this->differenceCartesian(point0, point1);
        euclideanNorms.push_back(norm(difference));
    }
    
    float epsilon = 0.001;
    vector<float> speeds;
    if (euclideanNorms.empty()) {
        return -1;
    } else {
        for (const auto& norm : euclideanNorms) {
            float speed = (norm / timeDelta) * 1000;
            if (speed < epsilon) {
                continue;
            }
            speeds.push_back(speed);
        }
    }
    if (speeds.empty()) {
        return -1;
    }

    vector<float> filteredSpeeds = this->filterSpeeds(speeds, epsilon);
    return SpeedExtractor::median(filteredSpeeds);
    
}

/*
 Calculates the difference between the two matrices (in homogenous form) and  returns a matrix in cat form
 */
Point3f SpeedExtractor::differenceCartesian(Mat& point0, Mat& point1) {
    Point3f difference;
    this->normalizeHomogeneous(point0);
    this->normalizeHomogeneous(point1);
    difference.x = point0.at<float>(0, 0) - point1.at<float>(0, 0);
    difference.y = point0.at<float>(1, 0) - point1.at<float>(1, 0);
    difference.z = point0.at<float>(2, 0) - point1.at<float>(2, 0);
    return difference;
}

void SpeedExtractor::filterMatches(vector<SpeedExtractor::MatchFilterPair>& matchesL,
                                   vector<SpeedExtractor::MatchFilterPair>& matchesR,
                                   vector<SpeedExtractor::MatchFilterPair>& matches0,
                                   vector<SpeedExtractor::MatchFilterPair>& matches1,
                                   vector<pair<SpeedExtractor::MatchFilterPair, SpeedExtractor::MatchFilterPair> >& filteredMatches) {
    map<uint, uint> query0, train0, query1, train1, queryL, trainL, queryR, trainR;
    SpeedExtractor::fillMaps(query0, train0, matches0);
    SpeedExtractor::fillMaps(query1, train1, matches1);
    SpeedExtractor::fillMaps(queryL, trainL, matchesL);
    SpeedExtractor::fillMaps(queryR, trainR, matchesR);
    for (auto matchL : matchesL) {
        try {
            int indexL0 = matchL.queryIndex;
            int indexR0 = matches0[query0.at(indexL0)].trainIndex;
            int indexR1 = matchesR[queryR.at(indexR0)].trainIndex;
            pair<SpeedExtractor::MatchFilterPair, SpeedExtractor::MatchFilterPair> match(matchL, matchesR[trainR.at(indexR1)]);
            filteredMatches.push_back(match);
        } catch (out_of_range) {
            continue;
        }
    }
    for (auto matchL : matchesL) {
        try {
            int indexL1 = matchL.trainIndex;
            int indexR1 = matches1[query1.at(indexL1)].trainIndex;
            int indexR0 = matchesR[queryR.at(indexR1)].queryIndex;
            pair<SpeedExtractor::MatchFilterPair, SpeedExtractor::MatchFilterPair> match(matchL, matchesR[queryR.at(indexR0)]);
            filteredMatches.push_back(match);
        } catch (out_of_range) {
            continue;
        }
    }
}

void SpeedExtractor::findDumbMatches(Image<uchar>& imageA, Image<uchar>& imageB, Mat descA, Mat descB,
                                     vector<KeyPoint> keyPointsA, vector<KeyPoint> keyPointsB,
                                     vector<SpeedExtractor::MatchFilterPair>& matches,
                                     string name) {
    
    assert(descA.size[0] != 0 && descB.size[0] != 0);
    
    vector<vector<DMatch> > knnMatches;
    this->matcher.knnMatch(descA, descB, knnMatches, 2);
    vector<DMatch> stereoMatches;
    vector<Point> matchPoints1, matchPoints2;
    for (vector<DMatch> knnMatch : knnMatches) {
        // distances cannot be too close
        if (knnMatch[0].distance < .8 * knnMatch[1].distance) {
            DMatch& bestMatch = knnMatch[0];
            SpeedExtractor::MatchFilterPair matchFilterPair = {
                bestMatch.queryIdx,
                bestMatch.trainIdx,
                keyPointsA[bestMatch.queryIdx].pt,
                keyPointsB[bestMatch.trainIdx].pt
            };
            matches.push_back(matchFilterPair);
            stereoMatches.push_back(bestMatch);
        }
    }

//    Mat imageMatches;
//    drawMatches(imageA, keyPointsA, imageB, keyPointsB, stereoMatches, imageMatches);
//    imshow(name, imageMatches);
//    waitKey();

}
