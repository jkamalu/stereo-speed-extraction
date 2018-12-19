#include "speed_extractor.h"

using namespace std;
using namespace cv;

SpeedExtractor::ImageQuad SpeedExtractor::loadImages(string l0, string r0, string l1, string r1) {
    Image<uchar> imageL0 = Image<uchar>(imread(l0, CV_LOAD_IMAGE_COLOR));
    Image<uchar> imageR0 = Image<uchar>(imread(r0, CV_LOAD_IMAGE_COLOR));
    Image<uchar> imageL1 = Image<uchar>(imread(l1, CV_LOAD_IMAGE_COLOR));
    Image<uchar> imageR1 = Image<uchar>(imread(r1, CV_LOAD_IMAGE_COLOR));
    SpeedExtractor::ImageQuad imageQuad = {imageL0, imageR0, imageL1, imageR1};
    return imageQuad;
}

float SpeedExtractor::estimateSpeed(string l0, string r0, string l1, string r1, int timeDelta) {
    SpeedExtractor::ImageQuad imageQuad = this->loadImages(l0, r0, l1, r1);
    return this->estimateSpeed(imageQuad, timeDelta);
}
float SpeedExtractor::filter_speed(vector<float> euclideanNorms)
{
	double speed;
	bool median = true;

	if (median) {
		sort(euclideanNorms.begin(), euclideanNorms.end());
		if (euclideanNorms.empty()) {
			speed = -1;
		}
		else if (euclideanNorms.size() % 2 == 0) {
			speed = euclideanNorms[euclideanNorms.size() / 2];
		}
		else {
			speed = euclideanNorms[(euclideanNorms.size() - 1) / 2];
		}
	}
	else {
		speed = mean(euclideanNorms)[0];
	}
	cout << "Speed" << speed << endl;
	int position = 0; 
	float min_aggregated_difference = 0; 
	float best_value = 0; 
	float aggregated_difference;

	for (auto difference_index : euclideanNorms) {
		cout << "Value: " << difference_index << endl; 
		aggregated_difference = 0; 
		int inner_position = 0; 
		for (auto difference_values : euclideanNorms) {
			aggregated_difference = aggregated_difference + (difference_index - difference_values) * (difference_index - difference_values); 
			inner_position++;
		}
		//cout << "Aggregated Difference" << aggregated_difference << endl;
		if (aggregated_difference < min_aggregated_difference || position == 0)
		{
			min_aggregated_difference = aggregated_difference; 
			best_value = difference_index; 
			//cout << "Min Value founded" << best_value << endl; 
		}
		position++; 
	}

	//cout << "Min difference value" << best_value <<endl;


	int position_ra = 0;
	int max_inliers = 0;
	float best_value_ra = 0;
	int aggregated_inliers;

	for (auto difference_index : euclideanNorms) {
		float under_border =  difference_index - 0.15;
		float upper_boarder = difference_index +0.15;
		aggregated_inliers = 0;
		int inner_position = 0;
		for (auto difference_values : euclideanNorms) {
			if (under_border <= difference_values && difference_values <= upper_boarder)
			{
				aggregated_inliers++;
			}
		}
		//cout << "Aggregated Difference" << aggregated_inliers << endl;
		if (aggregated_inliers > max_inliers || position_ra == 0 && difference_index > 0.4)
		{
			max_inliers = aggregated_inliers;
			best_value_ra = difference_index;
			cout << "Min Value founded" << best_value_ra << endl;
		}
		position_ra++;
	}

	cout << "Min difference value" << best_value_ra << endl;

	return speed;


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
    
    vector<float> euclideanNorms; //Why int 
    
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
	return this->filter_speed(euclideanNorms);
    }

Point3f SpeedExtractor::differenceCartesian(Mat& point0, Mat& point1) {
    /*
     Calculates the difference between the two matrices (in homogenous form) and  returns a matrix in cat form
     */
    Point3f difference;
    
    if (point0.at<float>(3, 0) != 1.0 || point1.at<float>(3, 0) != 1.0) {
        this->normalizeHomogeneous(point0);
        this->normalizeHomogeneous(point1);
    }
    
    difference.x = point0.at<float>(0, 0) - point1.at<float>(0, 0);
    difference.y = point0.at<float>(1, 0) - point1.at<float>(1, 0);
    difference.z = point0.at<float>(2, 0) - point1.at<float>(2, 0);
    
    return difference;
}

void SpeedExtractor::normalizeHomogeneous(Mat& matrix) {
    /* Normalises the last value of the vector to 1 */
    if (matrix.at<float>(3, 0) != 0)
    {
        matrix.at<float>(0, 0) = matrix.at<float>(0, 0) / matrix.at<float>(3, 0);
        matrix.at<float>(1, 0) = matrix.at<float>(1, 0) / matrix.at<float>(3, 0);
        matrix.at<float>(2, 0) = matrix.at<float>(2,0) / matrix.at<float>(3,0);
        matrix.at<float>(3, 0) = 1.0;
    }
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
            int indexR0 = matches0[query0[indexL0]].trainIndex;
            int indexR1 = matchesR[queryR[indexR0]].trainIndex;
            int indexL1 = matches1[query1[indexR1]].queryIndex;
            if (indexL1 == matchL.trainIndex) {
                continue;
            }
            pair<SpeedExtractor::MatchFilterPair, SpeedExtractor::MatchFilterPair> match(matchL, matchesR[queryR[indexR1]]);
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
    vector<DMatch> stereoMatches;
    
    assert(descA.size[0] != 0 && descB.size[0] != 0);
    
    this->matcher.match(descA, descB, stereoMatches);
    
    Mat imageMatches;
    drawMatches(imageA, keyPointsA, imageB, keyPointsB, stereoMatches, imageMatches);
    // imshow(name, imageMatches);
    
    double distanceThreshold = 100;
    for (auto match : stereoMatches) {
        if (match.distance < distanceThreshold) {
            SpeedExtractor::MatchFilterPair matchFilterPair = {
                match.queryIdx,
                match.trainIdx,
                keyPointsA[match.queryIdx].pt,
                keyPointsB[match.trainIdx].pt
            };
            matches.push_back(matchFilterPair);
        }
    }
}
