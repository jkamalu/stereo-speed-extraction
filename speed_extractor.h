#ifndef speed_extractor_h
#define speed_extractor_h

#include <map>
#include <stdio.h>
#include <iostream>
#include <random>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video.hpp>
#include "image.h"

using namespace std;
using namespace cv;

class SpeedExtractor {
    
private:
    
    // type def
    struct ImageQuad {
        Image<uchar> l0;
        Image<uchar> r0;
        Image<uchar> l1;
        Image<uchar> r1;
    };
    
    struct MatchFilterPair {
        int queryIndex;
        int trainIndex;
        Point2f queryPoint;
        Point2f trainPoint;
    };
    
    // class variables
    Mat cameraRight = (Mat_<double>(3, 4) << 1.0, 0, 0, 0.5, 0, 1.0, 0, -1.0, 0, 0, -1, -0.5);
    
    Mat cameraLeft = (Mat_<double>(3, 4) << 1.0, 0, 0, -0.5, 0, 1.0, 0, -1.0, 0, 0, -1, -0.5);
    
    Ptr<AKAZE> descriptor = AKAZE::create(AKAZE::DESCRIPTOR_MLDB);
    
    BFMatcher matcher = BFMatcher(NORM_HAMMING, false);
    
    // functions
    ImageQuad loadImages(string l0, string r0, string l1, string r1);
    
    void findDumbMatches(Image<uchar>& imageA, Image<uchar>& imageB, Mat descA, Mat descB,
                         vector<KeyPoint> keyPointsA, vector<KeyPoint> keyPointsB,
                         vector<MatchFilterPair>& matches,
                         string name);
    
    void filterMatches(vector<MatchFilterPair>& matchesL,
                       vector<MatchFilterPair>& matchesR,
                       vector<MatchFilterPair>& matches0,
                       vector<MatchFilterPair>& matches1,
                       vector<pair<MatchFilterPair, MatchFilterPair> >& filteredMatches);
    
    inline void fillMaps(map<uint, uint>& queryMap, map<uint, uint>& trainMap, vector<SpeedExtractor::MatchFilterPair>& matches) {
        for (uint i = 0; i < matches.size(); i++) {
            queryMap[matches[i].queryIndex] = i;
            trainMap[matches[i].trainIndex] = i;
        }
    };
    
    inline void normalizeHomogeneous(Mat& matrix) {
        /* Normalises the last value of the vector to 1 */
        if (matrix.at<float>(3, 0) != 0)
        {
            matrix.at<float>(0, 0) = matrix.at<float>(0, 0) / matrix.at<float>(3, 0);
            matrix.at<float>(1, 0) = matrix.at<float>(1, 0) / matrix.at<float>(3, 0);
            matrix.at<float>(2, 0) = matrix.at<float>(2,0) / matrix.at<float>(3,0);
            matrix.at<float>(3, 0) = 1.0;
        }
    }
    
    Point3f differenceCartesian(Mat& m1, Mat& m2);
    
    vector<float> filterSpeeds(vector<float> euclideanNorms, float epsilon, float threshold);
    
public:
    
    // functions
    float estimateSpeed(ImageQuad& imageQuad, int timeDelta);
    
    float estimateSpeed(string l0, string r0, string l1, string r1, int timeDelta);

	ImageQuad BackgroundSubtraction(Image<uchar> L0, Image<uchar>R0, Image<uchar>L1, Image<uchar>R1);
	Mat BackgroundSubtraction_image(Image<uchar> image1, Image<uchar>image2);

	float filterSpeeds(vector<float> euclideanNorms);


    
    static inline float median(vector<float>& vals) {
        float median;
        sort(vals.begin(), vals.end());
        if (vals.size() % 2 == 0 && vals.size() > 0) {
            median = vals[vals.size() / 2];
        } else {
            median = vals[(vals.size() - 1) / 2];
        }
        return median;
    }
    
};


#endif /* speed_extractor_h */
