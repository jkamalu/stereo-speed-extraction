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
    
    // four images to treat at each timestep
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
    
    // cameras derived directly from Unity
    Mat cameraRight = (Mat_<double>(3, 4) << 1.0, 0, 0, 0.5, 0, 1.0, 0, -1.0, 0, 0, -1, -0.5);
    Mat cameraLeft = (Mat_<double>(3, 4) << 1.0, 0, 0, -0.5, 0, 1.0, 0, -1.0, 0, 0, -1, -0.5);
    // the AKAZE descriptor for keypoint identification
    Ptr<AKAZE> descriptor = AKAZE::create(AKAZE::DESCRIPTOR_MLDB);
    // the matcher for knn thresholding keypoint matching
    BFMatcher matcher = BFMatcher(NORM_HAMMING, false);
    
    /*
     * Load the images
     */
    ImageQuad loadImages(string l0, string r0, string l1, string r1);
    
    /*
     * Find all knn-keypoint matches between two images.
     *
     * Alias: knn-thresholded keypoint matches
     */
    void findDumbMatches(Image<uchar>& imageA, Image<uchar>& imageB, Mat descA, Mat descB,
                         vector<KeyPoint> keyPointsA, vector<KeyPoint> keyPointsB,
                         vector<MatchFilterPair>& matches,
                         string name);
    
    /*
     * Take the match-intersection between matches across perspectives
     * and time steps.
     *
     * Alias: relaxed match filtering
     */
    void filterMatches(vector<MatchFilterPair>& matchesL,
                       vector<MatchFilterPair>& matchesR,
                       vector<MatchFilterPair>& matches0,
                       vector<MatchFilterPair>& matches1,
                       vector<pair<MatchFilterPair, MatchFilterPair> >& filteredMatches);

    /*
     * Bookeeping for match filtering
     */
    inline void fillMaps(map<uint, uint>& queryMap, map<uint, uint>& trainMap, vector<SpeedExtractor::MatchFilterPair>& matches) {
        for (uint i = 0; i < matches.size(); i++) {
            queryMap[matches[i].queryIndex] = i;
            trainMap[matches[i].trainIndex] = i;
        }
    };
    
    /*
     * Bring to cartesian coordinates
     */
    inline void normalizeHomogeneous(Mat& matrix) {
        matrix.at<float>(0, 0) = matrix.at<float>(0, 0) / matrix.at<float>(3, 0);
        matrix.at<float>(1, 0) = matrix.at<float>(1, 0) / matrix.at<float>(3, 0);
        matrix.at<float>(2, 0) = matrix.at<float>(2, 0) / matrix.at<float>(3, 0);
        matrix.at<float>(3, 0) = 1.0;
    }
    
    /*
     * Get the euclidean distance
     */
    Point3f differenceCartesian(Mat& m1, Mat& m2);
    
    /*
     * Filter the speeds
     *
     * Alias: median absolute deviation speed filtering
     */
    vector<float> filterSpeeds(vector<float> euclideanNorms, float epsilon);
    
public:
    
    /*
     * Estimate speed given image quad
     */
    float estimateSpeed(ImageQuad& imageQuad, int timeDelta);
    
    /*
     * Estimate speed given image names
     */
    float estimateSpeed(string l0, string r0, string l1, string r1, int timeDelta);

    /*
     * Subtract the background from the images
     */
	ImageQuad BackgroundSubtraction(Image<uchar> L0, Image<uchar>R0, Image<uchar>L1, Image<uchar>R1);
    
    /*
     * Helper function for background subtraction
     */
	Mat BackgroundSubtraction_image(Image<uchar> image1, Image<uchar>image2);
    
    /*
     * Return the median; inefficient due to in-place sort.
     *
     * TODO: change to out of place or undo sort to preserve
     * ordering of vals reference
     */
    static inline float median(vector<float> vals) {
        float median;
        sort(vals.begin(), vals.end());
        if (vals.size() % 2 == 0 && vals.size() > 0) {
            median = vals[vals.size() / 2];
        } else {
            median = vals[(vals.size() - 1) / 2];
        }
        return median;
    }
    
    /*
     * Median absolute deviation of speeds with epsilon cut-off
     */
    static inline vector<float> medianAbsoluteDeviations(vector<float>& speeds, float epsilon) {
        float median = SpeedExtractor::median(speeds);
        vector<float> MAD;
        for (size_t i = 0; i < speeds.size(); i++) {
            MAD.push_back(abs(speeds[i] - median));
        }
        return MAD;
    }
    
};


#endif /* speed_extractor_h */
