#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <list>
#include "image.h"


typedef struct {
	vector<Point2f> kp1_l, kp1_r, kp2_l, kp2_r;
	cv::Mat pnts3D;
	cv::Mat pnts3D_2;
	Vec3d difference;
}position_data;

void homnormal(Mat & matrix);
float euclidean_distance(Vec3d vec);
Vec3d difference_cat(Mat matrix1, Mat matrix2);
void matcher(Mat desc1, Mat desc2, vector<Point2f>& kp1, vector<Point2f>& kp2, Image<uchar> I1, Image<uchar> I2, vector<KeyPoint> m1, vector<KeyPoint> m2, String name);
list<position_data> position_calculating(String I1_l_s, String I1_r_s, String I2_l_s, String I2_r_s);
float mean_speed(String I1_l_s, String I1_r_s, String I2_l_s, String I2_r_s);