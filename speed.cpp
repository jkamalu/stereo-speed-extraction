#include "speed.h"



using namespace std;
using namespace cv;
void homnormal(Mat & matrix)
{
	/* Normalises the last value of the vector to 1 */
	if (matrix.at<float>(3, 0) != 0)
	{
		matrix.at<float>(0, 0) = matrix.at<float>(0, 0) / matrix.at<float>(3, 0);
			matrix.at<float>(1, 0) = matrix.at<float>(1, 0) / matrix.at<float>(3, 0);
			matrix.at<float>(2, 0) = matrix.at<float>(2,0) / matrix.at<float>(3,0);
			matrix.at<float>(3, 0) = 1.0; 
	}
}
float euclidean_distance(Vec3d vec)
{
	float tmp = 0; 
	tmp = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
	return tmp; 
}
Vec3d difference_cat(Mat matrix1, Mat matrix2)
{
	/*
	Calculates the difference between the two matrices (in homogenous form) and  returns a matrix in cat form
	*/
	//Mat difference(1, 3, CV_64F); //= Mat::zeros(1,3, CV_64F);
	Vec3d difference;

	if (matrix1.at<float>(3, 0) != 1.0 || matrix2.at<float>(3, 0) != 1.0)
	{
		homnormal(matrix1);
		homnormal(matrix2);
	}
	
	difference(0) = matrix1.at<float>(0, 0) - matrix2.at<float>(0, 0);
	difference(1) = matrix1.at<float>(1, 0) - matrix2.at<float>(1, 0);
	difference(2) = matrix1.at<float>(2, 0) - matrix2.at<float>(2, 0);

	return difference;
}
void matcher(Mat desc1, Mat desc2, vector<Point2f>& kp1, vector<Point2f>& kp2, Image<uchar> I1, Image<uchar> I2, vector<KeyPoint> m1, vector<KeyPoint> m2, String name)
{
	BFMatcher M = BFMatcher::BFMatcher(NORM_HAMMING, true);

	vector<DMatch> matches_stereo;
	M.match(desc1, desc2, matches_stereo);
	Mat img_matches;
	
	drawMatches(I1, m1, I2, m2, matches_stereo, img_matches);
	
	imshow(name , img_matches);

	double distance_threshold = 100;
	double ransacReprojThreshold = 3;
	// Order the matching points 
	for (int i = 0; i < matches_stereo.size(); i++)
	{
		if (matches_stereo[i].distance < distance_threshold)
		{
						
			kp1.push_back(m1[matches_stereo[i].queryIdx].pt);
			kp2.push_back(m2[matches_stereo[i].trainIdx].pt);
		}

	}

}
list<position_data> positions; // Is this good or bad to return a list ??? --> would it be better if list is given as Reference ?  
list<position_data> position_calculating(String I1_l_s, String I1_r_s, String I2_l_s, String I2_r_s)
{
	cout << I1_l_s << endl; 
	Image<uchar> I1_l = Image<uchar>(imread(I1_l_s, CV_LOAD_IMAGE_COLOR));
	Image<uchar> I1_r = Image<uchar>(imread(I1_r_s, CV_LOAD_IMAGE_COLOR));
	Image<uchar> I2_l = Image<uchar>(imread(I2_l_s, CV_LOAD_IMAGE_COLOR));
	Image<uchar> I2_r = Image<uchar>(imread(I2_r_s, CV_LOAD_IMAGE_COLOR));

	Mat cam_right = (Mat_<double>(3, 4) << 1.0, 0, 0, 0.5, 0, 1.0, 0, -1.0, 0, 0, -1, -0.5);
	Mat cam_left = (Mat_<double>(3, 4) << 1.0, 0, 0, -0.5, 0, 1.0, 0, -1.0, 0, 0, -1, -0.5);

	cout << "Mat Cam0 " << cam_right << endl;
	cout << "Mat Cam1 " << cam_left << endl;
	//Detect Keypoints with AKAZE
	Ptr<AKAZE> D = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, //descriptor_type =
		0, //int 	descriptor_size
		3, //int 	descriptor_channels 
		0.01f, //float 	threshold
		11, //int 	nOctaves
		11, //int 	nOctaveLayers
		KAZE::DIFF_PM_G2 //int diffusivity 
	);

	
	vector<KeyPoint> m1_l, m1_r, m2_l, m2_r; //Key points
	vector<Point2f> kp1_l, kp1_r, kp2_l, kp2_r, kp12_l, kp12_l2, kp12_r, kp12_r2; //Vectors for the matched key points --> at each position x in kp1(x) and kp2(x) are the coordinates of the matched Keypoints  
	Mat desc1_l, desc1_r, desc2_l, desc2_r;
	// Detect the keypoints for the different images
	
	D->detectAndCompute(I1_l, noArray(), m1_l, desc1_l);
	D->detectAndCompute(I1_r, noArray(), m1_r, desc1_r);
	D->detectAndCompute(I2_l, noArray(), m2_l, desc2_l);
	D->detectAndCompute(I2_r, noArray(), m2_r, desc2_r);

	// Match the different keypoints
	matcher(desc1_l, desc1_r, kp1_l, kp1_r, I1_l, I1_r, m1_l, m1_r, "First");
	matcher(desc2_l, desc2_r, kp2_l, kp2_r, I2_l, I2_r, m2_l, m2_r, "Second");
	matcher(desc1_l, desc2_l, kp12_l, kp12_l2, I1_l, I2_l, m1_l, m2_l, "Left Match");
	matcher(desc1_r, desc2_r, kp12_r, kp12_r2, I1_r, I2_r, m1_r, m2_r, "Right Match");
	int N = kp1_l.size();
	int N2 = kp2_l.size();

	list<position_data> position_data_list;
	
	RNG rng(3403);
	int counter = 0; 
	//Going through all keypoints finding the one that can be found in all 
	for (int h = 0; h < kp12_r.size(); h++)
	{
		for (int i = 0; i < kp12_l.size(); i++)
		{
			for (int j = 0; j < kp1_l.size(); j++)
			{

				for (int k = 0; k < kp2_l.size(); k++)
				{

					if ((kp1_l[j] == kp12_l[i] && kp2_l[k] == kp12_l2[i]) &&
						(kp1_r[j] == kp12_r[h] && kp2_r[k] == kp12_r2[h]))
					{
						//cout << "Match: " << endl;
						cv::Mat pnts3D(1, 3, CV_64F);
						cv::Mat pnts3D_2(1, 3, CV_64F);
						Vec3d difference;
						Mat match_point1_l(kp1_l[j]);
						Mat match_point1_r(kp1_r[j]);
						Mat match_point2_l(kp2_l[k]);
						Mat match_point2_r(kp2_r[k]);
						cv::triangulatePoints(cam_left, cam_right, match_point1_l, match_point1_r, pnts3D);
						cv::triangulatePoints(cam_left, cam_right, match_point2_l, match_point2_r, pnts3D_2);

						difference = difference_cat(pnts3D, pnts3D_2);
						//cout << difference << endl;
						
						//Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
						Scalar color[2] = { Scalar(187, 255, 0), Scalar(80, 0, 225) };
						circle(I1_l, kp1_l[j], 15, color[counter],2);
						circle(I2_l, kp2_l[k], 15, color[counter],2);
						counter++; 
						position_data one_position = {kp1_l, kp1_r, kp2_l,kp2_l, pnts3D, pnts3D_2, difference};
						position_data_list.push_back(one_position);

					}
				}

			}
		}
	}
	//Show the results  
	imshow("Circel 1 ", I1_l);
	imshow("Circel 2 ", I2_l);
	waitKey(0);
	return position_data_list;
}

float mean_speed(String I1_l_s, String I1_r_s, String I2_l_s, String I2_r_s){
	std::list<position_data> all_positions;
	all_positions = position_calculating(I1_l_s, I1_r_s, I2_l_s, I2_r_s);

	float mean = 0; 
	int counter = 0; 
	std::list<position_data>::iterator it;
	cout << "Difference in x, y, z" << endl;
	for (it = all_positions.begin(); it != all_positions.end(); ++it)
	{


		std::cout << euclidean_distance(it->difference) << endl;
		mean += euclidean_distance(it->difference);
		counter++; 
	}
	if (counter > 0)
	{
		return mean / counter;
	}

	else
	{
		return 0.0;
	}
	
}
/*int main()
{
	const int good_pairs = 100; 
	String I1_l_s = "../IMG_1_l.PNG";
	String I1_r_s = "../IMG_1_r.PNG";
	String I2_l_s = "../IMG_2_l.PNG";
	String I2_r_s = "../IMG_2_r.PNG";
	cout << mean_speed(I1_l_s,  I1_r_s,  I2_l_s,  I2_r_s) << endl;
	
	
	
	
	

	waitKey(0);
	return 0;
}*/
