#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>
#include "./includes/point_visualize.hpp"

#define IMG_WIDTH 2560
#define IMG_HEIGHT 720

using namespace std;
using namespace cv;

int main()
{
    // 读取标定参数
    FileStorage fs("../calibration/calib_param.yml", FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open calibration file." << endl;
        return -1;
    }

    Mat K_left, D_left, K_right, D_right, R, T;
    fs["cameraMatrixL"] >> K_left;
    fs["distCoeffsL"] >> D_left;
    fs["cameraMatrixR"] >> K_right;
    fs["distCoeffsR"] >> D_right;
    fs["R"] >> R;
    fs["T"] >> T;
    fs.release();

    // 打开双相机
    VideoCapture cap(4);
    if (!cap.isOpened()) {
        cerr << "Unable to open camera." << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, IMG_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT);

    Size imageSize(IMG_WIDTH / 2, IMG_HEIGHT);

    // 立体校正参数
    Mat R1, R2, P1, P2, Q;
    stereoRectify(K_left, D_left, K_right, D_right, imageSize, R, -T,
                  R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 0);

    // 创建映射地图
    Mat map1x, map1y, map2x, map2y;
    initUndistortRectifyMap(K_left, D_left, R1, P1, imageSize, CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(K_right, D_right, R2, P2, imageSize, CV_32FC1, map2x, map2y);

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cerr << "Empty frame." << endl;
            break;
        }

        // 分割左右图像
        Mat frame_left = frame(Rect(0, 0, IMG_WIDTH / 2, IMG_HEIGHT));
        Mat frame_right = frame(Rect(IMG_WIDTH / 2, 0, IMG_WIDTH / 2, IMG_HEIGHT));

        // 校正图像
        Mat rectified_left, rectified_right;
        remap(frame_left, rectified_left, map1x, map1y, INTER_LINEAR);
        remap(frame_right, rectified_right, map2x, map2y, INTER_LINEAR);

        // 相机内参
        double fx = K_left.at<double>(0, 0);
        double fy = K_left.at<double>(1, 1);
        double cx = K_left.at<double>(0, 2);
        double cy = K_left.at<double>(1, 2);
        double baseline = norm(T, NORM_L2) / 1000.0; // mm 转米

        // 转为灰度图
        Mat gray_left, gray_right;
        cvtColor(rectified_left, gray_left, COLOR_BGR2GRAY);
        cvtColor(rectified_right, gray_right, COLOR_BGR2GRAY);

        // 计算观差图
        Mat disparity_raw;
        Ptr<StereoSGBM> sgbm = StereoSGBM::create(
            0, 96, 9, 8 * 9 * 9, 32 * 9 * 9,
            1, 63, 10, 100, 32);

        sgbm->compute(gray_left, gray_right, disparity_raw);

        Mat disparity;
        disparity_raw.convertTo(disparity, CV_32F, 1.0 / 16.0f);

        // 生成点云
        vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;

        for (int v = 0; v < gray_left.rows; ++v) {
            for (int u = 0; u < gray_left.cols; ++u) {
                float disp = disparity.at<float>(v, u);
                if (disp <= 0.0f || disp >= 96.0f) continue;

                Vector4d point;
                double z = fx * baseline / disp;
                double x = (u - cx) * z / fx;
                double y = (v - cy) * z / fy;

                point << x, y, z, gray_left.at<uchar>(v, u) / 255.0; // 后面是灰度值
                pointcloud.push_back(point);
            }
        }

        // 显示观差图
        imshow("Disparity", disparity / 96.0);

        waitKey(0);  // 一次性显示

        // 显示点云
        showPointCloud(pointcloud);

        if (waitKey(15) >= 0) break;
    }

    return 0;
}