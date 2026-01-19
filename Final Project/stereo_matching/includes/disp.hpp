#ifndef DISPARITY_H
#define DISPARITY_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>

using namespace std;
using namespace cv;

// 用于外部访问的视差图变量（原图和彩色图）
extern cv::Mat disp_11;  // BM 灰度视差图
extern cv::Mat disp_12;  // BM 彩色视差图
extern cv::Mat disp_21;  // SGBM 灰度视差图
extern cv::Mat disp_22;  // SGBM 彩色视差图

/**
 * @brief 计算视差图（Block Matching 算法）
 * @param lpng 左图像（灰度或彩色）
 * @param rpng 右图像（灰度或彩色）
 * @param disp 生成的视差图（CV_16S 格式）
 */
void stereoBM(cv::Mat lpng, cv::Mat rpng, cv::Mat &disp)
{
    // 图像预处理（模糊 + 下采样 + 再模糊）
    cv::GaussianBlur(lpng, lpng, cv::Size(5, 5), 1.5);
    cv::GaussianBlur(rpng, rpng, cv::Size(5, 5), 1.5);
    cv::pyrDown(lpng, lpng);
    cv::pyrDown(rpng, rpng);
    cv::GaussianBlur(lpng, lpng, cv::Size(5, 5), 1.5);
    cv::GaussianBlur(rpng, rpng, cv::Size(5, 5), 1.5);

    // 创建视差图容器
    disp.create(lpng.rows, lpng.cols, CV_16S);
    cv::Mat disp8u(lpng.rows, lpng.cols, CV_8UC1);  // 可视化用灰度图

    // 图像尺寸
    cv::Size imgSize = lpng.size();
    int numDisparities = ((imgSize.width / 8) + 15) & -16; // 16的整数倍

    // 创建 BM 匹配器
    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(16, 9);
    cv::Rect roi1, roi2;  // BM 支持 ROI 区域匹配

    // 设置 BM 参数
    bm->setPreFilterType(0);
    bm->setPreFilterSize(15);
    bm->setPreFilterCap(31);
    bm->setBlockSize(9);
    bm->setMinDisparity(0);
    bm->setNumDisparities(numDisparities);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(5);
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setROI1(roi1);
    bm->setROI2(roi2);
    bm->setDisp12MaxDiff(1);

    // 计算视差图（输出为 CV_16S）
    bm->compute(lpng, rpng, disp);

    // 将 CV_16S 视差图转换为 CV_8U（缩放系数：255/(视差范围 * 16)）
    disp.convertTo(disp8u, CV_8U, 255.0 / (numDisparities * 16.0));

    // 彩色视差图可视化
    cv::Mat dispColor;
    cv::applyColorMap(disp8u, dispColor, cv::COLORMAP_JET);

    // 显示结果
    cv::imshow("BM_img", disp8u);
    cv::imshow("BM_color", dispColor);

    // 保存到全局变量
    disp_11 = disp8u;
    disp_12 = dispColor;
}


/**
 * @brief 计算视差图（Semi-Global Block Matching 算法）
 * @param lpng 左图像
 * @param rpng 右图像
 * @param disp_8u 输出的视差图（8位灰度）
 */
void stereoSGBM(cv::Mat lpng, cv::Mat rpng, cv::Mat &disp_8u)
{
    // 图像预处理（模糊 + 下采样 + 再模糊）
    cv::GaussianBlur(lpng, lpng, cv::Size(5, 5), 1.5);
    cv::GaussianBlur(rpng, rpng, cv::Size(5, 5), 1.5);
    cv::pyrDown(lpng, lpng);
    cv::pyrDown(rpng, rpng);
    cv::GaussianBlur(lpng, lpng, cv::Size(5, 5), 1.5);
    cv::GaussianBlur(rpng, rpng, cv::Size(5, 5), 1.5);

    // 创建视差图容器
    cv::Mat disp(lpng.rows, lpng.cols, CV_16S);
    disp_8u.create(lpng.rows, lpng.cols, CV_8UC1);

    // 图像尺寸及参数
    cv::Size imgSize = lpng.size();
    int numDisparities = ((imgSize.width / 8) + 15) & -16;
    int winSize = 9;
    int channels = lpng.channels();

    // 创建 SGBM 匹配器
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create();
    sgbm->setPreFilterCap(31);
    sgbm->setBlockSize(winSize);
    sgbm->setP1(16 * channels * winSize * winSize);
    sgbm->setP2(32 * channels * winSize * winSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numDisparities);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(200);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(cv::StereoSGBM::MODE_SGBM); // 全尺寸双通道模式

    // 计算视差图（输出为 CV_16S）
    sgbm->compute(lpng, rpng, disp);

    // 转换为 8 位灰度图用于显示
    disp.convertTo(disp_8u, CV_8U, 255.0 / (numDisparities * 16.0));
    cv::GaussianBlur(disp_8u, disp_8u, cv::Size(5, 5), 1.5);  // 平滑处理

    // 彩色视差图可视化
    cv::Mat dispColor;
    cv::applyColorMap(disp_8u, dispColor, cv::COLORMAP_JET);

    // 显示结果
    cv::imshow("SGBM_img", disp_8u);
    cv::imshow("SGBM_color", dispColor);

    // 保存到全局变量
    disp_21 = disp_8u;
    disp_22 = dispColor;
}

#endif  // DISPARITY_H
