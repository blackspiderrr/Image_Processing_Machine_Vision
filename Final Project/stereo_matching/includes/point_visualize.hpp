#ifndef SHOW_POINTCLOUD_H
#define SHOW_POINTCLOUD_H

#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>

using namespace std;
using namespace cv;
using namespace Eigen;

/**
 * @brief 使用 Pangolin 可视化点云
 * @param pointcloud 存储点云的向量，每个点为 Eigen::Vector4d（XYZ + 强度/灰度）
 */
void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>>& pointcloud) {
    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    // 创建窗口
    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);

    // 启用深度测试和混合（用于透明度）
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 设置投影和观察视角
    pangolin::OpenGlRenderState camera_settings(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(
            0.0, -0.1, -1.8,   // 相机位置 (x, y, z)
            0.0,  0.0,  0.0,   // 目标点 (look at)
            0.0, -1.0,  0.0    // 相机上方向 (up vector)
        )
    );

    // 创建交互式显示窗口
    pangolin::View& display = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(camera_settings));

    // 渲染循环
    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);      // 清空颜色与深度缓存
        display.Activate(camera_settings);                        // 激活相机视角
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);                      // 设置背景为白色

        glPointSize(2);
        glBegin(GL_POINTS);
        for (const auto& point : pointcloud) {
            float intensity = static_cast<float>(point[3]);

            // 如果 intensity 是以 [0, 255] 存储的，归一化为 [0,1]
            intensity = std::max(0.f, std::min(1.f, intensity > 1.0f ? intensity / 255.0f : intensity));

            // 彩虹色映射（类似 Jet colormap）
            float r = std::min(1.0f, std::max(0.0f, 1.5f - std::abs(4.0f * intensity - 3.0f)));
            float g = std::min(1.0f, std::max(0.0f, 1.5f - std::abs(4.0f * intensity - 2.0f)));
            float b = std::min(1.0f, std::max(0.0f, 1.5f - std::abs(4.0f * intensity - 1.0f)));

            glColor3f(r, g, b);  // 彩色映射
            glVertex3d(point[0], point[1], point[2]);
        }
        glEnd();

        pangolin::FinishFrame();
        usleep(5000);
    }

    pangolin::DestroyWindow("Point Cloud Viewer");                // 关闭窗口
}

#endif  // SHOW_POINTCLOUD_H
