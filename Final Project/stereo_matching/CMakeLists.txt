cmake_minimum_required(VERSION 3.5.1)

# 项目名称
project(StereoCamera)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找库 
# 2 versions of opencv for cyj and rxm
find_package(OpenCV 4 REQUIRED)
# find_package(OpenCV 4.9.0 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

# 包含头文件路径
include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${Pangolin_INCLUDE_DIRS})

# 设置OpenCV库路径
if (NOT OpenCV_LIBS)
  set(OpenCV_LIBS "opencv::opencv")
endif()

# 深度图
add_executable(disp_calculate disp_calculate.cpp includes/disp.hpp)
target_link_libraries(disp_calculate  ${OpenCV_LIBS})


# 稠密点云图
# add_executable(3d_pointcloud 3d_pointcloud.cpp includes/point_visualize.hpp)
# target_link_libraries(3d_pointcloud ${OpenCV_LIBS} ${Pangolin_LIBRARIES} Eigen3::Eigen)
