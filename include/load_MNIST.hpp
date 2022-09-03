#ifndef MNIST_H
#define MNIST_H

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

cv::Mat ReadImages(std::string& FileName);
cv::Mat ReadLabels(std::string& FileName);
void show100Images(cv::Mat images);

#endif