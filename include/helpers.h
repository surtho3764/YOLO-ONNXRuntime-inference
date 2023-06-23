
//#pragma once
#ifndef HELPERS_H_
#define HELPERS_H_

#include <codecvt>
#include <fstream>
#include <opencv2/opencv.hpp>

struct Detection
{
    cv::Rect box;
    float conf{};
    int classId{};
};

namespace helpers
{   
    
    bool isImage(const std::string& path);
    std::string splitExtension(std::string fileNamePath);
    std::vector<std::string> loadNames(const std::string& path);
    const bool isGPU_fu(std::string &);
    void letterbox(const cv::Mat& image, cv::Mat& outImage,
                   const cv::Size& newShape,
                   const cv::Scalar& color,
                   bool auto_,
                   bool scaleFill,
                   bool scaleUp,
                   int stride);
    
    size_t vectorProduct(const std::vector<int64_t>& vector);
    std::wstring charToWstring(const char* str);
    void visualizeDetection(cv::Mat& image, std::vector<Detection>& detections,
                            const std::vector<std::string>& classNames);
    void scaleCoords(const cv::Size& imageShape, cv::Rect& box, const cv::Size& imageOriginalShape);
    template <typename T>
    T clip(const T& n, const T& lower, const T& upper);
}

#endif // HELPERS_H_