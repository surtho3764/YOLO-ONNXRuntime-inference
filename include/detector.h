// #pragma once
#ifndef DETECTOR_H_
#define DETECTOR_H_

#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"
#include <utility>

#include "helpers.h"


class YOLODetector
{
public:
    explicit YOLODetector(std::nullptr_t) {};
    YOLODetector(const std::string& modelPath,
                 const bool& isGPU,
                 const cv::Size& inputSize);
    int num_class;
    std::vector<Detection> detect(cv::Mat &image, const float& confThreshold, const float& iouThreshold);

private:
    // =================================================================
    // YOLODetector
    // =================================================================
    Ort::Env env{nullptr};
    // session options
    Ort::SessionOptions sessionOptions{nullptr};
    // session
    Ort::Session session{nullptr};

    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    #if ORT_API_VERSION >= 13 
        std::vector<std::string> inputNamesString;
        std::vector<std::string> outputNamesString;
    #endif

    bool isDynamicInputShape{};
    cv::Size2f inputImageShape;

    void preprocessing(cv::Mat &image, float*& blob, std::vector<int64_t>& inputTensorShape);
    std::vector<Detection> postprocessing(const cv::Size& resizedImageShape,
                                          const cv::Size& originalImageShape,
                                          std::vector<Ort::Value>& outputTensors,
                                          const float& confThreshold, const float& iouThreshold);
    static void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
                                 float& bestConf, int& bestClassId);

};

#endif // DETECTOR_H_