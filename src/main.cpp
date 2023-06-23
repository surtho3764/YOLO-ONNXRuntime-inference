#include <iostream>
#include <opencv2/opencv.hpp>

#include "cmdline.h"
#include "helpers.h"
#include "detector.h"
#include <string>


int main(int argc, char* argv[]) {
    
    // 參數處理省略
    cmdline::parser arg;

    arg.add<std::string>("modelPath", 'm', "Path to onnx model.", true, "yolov5.onnx");
    arg.add<std::string>("imagePath", 's', "Data source to be detected.", true, "video.mp4");
    arg.add<std::string>("classNamePath", 'c', "Path to class names file.", true, "coco.names");
    arg.add<std::string>("score_thres", '\0', "Confidence threshold for categories.", false, "0.3f");
    arg.add<std::string>("iou_thres", '\0', "Overlap threshold.", false, "0.4f");

    arg.parse_check(argc, argv);
    std::string modelPath = arg.get<std::string>("modelPath");
    std::string imagePath = arg.get<std::string>("imagePath");
    std::string classNamePath = arg.get<std::string>("classNamePath");
    std::string score_thres = arg.get<std::string>("score_thres");
    std::string iou_thres = arg.get<std::string>("iou_thres");

    //std::string modelPath = "../data/models/yolov5s.onnx";
    //std::string imagePath = "../data/images/traffic.jpg";
    
    //std::string imagePath = "../data/images/video.mp4";
    //std::string classNamePath = "../data/labels/coco.names";
    //std::string score_thres = "0.3f";
    //std::string iou_thres = "0.4f";
    std::string gpu = "GPU";
    
    //const std::string modelPath = model_path;
    //const std::string classNamePath = class_names;

    const float confThreshold = std::stof(score_thres);
    const float iouThreshold = std::stof(iou_thres);
    // session config
    const bool isGPU = helpers::isGPU_fu(gpu);
    
    bool isImage = helpers::isImage(imagePath);
    //std::string sourcePath = imagePath;
    // 輸出檔案的路徑
    std::string outputPath = "";
    if (isImage)
    {
      outputPath = helpers::splitExtension(imagePath) + "_result.jpg";
    }else {
      outputPath = helpers::splitExtension(imagePath) + "_result.mp4";
    }

    const std::vector<std::string> classNames = helpers::loadNames(classNamePath);
    if (classNames.empty())
    {
        std::cout<< "Error:Empty class names file." << std::endl;
        return -1;
    }


    // 宣告yolo ddetector
    YOLODetector detector {nullptr};
    cv::Mat image;
    std::vector<Detection> result;

    try 
    {   // 創建目標識別物件
        detector = YOLODetector(modelPath, isGPU, cv::Size(640, 640));
        std::cout << "Model was initialized." << std::endl;
    } catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    
    if(!isImage)
    {
        std::cout << "=========== Video detection =========== " << std::endl;
        cv::VideoCapture cap;

        // Check if source is webcam
        if (imagePath == "0"){
            cap = cv::VideoCapture(0);
        } else{
            cap = cv::VideoCapture(imagePath);
        }

        if (!cap.isOpened()){
            std::cout << "Cannot open video.\n" << std::endl;
            return -1;
        }

        // 得到影片的資訊每frame為多少width和height
        cv::Size S = cv::Size((int)cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH), 
                      (int)cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT));
        // 得到每秒有多少frame
        int fps = cap.get(cv::VideoCaptureProperties::CAP_PROP_FPS);
        std::cout << "=============== origianl Video Info =============== " << fps << std::endl;
        std::cout << "Video PROP FRAME (width,height) :" << S << std::endl;
        std::cout << "Video FPS : " << fps << std::endl;
        std::cout << "=================================================== " << fps << std::endl;

        cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, S, true);
        while (true) {
            // 將frame讀到image中
            bool ret = cap.read(image); 
            if (!ret) {
                std::cout << "Can't receive frame (stream end?). Exiting ...\n" << std::endl;
                break;
            }
            result = detector.detect(image, confThreshold, iouThreshold);
            helpers::visualizeDetection(image, result, classNames);
            cv::imshow("result", image);
            writer.write(image);
            if (cv::waitKey(33) == 'q') {
                break;
            }
        }
        cap.release();
        writer.release();
        return 0;

    } else {
        std::cout << "=========== Image detection =========== " << std::endl;
        // 讀取影像
        // opencv讀取的影像資訊為BGR，而且為[weight,height,,channels]，只有Python版本的opencv，才是(height,weight)
        image = cv::imread(imagePath);
        std::cout << "=========== original Image Info ============== " <<  std::endl;
        std::cout << "original image " << image.size() << std::endl;
        std::cout << "=============================================== " <<  std::endl;
        
        // 執行目標偵測
        result = detector.detect(image, confThreshold, iouThreshold);
        // 視覺化偵測結果
        helpers::visualizeDetection(image, result, classNames);
        // 顯示預測結果
        cv::imshow("result", image);
        cv::imwrite(outputPath, image);
        cv::waitKey(0);
        return 0;
    }

 
}
