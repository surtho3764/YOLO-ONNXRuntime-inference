# YOLO-ONNXRuntime-inference
The C++ implementation of YOLO v5 based on ONNXRuntime for performing object detection in real-time.


## Dependecies:
- OpenCV 4.x
- ONNXRuntime 1.3+

## Build Repository
```bash
mkdir build
cd build
cmake ..
```

## Run
Run from CLI:
```bash
## image
./demo --modelPath ../data/models/yolov5s.onnx --imagePath ../data/images/426342.jpg --classNamePath ../data/labels/coco.names

## Video
./demo --modelPath ../data/models/yolov5s.onnx --imagePath ../data/images/video2.mp4 --classNamePath ../data/labels/coco.names
```


## Demo

![image](https://github.com/surtho3764/YOLO-ONNXRuntime-inference/blob/main/demo/426342_result.jpg)

![image](https://github.com/surtho3764/YOLO-ONNXRuntime-inference/blob/main/demo/4h14w-nmce3.gif)

