# Navivision
## Introduction
The spatial awareness of blind individuals is limited, making navigation through their environment challenging. This project aims to address this issue by developing a real-time tool that employs computer vision techniques, including Stereo Depth Estimation and Object Detection, to detect objects in the path of the user and their respective distance.

We use YOLOv7 for object detection and STereo TRansformer for stereo depth estimation. We apply YOLOv7 and STereo TRansformer to the images to get the objects in the frame and their depths. The combination of these techniques provides an effective and cost-efficient solution for navigation assistance for blind people. We trained our model on the KITTI object detection dataset for navigation.

STTR generates the disparity map, and from that we can get the depth using the formula:
>𝑑𝑒𝑝𝑡ℎ=  (𝑏𝑎𝑠𝑒𝑙𝑖𝑛𝑒×𝑓𝑜𝑐𝑎𝑙 𝑙𝑒𝑛𝑔𝑡ℎ)/𝑑𝑖𝑠𝑝𝑎𝑟𝑖𝑡𝑦

## Usage
1. First install the requirements:
   > pip install -r requirements.txt
2. Next create the images for navigation using: *(make sure that the input folder has both left and right folders)*
	> python .\navivision.py --focal_length 2063.2 --baseline 0.267145 --input_folder "D:/Downloads/Sampler/drivstereo/" --output_folder "output"
3. Next to create a video from the output images use:
	> python .\generate_video.py --image_folder output --video_name output.mp4 --frame_rate 12

## Results
For object detection, we used recall, precision, and mAp to evaluate our model. Once the metrics had plateaued, we stopped training the model.

As for the Stereo Depth estimation we used 3px Error, EPE and Occ IOU. Our values were close enough to the optimal STTR model, that we were satisfied with our training.
| | 3px Error  | EPE | Occ IOU |
| ------------- | ------------- | ------------- | ------------- |
| Our Model | 8.12  | 1.67  | 0.91 |
|STTR Model | 6.74  | 1.50  | 0.98 |

Here is the demo video for the final output:
