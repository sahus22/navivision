import cv2
import os
import argparse

parser = argparse.ArgumentParser(
    description='Create a video from a folder of images.')
parser.add_argument('--image_folder', type=str, required=True,
                    help='path to the folder containing images')
parser.add_argument('--video_name', type=str, required=True,
                    help='name of the output video file')
parser.add_argument('--frame_rate', type=int, required=True,
                    help='The frame rate of the output video.')
args = parser.parse_args()
if not os.path.isdir(args.image_folder):
    raise ValueError("Image folder does not exist")
image_folder = args.image_folder
video_name = args.video_name
frame_rate = args.frame_rate


images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort()

first_image = cv2.imread(os.path.join(image_folder, images[0]))
frame_size = (first_image.shape[1], first_image.shape[0])

video_writer = cv2.VideoWriter(
    video_name, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, frame_size)

for image in images:
    img = cv2.imread(os.path.join(image_folder, image))
    img = cv2.resize(img, frame_size)
    video_writer.write(img)
video_writer.release()
