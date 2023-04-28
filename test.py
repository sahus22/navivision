import cv2
import os

image_folder = "./outpp"
video_name = "my_video_slow.avi"

# Define the video frame size and frame rate
frame_size = (881, 400)
frame_rate = 12

# Get the list of image filenames in the folder
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

# Sort the images by filename
images.sort()

# Create a video writer object
video_writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"XVID"), frame_rate, frame_size)

# Loop through the images and add them to the video writer
for image in images:
    # Read the image file
    img_path = os.path.join(image_folder, image)
    img = cv2.imread(img_path)

    # Resize the image to the video frame size
    img = cv2.resize(img, frame_size)

    # Write the image to the video writer
    video_writer.write(img)

# Release the video writer and close the video file
video_writer.release()