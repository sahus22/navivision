First install the requirements:
	pip install -r requirements.txt
	
Next create the images for navigation using:
	python .\navivision.py --focal_length 2063.2 --baseline 0.267145 --input_folder "D:/Downloads/Sampler/drivstereo/" --output_folder "output"
(make sure that the input folder has both left and right folders)

Next to create a video from the output images use:
	python .\generate_video.py --image_folder output --video_name output.mp4 --frame_rate 12