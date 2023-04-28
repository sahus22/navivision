
from module.sttr import STTR
from dataset.preprocess import normalization, compute_left_occ_region
from utilities.misc import NestedTensor
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os
from PIL import Image
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys

# sttr
args = type('', (), {})()  # create empty args
args.channel_dim = 128
args.position_encoding = 'sine1d_rel'
args.num_attn_layers = 6
args.nheads = 4
args.regression_head = 'ot'
args.context_adjustment_layer = 'cal'
args.cal_num_blocks = 8
args.cal_feat_dim = 16
args.cal_expansion_ratio = 4
model_sttr = STTR(args).cuda().eval()
model_file_name = "./sttr_light_sceneflow_pretrained_model.pth.tar"
checkpoint = torch.load(model_file_name)
pretrained_dict = checkpoint['state_dict']
# prevent BN parameters from breaking the model loading
model_sttr.load_state_dict(pretrained_dict, strict=False)
print("Pre-trained model successfully loaded.")
focal_length = 2063.200
baseline = 0.545

# yolo
conf_thres = 0.8
iou_thres = 0.45
save_dir = Path("./outpp")
# os.mkdir(save_dir)
# Initialize
img_size = 640
set_logging()
device = select_device()
half = device.type != 'cpu'  # half precision only supported on CUDA
augment = False
# Load model
weights = ['yolov7.pt']
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(img_size, s=stride)  # check img_size
model = TracedModel(model, device, img_size)
if half:
    model.half()
vid_path, vid_writer = None, None
# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
        next(model.parameters())))  # run once
old_img_w = old_img_h = imgsz
old_img_b = 1
t0 = time.time()


folder_path = "D:/Downloads/Sampler/drivstereo/left/"
for filename in os.listdir(folder_path):
    print(filename)
    left = np.array(Image.open('D:/Downloads/Sampler/drivstereo/left/'+filename))
    right = np.array(Image.open('D:/Downloads/Sampler/drivstereo/right/'+filename))
    input_data = {'left': left, 'right': right}
    input_data = normalization(**input_data)
    h, w, _ = left.shape
    input_data = NestedTensor(input_data['left'].cuda()[None,], input_data['right'].cuda()[None,])
    with torch.no_grad():
        output = model_sttr(input_data)
        torch.cuda.synchronize()
    disp_pred = output['disp_pred'].data.cpu().numpy()[0]
    occ_pred = output['occ_pred'].data.cpu().numpy()[0] > 0.5
    disp_pred[occ_pred] = 0.0

    source = "D:/Downloads/Sampler/drivstereo/left/"+filename
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes=None, agnostic=False)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    mid_x = int((int(xyxy[0]) + int(xyxy[2])) / 2)
                    mid_y = int((int(xyxy[1]) + int(xyxy[3])) / 2)
                    # if(mid_x)
                    depth = focal_length * baseline / disp_pred[mid_y, mid_x]
                    label = names[int(cls)] + "(dist:" + format(depth, ".2f") + ")"
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(
                f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            cv2.imwrite(save_path, im0)
            print(f" The image with the result is saved in: {save_path}")

print(f'Done. ({time.time() - t0:.3f}s)')
