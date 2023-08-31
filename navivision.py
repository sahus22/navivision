
from module.sttr import STTR
from dataset.preprocess import normalization
from utilities.misc import NestedTensor
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel
import time
from pathlib import Path
import cv2
import torch
from numpy import random
import os
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Create images that contain the objects in frame and their depth.')
parser.add_argument("--focal_length", type=float, required=True,
                    help="Focal length of the stereo camera")
parser.add_argument("--baseline", type=float, required=True,
                    help="Baseline of the stereo camera")
parser.add_argument("--output_folder", type=str, required=True,
                    help="Folder to store the output images")
parser.add_argument("--input_folder", type=str, required=True,
                    help="Folder that contains the left and right images")

args = parser.parse_args()

focal_length = args.focal_length
baseline = args.baseline
output_folder = args.output_folder
input_folder = args.input_folder

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(input_folder):
    raise ValueError('Input folder does not exist.')
if not os.path.exists(os.path.join(args.input_folder, 'left')) or not os.path.exists(os.path.join(args.input_folder, 'right')):
    raise ValueError('Input folder must contain left and right subfolders.')

output_folder = Path(output_folder)

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
model_file_name = "./stereo_depth_estimation_model.pth.tar"
checkpoint = torch.load(model_file_name)
pretrained_dict = checkpoint['state_dict']
model_sttr.load_state_dict(pretrained_dict, strict=False)
print("Stereo depth estimation model successfully loaded.")
focal_length = 2063.2
baseline = 0.267145

# yolo
conf_thres = 0.8
iou_thres = 0.45
img_size = 640
set_logging()
device = select_device()
half = device.type != 'cpu'
augment = False
weights = ['object_detection_model.pt']
model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(img_size, s=stride)
model = TracedModel(model, device, img_size)
if half:
    model.half()
vid_path, vid_writer = None, None
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
        next(model.parameters())))
old_img_w = old_img_h = imgsz
old_img_b = 1
t0 = time.time()


for filename in os.listdir(os.path.join(input_folder, 'left')):
    left = np.array(Image.open(os.path.join(input_folder, 'left', filename)))
    right = np.array(Image.open(os.path.join(input_folder, 'right', filename)))
    input_data = {'left': left, 'right': right}
    input_data = normalization(**input_data)
    h, w, _ = left.shape
    input_data = NestedTensor(input_data['left'].cuda()[
                              None,], input_data['right'].cuda()[None,])
    with torch.no_grad():
        output = model_sttr(input_data)
        torch.cuda.synchronize()
    disp_pred = output['disp_pred'].data.cpu().numpy()[0]
    occ_pred = output['occ_pred'].data.cpu().numpy()[0] > 0.5
    disp_pred[occ_pred] = 0.0

    source = os.path.join(input_folder, 'left', filename)
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes=None, agnostic=False)
        t3 = time_synchronized()

        for i, det in enumerate(pred):
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(output_folder / p.name)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    mid_x = int((int(xyxy[0]) + int(xyxy[2])) / 2)
                    mid_y = int((int(xyxy[1]) + int(xyxy[3])) / 2)
                    depth = focal_length * baseline / disp_pred[mid_y, mid_x]
                    label = names[int(cls)] + "(dist:" + \
                        format(depth, ".2f") + ")"
                    plot_one_box(xyxy, im0, label=label,
                                 color=colors[int(cls)], line_thickness=1)
                    if (depth < 8):
                        print(
                            "\033[91mWarning: {names[int(cls)]} close by\033[0m")

            print(f'{s} found')
            cv2.imwrite(save_path, im0)
print(f'Done. ({time.time() - t0:.3f}s)')
