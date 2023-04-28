from PIL import Image
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# sys.path.append('../') # add relative path

from module.sttr import STTR
from dataset.preprocess import normalization, compute_left_occ_region
from utilities.misc import NestedTensor

# Default parameters
args = type('', (), {})() # create empty args
args.channel_dim = 128
args.position_encoding='sine1d_rel'
args.num_attn_layers=6
args.nheads=4
args.regression_head='ot'
args.context_adjustment_layer='cal'
args.cal_num_blocks=8
args.cal_feat_dim=16
args.cal_expansion_ratio=4

model = STTR(args).cuda().eval()

# Load the pretrained model
model_file_name = "./sttr_light_sceneflow_pretrained_model.pth.tar"
checkpoint = torch.load(model_file_name)
pretrained_dict = checkpoint['state_dict']
model.load_state_dict(pretrained_dict, strict=False) # prevent BN parameters from breaking the model loading
print("Pre-trained model successfully loaded.")

folder_path = "D:/Downloads/Sampler/drivstereo/left/"
for filename in os.listdir(folder_path):
    print(filename)
    left = np.array(Image.open('D:/Downloads/Sampler/drivstereo/left/'+filename))
    right = np.array(Image.open('D:/Downloads/Sampler/drivstereo/right/'+filename))
# left = np.array(Image.open('./sample_data/KITTI_2015/training/image_2/000046_10.png'))
# right = np.array(Image.open('./sample_data/KITTI_2015/training/image_3/000046_10.png'))
# left = np.array(Image.open('D:/Downloads/Sampler/drivstereo/left/2018-07-10-09-54-03_2018-07-10-10-06-55-366.jpg'))
# right = np.array(Image.open('D:/Downloads/Sampler/drivstereo/right/2018-07-10-09-54-03_2018-07-10-10-06-55-366.jpg'))
# disp = np.array(Image.open('./sample_data/KITTI_2015/training/disp_occ_0/000046_10.png')).astype(float) / 256.

# Visualize image
# plt.figure(1)
# plt.imshow(left)
# plt.figure(2)
# plt.imshow(right)
# plt.figure(3)
# plt.imshow(disp)

# normalize
    input_data = {'left': left, 'right':right}
    input_data = normalization(**input_data)

    h, w, _ = left.shape

    # build NestedTensor
    input_data = NestedTensor(input_data['left'].cuda()[None,],input_data['right'].cuda()[None,])

    with torch.no_grad():
        output = model(input_data)
        torch.cuda.synchronize()

    # set disparity of occ area to 0
    disp_pred = output['disp_pred'].data.cpu().numpy()[0]
    occ_pred = output['occ_pred'].data.cpu().numpy()[0] > 0.5
    disp_pred[occ_pred] = 0.0

# visualize predicted disparity and occlusion map
# plt.figure(3)
# plt.imshow(disp_pred)
# plt.figure(4)
# plt.imshow(occ_pred)


# focal_length = 2063.200  # In pixels
# # 2063.400
# baseline = 0.545  # In meters
# depth = focal_length * baseline / disp_pred[200,200]
# print(disp_pred.shape)
# # plt.figure(5)
# # plt.imshow(depth_image)
# print(depth)
# plt.show()