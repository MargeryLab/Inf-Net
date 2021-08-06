# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.

First Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc
import imageio
from PIL import Image
from Code.model_lung_infection.InfNet_Res2Net import Inf_Net as Network
from Code.utils.dataloader_LungInf import test_dataset
from loss import dice_bce_loss
from torchvision import transforms
import matplotlib.pyplot as plt

threshold = 0.5

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def plot_img_and_mask(pred, mask):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Prediction image')
    ax[0].imshow(pred)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Ground Truth')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=256, help='testing size')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=True)
    parser.add_argument('--data_path', type=str, default='./Dataset/TestingSet/LungInfection-Test/',
                        help='Path to test data')
    parser.add_argument('--pth_path', type=str, default='./snapshots/save_weights/Inf-Net/Inf-Net-100.pth',
                        help='Path to weights file. If `semi-sup`, edit it to `Semi-Inf-Net/Semi-Inf-Net-100.pth`')
    parser.add_argument('--save_path', type=str, default='./Results/Lung infection segmentation/Inf-Net/',
                        help='Path to save the predictions. if `semi-sup`, edit it to `Semi-Inf-Net`')
    opt = parser.parse_args()

    print("#" * 20, "\nStart Testing (Inf-Net)\n{}\nThis code is written for 'Inf-Net: Automatic COVID-19 Lung "
                    "Infection Segmentation from CT Scans', 2020, arXiv.\n"
                    "----\nPlease cite the paper if you use this code and dataset. "
                    "And any questions feel free to contact me "
                    "via E-mail (gepengai.ji@gamil.com)\n----\n".format(opt), "#" * 20)

    model = Network()
    # model = torch.nn.DataParallel(model, device_ids=[0, 1]) # uncomment it if you have multiply GPUs.
    # model.load_state_dict(torch.load(opt.pth_path, map_location={'cuda:1':'cuda:0'}))
    model.load_state_dict(torch.load(opt.pth_path),False)
    model.cuda()
    model.eval()

    image_root = '{}/Imgs/'.format(opt.data_path)
    gt_root = '{}/GT/'.format(opt.data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    os.makedirs(opt.save_path, exist_ok=True)

    total_dice = []
    for i in range(test_loader.size):
        image, gt, name, ori_size = test_loader.load_data()
        gt_arr = gt.numpy() #(1,512,512)(0,1)
        image = image.cuda()

        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(image)

        res = lateral_map_2 #(1,1,256,256)
        # res = F.upsample(res, size=(ori_size[1],ori_size[0]), mode='bilinear', align_corners=False)
        # res = res.sigmoid().data.cpu().numpy().squeeze()    #(256,256)
        res = torch.sigmoid(res)
        res = (res > threshold).float().squeeze(0) #(1,1,256,256)
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((ori_size[1],ori_size[0])),
                transforms.ToTensor()
            ]
        )
        res = tf(res.cpu())
        res = res.squeeze().cpu().numpy() #(512,512)

        dice = dice_bce_loss().soft_dice_coeff(torch.from_numpy(gt_arr[0]),torch.from_numpy(res))
        print(name.split('.')[0], dice.item())
        total_dice.append(dice)

        result = mask_to_image(res)
        result.save(os.path.join(opt.save_path + name))

        if opt.viz:
            plot_img_and_mask(result, gt[0])

    print('Test Done!')
    print("dice_mean: {}, dice_std: {}".format(np.mean(total_dice), np.std(total_dice)))


if __name__ == "__main__":
    inference()
