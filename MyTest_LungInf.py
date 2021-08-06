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
from Code.model_lung_infection.InfNet_Res2Net import Inf_Net as Network
from Code.utils.dataloader_LungInf import test_dataset
from loss import dice_bce_loss

threshold = 0.4

def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=256, help='testing size')
    parser.add_argument('--data_path', type=str, default='./Dataset/TestingSet/LungInfection-Test/',
                        help='Path to test data')
    parser.add_argument('--pth_path', type=str, default='./snapshots/save_weights/Inf-Net/Inf-Net-60.pth',
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

        image = image.cuda()

        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, x_layer1,x_layer2,lateral_edge = model(image)
        edge_l1 = F.upsample(x_layer1, size=(ori_size[1],ori_size[0]), mode='bilinear', align_corners=False)
        edge_l1 = edge_l1.sigmoid().data.cpu().numpy().squeeze()
        edge_l1[edge_l1>=threshold] = 1.0
        edge_l1[edge_l1<threshold] = 0
        imageio.imwrite(os.path.join('Results/Lung infection segmentation/Edge-test-layer1',name), edge_l1)

        edge_l2 = F.upsample(x_layer2, size=(ori_size[1], ori_size[0]), mode='bilinear', align_corners=False)
        edge_l2 = edge_l2.sigmoid().data.cpu().numpy().squeeze()
        edge_l2[edge_l2>=threshold] = 1.0
        edge_l2[edge_l2<threshold] = 0
        imageio.imwrite(os.path.join('Results/Lung infection segmentation/Edge-test-layer2', name),
                        edge_l2)

        # edge_l3 = F.upsample(x_layer3, size=(ori_size[1], ori_size[0]), mode='bilinear', align_corners=False)
        # edge_l3 = edge_l3.sigmoid().data.cpu().numpy().squeeze()
        # edge_l3[edge_l3>=threshold] = 1.0
        # edge_l3[edge_l3<threshold] = 0
        # imageio.imwrite(os.path.join(
        #     r'D:\pythonDemo\medical_image_segmentation\Inf-Net\Results\Lung infection segmentation\Edge-test-layer3', name),
        #                 edge_l3)

        edge = lateral_edge
        edge = F.upsample(edge, size=(ori_size[1],ori_size[0]), mode='bilinear', align_corners=False)
        edge = edge.sigmoid().data.cpu().numpy().squeeze()
        edge[edge>=threshold] = 1.0
        edge[edge<threshold] = 0
        imageio.imwrite(os.path.join('Results/Lung infection segmentation/Edge-test-layer4',name), edge)

        lateral_map_5 = F.upsample(lateral_map_5, size=(ori_size[1], ori_size[0]), mode='bilinear', align_corners=False)
        lateral_map_5 = lateral_map_5.sigmoid().data.cpu().numpy().squeeze()
        lateral_map_5[lateral_map_5>=threshold] = 1.0
        lateral_map_5[lateral_map_5<threshold] = 0
        imageio.imwrite(os.path.join('Results/Lung infection segmentation/lateral_map_5',name), lateral_map_5)

        lateral_map_4 = F.upsample(lateral_map_4, size=(ori_size[1], ori_size[0]), mode='bilinear', align_corners=False)
        lateral_map_4 = lateral_map_4.sigmoid().data.cpu().numpy().squeeze()
        lateral_map_4[lateral_map_4>=threshold] = 1.0
        lateral_map_4[lateral_map_4<threshold] = 0
        imageio.imwrite(os.path.join('Results/Lung infection segmentation/lateral_map_4',name), lateral_map_4)

        lateral_map_3 = F.upsample(lateral_map_3, size=(ori_size[1],ori_size[0]), mode='bilinear', align_corners=False)
        lateral_map_3 = lateral_map_3.sigmoid().data.cpu().numpy().squeeze()
        lateral_map_3[lateral_map_3>=threshold] = 1.0
        lateral_map_3[lateral_map_3<threshold] = 0
        imageio.imwrite(os.path.join('Results/Lung infection segmentation/lateral_map_3',name), lateral_map_3)

        res = lateral_map_2 #(1,1,256,256)
        res = F.upsample(res, size=(ori_size[1],ori_size[0]), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()    #(256,256)
        res[res>=threshold] = 1.0
        res[res<threshold] = 0
        dice = dice_bce_loss().soft_dice_coeff(gt[0],torch.from_numpy(res))
        print(name.split('.')[0], dice.item())
        total_dice.append(dice)

        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imwrite(opt.save_path + name, res)

    print('Test Done!')
    print("dice_mean: {}, dice_std: {}".format(np.mean(total_dice), np.std(total_dice)))


if __name__ == "__main__":
    inference()
