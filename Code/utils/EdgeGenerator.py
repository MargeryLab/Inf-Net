import os, cv2
from Code.utils.format_conversion import binary2edge
import numpy as np


mask_path = '/media/margery/b44160de-00cb-402a-ba45-d81240edf8a4/DeepLearningDemo/CE-Net/submits/47TumorWallMask'
edge_path = '/media/margery/b44160de-00cb-402a-ba45-d81240edf8a4/DeepLearningDemo/CE-Net/submits/47TumorWallContour'


# def mask2edge():
#     for gt in os.listdir(mask_path):
#         edge_tmp = binary2edge(os.path.join(mask_path,gt))
#         cv2.imwrite(os.path.join(edge_path,gt), edge_tmp)

def mask2edge():
    for ID in os.listdir(mask_path):
        ID_path = os.path.join(mask_path, ID)
        if not os.path.exists(os.path.join(edge_path,ID)):
            os.mkdir(os.path.join(edge_path,ID))
        for gt in os.listdir(ID_path):
            wall_edge_tmp, tumor_edge_tmp = binary2edge(os.path.join(ID_path,gt))
            tumor_edge_tmp[tumor_edge_tmp==255] = 127
            edge_tmp = wall_edge_tmp + tumor_edge_tmp
            if np.max(edge_tmp)>255:
                print('error')
            cv2.imwrite(os.path.join(edge_path,ID, gt), edge_tmp)


if __name__ == '__main__':
    mask2edge()