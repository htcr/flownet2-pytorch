import torch
import numpy as np
import cv2

import os

from models import FlowNet2

import importlib.util
spec = importlib.util.spec_from_file_location("utils", "./utils.py")
global_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(global_utils)
limit_img_size = global_utils.limit_img_size
block_img_size = global_utils.block_img_size
resize_img = global_utils.resize_img

# from global_utils import limit_img_size, block_img_size, resize_img

class FlowNet2Args(object):
    def __init__(self, fp16=False, rgb_max=255.):
        self.fp16 = fp16
        self.rgb_max = rgb_max
        

def flownet2_preprocess(img):
    ori_size = img.shape[:2]
    new_size = limit_img_size(ori_size, 1024)
    new_size = block_img_size(new_size, 64) 
    new_img = resize_img(img, new_size)
    return new_img


class FlowNet2DiffMasker(object):
    def __init__(self):
        self.net = FlowNet2(FlowNet2Args()).cuda()
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FlowNet2_checkpoint.pth.tar")
        loaded_dict = torch.load(model_path)
        self.net.load_state_dict(loaded_dict["state_dict"])

    def get_mask(self, img, bk):
        assert img.shape == bk.shape
        ori_size = img.shape[:2]
        img = flownet2_preprocess(img)
        bk = flownet2_preprocess(bk)

        images = [img, bk]
        images = np.array(images).transpose(3, 0, 1, 2)
        im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()
        
        result = self.net(im).squeeze()
        flow = result.data.cpu().numpy().transpose(1, 2, 0)

        mag = np.sum(flow**2, axis=2)**0.5
        
        avg_mag = np.mean(mag)
        
        mag = resize_img(mag, ori_size)

        mask = mag > 0.1*avg_mag
        mask = mask.astype(np.uint8) * 255

        return mask