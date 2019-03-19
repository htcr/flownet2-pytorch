import torch
import numpy as np
import argparse
import cv2
import cvbase as cvb

from models import FlowNet2#the path is depended on where you create this module
from utils.frame_utils import read_gen#the path is depended on where you create this module 


def block_img_size(ori_size, block_size):
    ori_h, ori_w = ori_size
    new_h, new_w = ori_h // block_size * block_size, ori_w // block_size * block_size
    return (new_h, new_w)

def limit_img_size(ori_size, max_edge):
    ori_h, ori_w = ori_size
    ori_max = max(ori_h, ori_w)
    scale = min(max_edge / ori_max, 1.0)
    new_h, new_w = int(scale * ori_h), int(scale * ori_w)
    return (new_h, new_w)
    
def resize_img(img, new_size):
    new_h, new_w = new_size
    new_img = cv2.resize(img, (new_w, new_h))
    return new_img

def flownet2_preprocess(img):
    ori_size = img.shape[:2]
    new_size = limit_img_size(ori_size, 1024)
    new_size = block_img_size(new_size, 64)
    new_img = resize_img(img, new_size)
    return new_img


if __name__ == '__main__':
    #obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    #initial a Net
    net = FlowNet2(args).cuda()
    #load the state_dict
    dict = torch.load("FlowNet2_checkpoint.pth.tar")
    net.load_state_dict(dict["state_dict"])
    
    #load the image pair, you can find this operation in dataset.py
    pim1 = read_gen("data/test_fg.png")
    pim2 = read_gen("data/test_bg.png")

    pim1 = flownet2_preprocess(pim1)
    pim2 = flownet2_preprocess(pim2)

    images = [pim1, pim2]
    images = np.array(images).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

    #process the image pair to obtian the flow 
    result = net(im).squeeze()

    
    #save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project 
    def writeFlow(name, flow):
        f = open(name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()

    data = result.data.cpu().numpy().transpose(1, 2, 0)
    cvb.show_flow(data)
    writeFlow("test_out.flo", data)