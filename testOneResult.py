import torch
import argparse
import torchvision.transforms as transforms
from network import SRNDeblurNet
from data import TestDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import load_model,set_requires_grad,compute_psnr
from time import time
from metric import PSNR,SSIM
import os
from PIL import Image
from skimage.io import imsave
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-resume',default='./')
    parser.add_argument('-input_list')
    parser.add_argument('-output_dir')
    parser.add_argument('--output_list',type=str,default='./output.list')
    parser.add_argument('--preserve_dir_layer',type=int,default=0)
    parser.add_argument('--batch_size' , type=int,default=1)
    parser.add_argument('--resume_epoch',default=None)
    return parser.parse_args()

if __name__ == '__main__' :
    p = PSNR()
    s = SSIM()
    to_tensor = transforms.ToTensor()
    args = parse_args()
    img_list = open('test.list', 'r').read().strip().split('\n')
    print(img_list)
    dataset = TestDataset(img_list)
    dataloader = DataLoader( dataset , batch_size = args.batch_size , shuffle = False , drop_last = False , num_workers = 8 , pin_memory = True )
    print(len(dataloader))
    net = SRNDeblurNet().cuda()
    set_requires_grad(net,False)
    last_epoch = load_model( net , args.resume , epoch = 1999  ) 
    batch = []
    for b in dataloader:
        batch = b
    print(batch)
    for k,v in batch.items():
        batch[k] = v.cuda()
    y, _ , _ = net( batch['img256'] , batch['img128'] , batch['img64'] )
    y.detach_() 
    y = ((y.clamp(-1,1) + 1.0) / 2.0 * 255.999).byte()
    y = y.permute( 0 , 2 , 3 , 1   ).cpu().numpy()#NHWC
    y = y.squeeze()
    sharpe = img_list[0].split(" ")[1]
    print(sharpe)
    simg = to_tensor(Image.open(sharpe))
    ty = to_tensor(y)
    psnritem = p(ty,simg)
    ssimitem = s(ty,simg)
    print("Psnr：",psnritem,"  ssim：",ssimitem)
    imsave( './result.jpg' , y)