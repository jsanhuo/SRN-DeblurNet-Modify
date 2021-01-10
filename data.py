import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_list, crop_size=(256, 256)):
        super(type(self), self).__init__()
        self.img_list = img_list
        # 图片应被修建的尺寸
        self.crop_size = crop_size
        # numpy转tensor的方法
        self.to_tensor = transforms.ToTensor()

    # 裁剪图片并且将图片转化为tensor
    def crop_resize_totensor(self, img, crop_location):
        img256 = img.crop(crop_location)
        # Image.NEAREST ：低质量
        # Image.BILINEAR：双线性
        # Image.BICUBIC ：三次样条插值
        # Image.ANTIALIAS：高质量
        # resize 改变图像大小
        img128 = img256.resize((self.crop_size[0] // 2, self.crop_size[1] // 2), resample=Image.BILINEAR)
        img64 = img128.resize((self.crop_size[0] // 4, self.crop_size[1] // 4), resample=Image.BILINEAR)
        return self.to_tensor(img256), self.to_tensor(img128), self.to_tensor(img64)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # filename processing
        # 获得模糊图片的地址
        blurry_img_name = self.img_list[idx].split(' ')[-2]
        # 获得清晰图像的地址
        clear_img_name = self.img_list[idx].split(' ')[-1]
        # 打开清晰和模糊图像
        blurry_img = Image.open(blurry_img_name)
        clear_img = Image.open(clear_img_name)
        # 如果清晰和模糊图像尺寸不一样，说明数据有问题，那么就退出
        assert blurry_img.size == clear_img.size
        # crop_size[0] = 256,crop_size[1] = 256,
        # 随机裁剪图片
        crop_left = int(np.floor(np.random.uniform(0, blurry_img.size[0] - self.crop_size[0] + 1)))
        crop_top = int(np.floor(np.random.uniform(0, blurry_img.size[1] - self.crop_size[1] + 1)))
        # left，up, right, below
        crop_location = (crop_left, crop_top, crop_left + self.crop_size[0], crop_top + self.crop_size[1])
        # 将模糊图像和清晰图像都转化为tensor
        img256, img128, img64 = self.crop_resize_totensor(blurry_img, crop_location)
        label256, label128, label64 = self.crop_resize_totensor(clear_img, crop_location)
        batch = {'img256': img256, 'img128': img128, 'img64': img64, 'label256': label256, 'label128': label128,
                 'label64': label64}
        for k in batch:
            batch[k] = batch[k] * 2 - 1.0  # in range [-1,1]
        return batch


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_list):
        super(type(self), self).__init__()
        self.img_list = img_list
        self.to_tensor = transforms.ToTensor()

    def resize_totensor(self, img):
        img_size = img.size
        img256 = img
        img128 = img256.resize((img_size[0] // 2, img_size[1] // 2), resample=Image.BILINEAR)
        img64 = img128.resize((img_size[0] // 4, img_size[1] // 4), resample=Image.BILINEAR)
        return self.to_tensor(img256), self.to_tensor(img128), self.to_tensor(img64)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # filename processing
        blurry_img_name = self.img_list[idx].split(' ')[-2]
        clear_img_name = self.img_list[idx].split(' ')[-1]

        blurry_img = Image.open(blurry_img_name)
        clear_img = Image.open(clear_img_name)
        assert blurry_img.size == clear_img.size

        img256, img128, img64 = self.resize_totensor(blurry_img)
        label256 = self.to_tensor(clear_img)
        batch = {'img256': img256, 'img128': img128, 'img64': img64, 'label256': label256}
        for k in batch:
            batch[k] = batch[k] * 2 - 1.0  # in range [-1,1]
        return batch
