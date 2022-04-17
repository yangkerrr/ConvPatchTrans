import os
import cv2
import pandas as pd
import numpy as np
import random
import paddle
from paddle.io import Dataset
from paddle.io import DataLoader
from paddle.io import Sampler
import paddle.vision.transforms as transforms
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class MyData(Dataset):
    def __init__(self, root_dir, list_txt, tolength, transf, aug=False):
        super(MyData, self).__init__()
        self.imglabels = pd.read_table(os.path.join(root_dir, list_txt), delim_whitespace=True, header=None)
        self.rootdir = root_dir
        self.tolength = tolength
        self.transf = transf
        self.aug = aug

    def __len__(self):
        return self.imglabels.shape[0]

    def image_process(self, image):
        #print("1",image.shape)
        if self.aug:
            image = random_augmentation(image)
            #print("2",image.shape)
        image = cv2.resize(image, (self.tolength, 32))
        #print("3",image.shape)
        return self.transf(image)



    def __getitem__(self, idx):    # Get through here every iteration when refreshing the dataloader
        img_name, label = self.imglabels.iloc[idx, :]
        img_name = img_name.replace('\\', '/')
        img_dir = os.path.join(self.rootdir, img_name)
        image = cv2.imread(img_dir, 1)
        #print(img_dir)
        image = self.image_process(image)
        #print("4",image.shape)
        return image, label - 1

def random_augmentation(image, allow_crop=True):
    f = ImageTransfer(image)
    seed = random.randint(0, 5)     # 0: original image used
    switcher = random.random() if allow_crop else 1.0
    if seed == 1:
        image = f.add_noise()
    elif seed == 2:
        image = f.change_contrast()
    elif seed == 3:
        image = f.change_hsv()
    elif seed >= 4:
        f1 = ImageTransfer(f.add_noise())
        f2 = ImageTransfer(f1.change_hsv())
        image = f2.change_contrast()
    if switcher < 0.4:
        fn = ImageTransfer(image)
        image = fn.slight_crop()
    elif switcher < 0.8:
        fn = ImageTransfer(image)
        image = fn.perspective_transform()
    return image


class ImageTransfer(object):
    """crop, add noise, change contrast, color jittering"""
    def __init__(self, image):
        """image: a ndarray with size [h, w, 3]"""
        self.image = image

    def slight_crop(self):
        h, w = self.image.shape[:2]
        k0 = 6 if w / h > 3 else 8
        k = random.randint(k0, 10) / 10
        ch, cw = int(h * 0.9), int(w * k)     # cropped h and w
        hs = random.randint(0, h - ch)      # started loc
        ws = random.randint(0, w - cw)
        return self.image[hs:hs+ch, ws:ws+cw]

    def add_noise(self):
        img = self.image * (np.random.rand(*self.image.shape) * 0.3 + 0.7)
        img = img.astype(np.uint8)
        return img

    def change_contrast(self):
        if random.random() < 0.5:
            k = random.randint(6, 9) / 10.0
        else:
            k = random.randint(11, 14) / 10.0
        b = 128 * (k - 1)
        img = self.image.astype(np.float)
        img = k * img - b
        img = np.maximum(img, 0)
        img = np.minimum(img, 255)
        img = img.astype(np.uint8)
        return img

    def perspective_transform(self):
        h, w = self.image.shape[:2]
        short = min(h, w)
        gate = int(short * 0.3)
        mrg = []
        for _ in range(8):
            mrg.append(random.randint(0, gate))
        pts1 = np.float32(
            [[mrg[0], mrg[1]], [w - 1 - mrg[2], mrg[3]], [mrg[4], h - 1 - mrg[5]], [w - 1 - mrg[6], h - 1 - mrg[7]]])
        pts2 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(self.image, M, (w, h))

    def change_hsv(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        s = random.random()
        def ch_h():
            dh = random.randint(2, 11) * random.randrange(-1, 2, 2)
            img[:, :, 0] = (img[:, :, 0] + dh) % 180
        def ch_s():
            ds = random.random() * 0.25 + 0.7
            img[:, :, 1] = ds * img[:, :, 1]
        def ch_v():
            dv = random.random() * 0.35 + 0.6
            img[:, :, 2] = dv * img[:, :, 2]
        if s < 0.25:
            ch_h()
        elif s < 0.50:
            ch_s()
        elif s < 0.75:
            ch_v()
        else:
            ch_h()
            ch_s()
            ch_v()
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def getloaders(root_dir, mid_dir, widths, txts, transf, shuffle=True, bs=32, aug=False):
    #mid_dir = os.path.join(root_dir, mid_dir)
    loaders = []
    for i in range(len(widths)):
        list_txt = os.path.join(mid_dir, txts[i])
        mydataset = MyData(root_dir, list_txt, widths[i], transf, aug=aug)
        loaders.append(DataLoader(mydataset, batch_size=bs, shuffle=shuffle, num_workers=3))
    return loaders

def loaders_default(istrainset, batchsize=16):
    # Get loaders
    root_dir = 'data'
    tmp = "TrainDataset_CVSI2015" if istrainset else "ValidationDataset_CVSI2015"
    root_dir = os.path.join(root_dir, tmp)
    mid_dir = "z_grp_ccx"
    widths = [32, 64, 128]
    txts = ["grp32.txt", "grp64.txt", "grp128.txt"]
    transf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return getloaders(root_dir, mid_dir, widths, txts, transf, shuffle=istrainset, bs=batchsize, aug=istrainset)


if __name__ == "__main__":
    root_dir = 'work/script/data/cvsi/TrainDataset_CVSI2015'
    mid_dir = "z_grp_ccx"
    widths = [32, 64, 128]
    txts = ["grp32.txt", "grp64.txt", "grp128.txt"]
    transf = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5,0.5))
    myloaders = getloaders(root_dir, mid_dir, widths, txts, transf, shuffle=True, bs=16, aug=False)
    dataiter = iter(myloaders[1])
    images, label = dataiter.next()