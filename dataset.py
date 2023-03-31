#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import sys
from PIL import Image
import numpy as np
import io
import os

class loadDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        
        self.imagePathList = []
        self.labelList = []
        with open(os.path.join(root,'label.txt'), 'r', encoding='utf-8') as file:
            for item in file.readlines():
                image, word = item.rstrip().split()	
                image = os.path.join(root, image)
                self.imagePathList.append(image)
                self.labelList.append(word)
        
        self.nSamples = len(self.labelList)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        #print(index)
        assert index <= len(self), 'index range error'

        try:
            img = Image.open(self.imagePathList[index]).convert('L')
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index+1]
            #return self.__getitem__(index+1)

        if self.transform is not None:
            img = self.transform(img)

        label = self.labelList[index]
        if len(label) > 25:
            return self[index+1]

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        #img.sub_(0.5).div_(0.5)
        
        #new code img -> 100*100
        if self.size[0] != self.size[1]:
            img_PIL = transforms.ToPILImage() 
            img = img_PIL(img)
            new_img = Image.new(img.mode,(100,100))
            new_img.paste(img)
            new_img = self.toTensor(new_img)
            img = new_img
        
        #new_img = ImageOps.pad(img,(100,100), color = [0,0,0], centering=(0,0))
        img.sub_(0.5).div(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        #new code
        hori_transform = resizeNormalize((imgW, imgH))
        verti_transform = resizeNormalize((imgH, imgW))
        square_transform = resizeNormalize((100,100))
        #images = [transform(image) for image in images]
        temp = []
        for image in images:
            w, h = image.size
            if abs(w-h) <= 20:
                temp.append(square_transform(image))
            else:
                if w >= h:
                    temp.append(hori_transform(image))
                else:
                    temp.append(verti_transform(image))
        
        #images = torch.cat([t.unsqueeze(0) for t in images], 0)
        images = torch.cat([t.unsqueeze(0) for t in temp], 0)

        return images, labels