import torch
from torch.autograd import Variable
import utils
from PIL import Image
import os
from nltk.metrics.distance import edit_distance
import argparse
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
#import lmdb
import sys
from PIL import Image
import numpy as np
import io
import torch
from torch.autograd import Variable
import utils
from PIL import Image
import model as am
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Model to test on')
parser.add_argument('--test_data', required=True, help='path to testing dataset')
parser.add_argument('--display_result', required=True, help='text file to display result')
parser.add_argument('--lexicon', required=True, help='path to the lexicon file')
opt = parser.parse_args()
print(opt)

class load_testDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        self.imagePathList = []
        self.labelList = []
        with open(os.path.join(root,'label.txt'), 'r', encoding='utf-8') as file:
            for item in file.readlines():
                image, word = item.rstrip().split()	
                #image = image.split('/')	
                image = os.path.join(root, image)
                self.imagePathList.append(image)
                self.labelList.append(word)
        
        self.nSamples = len(self.labelList)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= self.nSamples, 'index range error'

        try:
            img_name = self.imagePathList[index]
            img = Image.open(self.imagePathList[index])
            img = img.convert('L')
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        if self.transform is not None:
            img = self.transform(img)

        label = self.labelList[index]

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img_name, img, label)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)

        #new code img -> 100*100
        if self.size[0] != self.size[1]:
            img_PIL = transforms.ToPILImage()
            img = img_PIL(img)
            new_img = Image.new(img.mode,(100,100))
            new_img.paste(img)
            new_img = self.toTensor(new_img)
            img = new_img

        img.sub_(0.5).div(0.5)
        return img

class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        img_name, images, labels = zip(*batch)

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

        images = torch.cat([t.unsqueeze(0) for t in temp], 0)

        return img_name, images, labels

model_path = opt.model
lexicon_filename = opt.lexicon
p = open(lexicon_filename, 'r').readlines()
alphabet = p

converter = utils.AttnLabelConverter(alphabet)

nclass = len(converter.character)
print(nclass)

model = am.Attn_model(32, 100, 1, nclass, 256)

if torch.cuda.is_available():
    model = model.cuda()

model.apply(weights_init)
model_dict = model.state_dict()
checkpoint = torch.load(model_path)

model_parameters = filter(lambda p:p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
#print('Total parameters:', sum(p.numel() for p in model.parameters()))
print('Trainable parameters:', params)


for key in list(checkpoint.keys()):
  if 'module.' in key:
    checkpoint[key.replace('module.', '')] = checkpoint[key]
    del checkpoint[key]

model_dict.update(checkpoint)
model.load_state_dict(checkpoint)

vocab = []
for i in alphabet:
  vocab.append(i.strip())

image = torch.FloatTensor(32, 3, 32, 32)
text = torch.LongTensor(32 * 5)
length = torch.LongTensor(32)

model = model.cuda()
model = torch.nn.DataParallel(model, device_ids=range(1))
image = image.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)
loss_avg = utils.averager()

def test_batch(model, test_dataset):
    
    print('Start test')
    #new
    batchSize = 32
    batch_max_length = 25
    
    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    data_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, batch_size=batchSize, num_workers=1,
        collate_fn=alignCollate(imgH=32, imgW=100, keep_ratio=False))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()
    norm_ED = 0
    #max_iter = min(max_iter, len(data_loader))
    max_iter = len(data_loader)
    result = []
    for i in range(max_iter):
        data = next(val_iter)
        i += 1
        (img_names, cpu_images, cpu_texts) = data
        batch_size = cpu_images.size(0)
        
        # For max length prediction
        length_for_pred = torch.IntTensor([batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).to(device)


        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = model(image, text_for_pred, is_train=False)
        preds = preds[:, :t.shape[1] - 1, :]
        target = t[:, 1:]  # without [GO] Symbol

        #loss_avg.add(cost)

        _, preds = preds.max(2)
        sim_preds = converter.decode(preds, length_for_pred)
        cpu_texts = converter.decode(t[:, 1:], l)
        flag = 0
        for img_name, pred, target in zip(img_names, sim_preds, cpu_texts):
            target = target.strip()
            gt = target.strip()
            gt = gt[:gt.find('[s]')]
            pred = pred[:pred.find('[s]')]

            if pred == gt:
                n_correct += 1
                flag = 1
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)
            pred = ''.join([i for i in pred])
            gt = ''.join([i for i in gt])
            pred = pred.replace('\n','')
            gt = gt.replace('\n','')
            result.append([img_name,gt,pred,flag])
            flag=0

    for pred, gt in zip(sim_preds, cpu_texts):
        pred = pred[:pred.find('[s]')]
        pred = ''.join([i for i in pred])
        gt = gt[:gt.find('[s]')]
        gt = ''.join([i for i in gt])
        
        pred = pred.replace('\n','')
        gt = gt.replace('\n','')

        print('pred: %-20s, gt: %-20s' % (pred.strip(), gt.strip()))
    print("Samples Correctly recognised = " + str(n_correct))
    accuracy = n_correct / float(max_iter * batchSize)
    crr = norm_ED / float(max_iter * batchSize)
    return accuracy, crr, result

test_dataset = load_testDataset(opt.test_data) 

wrr, crr, result = test_batch(model,test_dataset)
print("FINAL RESULTS")
print("Word Recognition Rate :" + str(wrr))
print("Character Recognition Rate :" + str(crr))

with open(opt.display_result,'w') as fp:
    fp.write("\n".join(str(item).strip() for item in result))

print("Results stored in a text file")