from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import os
import utils
import dataset

from dict_trie import Trie
from editdistance import eval
import operator

import torch.nn as nn
import copy
import LaSA as am
import string
from nltk.metrics import edit_distance
import torchvision
import cv2
import pdb
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#CUDA_LAUNCH_BLOCKING=1

parser = argparse.ArgumentParser()
parser.add_argument('--trainRoot', required=True, help='path to dataset')
parser.add_argument('--valRoot', required=True, help='path to dataset')
parser.add_argument('--lan', required=True, help='language to train on')
parser.add_argument('--arch',required=True, help='select one of these - crnn or starnet')
parser.add_argument('--charlist',required=True, help='path to the character list')
parser.add_argument('--dict',required=True, help='path to the dictionary (total words)')
parser.add_argument('--finetune', required=False, default=False, help='finetune on real data')
parser.add_argument('--savedir', default='tensorboard_runs', help='where to store the tensorboard logs')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')#32
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--alphabet')
parser.add_argument('--expr_dir', default='output_results', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=1000, help='Interval to be displayed')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, not used by adadelta')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
parser.add_argument('--deal_with_lossnan', action='store_true',help='whether to replace all nan/inf in gradients to zero')

opt = parser.parse_args()

if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
if torch.cuda.is_available() and opt.cuda:
    print('Nothing wrong with cuda')

train_dataset = dataset.loadDataset(root=opt.trainRoot)
test_dataset = dataset.loadDataset(root=opt.valRoot)
assert train_dataset

if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    sampler=sampler,shuffle=True, num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=False))

charlist_fname = opt.charlist 
opt.alphabet = open(charlist_fname,'r').readlines()

converter = utils.AttnLabelConverter(opt.alphabet)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

dictionary = open(opt.dict).read().split("\n")
trie = Trie(dictionary)

nclass = len(converter.character)
nc = 1

# custom weights initialization called on model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

model = am.Attn_model(opt.imgH, opt.imgW, nc, nclass, opt.nh)

model.apply(weights_init)
model_dict = model.state_dict()
if opt.pretrained != '':
    model_path = opt.pretrained
    checkpoint = torch.load(model_path)
    for key in list(checkpoint.keys()):
        if 'module.' in key:
            checkpoint[key.replace('module.', '')] = checkpoint[key]
            del checkpoint[key]
    print('loading pretrained model from %s' % opt.pretrained)
    pretrained_dict = {k: v for k, v in checkpoint.items() if checkpoint[k].size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.LongTensor(opt.batchSize * 5,10)
length = torch.LongTensor(opt.batchSize,10)

#opt.cuda = False
if opt.cuda:
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(opt.ngpu))
    image = image.cuda()
    text = text.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(model.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(model.parameters())
else:
    optimizer = optim.RMSprop(model.parameters(), lr=opt.lr)

def val(net, test_dataset, criterion, max_iter=100):
    print('Start val')
    #new
    batch_max_length = 25

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    data_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers),
        collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=False))
     
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()
    norm_ED = 0
    #max_iter = min(max_iter, len(data_loader))
    max_iter = len(data_loader)
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        
        # For max length prediction
        length_for_pred = torch.IntTensor([batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).to(device)

        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        
        utils.loadData(text, t)
        utils.loadData(length, l)

        # align with Attention.forward
        preds, cost = model(image, text_for_pred, distances=None, is_train=False)
        preds = preds[:, :t.shape[1] - 1, :]
        
        target = t[:, 1:]  # without [GO] Symbol

        _, preds = preds.max(2)
        
        pred_texts = converter.decode(preds, length_for_pred)
        cpu_texts = converter.decode(t[:, 1:], l)
        
        for pred, target in zip(pred_texts, cpu_texts):
            target = target.strip()
            gt = target.strip()
            gt = gt[:gt.find('[s]')]
            pred = pred[:pred.find('[s]')]

            if pred == gt:
                n_correct += 1
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)
    
    for pred, gt in zip(pred_texts, cpu_texts):
        pred = pred[:pred.find('[s]')]
        pred = ''.join([i for i in pred])
        gt = gt[:gt.find('[s]')]
        gt = ''.join([i for i in gt])
        
        pred = pred.replace('\n','')
        gt = gt.replace('\n','')

        print('pred: %-20s, gt: %-20s' % (pred.strip(), gt.strip()))
    print("Samples Correctly recognised = " + str(n_correct))
    accuracy = n_correct / float(max_iter * opt.batchSize)
    crr = norm_ED / float(max_iter * opt.batchSize)
    lossval = loss_avg.val()
    print('accuracy(wrr): %f, crr: %f' % (accuracy, crr))
    return accuracy, crr

def give_candidates(target, how_many=10):
    candidates_list = list(trie.all_levenshtein(target, 3))
    candidates_list.append(target)
    candidates_list = list(set(candidates_list))
    candidates = {}
    for candidate in candidates_list:
        candidates[candidate] = eval(target, candidate)
    candidates = sorted(candidates.items(), key=operator.itemgetter(1))
    dist_sharp = eval("###", target)
    while len(candidates) < 10:
        candidates.append(("###", dist_sharp))
    candidates = candidates[:how_many]
    return candidates

def give_candidates_for_batches(target_list):
    batch_size = len(target_list)
    candidate_list_raw = []
    candidate_list = []
    edit_distance_list = []

    #getting cadidate words (default 10) and corresponding edit distance for each word
    for i in range(batch_size):
        candidate_list_raw.append(give_candidates(target_list[i]))

    #organising candidate words and its edit distance
    number_of_candidates = len(candidate_list_raw[0])
    for i in range(number_of_candidates):
        candidate_words = []
        edit_distance = []
        for j in range(batch_size):
            candidate_words.append(candidate_list_raw[j][i][0])
            edit_distance.append(candidate_list_raw[j][i][1])
        candidate_list.append(candidate_words)
        edit_distance_list.append(edit_distance)

    return candidate_list, edit_distance_list

def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    
    #dicitionary
    candidate_list_batches, edit_distance_batches = give_candidates_for_batches(cpu_texts)
    
    t = []
    l = []
    for candidate in candidate_list_batches:
        temp_t, temp_l = converter.encode(candidate)
        t.append(temp_t)
        l.append(temp_l)

    optimizer.zero_grad()
    #print(text.size())
        
    pred, cost = model(image, t, distances=edit_distance_batches, is_train=True)

    model.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

losses_per_epoch = []
acc_per_epoch = []
best_acc = 0.0
is_best = 0
l_avg = utils.averager()

writer = SummaryWriter('{0}/runs_{1}_{2}'.format(opt.savedir,opt.lan,opt.arch))

for epoch in range(opt.nepoch):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader)-1:
        for p in model.parameters():
            p.requires_grad = True
        model.train()

        cost = trainBatch(model, criterion, optimizer)
        loss_avg.add(cost)
        l_avg.add(cost)
        i += 1
        writer.add_scalar('Loss/train', loss_avg.val(), epoch*len(train_loader) + i)

        if i % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.nepoch, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if i % opt.valInterval == 0:
            acc, crr = val(model, test_dataset, criterion)
            # pdb.set_trace()	
            #writer.add_scalar('Loss/val', lossval, (epoch*len(train_loader)+i)/opt.valInterval)
            writer.add_scalar('Acc-WRR/accuracy_val', acc, (epoch*len(train_loader)+i)/opt.valInterval)
            writer.add_scalar('CRR/char_val', crr, (epoch*len(train_loader)+i)/opt.valInterval)

            is_best = acc >= best_acc
            if is_best:
                best_acc = acc
                filename = '{0}/best_model_{2}_{1}.pth'.format(opt.expr_dir, opt.arch, opt.lan)
                torch.save(model.state_dict(), filename)
                is_best = 0