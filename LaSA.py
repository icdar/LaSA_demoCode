import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.resnet import ResNet_FeatureExtractor
from modules.attention import Attention
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Attn_model(nn.Module):

    def __init__(self, imgH, imgW, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(Attn_model, self).__init__()
        
        input_channel = nc
        output_channel = 512
        hidden_size = nh
        class_size = nclass
        
        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        
        self.FeatureExtraction_output = 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        self.Attention = Attention(self.FeatureExtraction_output, hidden_size, class_size)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
        self.divergence = nn.KLDivLoss(reduction="batchmean").to(device)
        self.criterion = self.criterion.cuda()
        self.divergence = self.divergence.cuda()
    
    def forward(self, input, text, distances, is_train=True):

        visual_feature = self.FeatureExtraction(input)
        batch_size, c, h, w = visual_feature.size()
        #print(visual_feature.size())
        
        visual_feature = visual_feature.permute(0,2,3,1)
        visual_feature = visual_feature.view(batch_size,h*w,c)
        
        no_of_candidates = len(text)
        ce_loss_batch = torch.zeros([no_of_candidates,batch_size], dtype=torch.float, device=device)
        dict_loss = torch.zeros([batch_size], dtype=torch.float, device=device)
        cost = torch.zeros([batch_size], dtype=torch.float, device=device)
        ce_loss_per_image = torch.zeros([no_of_candidates], dtype=torch.float, device=device)
        distances_per_image = torch.zeros([no_of_candidates], dtype=torch.float, device=device)
        
        if is_train:
            assert distances != None, 'Distance values of candidate words are not provided.'
            distances = torch.Tensor(distances)
            
            preds = self.Attention(visual_feature.contiguous(), text[0][:, :-1], is_train)
            #for each candidate word, corresponding prediction is recorded
            for i in range(no_of_candidates):
                target = text[i][:, 1:] #without [GO] symbol

                for batch in range(batch_size):
                    cost[batch] = self.criterion(preds[batch],target[batch].contiguous())
                
                if i == 0:
                    total_cost = cost
                
                cost = 1/cost
                ce_loss_batch[i] = cost

            for batch in range(batch_size):
                ce_loss_per_image = ce_loss_batch[:,batch]
                ce_loss_per_image = nn.functional.log_softmax(ce_loss_per_image, dim=0)
                
                distances_per_image = 1/(distances[:,batch]+0.1)
                distances_per_image = nn.functional.softmax(distances_per_image, dim=0)

                ce_loss_per_image = torch.unsqueeze(ce_loss_per_image, dim=0).to(device)
                distances_per_image = torch.unsqueeze(distances_per_image, dim=0).to(device)
                dict_loss[batch] = self.divergence(ce_loss_per_image, distances_per_image)
            
            total_cost += dict_loss
            total_cost = sum(total_cost)/batch_size
            return preds, total_cost
        
        else:
            #testing and validating
            preds = self.Attention(visual_feature.contiguous(), text, is_train)
            return preds, None
    

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0   # replace all nan/inf in gradients to zero