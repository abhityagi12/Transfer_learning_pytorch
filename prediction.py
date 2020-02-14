import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision
from torchvision import datasets, models, transforms,utils
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time
import copy

m=models.resnet50(num_classes=2,pretrained=None)
weight=torch.load('/media/myfiles2/ml/projects/user/abhinav/model/best_val_acc.pth')
m.load_state_dict(weight)
m.cuda()
m.eval();

input_size=224
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_transforms_pred = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
get_label =lambda x: 0 if x=='unsafe' else 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import shutil
softmax = torch.nn.Softmax(dim=1)


ind=0
s_t=time.time()
for root,dirs,files in os.walk('/media/myfiles2/ml/projects/user/abhinav/model/data/test/'):
    for file in files:
        if ind%100==0:
            print(ind,time.time()-s_t)
        ind+=1
        f=os.path.join(root,file)
        try:
#         act=get_label(f.split('/')[3])
            img=torch.unsqueeze(data_transforms_pred(Image.open(f)),0)
            img = img.to(device)
            pred=m(img)
            x=softmax(pred).tolist()[0][0]
            dst='/media/myfiles2/ml/projects/user/abhinav/model/tested/data/prediction/'+str(int(x*10))
            os.makedirs(dst,exist_ok=True)
            shutil.copy(f,dst+'/'+str(round(x,3))+','+f.split('/')[-1])
        except:
            print(f)
            shutil.move(f,'error_files/'+f.split('/')[-1])
