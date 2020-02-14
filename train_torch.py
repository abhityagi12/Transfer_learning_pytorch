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
import pickle
import cv2 
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb


best_path='./model/best_val_acc_1.pth'
data_dir = "./data/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 2

batch_size = 128

num_epochs = 50

feature_extract = True

get_label =lambda x: 0 if x=='safe' else 1

class BloodDataset(Dataset):
    def __init__(self, data_root,data_transform=None):
        self.imgs = []
        self.labels=[]
        self.data_transform=data_transform
        for root,dirs,files in os.walk(data_root):
            for file in files:
                f=os.path.join(root,file)
                self.labels.append(f.split('/')[3])
                self.imgs.append(f)

    def extract_colour_features(self,img): #add mask to blood region in image
        # img=cv2.imread(f)
        # try:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # except:
        #     img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    #         r, g, b = cv2.split(img)
    #         fig = plt.figure()
    #         axis = fig.add_subplot(1, 1, 1, projection="3d")
        img = np.array(img) 
# Convert RGB to BGR 
        img = img[:, :, ::-1].copy() 
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
        norm = colors.Normalize(vmin=-1.,vmax=1.)
        norm.autoscale(pixel_colors)
        pixel_colors = norm(pixel_colors).tolist()
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_red = np.array([0,120,70])
        upper_red = np.array([10,255,255])
        mask1 = cv2.inRange(hsv_img, lower_red, upper_red)
        lower_red = np.array([170,120,70])
        upper_red = np.array([180,255,255])
        mask2 = cv2.inRange(hsv_img,lower_red,upper_red)
        mask = mask1+mask2
        mask.resize(1,224,224)
        return torch.tensor(mask,dtype=torch.float32)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if self.data_transform:
            img=Image.open(self.imgs[idx]).convert('RGB')
            first_tensor=self.data_transform(img)
            second_tensor=self.extract_colour_features(img)
            x=torch.cat((first_tensor, second_tensor), 0)
            return self.data_transform(Image.open(self.x)),torch.tensor(get_label(self.labels[idx]))
        else:
            Image.open(self.imgs[idx]),torch.tensor(get_label(self.labels[idx]))


input_size=224
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((input_size,input_size)),        
        transforms.RandomRotation([-30, 30]),
        transforms.Resize((input_size,input_size)),
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
train_dataset=BloodDataset('./data/train/',data_transforms['train'])
valid_dataset=BloodDataset('./data/valid/',data_transforms['val'])

print('Training Dataset Length:',len(train_dataset))
print('Validation Dataset Length:',len(valid_dataset))

train_dl=DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dl=DataLoader(valid_dataset, batch_size=64, shuffle=True)

data_loaders={'train':train_dl,'val':valid_dl}

model = models.resnet50(pretrained=True)
#adding Convolutional layer to make the model take 4 channel input
model.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False) 
model.fc = torch.nn.Linear(512,2) #making a 2-feature linear layer fo binary classification  
# print(model)
model.cuda();

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, log_pred_prob_onehot, target):
        pred_prob_oh = torch.exp(log_pred_prob_onehot)
        pt = Variable(pred_prob_oh.data.gather(1, target.data.view(-1, 1)), requires_grad=True)
        modulator = (1 - pt) ** self.gamma
        mce = modulator * (-torch.log(pt))

        return mce.mean()

criterion=FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def train_model(model, dataloaders, criterion, optimizer, num_epochs=30, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # print('Sailabh')
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
#                 print('1st batch')
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), best_path)
            if phase == 'val':
                val_acc_history.append(epoch_acc)


            if((epoch%5)==0):
                torch.save(model.state_dict(), './resnet_50/till_'+str(epoch)+'_epoch.pth')


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


model, val_acc_history=train_model(model,data_loaders, criterion, optimizer, num_epochs=30, is_inception=False)


torch.save(model.state_dict(), 'model/last_epoch1.pth')

with open('resnet_50/model1.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('resnet_50/val_acc_hist.pkl', 'wb') as f:
    pickle.dump(val_acc_history, f)

