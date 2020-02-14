import shutil, os
from PIL import Image
from torchvision import transforms
input_size=224
data_transforms = {
    'train': transforms.Compose([
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

ind=0
error_count=0
for root,dirs,files in os.walk('./data'):
    for file in files:
        f=os.path.join(root,file)
        ind+=1
        
        if ind%1000==0:
            print(ind,'-------------------------------------')
        try:
            img=Image.open(f).convert('RGB')
            if not list(data_transforms['train'](img).size())==[3, 224, 224]:s
                print('error')
                shutil.move(f,'./error_files/'+file)
        except Exception as e:
            error_count+=1
            shutil.move(f,'./error_files/'+f.split('/')[3]+'/'+file)
            print(e)
