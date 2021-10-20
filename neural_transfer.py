from __future__ import print_function
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import torchvision
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path,transform=None,max_size=None,shape=None):
    image=Image.open(image_path)

    if max_size:
        scale=max_size/max(image.size)
        size=np.array(image.size)*scale
        image=image.resize(size.astype(int),Image.ANTIALIAS)

    if shape:
        image=image.resize(shape,Image.ANTIALIAS)

    if transform:
        image=transform(image).unsqueeze(0)

    return image

class VGGNet(nn.Module):
    def __init__(self,type=1):
        super().__init__()
        self.vgg=models.vgg19(pretrained=True).features
        if (type=="content") or (type==0):
            self.select=["10"]
        else:
            self.select = ["0", "5", "10", "19", "28"]

    def forward(self,x):
        features=[]
        for name,layer in self.vgg._modules.items():
            x=layer(x)
            if name in self.select:
                features.append(x)
        return features

def main():
    mean=(0.485,0.456,0.406)
    std=(0.229,0.224,0.225)
    transform=transforms.Compose([transforms.ToTensor()
                                  ,transforms.Normalize(mean=mean,std=std)])
    content=load_image(r"G:\深度学习代码\torch实现神经网络风格迁移\content.jpg",transform,max_size=400)
    style=load_image(r"G:\深度学习代码\torch实现神经网络风格迁移\style.jpg",transform,shape=(content.size(2),content.size(3)))
    target=content.clone().requires_grad_(True)
    vgg_content=VGGNet(0).eval()
    vgg_style=VGGNet().eval()
    content_feature=vgg_content(content)
    style_feature=vgg_style(content)
    loss_func=nn.MSELoss()
    optim=torch.optim.Adam([target],lr=0.01)
    for step in range(200):
        target_content=vgg_content(target)
        target_style=vgg_style(target)
        content_loss=loss_func(target_content[0],content_feature[0])
        style_loss=0
        for f1,f2 in zip(style_feature,target_style):
            n,c,h,w=f1.size()
            f1=f1.view(n*c,h*w)
            f2=f2.view(n*c,h*w)
            f1_gram_mat=torch.mm(f1,f1.t())
            f2_gram_mat=torch.mm(f2,f2.t())
            style_loss+=loss_func(f1,f2) / (n*c*h*w)
        loss=content_loss+style_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (step+1) % 50 == 0 :
            print("step :[{}/{}],content_loss:{:.4f},style_loss:{:.4f}".format(step+1,200,content_loss.item(),style_loss.item()))
            img=target.squeeze().detach()
            img_tensor=torch.stack([img[i]*std[i]+mean[i] for i in range(3)]).clip(0,1)
            plt.imshow(img_tensor.numpy().transpose(1,2,0))
            torchvision.utils.save_image(img_tensor,"output-{}.jpg".format(step+1))



