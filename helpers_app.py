# -*- coding: utf-8 -*-
"""
Helper funtions


"""


#Imports
import os
import numpy as np
import cv2                
import matplotlib.pyplot as plt                        
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
# Set PIL to be tolerant of image files that are truncated.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True                              


#Humans
#-------
# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
 

#Dogs (pretrained VGG16 model)
#-----------------------------
def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    # check if CUDA is available
    use_cuda = torch.cuda.is_available()

    
    # define VGG16 model
    VGG16 = models.vgg16(pretrained=True)
	
    if use_cuda:
        VGG16 = VGG16.cuda()


    img = Image.open(img_path)

    test_transforms = transforms.Compose([transforms.Resize((224,224),interpolation=Image.NEAREST),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])

    image_tensor = test_transforms(img)
    input = image_tensor.unsqueeze_(0)
    
    if use_cuda:
        input = input.cuda()

    VGG16.eval()
    output = VGG16(input)
    out_cpu = output.cpu()
    index = np.argmax(np.exp(out_cpu.detach().numpy()))
    
    return index    


#Dog detector
#------------

def dog_detector(img_path):
    pred = VGG16_predict(img_path)
    testlist = list(range(151, 269))
    return True if pred in testlist else False 
    

