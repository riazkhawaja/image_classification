# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
from model import Net
import os
import PIL
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Image classification demo')
parser.add_argument('--image', help='the name of the image in the inputs folder', required=True)
parser.add_argument('--label', help='the actual label of the image', required=True)
args = vars(parser.parse_args())

MODEL_PATH = './models/test.pth'
net = Net()
net.load_state_dict(torch.load(MODEL_PATH))

# load image and normalize
image_dir = PIL.Image.open(os.path.join(os.getcwd(), "inputs", args['image']))

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(128, antialias=True), transforms.functional.rgb_to_grayscale])
normalized_image_tensor = transform(image_dir).unsqueeze(0) # set batch size to 1

labels = pd.read_csv("./labels/labels.csv")
classes = pd.read_csv("./labels/classes.csv")

outputs = net(normalized_image_tensor)
predicted = torch.argmax(outputs)

print("actual: " + args['label'])
print("predicted: " + classes.iloc[predicted.item(), 1])
