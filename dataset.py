import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os,sys
import numpy as np

class FlowerDataset(Dataset):
	def __init__(self,dataset_path,input_shape,labels_map,scale=256):
		super().__init__()
		self.input_shape=input_shape
		self.scale=scale
		images = []
		for key in labels_map.keys():
			for image in os.listdir(os.path.join(dataset_path,labels_map[key])):
				images.append((os.path.join(dataset_path,labels_map[key],image),key))
		self.images = pd.DataFrame(images)
		self.images = self.images.rename(columns={0:'image',1:'classes'})
	def __len__(self):
		return len(self.images)

	def __getitem__(self,idx):
		img = self.images.iloc[idx].image
		clas = self.images.iloc[idx].classes
		im = cv2.imread(img)
		im = cv2.resize(im,self.input_shape)
		im = np.transpose(im,(2,0,1)).astype(np.float32)
		im = im/self.scale
		# cv2.imshow("Image",im)
		# cv2.waitKey(0)

		return im,clas