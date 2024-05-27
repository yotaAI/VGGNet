import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os,sys
import numpy as np

class FlowerDataset(Dataset):
	def __init__(self,dataset_path,input_shape,labels_map,scale=256,num_class=5):
		super().__init__()
		self.input_shape=input_shape
		self.scale=scale
		self.num_class=num_class
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
		clas = F.one_hot(torch.tensor(clas), num_classes=self.num_class).to(torch.float32)

		return im,clas

class ImageNetDataset(Dataset):
	def __init__(self,dataset_path,input_shap,scale,num_classes,dadtaset_type='train'):
		super().__init__()
		self.input_shape=input_shape
		self.scale=scale
		self.num_classes=num_classes
		synsets_path="synsets.txt"
		labels_path="labels.txt"

		with open(os.path.join(dataset_path,synsets_path),'r') as f:
			synsets = [i.strip() for i in f.readlines()]

		with open(os.path.join(dataset_path,labels_path),'r') as f:
			label = [i.strip().split(":") for i in f.readlines()]
			labels = [[synsets[n],int(ids)-1,clss] for n,(ids,clss) in enumerate(label)]
			self.map = dict(label)

		dataset = []
		for (folder,ids,clss) in labels:
			dataset +=[[os.path.join(dataset_path,'train',folder,file),ids,clss] for file in os.listdir(os.path.join(dataset_path,'train',folder))]

		self.df = pd.DataFrame(dataset)
		self.df.columns = ["File",'ID','Class']
	def __len__(self):
		return len(self.df)

	def __getitem__(self,index):
		src = self.df.iloc[index]
		img = cv2.imread(src.File)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RBG)
		img = cv2.resize(img,(224,224))
		img = img.transpose((2,0,1))
		img = img / self.scale
		img = torch.from_numpy(img).to(torch.float32)
		clss = F.one_hot(torch.tensor(src.ID),num_classes=self.num_classes).to(torch.float32)

		return (img,clss)


	def getMap(self):
		return self.maps