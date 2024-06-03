import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


import cv2
import pandas as pd
import numpy as np
import os


class ImageNetDataset(Dataset):
	def __init__(self,dataset_path='../dataset/imagenet/',input_shape=(224,224),scale=255,num_classes = 1000,dataset_type='train'):
		super().__init__()
		self.input_shape=input_shape
		self.scale=scale
		self.num_classes = num_classes
		synsets_path="synsets.txt"
		labels_path = "labels.txt"

		with open(os.path.join(dataset_path,synsets_path),'r') as f:
			synsets = [i.strip() for i in f.readlines()]

		with open(os.path.join(dataset_path,labels_path),'r') as f:
			label = [i.strip().split(':') for i in f.readlines()]
			labels = [[synsets[n],int(ids) - 1,cls] for n,(ids,cls) in enumerate(label)]
			self.map = dict(label)
		dataset = []
		for (folder,ids,cls) in labels:
			dataset +=[[os.path.join(dataset_path,'train',folder,file),ids,cls] for file in os.listdir(os.path.join(dataset_path,'train',folder))]

		df = pd.DataFrame(dataset)
		df.columns = ['File','ID','Class']

		X = df["File"].to_list()
		Y = df["ID"].to_list()
		X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.01,random_state=4)

		if dataset_type=='train':
			self.X = X_train
			self.y = y_train

		elif dataset_type =='test':
			self.X = X_test
			self.y = y_test

	def getMap(self):
		return self.map

	def __len__(self):
		return len(self.X)

	def __getitem__(self,index):
		X = self.X[index]
		y = self.y[index]
		img = cv2.imread(X)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		img = cv2.resize(img,(224,224))
		img = img.transpose((2,0,1))
		img = img /self.scale
		img = torch.from_numpy(img).to(torch.float32)

		return(img,torch.tensor(y))


if __name__ =='__main__':
	dataset = ImageNetDataset()
