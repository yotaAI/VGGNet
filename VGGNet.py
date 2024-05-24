import torch
import os
import torch.nn as nn
from torchvision import models
from torchsummary import summary


vgg_a_architecture = [
	('conv',  64, (3,3), 1, 'same'), #[layer,out_channel,kernel_size,stride,padding]
	'maxpool',
	('conv', 128, (3,3), 1, 'same'),
	'maxpool',
	('conv', 256, (3,3), 1, 'same'),
	('conv', 256, (3,3), 1, 'same'),
	'maxpool',
	('conv', 512, (3,3), 1, 'same'),
	('conv', 512, (3,3), 1, 'same'),
	'maxpool',
	('conv', 512, (3,3), 1, 'same'),
	('conv', 512, (3,3), 1, 'same'),
	'maxpool'

]

class VGG(nn.Module):
	def __init__(self,in_channel,architecture):
		super().__init__()
		self.conv_layers = nn.Sequential()
		self.in_channel=in_channel

		for layer in architecture:
			if type(layer)==tuple:
				name,out_channel,kernel,stride,padding=layer
				
				self.conv_layers.append(nn.Conv2d(in_channel,out_channel,kernel,stride,padding))

				in_channel=out_channel

			elif type(layer)==str:
				self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

		self.fc = nn.Sequential(
			nn.Flatten(),
			nn.Linear(25088,4096),
			nn.Linear(4096,4096),
			nn.Linear(4096,1000),
			nn.ReLU(),
			)

	def forward(self,x):
		return self.fc(self.conv_layers(x))


vgg11 = VGG(3,vgg_a_architecture)

print(summary(vgg11,(3,224,224)))	