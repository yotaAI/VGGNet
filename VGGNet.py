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

vgg_b_architecture = [
	('conv',  64, (3,3), 1, 'same'), #[layer,out_channel,kernel_size,stride,padding]
	('conv',  64, (3,3), 1, 'same'),
	'maxpool',
	('conv', 128, (3,3), 1, 'same'),
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

vgg_c_architecture = [
	('conv',  64, (3,3), 1, 'same'), #[layer,out_channel,kernel_size,stride,padding]
	('conv',  64, (3,3), 1, 'same'),
	'maxpool',
	('conv', 128, (3,3), 1, 'same'),
	('conv', 128, (3,3), 1, 'same'),
	'maxpool',
	('conv', 256, (3,3), 1, 'same'),
	('conv', 256, (3,3), 1, 'same'),
	('conv', 256, (3,3), 1, 'same'),
	'maxpool',
	('conv', 512, (3,3), 1, 'same'),
	('conv', 512, (3,3), 1, 'same'),
	('conv', 512, (3,3), 1, 'same'),
	'maxpool',
	('conv', 512, (3,3), 1, 'same'),
	('conv', 512, (3,3), 1, 'same'),
	('conv', 512, (3,3), 1, 'same'),
	'maxpool'

]

vgg_d_architecture = [
	('conv',  64, (3,3), 1, 'same'), #[layer,out_channel,kernel_size,stride,padding]
	('conv',  64, (3,3), 1, 'same'),
	'maxpool',
	('conv', 128, (3,3), 1, 'same'),
	('conv', 128, (3,3), 1, 'same'),
	'maxpool',
	('conv', 256, (3,3), 1, 'same'),
	('conv', 256, (3,3), 1, 'same'),
	('conv', 256, (3,3), 1, 'same'),
	('conv', 256, (3,3), 1, 'same'),
	'maxpool',
	('conv', 512, (3,3), 1, 'same'),
	('conv', 512, (3,3), 1, 'same'),
	('conv', 512, (3,3), 1, 'same'),
	('conv', 512, (3,3), 1, 'same'),
	'maxpool',
	('conv', 512, (3,3), 1, 'same'),
	('conv', 512, (3,3), 1, 'same'),
	('conv', 512, (3,3), 1, 'same'),
	('conv', 512, (3,3), 1, 'same'),
	'maxpool'

]

class VGG(nn.Module):
	def __init__(self,in_channel,num_classes,architecture):
		super().__init__()
		self.conv_layers = nn.Sequential()
		self.in_channel=in_channel

		for layer in architecture:
			if type(layer)==tuple:
				name,out_channel,kernel,stride,padding=layer
				
				self.conv_layers.append(nn.Conv2d(in_channel,out_channel,kernel,stride,padding))
				self.conv_layers.append(nn.ReLU())

				in_channel=out_channel

			elif type(layer)==str:
				self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

		self.fc = nn.Sequential(
			nn.Flatten(),
			nn.Linear(25088,4096),
			nn.Dropout(p=0.5),
			nn.Linear(4096,4096),
			nn.Dropout(p=0.5),
			nn.Linear(4096,num_classes),
			nn.Dropout(p=0.5),
			nn.Softmax(dim=1),

			)
		#Initializing weights with normal distribution.
		self.init_weights()

	def forward(self,x):
		return self.fc(self.conv_layers(x)) #[batch_size,num_classes]

	def init_weights(self):
		for layer in self.conv_layers:
			if isinstance(layer,nn.Conv2d):
				nn.init.normal_(layer.weight,std=0.1)
				nn.init.constant_(layer.bias,0)


if __name__=='__main__':
	vgg11 = VGG(3,1000,vgg_a_architecture)
	print(summary(vgg11,(3,224,224)))	
	print(vgg11(torch.rand(16,3,224,224)))