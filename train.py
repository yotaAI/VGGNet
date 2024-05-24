import torch
import torch.nn as nn
from torchsummary import summary
from tqdm import tqdm



from VGGNet import vgg_a_architecture,VGG
from dataset import FlowerDataset
from utils import EucladianLoss


if __name__=="__main__":
	# HyperParameter
	EPOCH=1
	BATCH_SIZE=4
	LR=0.01
	MOMENTUM=0.9
	dataset_path = "./dataset/flowers/"
	INPUT_SHAPE = (224,224)
	labels_map = {
		0 : "daisy",
		1 : "rose",
		2 : "tulip",
		3 : "dandelion",
		4 : "sunflower",
	}

	# Model Load
	vgg11 = VGG(in_channel=3,num_classes=5,architecture=vgg_a_architecture)
	# print(summary(vgg11,(3,224,224)))	
	# Loss Function
	eucladian_loss=EucladianLoss()
	# Optimizer
	optimizer = torch.optim.SGD(vgg11.parameters(),lr=LR,momentum=MOMENTUM)
	
	# DataSet
	flower_dataset = FlowerDataset(dataset_path,input_shape=INPUT_SHAPE,labels_map=labels_map)
	# Dataloader
	training_loader = torch.utils.data.DataLoader(flower_dataset,batch_size=BATCH_SIZE,shuffle=True)

	#Training Loop
	for epoch in range(EPOCH):
		current_loss = 0

		for i, data in enumerate(tqdm(training_loader)):
			inputs,labels = data

			optimizer.zero_grad()
			outputs = vgg11(inputs)

			loss = eucladian_loss(outputs,labels)

			loss.backward()

			optimizer.step()

			current_loss +=loss.item()

			if i!=0 and i%30==0:
				print(f'Loss after Epoch {epoch} & Iteration {i}  : {current_loss/i}')


	#Save Model