import torch
import torch.nn as nn
from torchsummary import summary
from tqdm import tqdm
import os


from VGGNet import vgg_a_architecture,VGG
from dataset import FlowerDataset
from utils import EucladianLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Current Device : {device}')

if __name__=="__main__":
	# ======================HyperParameter=====================
	EPOCH=80
	BATCH_SIZE=4
	LR=0.01
	MOMENTUM=0.9
	L2_REG = 5e-4
	dataset_path = "./dataset/train/"
	INPUT_SHAPE = (224,224)
	labels_map = {
		0 : "daisy",
		1 : "rose",
		2 : "tulip",
		3 : "dandelion",
		4 : "sunflower",
	}
	loss_path="vgg_11loss.txt"
	model_path = './vgg_11/'
	pre_tained_model=None
	#==========================================================

	os.makedirs(model_path,exist_ok=True)
	#Clean the file
	with open(loss_path,'w+') as loss_file:
		pass

	curr_epoch=0


	# Model Load
	vgg11 = VGG(in_channel=3,num_classes=5,architecture=vgg_a_architecture).to(device)
	vgg11.init_weights()		#Initialize the weights with Gausan Distrubution.
	# print(summary(vgg11,(3,224,224)))	
	# Loss Function
	eucladian_loss=EucladianLoss()
	# Optimizer
	optimizer = torch.optim.SGD(vgg11.parameters(),lr=LR,momentum=MOMENTUM,weight_decay=L2_REG)
	#Loading model if exist
	if pre_tained_model is not None and  os.path.isfile(pre_tained_model):
		print("Loading Pretrained Model .....")
		checkpoint = torch.load(pre_tained_model)
		vgg11.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		curr_epoch=checkpoint['epoch']

	# DataSet
	flower_dataset = FlowerDataset(dataset_path,input_shape=INPUT_SHAPE,labels_map=labels_map)
	# Dataloader
	training_loader = torch.utils.data.DataLoader(flower_dataset,batch_size=BATCH_SIZE,shuffle=True)

	#Training Loop
	
	for epoch in range(curr_epoch,EPOCH):
		current_loss = 0
		for i, data in enumerate(tqdm(training_loader)):
			inputs,labels = data
			inputs =inputs.to(device)

			optimizer.zero_grad()
			outputs = vgg11(inputs)

			loss = eucladian_loss(outputs,labels)

			loss.backward()

			optimizer.step()

			current_loss +=loss.item()

			if i!=0 and i%50==0:
				with open(loss_path,'a+') as loss_file:
					loss_file.write(str(current_loss/i)+'\n')
		
		#Save Model
		state_dict = {
			'epoch':epoch,
			'model' : vgg11.state_dict(),
			'optimizer': optimizer.state_dict(),

		}
		torch.save(state_dict,os.path.join(model_path,f'vgg11_e{epoch}.pt'))
