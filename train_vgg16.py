import torch
import torch.nn as nn
from torchsummary import summary
from tqdm import tqdm
import os


from VGGNet import vgg_a_architecture,vgg_c_architecture, VGG
from dataset import FlowerDataset
from utils import EucladianLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Current Device : {device}')

if __name__=='__main__':
	#======================Hyper Parameter=================
	vgg_11_path = 'vgg_11/vgg11_e0.pt'
	EPOCH=80
	BATCH_SIZE=4
	LR=0.01
	MOMENTUM=0.9
	dataset_path = "./dataset/train/"
	INPUT_SHAPE = (224,224)
	labels_map = {
		0 : "daisy",
		1 : "rose",
		2 : "tulip",
		3 : "dandelion",
		4 : "sunflower",
	}
	loss_path="vgg16_loss.txt"
	model_path = './vgg_16/'
	LOAD_FROM_STATEDICT=False          #If you have already pretrained vgg16.
	#=====================================================
	os.makedirs(model_path,exist_ok=True)
	#Clean the file
	with open(loss_path,'w+') as loss_file:
		pass
	
	# DataSet
	flower_dataset = FlowerDataset(dataset_path,input_shape=INPUT_SHAPE,labels_map=labels_map)
	# Dataloader
	training_loader = torch.utils.data.DataLoader(flower_dataset,batch_size=BATCH_SIZE,shuffle=True)
	curr_epoch=0

	#Loading VGG16 Model
	vgg16 = VGG(in_channel=3,num_classes=5,architecture=vgg_c_architecture)
	# Loss Function
	eucladian_loss=EucladianLoss()
	# Optimizer
	optimizer = torch.optim.SGD(vgg16.parameters(),lr=LR,momentum=MOMENTUM)
	
	if not LOAD_FROM_STATEDICT:
		vgg16_dict = vgg16.state_dict()

		#Loading VGG11 Pretrained model.
		vgg11_dict = torch.load(vgg_11_path)['model']

		#The Layers to update in VGG16 from pretrained VGG11 model.
		layers_to_update = ['conv_layers.0','fc.']

		for layer in layers_to_update:
			for k,v in vgg16_dict.items():
				if k.startswith(layer):
					print("Updating Layer : ",k)
					vgg16_dict[k] = vgg11_dict[k]

		vgg16.load_state_dict(vgg16_dict)

	else:
		if len(os.listdir(model_path))!=0:
			print("Loading Pretrained Model .....")
			checkpoint = torch.load(os.path.join(model_path,f'vgg16_e{len(os.listdir(model_path))}.pt'))
			vgg16.load_state_dict(checkpoint['model'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			curr_epoch=checkpoint['epoch']
		else:
			raise Exception("Pretrained Model not found!")
	
	#Training Loop
	for epoch in range(curr_epoch,EPOCH):
		current_loss = 0
		for i, data in enumerate(tqdm(training_loader)):
			inputs,labels = data
			inputs =inputs.to(device)

			optimizer.zero_grad()
			outputs = vgg16(inputs)

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
			'model' : vgg16.state_dict(),
			'optimizer': optimizer.state_dict(),

		}
		torch.save(state_dict,os.path.join(model_path,f'vgg16_e{epoch+1}.pt'))