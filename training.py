import torch
import torch.nn as nn
from torchsummary import summary
from tqdm import tqdm
import os
import argparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Current Device : {device}')

from VGGNet import vgg_a_architecture,vgg_b_architecture,vgg_c_architecture,vgg_d_architecture,VGG
from dataset import FlowerDataset
from utils import EucladianLoss, ClassificationLoss

parser = argparse.ArgumentParser(description ="VGG Model Training")
parser.add_argument('-m','--model',type=str,choices	=['VGG11','VGG13','VGG16','VGG19'],required=True,help='Provide Which VGG Model you want to train.')
parser.add_argument('-mp','--model_path',type=str,required=True,help='Path to store the trained Model.')
parser.add_argument('-l','--loss',type=str,required=True,help='File where losses will be stored [Required][EG: vgg11_loss.txt].')

parser.add_argument('-s','--scale',type=int,default=256,help='Scale the Image with. [Default 256].')
parser.add_argument('-e','--epoch',type=int,default=80,help='Number of Epoch the model will train [Default 80].')
parser.add_argument('-b','--batch_size',type=int,default=4,help='Size of Training Batch Size [Default 4].')
parser.add_argument('-lr','--learning_rate',type=int,default=0.001,help='Learning Rate [Default 0.001].')
parser.add_argument('-d','--dataset_path',type=str,default="./dataset/train/",help='Path of the training Dataset [Default "./dataset/train/"].')
parser.add_argument('-labels','--labels_map',type=dict,default=None,help='Labels of the training Images. [Default is {0 : "daisy", 1 : "rose", 2 : "tulip", 3 : "dandelion", 4 : "sunflower"}].')
parser.add_argument('-pm','--pretrained_model',type=str,default=None,help='Pretraine Model path, If you have any pretrained model pass the model path else None.')




if __name__=="__main__":
	args = parser.parse_args()

	# ======================HyperParameter=====================
	EPOCH=args.epoch
	BATCH_SIZE=args.batch_size
	LR=args.learning_rate
	MOMENTUM=0.9
	L2_REG = 5e-4
	dataset_path = "./dataset/train/" if not args.dataset_path else args.dataset_path
	INPUT_SHAPE = (224,224)
	labels_map = {0 : "daisy",1 : "rose",2 : "tulip",3 : "dandelion",4 : "sunflower",} if not args.labels_map else args.labels_map
	loss_path= args.loss
	model_path = args.model_path
	pre_tained_model=args.pretrained_model
	#==========================================================

	os.makedirs(model_path,exist_ok=True)
	curr_epoch=0

	#----------------------Model Loading--------------------
	if args.model=='VGG11' :
		vgg= VGG(in_channel=3,num_classes=5,architecture=vgg_a_architecture).to(device)

	elif args.model=='VGG13' : 
		vgg= VGG(in_channel=3,num_classes=5,architecture=vgg_b_architecture).to(device)

	elif args.model=='VGG16' : 
		vgg= VGG(in_channel=3,num_classes=5,architecture=vgg_c_architecture).to(device)

	elif args.model=='VGG19' : 
		vgg= VGG(in_channel=3,num_classes=5,architecture=vgg_d_architecture).to(device)
	
	#----------------------Loding Loss and Optimizer--------------------
	# eucladian_loss=EucladianLoss()
	# classification_loss=ClassificationLoss()
	cross_entropy_loss = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(vgg.parameters(),lr=LR,momentum=MOMENTUM,weight_decay=L2_REG)
	
	#----------------------Loding Pretrained Model--------------------
	if pre_tained_model is not None and  os.path.isfile(pre_tained_model):

		print("Loading Pretrained Model .....")
		checkpoint = torch.load(pre_tained_model)
		if checkpoint['model_name']!=args.model:
			raise Exception(f"Trained Model is {checkpoint['model_name']} and you are loading for {args.model}")
		
		vgg.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		curr_epoch=checkpoint['epoch'] + 1
		print(f"Pretraine Model Epoch : {curr_epoch	}")
	else:
		with open(loss_path,'w+') as loss_file:
			pass

	#----------------------Loding Dataset --------------------
	flower_dataset = FlowerDataset(dataset_path,input_shape=INPUT_SHAPE,labels_map=labels_map,num_class=len(labels_map))
	training_loader = torch.utils.data.DataLoader(flower_dataset,batch_size=BATCH_SIZE,shuffle=True)



	#----------------------Training Loop --------------------
	for epoch in range(curr_epoch,EPOCH):
		print(f"Current Epoch : {epoch	}")
		current_loss = 0
		for i, data in enumerate(tqdm(training_loader)):
			inputs,labels = data
			inputs =inputs.to(device)

			optimizer.zero_grad()
			outputs = vgg(inputs)
			# print(outputs,labels)
			# break
			loss = cross_entropy_loss(outputs,labels)

			loss.backward()

			optimizer.step()

			current_loss +=loss.item()
	#----------------------Loss Printing --------------------
			if i!=0 and i%50==0:
				print("Loss :",loss.item())
				with open(loss_path,'a+') as loss_file:
					# loss_file.write(str(current_loss/len(training_loader))+'\n')
					loss_file.write(str(current_loss/i)+'\n')

	#----------------------Saving Model --------------------
		if epoch%5==0:
			state_dict = {
				'model_name': args.model,
				'epoch':epoch,
				'model' : vgg.state_dict(),
				'optimizer': optimizer.state_dict(),

			}
			torch.save(state_dict,os.path.join(model_path,f'{args.model}_e{epoch}.pt'))