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
parser.add_argument('-lr','--learning_rate',type=float,default=0.001,help='Learning Rate [Default 0.001].')
parser.add_argument('-d','--dataset_path',type=str,default="./dataset/train/",help='Path of the training Dataset [Default "./dataset/train/"].')
parser.add_argument('-pm','--pretrained_model',type=str,default=None,help='Pretraine Model path, If you have any pretrained model pass the model path else None.')
parser.add_argument('-nc','--number_of_classes',type=int,default=1000,help='Number of Output Classes of the dataset[Default 1000]')




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
	loss_path= args.loss
	model_path = args.model_path
	pre_tained_model=args.pretrained_model
	num_classes =args.num_classes
	scale = args.scale
	#==========================================================

	os.makedirs(model_path,exist_ok=True)
	curr_epoch=0

	#----------------------Model Loading--------------------
	if args.model=='VGG11' :
		vgg= VGG(in_channel=3,num_classes=num_classes,architecture=vgg_a_architecture).to(device)

	elif args.model=='VGG13' : 
		vgg= VGG(in_channel=3,num_classes=num_classes,architecture=vgg_b_architecture).to(device)

	elif args.model=='VGG16' : 
		vgg= VGG(in_channel=3,num_classes=num_classes,architecture=vgg_c_architecture).to(device)

	elif args.model=='VGG19' : 
		vgg= VGG(in_channel=3,num_classes=num_classes,architecture=vgg_d_architecture).to(device)
	
	summary(vgg,(3,224,2224))

	#----------------------Loding Loss and Optimizer--------------------
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(vgg.parameters(),lr=LR,momentum=MOMENTUM,weight_decay=L2_REG)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',0.1,2)
	
	train_dataset = ImageNetDataset(dataset_path,dataset_type='train')
	train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)

	test_dataset = ImageNetDataset(dataset_path,dataset_type='test')
	test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)

	if pretrained_model!=None:
		checkpoint=torch.load(pretrained_model)
		vgg.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		curr_epoch = checkpoint['epoch']
		print(f'Pretrained Epoch : {curr_epoch}')
		optimizer.param_groups[0]['lr'] = LR

	total_loss = []
	total_accuracy = []

	for epoch in range(curr_epoch+1,EPOCH):
		current_loss = []
		current_accuracy = []

		with tqdm(training_loader,ncols=150) as tepoch:
			for i,data in enumerate(tepoch):
				vgg.train()
				inputs,labels = data
				inputs = inputs.to(device)
				labels = labels.to(device)

				output = vgg(inputs)

				loss = loss_fn(output,labels)
				current_loss.append(loss.item())
				optimizer.zero_Grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					accuracy = accuracy_calculate(labels,output)/BATCH_SIZE
					current_accuracy.append(accuracy)

					if i%calc==0 and i>=calc:
						tepoch.set_description(f'EP {epoch}')
						tepoch.set_postfix(Loss = loss.item(), A = f'{accuracy:.4f}', LR = optimizer.param_groups[0]['lr'])

						if loss.item()==float('nan'):
							exit(0)

			with torch.no_grad():
				total_loss.append(np.average(current_loss))
				total_accuracy.append(np.average(torch.tensor(current_accuracy).cpu()))

				#======================== Testing==================

				test_accuracy = []
				test_loss = []
				vgg.eval()

				for (X_test,y_test) in tqdm(test_loader):
					pred = vgg(X_test.to(device))
					pred = pred.cpu()
					test_loss.append(loss_fn(pred,y_test))
					test_accuracy.append(accuracy_calculate(y_test,pred))

				total_test_loss = np.average(test_loss)
				total_test_accuracy = np.average(test_accuracy)

				print(f'Test Loss : {total_test_loss} Accuracy : {total_test_accuracy}')

				#===================== Scheduler ===================
				scheduler.step(total_test_loss)
				LR = optimizer.param_groups[0]['lr']

			with open(loss_path,'a+') as l:
				l.write(f'Epoch : {epoch} LR : {LR} LOSS : {np.average(current_loss)} Accuracy : {np.average(torch.tensor(current_accuracy).cpu())} Test Loss : {total_test_loss} Test Accuracy : {total_test_accuracy}\n')
		state_dict {
		'model_name' : args.model,
		'epoch' : epoch,
		'model' : vgg.state_dict(),
		'optimizer' : optimizer.state_dict(),
		}

		torch.save(state_dict,os.path.join(model_path,f'final_model.pt'))
		print('Model Saved . . .')
