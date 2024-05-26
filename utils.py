import torch
import torch.nn as nn
import torchvision

class EucladianLoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self,output,target):
		# print("Loss : ",output,target)
		loss = torch.sqrt((output-target)**2).sum()
		print(loss)
		loss.requires_grad=True
		return loss

class ClassificationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target):
    	cls_loss=0
    	no_cls_loss=0
    	for i in range(len(target)):
    		no_cls_loss +=(torch.sum(inputs[i]) - inputs[i,target[i]])
    		cls_loss 	+=(1 - inputs[i,target[i]])
    	loss = cls_loss + 0.5*no_cls_loss
    	return loss