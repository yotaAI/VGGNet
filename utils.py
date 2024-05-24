import torch
import torch.nn as nn

class EucladianLoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self,output,target):
		loss = torch.sqrt((output-target)**2).sum()
		loss.requires_grad=True
		return loss

