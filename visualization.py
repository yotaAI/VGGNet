import os,sys
import matplotlib.pyplot as plt
#---------------Loss File---------------
LOSS_FILE="./vgg_11loss.txt"

with open(LOSS_FILE,'r') as l:
	losses = l.readlines()
	losses = [float(loss.strip())for loss in losses]

plt.title("Model Loss!")
plt.plot(losses)
plt.show()
