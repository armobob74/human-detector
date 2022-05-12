
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import torch.cuda

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def prepare_for_pyplot(img):
	'''(Tensor) -> np array
	Do various things to prepare Tensor to be plotted with pyplot
	Return an np array with all values between 0 to 1
	'''
	img = img.numpy() #convert to numpy array
	img = np.swapaxes(img,0,2) #reshape forconvention
	img = np.swapaxes(img,0,1) 
	img = img - img.min() #basically de-normalize so it can be viewed better 
	img = img / img.max()
	return img
	
def plot_from_dataset(dataset,n,m):
	''' (dataset, int, int, int) -> None
	Plot n*m random images from dataset.
	'''
	k_list = random.sample(range(len(dataset)),n*m) #generates a shuffled list of 0 to n*m, with unique values.
	fig, ax = plt.subplots(n,m)
	fig.set_figheight(n * 3)
	fig.set_figwidth(m * 3)
	plt.tight_layout()
	iters = 0;
	for i in range(n):
		for j in range(m):
			k = k_list[iters]
			img = dataset.__getitem__(k)[0] #select image (tuple) from dataset
			img = prepare_for_pyplot(img)
			ax[i,j].imshow(img)
			ax[i,j].axis('off')
			subplot_title = str(dataset.__getitem__(k)[1])
			ax[i,j].set_title(subplot_title)
			iters=iters+1
	plt.show()

def evaluate_epoch_loss(net,dataloader,criterion):
	''' (nn.Module,torch.utils.data.DataLoader,loss_criterion) -> float 
		Given a neural network, a dataloader, and a criterion, calculate loss over the dataset.
	'''
	bitter = iter(dataloader)
	epoch_loss = 0
	for i in range(len(dataloader)):
		batch,target = next(bitter)
		target = target.to(device)
		output = net(batch.to(device))
		loss = criterion(output,target/1) #divide by 1 so "target" is float and not int
		epoch_loss += loss.item()
	return epoch_loss / len(dataloader) #return normalized epoch_loss

def plot_tests(net,dataset,n,m):
	'''
	Same idea as plot_from_dataset(), but we include predicted label in the title.
	We also return a dictionary that's basically a confusion matrix 
	'''
	ret_dict = {'pred 0 label 1': 0, 'pred 0 label 0':0,'pred 1 label 1':0, 'pred 1 label 0':0}
	k_list = random.sample(range(len(dataset)),n*m)
	
	fig, ax = plt.subplots(n,m)
	fig.set_figheight(n * 3)
	fig.set_figwidth(m * 3)
	plt.tight_layout()
	
	iters = 0;
	for i in range(n):
		for j in range(m):
			k = k_list[iters]
			img = dataset.__getitem__(k)[0] #select image (tuple) from dataset
			pred_label = net(img.unsqueeze(0).to(device)) #need unsqueeze(0) to add dimension
			real_label = dataset.__getitem__(k)[1]
			abs_pred_label = round(float(pred_label))
			if(abs_pred_label == 1):
				if(real_label==1):
					ret_dict['pred 1 label 1'] += 1
				else:
					ret_dict['pred 1 label 0'] += 1
			else:
				if(real_label==1):
					ret_dict['pred 0 label 1'] += 1
				else:
					ret_dict['pred 0 label 0'] += 1
			img = prepare_for_pyplot(img) #convert from normalized tensor to viewable pyplot img
			ax[i,j].imshow(img)
			ax[i,j].axis('off')
			subplot_title = str("pred: " + str(float(pred_label))[0:4] + " label: " + str(real_label))
			ax[i,j].set_title(subplot_title)
			iters+=1
	plt.show()
	return ret_dict

def plot_loss(tst_loss,trn_loss,title="Loss Plot"):
		'''
		Generate a loss plot for the given inputs.
		'''
		n = range(len(tst_loss))
		m = range(len(trn_loss))
		plt.plot(n,tst_loss,label='Test loss')
		plt.plot(m,trn_loss,label = 'Training loss')
		plt.title(title)
		plt.xlabel('epoch')
		plt.ylabel('loss')
		plt.legend()
		
def save_loss_plot(optimizer,save_path):
	'''
	Saves loss plot with optimizer as title. 
	'''
	save_title=str(optimizer)
	save_title = save_title.replace('(','').replace(')','').replace(':','').replace('\n','')
	plot_num = 1; #add to avoid collisions
	proposed_str = save_path + save_title + ' plotnum '+ str(plot_num) + '.png'
	while os.path.exists(proposed_str): #make sure no collisions
		plot_num += 1
		proposed_str = save_path + save_title + ' plotnum '+ str(plot_num) + '.png'
	plt.savefig(proposed_str)
	
def rand_str(n):
	''' (int) -> str
		Generate a random string of length n
	'''
	S = 'abcdefghijklmnopqrstubwxyz'
	S = S+S.upper()+'1234567890'
	s='';
	for i in range(n):
		s = s + random.choice(S)
	return s
