import argparse, os, pathlib, numpy as np, torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from utils.preproc import proc
from dataclasses import dataclass
import torch.autograd.variable as Variable
from torchsummary import summary
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_density(label):
    if label.shape[1]>1:
        label = torch.from_numpy(label[:,:,0,0])
        density = torch.mean((label>0).float(),0)
    else:
        density = torch.Tensor([0])
    return density

def main():
	# Config options
	parser = argparse.ArgumentParser(description='CS2770 Project Train')
	parser.add_argument('data_set', type=str, help='Train on "vqa" or "vqg" questions')
	parser.add_argument('--config', type=pathlib.Path, default='config.ini', help='The config file')
	parser.add_argument('--categoryname', type=str, default='person', help='classification category')

	args = parser.parse_args()
	root_dir = os.path.dirname(os.path.realpath(__file__))

	encoder, decoder, data_loader, config = proc(args, 'train', root_dir, 'train.py')

	# Create model directory
	if not os.path.exists(config['model_dir']):
		os.makedirs(config['model_dir'])

	# Put models on device
	encoder = encoder.to(device)
	decoder = decoder.to(device)

	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	params = list(decoder.parameters()) #+ list(encoder.linear.parameters()) + list(encoder.bn.parameters())
	optimizer = torch.optim.Adam(params, lr=config['learning_rate'])

	# Train the models
	total_step = len(data_loader)
	category_id_idx = int(config['categories'][args.categoryname])
	for epoch in range(1,config['num_epochs']+1):
		for i, (images, category, questions, lengths) in enumerate(data_loader):
			# Set mini-batch dataset
			targets = pack_padded_sequence(questions, lengths, batch_first=True, enforce_sorted=False)[0]
			targets = targets.to(device)
			# for image, category_list, question in zip(images, categories, questions):		
			images = images.to(device)
			questions = questions.to(device)
			lengths = torch.Tensor(np.array(lengths).reshape((len(lengths),1)))
			# Forward, backward and optimize
			density = get_density(category)
			features = encoder(images) #encoder(Variable(images), category, torch.Tensor([epoch + 1]),density) #encoder(images)
			print(summary(encoder, (3,299,299)))
			outputs = decoder(features, questions, lengths)
			print("target size",targets.size(),"output size", outputs.size())
			loss = criterion(outputs, targets)
			decoder.zero_grad()
			encoder.zero_grad()
			loss.backward()
			optimizer.step()

			# Print log info
			if i % config['log_step'] == 0:
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
					.format(epoch, config['num_epochs'], i, total_step, loss.item(), np.exp(loss.item()))) 
			
		torch.save(decoder.state_dict(), os.path.join(config['model_dir'], f'decoder-{epoch}.pth'))
		torch.save(encoder.state_dict(), os.path.join(config['model_dir'], f'encoder-{epoch}.pth'))


if __name__ == "__main__": main()