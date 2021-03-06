import argparse, os, pathlib, numpy as np, torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from utils.preproc import proc
from dataclasses import dataclass
import torch.autograd.variable as Variable
from torchsummary import summary
from icnn_resnet_18 import resnet_18

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

	icnn_encoder, decoder, data_loader, config = proc(args, 'train', root_dir, 'train.py')

	# Create model directory
	if not os.path.exists(config['model_dir']):
		os.makedirs(config['model_dir'])

	# Put models on device
	# encoder = encoder.to(device)
	decoder = decoder.to(device)
	icnn_encoder = icnn_encoder.to(device)

	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	params = list(decoder.parameters()) + list(icnn_encoder.linear.parameters()) + list(icnn_encoder.bn.parameters())
	optimizer = torch.optim.Adam(params, lr=config['learning_rate'])

	# Train the models
	total_step = len(data_loader)
	layers = [9,11]				## mask layers

	for epoch in range(1,config['num_epochs']+1):
		for i, (images, categories, questions, lengths) in enumerate(data_loader):
			# Set mini-batch dataset
			targets = pack_padded_sequence(questions, lengths, batch_first=True, enforce_sorted=False)[0]
			targets = targets.to(device)
			# for image, category_list, question in zip(images, categories, questions):		
			images = images.to(device)
			questions = questions.to(device)
			categories = categories.to(device)
			density = get_density(categories.cpu().detach().numpy())
			# category = np.array([category_list[category_id_idx] for category_list in categories])
			# category  = torch.from_numpy(category.reshape((1,category.shape[0],1,1))).to(device)
			lengths = torch.Tensor(np.array(lengths).reshape((len(lengths),1)))
			# print("category shape",categories.size())
			# Forward, backward and optimize
			icnn_encoder.create_forward_hooks(layers)
			# features = encoder(images) 
			features = icnn_encoder(Variable(images), categories, torch.Tensor([epoch + 1]),density)
			# summary(encoder, (3,7,7))
			# summary(icnn_encoder, ((3,7,7),categories.size(), torch.Tensor([epoch + 1]).size(),density.size()))
			layer_features = [icnn_encoder.extract_layer_features(i) for i in layers]
			icnn_encoder.close_forward_hooks()
			outputs = decoder(layer_features, questions, lengths)
			loss = criterion(outputs, targets)
			decoder.zero_grad()
			icnn_encoder.zero_grad()
			loss.backward()
			optimizer.step()

			# Print log info
			if i % config['log_step'] == 0:
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
					.format(epoch, config['num_epochs'], i, total_step, loss.item(), np.exp(loss.item()))) 
			
		torch.save(decoder.state_dict(), os.path.join(config['model_dir'], f'decoder-{epoch}.pth'))
		# torch.save(encoder.state_dict(), os.path.join(config['model_dir'], f'encoder-{epoch}.pth'))
		torch.save(icnn_encoder.state_dict(), os.path.join(config['model_dir'], f'icnn-encoder-{epoch}.pth'))


if __name__ == "__main__": main()