import argparse, os, pathlib, numpy as np, torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from utils.preproc import proc
from utils.convert_to_icnn_format import data_format_converter
from icnn.tools.classification import classification
from icnn.tools.classification_multi import classification_multi
from dataclasses import dataclass
from icnn.tools.lib import init_lr

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Config options
parser = argparse.ArgumentParser(description='CS2770 Project Train')
parser.add_argument('data_set', type=str, help='Train on "vqa" or "vqg" questions')
parser.add_argument('--config', type=pathlib.Path, default='config.ini', help='The config file')

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
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=config['learning_rate'])

# Train the models
total_step = len(data_loader)
for epoch in range(1,config['num_epochs']+1):
	for i, (images, questions, lengths) in enumerate(data_loader):
		
		# Set mini-batch dataset
		images = images.to(device)
		questions = questions.to(device)
		targets = pack_padded_sequence(questions, lengths, batch_first=True)[0]
		
		# Forward, backward and optimize
		features = encoder(images)
		outputs = decoder(features, questions, lengths)
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




def ICNN(categoryname):

    root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'icnn') #path.join(os.getcwd(),'icnn') #'/data2/lqm/pytorch_interpretable/py_icnn        

    @dataclass
    class args:
        gpu_id: int  = 0
        task_name: str = 'classification'
        task_id:int = 0
        dataset:str = 'voc2010_crop'
        imagesize:int = 224
        label_name:str = categoryname
        label_num:int = 1
        model:str = 'resnet_18'
        losstype:str = 'logistic'
        batchsize:int = 16
        dropoutrate:int = 0
        lr:int = 0
        epochnum:int = 2
        weightdecay:int = 0.0005
        momentum:int = 0.09
    
    args.lr, args.epochnum = init_lr(args.model,args.label_num,args.losstype) #init lr and epochnum
    if(args.task_name=='classification'):
        if args.dataset == 'celeba':
            args.label_num = 40
        return classification(root_path, args)
    else:
        if args.dataset == 'vocpart':
            args.label_name = [''.join(categoryname.split()).lower()]
            args.label_num = 6
        return classification_multi(root_path,args)

source_data_path = os.path.join(os.path.realpath(__file__),'data/data_files/{0}') 
dest_data_path = os.path.join(os.path.realpath(__file__),'datasets/voc2010_crop/{0}_info.txt')

train_lines = open(source_data_path.format('train_data.txt'),'r').readlines()
val_lines = open(source_data_path.format('val_data.txt'),'r').readlines()

for categoryname in list(config['categories'].keys()):
    ## create text files in ICNN format
    data_format_converter(config['categories']['_'.join(categoryname.split())],dest_data_path,train_lines,val_lines)
    net = ICNN(categoryname)
    torch.save(net.state_dict(), os.path.join(config['model_dir'], f"icnn-{''.join(categoryname.split())}.pth"))
