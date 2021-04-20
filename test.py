import argparse, copy, nltk, os, pathlib, torch, numpy as np
from utils.preproc import proc

def get_density(label):
    if label.shape[1]>1:
        label = torch.from_numpy(label[:,:,0,0])
        density = torch.mean((label>0).float(),0)
    else:
        density = torch.Tensor([0])
    return density

def test(icnn_encoder, decoder, data_loader, id_to_word, epoch, doOutputQuestions=False):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	c = nltk.translate.bleu_score.SmoothingFunction()
	
	total_bleu_score = 0.0
	layers = [9,11]
	for i, (images, categories, questions, lengths) in enumerate(data_loader):
		print(f'Testing step {i} of {len(data_loader)}')
		images = images.to(device)
		questions = questions.to(device)
		icnn_encoder.create_forward_hooks(layers)
		feature = icnn_encoder(images,categories, torch.Tensor([epoch+1]), get_density(categories.detach().cpu().numpy()))
		layer_features = [icnn_encoder.extract_layer_features(i) for i in layers]
		icnn_encoder.close_forward_hooks()
		sampled_ids = decoder.sample(layer_features,questions,lengths)
		sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
		questions = questions.detach().cpu().numpy()
		references = []
		for item in questions:
			references.append(id_to_word(item))
		generated = id_to_word(sampled_ids)
		
		bleu_score = nltk.translate.bleu_score.sentence_bleu(references,generated,smoothing_function=c.method7)
		total_bleu_score += bleu_score
		
		if doOutputQuestions:
			print(f'{i} of {len(data_loader)}: bleu score: {bleu_score} | references: {references} | generated: {generated}')	
	
	return total_bleu_score / float(len(data_loader))

if __name__ == '__main__':

	# Device configuration
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Config options
	parser = argparse.ArgumentParser(description='CS2770 Project Eval')
	parser.add_argument('data_set', type=str, help='Eval using "vqa" or "vqg" questions')
	parser.add_argument('--config', type=pathlib.Path, default='config.ini', help='The config file')

	args = parser.parse_args()
	root_dir = os.path.dirname(os.path.realpath(__file__))

	icnn_encoder, decoder, data_loader, config = proc(args, 'test', root_dir, 'test.py')

	icnn_encoder_path = os.path.join(config['model_dir'], 'best_encoder.pth')
	decoder_path = os.path.join(config['model_dir'], 'best_decoder.pth')
	if not os.path.exists(icnn_encoder_path):
		raise Exception(f'Encoder does not exist: {encoder_path}')
	if not os.path.exists(decoder_path):
		raise Exception(f'Decoder does not exist: {decoder_path}')
	
	# encoder = encoder.to(device)
	decoder = decoder.to(device)
	icnn_encoder = icnn_encoder.to(device)
	icnn_encoder.load_state_dict(torch.load(icnn_encoder_path))
	decoder.load_state_dict(torch.load(decoder_path))

	bleu_score = test(encoder, decoder, data_loader, config['id_to_word'], epoch, True)
	print(f'Average bleu score for test set: {bleu_score}')




