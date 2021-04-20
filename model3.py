import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
from torchvision.models.utils import load_state_dict_from_url
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.functional import pad

class SaveFeatures:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output.clone().detach()
    def close(self):
        self.hook.remove()

# This is a simple model that returns that last fully-connected layer of a Resnet 18 CNN      
class EncoderCNN(resnet.ResNet):
    def __init__(self):
        super().__init__(resnet.BasicBlock, [2, 2, 2, 2])
        state_dict = load_state_dict_from_url(resnet.model_urls['resnet18'], progress=True)
        self.load_state_dict(state_dict)
        self.activation = SaveFeatures(list(self.children())[-1])
     
    def __call__(self, inputs):
        super().__call__(inputs)
        return self.activation.features
    
    # Pass a list of ints representing the modules of the encoder for which you want to extract features
    def create_forward_hooks(self, layer_list):
        modules = list(self.modules())
        self.activations = {i: SaveFeatures(modules[i]) for i in layer_list}
    
    # Pass the int value of the layer for which you want a feature map.  Only call this after passing 
    # inputs to the encoder and you will get the output associated with those inputs
    def extract_layer_features(self, layer):
        return self.activations[layer].features
    
    def close_forward_hooks(self):
        for activation in self.activations.values():
            activation.close()

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size 
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, embed_size, hidden_size)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, hidden_size)
        :return: attention weighted encoding, weights
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        att1 = self.encoder_att.to(device)(encoder_out)                    
        att2 = self.decoder_att.to(device)(decoder_hidden)                 
        # print("attention encoder",att1.mean(dim=0).mean(dim=0).size(), "attention decoder", att2.mean(dim=0).size())
        att = self.full_att.to(device)(att1.mean(dim=0).mean(dim=0).mean(dim=0) + att2.mean(dim=0))
        # print("full att", att)
        alpha = self.softmax(att)                                 
        attention_weighted_encoding = (alpha * encoder_out.mean()).sum()  

        return attention_weighted_encoding

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, batch_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.init_h = nn.Linear(embed_size,hidden_size)
        self.init_c = nn.Linear(embed_size,hidden_size)
        self.h = None
        self.c = None
        self.f_beta = nn.Linear(hidden_size, 1)  # linear layer to create a sigmoid-activated gate
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, layer_features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        layer_features_l = [l.size(2) for l in layer_features]
        encoder_dim = max(layer_features_l)
        layer_features = [pad(l,(int((encoder_dim-l.size(2))/2),int((encoder_dim-l.size(2))/2),int((encoder_dim-l.size(2))/2),int((encoder_dim-l.size(2))/2))) if l.size(2) < encoder_dim else l for l in layer_features]
        layer_features = torch.cat(layer_features)
        attention = Attention(encoder_dim,self.hidden_size,self.hidden_size)
        # print("layer_features size", layer_features[0].size())
        # embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths.flatten(), batch_first=True) 
        self.h, self.c = self.lstm(packed)
        attention_weighted_encoding = attention(layer_features, self.h[0])
        # attention_weighted_encoding = self.h[0]#
        attention_weighted_encoding = attention_weighted_encoding * self.sigmoid(self.h[0])
        outputs = self.linear(attention_weighted_encoding)
        # print("hidden size",attention_weighted_encoding.squeeze(2).size())
        # outputs = self.linear(attention_weighted_encoding)
        return outputs
    
    def sample(self, outputs, states=None):
        """Generate captions for given image features using greedy search."""
        _, predicted = outputs.max(1)                        # predicted: (batch_size)
        inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
        for i in range(self.max_seg_length):
            # packed = pack_padded_sequence(inputs, lengths.flatten(), batch_first=True) 
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            # attention_weighted_encoding = attention(layer_features, hiddens[0])
            # attention_weighted_encoding = attention_weighted_encoding * self.sigmoid(hiddens[0])
            # outputs = self.linear(hiddens[0])            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)
