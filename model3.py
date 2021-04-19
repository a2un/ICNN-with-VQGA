import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
from torchvision.models.utils import load_state_dict_from_url
from torch.nn.utils.rnn import pack_padded_sequence

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
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, embed_size, hidden_size)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, hidden_size)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)                    # (batch_size, -1, embed_size)
        att2 = self.decoder_att(decoder_hidden)                 # (batch_size, hidden_size)
        print("attention encoder",att1.size(), "attention decoder", att2.size())
        att = self.full_att(att1 + att2)                          # (batch_size, hidden_size)
        # alpha = self.softmax(att)                                 # (hidden_size, 1)
        attention_weighted_encoding = (att * encoder_out.mean())#.sum()  #  (batch_size, hidden_size)

        return attention_weighted_encoding

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, batch_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.attention = Attention(embed_size,hidden_size,batch_size)
        self.init_h = nn.Linear(embed_size,hidden_size)
        self.init_c = nn.Linear(embed_size,hidden_size)
        self.h = None
        self.c = None
        self.f_beta = nn.Linear(hidden_size, 1)  # linear layer to create a sigmoid-activated gate
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        # embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        # embeddings = self.embed(captions)
        # embeddings = self.embed(captions)
        # print(embeddings.size(),features.unsqueeze(1).size())
        # embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # packed = pack_padded_sequence(embeddings, lengths.flatten(), batch_first=True, enforce_sorted=False) 
        # self.h, self.c = self.lstm(packed)
        # print(self.h[0].size())
        # encoder_out = features.view(features.size(0), -1, features.size(1))
        # attention_weighted_encoding = self.attention(encoder_out, self.h[0])
        # print("hidden size",attention_weighted_encoding.squeeze(2).size())
        # outputs = self.linear(attention_weighted_encoding)
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
