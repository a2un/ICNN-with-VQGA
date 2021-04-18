import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        #resnet = models.inception_v3(pretrained=True)
        resnet = models.resnet18(pretrained=True)
        self.modules = list(resnet.children())[:-1]      # delete the last fc layer.
        for i in range(len(self.modules)):
            self.modules[i] = self.modules[i].to(device)
        self.linear = nn.Linear(2048, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
        
            """ This was the set-up from the inception_v3 model
            x = self.modules[0](images)
            x = self.modules[1](x)
            x = self.modules[2](x)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            x = self.modules[3](x)
            x = self.modules[4](x)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            x = self.modules[5](x)
            x = self.modules[6](x)
            x = self.modules[7](x)
            x = self.modules[8](x)
            x = self.modules[9](x)
            x = self.modules[10](x)
            x = self.modules[11](x)
            x = self.modules[12](x)
            x = self.modules[14](x)
            x = self.modules[15](x)
            x = self.modules[16](x)
            x = F.avg_pool2d(x, kernel_size=8)
            x = x.view(x.size(0), -1)    
            """
            
            x = self.modules[0](images)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            for i in range(1,len(self.modules)-1):	# I cut off the last layer because it was reducing the size of the matrix to 1x1
                x = self.modules[i](x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = x.view(x.size(0), -1)
            
        features = self.bn(self.linear(x))
        return features

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

        :param encoder_out: encoded images, a tensor of dimension (batch_size, embed_size)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, hidden_size)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        # alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_seg_length = max_seq_length
        self.attention = Attention(embed_size,hidden_size,hidden_size)

        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        batch_size = embeddings.size(0)
        embed_size = embeddings.size(-1)
        lengths, sortind = lengths.squeeze(1).sort(dim=0, descending=True)
        decode_lengths = (lengths - 1).tolist()
        embeddings = embeddings[sortind]
        h = nn.linear(embeddings.size(-1),self.hidden_size)(embeddings.mean(dim=1))
        c = nn.linear(embeddings.size(-1),self.hidden_size)(embeddings.mean(dim=1))
        outputs = torch.zeros(batch_size,max(decode_lengths))

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding = self.attention(embeddings[:batch_size_t],h[:batch_size_t])
            # packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
            h, c = self.lstm(torch.cat([embeddings[:batch_size_t, t], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))
            outputs[:batch_size_t,t] = self.linear(hiddens[0])
        
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
        return sampled_ids