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
        :param encoder_dim: feature size of encoded images
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

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderRNN(nn.Module):
    def __init__(self, attention_dim, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.attention = Attention(hidden_size, hidden_size, attention_dim)
        self.vocab_size = vocab_size
        self.init_h = nn.Linear(hidden_size, hidden_size)  # linear layer to find initial cell state of LSTMCell
        self.init_c = nn.Linear(hidden_size, hidden_size)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(hidden_size, hidden_size)  # linear layer to create a sigmoid-activated gate
        self.fc = nn.Linear(hidden_size, vocab_size)  # linear layer to find scores over vocabulary
        self.sigmoid = nn.Sigmoid()
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)    
    
    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, hidden_size)
        :return: hidden and cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, hidden_size)
        c = self.init_c(mean_encoder_out)  
        return h,c

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        encoder_out = features.view(features.size(0), -1, features.size(-1))
        num_pixels = features.size(1)
        caption_lengths, sort_ind = lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        outputs = torch.zeros(features.size(0), self.max_seg_length, self.vocab_size).to(device)
        h,c = self.init_hidden_state(encoder_out)      # (batch_size, decoder_dim)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(self.max_seg_length):
            batch_size_t = sum([l > t for l in lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding

            # packed = pack_padded_sequence(torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1), lengths.cpu().flatten(), batch_first=True,enforce_sorted=False) 
            
            h, c = self.lstm(torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.lstm(self.dropout(h))  # (batch_size_t, vocab_size)
            outputs[:batch_size_t, t, :] = preds

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
