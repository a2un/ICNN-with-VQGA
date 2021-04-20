import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet
from torchvision.models.utils import load_state_dict_from_url
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.functional import pad
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# This is a simple model that returns that last fully-connected layer of a Resnet 18 CNN      
class EncoderCNN(resnet.ResNet):
    def __init__(self,embed_size):
        super().__init__(resnet.BasicBlock, [2, 2, 2, 2])
        state_dict = load_state_dict_from_url(resnet.model_urls['resnet18'], progress=True)
        self.load_state_dict(state_dict)
        self.modules = list(self.children())[:-1]
        for i in range(len(self.modules)):
            self.modules[i] = self.modules[i].to(device)
        self.linear = nn.Linear(2048, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
    
    def forward(self,images):
        with torch.no_grad():
            x = self.modules[0](images)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            for i in range(1, len(self.modules)-1):
                x = self.modules[i](x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = x.view(x.size(0), -1)
        
        features = self.bn(self.linear(x))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, batch_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features, embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths.flatten(), batch_first=True) 
        h,_ = self.lstm(packed)
        outputs = self.linear(h[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.squeeze(0)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
