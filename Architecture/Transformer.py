import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


###### Understanding still in progress############
class Model(nn.Module):
    def __init__(self,ntoken,ninp,nahead,nhid,nlayers,dropout=.5):
        super(Model, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout) #Encode input then pass through pos_encoder
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)  #What is the difference between the encoder layer and the encoder
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):      # What is this mask?
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange,initrange)
        self.decoder.bias.dataa.zero_()
        self.decoder.weight.data.uniform_(-initrange,initrange)
        
    def forward(self, src, src_mask):
        src = self.encoder(src)*math.sqrt(self.ninp) # What is ninp?
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arrange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arrange(0,d_model,2)/float() * (-math.log(100000.0)/d_model))
