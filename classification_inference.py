import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerDecoder(nn.Module):

    def __init__(self, vocab_size, embed_dim=32, num_heads=2, hidden_dim=128, num_layers=1, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        encoded = self.pos_encoder(embedded)
        output = self.transformer_decoder(encoded, encoded)
        output = output.mean(dim=1)
        output = self.fc(output)
        return torch.sigmoid(output).squeeze()
    

def init_model(tokenizer, device):

    model = TransformerDecoder(tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(r"model_states\model_continued_epoch_20.pt"))

    return model

def inference(model, tokenizer, device, text):
    
    max_length = 256
    
    input = tokenizer(text,  max_length=max_length, truncation=True, padding='max_length', return_tensors="pt")
    
    input = input['input_ids'].squeeze(0)

    output = model(input.to(device))
    
    output = torch.mean(output)

    return output.item()
