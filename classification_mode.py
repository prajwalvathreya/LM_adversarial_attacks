from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = load_dataset('imdb', split='train')  # Load the IMDB dataset  

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_dataset(dataset, tokenizer):
    tokenized_input = []
    tokenized_labels = []
    for sample in dataset:

        if len(sample["text"]) > 1024:
            continue

        tokenized_text = tokenizer(sample["text"], return_tensors="pt", padding="max_length", max_length=1024).to(device)
        tokenized_label = torch.tensor([sample["label"]]).to(device)

        tokenized_labels.append(tokenized_label)
        tokenized_input.append(tokenized_text)

    return tokenized_input, tokenized_labels

train_dataset, train_labels = tokenize_dataset(dataset, tokenizer)

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, num_heads=8, hidden_dim=16, num_layers=1, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(device)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout).to(device)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout).to(device)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers).to(device)

        self.flatten = nn.Flatten().to(device)

        self.fc = nn.Linear(32768, 1).to(device)
        

    def forward(self, x):
        x = self.embedding(x).to(device)
        x = self.positional_encoding(x).to(device)

        memory = x.clone().to(device)
    
        # Pass through transformer decoder layers
        x = self.transformer_decoder(x, memory).to(device)

        x = self.flatten(x).to(device)

        x = self.fc(x).to(device)

        x = torch.sigmoid(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

model = TransformerDecoder(tokenizer.vocab_size)

def train(model, train_dataset, epochs=10):

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    overall_loss = 0
    # num = len(train_dataset)

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for idx, vals in enumerate(train_dataset):
            sample, labels = vals
            optimizer.zero_grad()
            output = model(sample["input_ids"]).to(device)
            loss = criterion(output, labels).to(device)
            loss.backward()
            optimizer.step()
            overall_loss += loss.item()
        print("Epoch {}".format(epoch)," -- ", "Loss : ",overall_loss)

# inference function
def inference(model, text):

    model.eval()
    tokenized_text = tokenizer(text, return_tensors="pt")
    output = model(tokenized_text["input_ids"])

    return output.argmax().item()

data = zip(train_dataset, train_labels)

train(model, data)
