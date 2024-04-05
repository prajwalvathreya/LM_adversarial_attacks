from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformers import GPT2Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

print("Script Started")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using {device} for training")

# Load the IMDB dataset
dataset = load_dataset('imdb', split='train')

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

with torch.no_grad():
    attack_model = GPT2Model.from_pretrained('openai-community/gpt2', output_hidden_states = True).to(device)

tokenizer.pad_token = tokenizer.eos_token

class IMDbDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=256):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        encoded_sample = self.tokenizer(sample['text'], max_length=self.max_length, truncation=True, padding='max_length', return_tensors="pt")
        input_ids = encoded_sample['input_ids'].squeeze(0)  # Remove batch dimension
        label = torch.tensor(sample['label'], dtype=torch.float)
        return input_ids, label

imdb_dataset = IMDbDataset(dataset, tokenizer)
data_loader = DataLoader(imdb_dataset, batch_size=32, shuffle=True, num_workers=2)

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

    def __init__(self, vocab_size, embed_dim=768, num_heads=2, hidden_dim=768, num_layers=1, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embed_dim)
        # self.pos_encoder = PositionalEncoding(embed_dim)
        self.embedding = attack_model.get_input_embeddings()
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, input_ids):
        # embedded = self.embedding(input_ids)
        # encoded = self.pos_encoder(embedded)
        embedded = None
        output = self.transformer_decoder(embedded, embedded)
        output = output.mean(dim=1)
        output = self.fc(output)
        return torch.sigmoid(output).squeeze()

model = TransformerDecoder(tokenizer.vocab_size).to(device)

# model.load_state_dict(torch.load(r"model_states\model_epoch_20.pt"), strict=False)

# for param in model.parameters():
#     param.requires_grad = True

def train(model, data_loader, epochs=20):
    
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    criterion = nn.BCELoss()

    for epoch in tqdm(range(epochs), desc="Epoch"):
        total_loss = 0
        for input_ids, labels in data_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            print(input_ids)
            break
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        break
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), r"updated_states/model_epoch_{}.pt".format(epoch+1))
            
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(data_loader)}")
        lr_scheduler.step()

if __name__ == '__main__':
    train(model, data_loader)