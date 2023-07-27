import tiktoken 
import torch 
import json 

with open('config.json', 'r') as file:
	config = json.load(file) 

block_size = config['block_size']
batch_size = config['batch_size']

# encoding = tiktoken.get_encoding('cl100k_base')
with open('data/friends.txt', 'r') as file: 
	text = file.read() 

characters = sorted(list(set(text))) 
vocab_size = len(characters) 

char_int = {c:i for i, c in enumerate(characters)}
int_char = {i:c for i, c in enumerate(characters)}

encode = lambda s: [char_int[c] for c in s]
decode = lambda l: ''.join([int_char[i] for i in l])

data = torch.tensor(encode(text))
train_data = data[:int(0.95*len(data))]
val_data = data[int(0.95*len(data)):]

def _init_data(): 
	return vocab_size, encode 

def dataloader(split): 
	data = train_data if split == 'train' else val_data
	ix = torch.randint(len(data) - block_size, (batch_size,))
	x = torch.stack([data[i:i+block_size] for i in ix])
	y = torch.stack([data[i+1:i+block_size+1] for i in ix])
	return x, y 