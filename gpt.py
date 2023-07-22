import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import json 

with open('config.json', 'r') as file: 
	config = json.load(file) 

block_size = config['block_size']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = config['n_embd']
n_head = config['n_head']
n_layer = config['n_layer']
dropout = config['dropout']

class adaptive_span(nn.Module): 
	def __init__(self, max_size, ramp_size, init_val=0, shape=(1,1,1)): 
		super().__init__() 
		self.max_size = max_size 
		self.ramp_size = ramp_size 
		self.current_val = nn.Parameter(torch.zeros(*shape) + init_val)
		mask_template = torch.linspace(1 - max_size, 0, steps=max_size)
		self.register_buffer('mask_template', mask_template)

	def forward(self, x): 
		mask = self.mask_template + self.current_val * self.max_size 
		mask = mask / self.ramp_size + 1 
		mask = mask.clamp(0, 1) 
		if x.size(-1) < self.max_size: 
			mask = mask[:, :, -x.size(-1):]
		x = x * mask 
		x = x / (x.sum(-1, keepdim=True) + 1e-8)
		return x 
	
	def clamp_param(self): 
		self.current_val.data.clamp_(0, 1)

class attention_head(nn.Module): 
	def __init__(self, enable_adaptive_span, head_size=n_embd//n_head): 
		super().__init__() 
		self.key = nn.Linear(n_embd, head_size, bias=False) 
		self.query = nn.Linear(n_embd, head_size, bias=False) 
		self.value = nn.Linear(n_embd, head_size, bias=False) 
		self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
		self.dropout = nn.Dropout(dropout) 

		self.enable_adaptive_span = enable_adaptive_span
		if enable_adaptive_span: 
			self.adaptive_span = adaptive_span(max_size=block_size, ramp_size=block_size)

	def forward(self, x): 
		B, T, C = x.shape 
		k = self.key(x) 
		q = self.query(x) 
		weight = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
		weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
		weight = F.softmax(weight, dim=-1)
		if self.enable_adaptive_span: 
			weight = self.adaptive_span(weight) 
		weight = self.dropout(weight) 
		v = self.value(x) 
		out = weight @ v 
		return out 

class multi_head_attention(nn.Module): 
	def __init__(self, enable_adaptive_span, num_heads=n_head, head_size=n_embd//n_head): 
		super().__init__() 
		self.heads = nn.ModuleList([attention_head(enable_adaptive_span) for _ in range(num_heads)])
		self.proj = nn.Linear(head_size * num_heads, n_embd)
		self.dropout = nn.Dropout(dropout) 

	def forward(self, x): 
		out = torch.cat([h(x) for h in self.heads], dim=-1)
		out = self.dropout(self.proj(out)) 
		return out 

class feed_forward(nn.Module): 
	def __init__(self, n_embd=n_embd):
		super().__init__() 
		self.net = nn.Sequential(
			nn.Linear(n_embd, 4*n_embd), 
			nn.ReLU(), 
			nn.Linear(4*n_embd, n_embd), 
			nn.Dropout(dropout) 
		)

	def forward(self, x): 
		return self.net(x) 

class block(nn.Module): 
	def __init__(self, enable_adaptive_span, n_embd=n_embd, n_head=n_head): 
		super().__init__() 
		self.sa = multi_head_attention(enable_adaptive_span) 
		self.ff = feed_forward() 
		self.ln1 = nn.LayerNorm(n_embd) 
		self.ln2 = nn.LayerNorm(n_embd) 

	def forward(self, x): 
		x = x + self.sa(self.ln1(x)) 
		x = x + self.ff(self.ln2(x))
		return x 

class gpt_model(nn.Module): 
	def __init__(self, vocab_size, enable_adaptive_span=True): 
		super().__init__() 
		self.token_embedding = nn.Embedding(vocab_size, n_embd)
		self.position_embedding = nn.Embedding(block_size, n_embd) 
		self.blocks = nn.Sequential(*[block(enable_adaptive_span) for _ in range(n_layer)])
		self.ln_f = nn.LayerNorm(n_embd) 
		self.lm_head = nn.Linear(n_embd, vocab_size) 

		self.apply(self._init_weights)
	
	def _init_weights(self, module): 
		if isinstance(module, nn.Linear): 
			nn.init.normal_(module.weight, mean=0.0, std=0.02) 
			if module.bias is not None: 
				nn.init.zeros_(module.bias) 
		elif isinstance(module, nn.Embedding): 
			nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def forward(self, idx):
		# sentence converted into indices of words
		B, T = idx.shape 

		tok_emb = self.token_embedding(idx) 
		pos_emb = self.position_embedding(torch.arange(T, device=device)) 
		x = tok_emb + pos_emb 
		x = self.blocks(x) 
		x = self.ln_f(x) 
		logits = self.lm_head(x) 

		return logits 
	
	def generate(self, idx, max_new_tokens): 
		for _ in range(max_new_tokens): 
			idx_cond = idx[:, -block_size:]
			logits = self(idx_cond) 
			logits = logits[:, -1, :]
			probs = F.softmax(logits, dim=-1)
			idx_next = torch.multinomial(probs, num_samples=1)
			idx = torch.cat((idx, idx_next), dim=1) 
		return idx 