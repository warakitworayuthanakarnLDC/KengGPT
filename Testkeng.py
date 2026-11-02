import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

# 1. Device Selection (Automatically uses NVIDIA CUDA, Apple MPS, or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"🚀 Running Solaris on target device: {device}")

# 2. RWKV Model Architecture Definition
@torch.jit.script
def rwkv_recurrent_scan(k: torch.Tensor, v: torch.Tensor, r: torch.Tensor, 
                        decay: torch.Tensor, time_first: torch.Tensor) -> torch.Tensor:
    B, T, C = k.size()
    wkv = torch.zeros(B, T, C, device=k.device, dtype=k.dtype)
    state = torch.zeros(B, C, device=k.device, dtype=k.dtype)

    for t in range(T):
        kt = k[:, t, :]
        vt = v[:, t, :]
        rt = r[:, t, :]
        wkv[:, t, :] = rt * (time_first * kt + state)
        state = decay * state + kt * vt

    return wkv

class RWKV_TimeMix(nn.Module):
    def __init__(self, n_embd, layer_id):
        super().__init__()
        self.n_embd = n_embd
        self.layer_id = layer_id
        
        self.time_decay = nn.Parameter(torch.ones(n_embd))
        self.time_first = nn.Parameter(torch.ones(n_embd) * 0.5)
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, n_embd) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, n_embd) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, n_embd) * 0.5)

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptive = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        xx = torch.cat([torch.zeros(B, 1, C, device=x.device), x[:, :-1, :]], dim=1)
        
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptive(xr))

        decay = torch.exp(-torch.exp(self.time_decay))
        wkv = rwkv_recurrent_scan(k, v, r, decay, self.time_first)

        return self.output(wkv)

class RWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, n_embd) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, n_embd) * 0.5)
        
        self.key = nn.Linear(n_embd, n_embd * 4, bias=False)
        self.receptive = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd * 4, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        xx = torch.cat([torch.zeros(B, 1, C, device=x.device), x[:, :-1, :]], dim=1)
        
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = torch.square(torch.relu(self.key(xk)))
        kv = self.value(k)
        r = torch.sigmoid(self.receptive(xr))
        return r * kv

class Block(nn.Module):
    def __init__(self, n_embd, layer_id):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.att = RWKV_TimeMix(n_embd, layer_id)
        self.ffn = RWKV_ChannelMix(n_embd)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class KengGPT_Solaris(nn.Module):
    def __init__(self, vocab_size=4096, n_embd=448, n_layer=12):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, i) for i in range(n_layer)])
        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, idx):
        x = self.emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        return self.head(x)

# 3. Load Tokenizer & Model Weights
tokenizer_path = "keng_tokenizer.json"
model_path = "kenggpt_solar_sft_v2_fp16.pth"

if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
    raise FileNotFoundError("Missing keng_tokenizer.json or kenggpt_solar_sft_v2_fp16.pth in local directory!")

tokenizer = Tokenizer.from_file(tokenizer_path)

stop_tokens = ["<|endoftext|>", "<|user|>", "<|system|>"]
stop_ids = {tokenizer.token_to_id(t) for t in stop_tokens if tokenizer.token_to_id(t) is not None}

model = KengGPT_Solaris(vocab_size=4096, n_embd=448, n_layer=12).to(device)

# Load state dict and handle FP16 / FP32 casting based on CPU/GPU hardware
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)

if device.type == "cpu":
    model = model.float()  # CPU execution runs most reliably in standard FP32
else:
    model = model.half()   # GPU/MPS runs fast in FP16

model.eval()
print("✅ KengGPT Solaris (SFT v2) successfully loaded on local PC!\n")

# 4. Generation Logic
def generate_response(user_input, max_new_tokens=64, temperature=0.3, top_k=5, rep_penalty=1.15):
    formatted_prompt = f"<|system|>\nYou are Keng, an AI assistant on the KARK K-5.\n<|user|>\n{user_input}\n<|assistant|>\n"
    tokens = tokenizer.encode(formatted_prompt).ids
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            curr_input = input_ids[:, -128:]
            
            if device.type == "cuda":
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(curr_input)
            else:
                logits = model(curr_input)
            
            next_logits = logits[0, -1, :].clone().float() / max(temperature, 1e-5)

            # Apply repetition penalty
            for t_id in set(generated_tokens):
                if next_logits[t_id] < 0:
                    next_logits[t_id] *= rep_penalty
                else:
                    next_logits[t_id] /= rep_penalty

            # Top-K filtering
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[-1]] = -float('Inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token in stop_ids:
                break

            generated_tokens.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)

    return tokenizer.decode(generated_tokens)

# 5. Interactive CLI Chat Loop
print("="*60)
print("🤖 KengGPT Solaris 1.5 Local Terminal")
print("Type your message and press Enter (or type 'exit' / 'quit' to stop).")
print("="*60 + "\n")

while True:
    try:
        user_query = input("You: ").strip()
        if not user_query:
            continue
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
            
        response = generate_response(user_query, max_new_tokens=256, temperature=0.3)
        print(f"KengGPT 1.5 SOLARIS: {response.strip()}\n")
    except KeyboardInterrupt:
        print("\nExiting...")
        break