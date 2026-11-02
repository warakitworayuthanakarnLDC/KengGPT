# =========================================================
# KengGPT Interactive Terminal with Claude-Style Spinner
# =========================================================
import os
import sys
import time
import random
import threading
import requests
import urllib3
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders

# Suppress SSL warnings for self-hosted DuckDNS server
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------
# 1. Download Model from Windows Server (SSL Bypass)
# ---------------------------------------------------------
model_path = "/content/kenggpt_rwkv_22m.pth"
model_url = "https://srrsoftwarehouse.duckdns.org/MUSIC/MUSIC/kenggpt_rwkv_22m.pth"

if not os.path.exists(model_path):
    print(f"Downloading model from {model_url}...")
    try:
        response = requests.get(model_url, verify=False, timeout=60)
        response.raise_for_status()
        with open(model_path, "wb") as f:
            f.write(response.content)
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Failed to download model automatically: {e}")
        print("Please upload 'kenggpt_rwkv_22m.pth' manually to /content/ before running.")

# ---------------------------------------------------------
# 2. Model Architecture Definitions
# ---------------------------------------------------------
class RWKV5_TimeMix(nn.Module):
    def __init__(self, n_embd, layer_id, n_layer):
        super().__init__()
        self.n_embd = n_embd
        self.layer_id = layer_id

        self.time_decay = nn.Parameter(torch.empty(n_embd))
        self.time_first = nn.Parameter(torch.empty(n_embd))
        self.time_mix_k = nn.Parameter(torch.empty(1, 1, n_embd))
        self.time_mix_v = nn.Parameter(torch.empty(1, 1, n_embd))
        self.time_mix_r = nn.Parameter(torch.empty(1, 1, n_embd))
        self.time_mix_g = nn.Parameter(torch.empty(1, 1, n_embd))

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.gate = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)
        self.ln_x = nn.GroupNorm(1, n_embd)

        nn.init.uniform_(self.time_decay, -6, -1)
        nn.init.uniform_(self.time_first, -3, 0)
        nn.init.zeros_(self.time_mix_k)
        nn.init.zeros_(self.time_mix_v)
        nn.init.zeros_(self.time_mix_r)
        nn.init.zeros_(self.time_mix_g)

    def forward(self, x):
        B, T, C = x.size()
        xx = F.pad(x, (0, 0, 1, -1))

        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))
        g = F.silu(self.gate(xg))

        w = torch.exp(-torch.exp(self.time_decay)).view(1, 1, C)
        u = self.time_first.view(1, 1, C)

        wkv = torch.zeros(B, T, C, device=x.device)
        state = torch.zeros(B, C, device=x.device)
        for t in range(T):
            kt, vt = k[:, t, :], v[:, t, :]
            wkv[:, t, :] = state + u * kt * vt
            state = w * state + kt * vt

        x_out = self.ln_x(wkv.view(B * T, C)).view(B, T, C)
        return self.output(r * x_out * g)

class RWKV5_ChannelMix(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.time_mix_k = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_mix_r = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.key = nn.Linear(n_embd, n_embd * 3, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd * 3, n_embd, bias=False)

    def forward(self, x):
        xx = F.pad(x, (0, 0, 1, -1))
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = torch.square(torch.relu(self.key(xk)))
        kv = self.value(k)
        r = torch.sigmoid(self.receptance(xr))
        return r * kv

class Block(nn.Module):
    def __init__(self, n_embd, layer_id, n_layer):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.att = RWKV5_TimeMix(n_embd, layer_id, n_layer)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = RWKV5_ChannelMix(n_embd)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class KengGPT_RWKV(nn.Module):
    def __init__(self, vocab_size=4096, n_embd=384, n_layer=8):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, i, n_layer) for i in range(n_layer)])
        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, idx):
        x = self.emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        return self.head(x)

# ---------------------------------------------------------
# 3. Load Tokenizer & Model Weights
# ---------------------------------------------------------
tokenizer_path = "/content/keng_tokenizer.json"
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Please upload your tokenizer to {tokenizer_path}!")

tokenizer = Tokenizer.from_file(tokenizer_path)
tokenizer.decoder = decoders.ByteLevel()
eos_id = tokenizer.token_to_id("<|endoftext|>")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KengGPT_RWKV(vocab_size=4096, n_embd=384, n_layer=8).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---------------------------------------------------------
# 4. Claude Code Style Spinner & 184 Status Phrases
# ---------------------------------------------------------
CLAUDE_FRAMES = ['·', '✻', '✽', '✶', '✳', '✢']

STATUS_PHRASES = [
    "combobulating", "discombobulating", "shenaniganing", "pondering", "synthesizing",
    "musing", "calculating", "hallucinating", "brewing coffee", "debugging reality",
    "quantum tunneling", "consulting the oracle", "crunching numbers", "consulting discord logs",
    "channeling Keng's soul", "spinning up neural pathways", "overthinking", "validating vibes",
    "touching grass", "escaping the matrix", "generating pure chaos", "estimating trajectories",
    "optimizing procrastination", "compiling memes", "consulting ancient scrolls",
    "reversing polarity", "calibrating flux capacitors", "interpreting dreams",
    "analyzing starlight", "surfing the ether", "brewing thoughts", "weaving narratives",
    "decrypting thoughts", "polishing pixels", "loading personality", "summoning inspiration",
    "untangling syntax", "stargazing", "whispering to tensors", "herding pixels",
    "baking data", "shuffling weights", "untangling matrices", "consulting destiny",
    "crafting sentences", "fine-tuning chaos", "simulating wisdom", "brewing ideas",
    "translating vibes", "summoning coffee", "debugging destiny", "aligning chakras",
    "parsing thoughts", "cultivating creativity", "gathering electrons", "sparking imagination",
    "inspecting gradients", "folding space-time", "channeling genius", "structuring sentences",
    "composing verse", "rendering output", "spinning thoughts", "consulting history",
    "building worlds", "shaping ideas", "formatting dreams", "polishing prose",
    "invoking logic", "channeling focus", "tracing paths", "optimizing output",
    "balancing entropy", "distilling knowledge", "crafting responses", "consulting wisdom",
    "weaving words", "harnessing lightning", "navigating syntax", "orchestrating thoughts",
    "tuning frequencies", "calibrating meaning", "generating brilliance", "structuring logic",
    "assembling grammar", "filtering noise", "polishing syntax", "refining tokens",
    "synthesizing context", "building narrative", "calculating probabilities", "tracing logic",
    "mapping thoughts", "formatting reality", "assembling fragments", "crafting sentences",
    "consulting archives", "decoding intent", "interpreting prompt", "analyzing context",
    "weighing options", "evaluating paths", "formulating reply", "composing response",
    "polishing output", "refining delivery", "optimizing flow", "structuring text",
    "weaving narrative", "generating prose", "rendering thought", "finalizing words",
    "reviewing drafts", "checking grammar", "polishing edges", "perfecting tone",
    "adjusting limits", "fine-tuning response", "smoothing syntax", "checking alignment",
    "validating logic", "ensuring coherence", "testing boundaries", "exploring concepts",
    "gathering context", "processing inputs", "evaluating meaning", "interpreting signals",
    "translating intent", "decoding messages", "parsing requests", "analyzing inputs",
    "reviewing parameters", "checking weights", "calibrating models", "tuning responses",
    "orchestrating flow", "managing tokens", "controlling entropy", "directing attention",
    "guiding generation", "shaping output", "molding text", "crafting meaning",
    "building context", "forming ideas", "generating logic", "compiling responses",
    "synthesizing data", "processing tokens", "calculating loss", "updating state",
    "running inference", "executing code", "evaluating logits", "sampling distribution",
    "selecting tokens", "appending sequence", "advancing step", "continuing generation",
    "looping logic", "iterating process", "progressing text", "finishing up"
]

class ClaudeSpinner:
    def __init__(self):
        self.stop_signal = threading.Event()
        self.phrase = random.choice(STATUS_PHRASES)
        self.thread = threading.Thread(target=self._animate)

    def _animate(self):
        i = 0
        while not self.stop_signal.is_set():
            frame = CLAUDE_FRAMES[i % len(CLAUDE_FRAMES)]
            sys.stdout.write(f"\r{frame} {self.phrase}...")
            sys.stdout.flush()
            time.sleep(0.08)
            i += 1

    def __enter__(self):
        self.stop_signal.clear()
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_signal.set()
        self.thread.join()
        # Clear the spinner line completely
        sys.stdout.write("\r" + " " * (len(self.phrase) + 10) + "\r")
        sys.stdout.flush()

# ---------------------------------------------------------
# 5. Optimized Generation Function with Repetition Penalty
# ---------------------------------------------------------
def generate_text(
    prompt, 
    max_new_tokens=80, 
    min_tokens=15, 
    temperature=0.35, 
    top_p=0.75,
    repetition_penalty=1.2
):
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    prompt_len = input_ids.shape[1]
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            idx_cond = input_ids[:, -128:]
            logits = model(idx_cond)
            next_token_logits = logits[0, -1, :]
            
            # Repetition Penalty
            for token_id in set(input_ids[0].tolist()):
                if next_token_logits[token_id] > 0:
                    next_token_logits[token_id] /= repetition_penalty
                else:
                    next_token_logits[token_id] *= repetition_penalty

            next_token_logits = next_token_logits / max(temperature, 1e-4)
            
            if eos_id is not None and step < min_tokens:
                next_token_logits[eos_id] = -float('Inf')
            
            next_token_logits = torch.nan_to_num(next_token_logits, nan=-1e9, posinf=1e4, neginf=-1e9)
            
            # Top-p (nucleus) filtering
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if eos_id is not None and step >= min_tokens and next_token.item() == eos_id:
                break
                
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
    generated_ids = input_ids[0][prompt_len:].tolist()
    return tokenizer.decode(generated_ids)

# ---------------------------------------------------------
# 6. Interactive Terminal Loop
# ---------------------------------------------------------
print("\n" + "="*65)
print("🤖 KengGPT Interactive Terminal Initialized Successfully!")
print("Type your prompt and press Enter. Type 'exit' or 'quit' to stop.")
print("="*65 + "\n")

while True:
    try:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break
        if not user_input:
            continue

        # Run generation wrapped in the Claude Code style animated spinner
        with ClaudeSpinner():
            output = generate_text(
                prompt=user_input,
                max_new_tokens=80,
                min_tokens=15,
                temperature=0.35,
                top_p=0.75,
                repetition_penalty=1.2
            )
        
        print(f"Keng: {output}\n")

    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye! 👋")
        break