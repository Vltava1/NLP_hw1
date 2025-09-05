# (b) Sampling Comparison with DistilGPT2
# Greedy + different temperatures, save outputs and a summary.

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

MODEL_NAME = "distilgpt2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "outputs"; os.makedirs(OUT_DIR, exist_ok=True)

PROMPT = "Once upon a time"
MAX_NEW_TOKENS = 500
TEMPS = [1e-6, 0.3, 0.6, 0.9, 1.2, 1.5]

def generate_text(tokenizer, model, prompt, temperature, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    if temperature is None:  # greedy
        out_ids = model.generate(
            **inputs, do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    else:  # temperature sampling
        out_ids = model.generate(
            **inputs, do_sample=True, temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)

def distinct_n(text: str, n: int) -> float:
    toks = text.split()
    if len(toks) < n: return 0.0
    total = len(toks) - n + 1
    grams = set(tuple(toks[i:i+n]) for i in range(total))
    return len(grams) / total

def save_txt(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def main():
    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    lines = ["# Sampling Comparison\n", f"Prompt: {PROMPT!r}\n"]

    # ---- Greedy (明确单独跑一遍) ----
    greedy_txt = generate_text(tokenizer, model, PROMPT, temperature=None, max_new_tokens=MAX_NEW_TOKENS)
    greedy_path = os.path.join(OUT_DIR, "gen_greedy.txt")
    save_txt(greedy_path, greedy_txt)
    g_d1, g_d2 = distinct_n(greedy_txt, 1), distinct_n(greedy_txt, 2)
    lines += [f"## Greedy (T=0)\n", f"- Distinct-1: {g_d1:.3f}, Distinct-2: {g_d2:.3f}\n", f"- Saved: {greedy_path}\n"]

    # ---- Temperature sampling ----
    for T in TEMPS:
        text = generate_text(tokenizer, model, PROMPT, temperature=T, max_new_tokens=MAX_NEW_TOKENS)
        fn = f"gen_T{T}".replace(".", "p") + ".txt"; path = os.path.join(OUT_DIR, fn)
        save_txt(path, text)
        d1, d2 = distinct_n(text, 1), distinct_n(text, 2)
        lines += [f"## T={T}\n", f"- Distinct-1: {d1:.3f}, Distinct-2: {d2:.3f}\n", f"- Saved: {path}\n"]

    save_txt(os.path.join(OUT_DIR, "sampling_summary.txt"), "\n".join(lines))
    print(f"Saved summary to {os.path.join(OUT_DIR, 'sampling_summary.txt')}")

if __name__ == "__main__":
    main()
