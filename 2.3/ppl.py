# (a) Perplexity Analysis with DistilGPT2
# Computes PPL for a paragraph vs. a shuffled version

import os, math, random, textwrap
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# ---------------- Config ----------------
MODEL_NAME = "distilgpt2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# 你可以换成任意 3-5 句英文段落
PARAGRAPH = (
    "Large language models have transformed natural language processing. "
    "They learn from vast corpora and can perform a surprising variety of tasks. "
    "However, they sometimes hallucinate facts and require careful evaluation. "
    "Researchers continue to study their limitations and opportunities. "
    "Responsible deployment is crucial for real-world applications."
)

# ---------------- Helpers ----------------
def shuffle_words(paragraph: str) -> str:
    words = paragraph.strip().split()
    random.shuffle(words)
    return " ".join(words)

def compute_ppl(tokenizer, model, text: str) -> float:
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    n_tokens = input_ids.numel()
    max_len = tokenizer.model_max_length
    stride = max_len
    nlls = []
    for i in range(0, n_tokens, stride):
        chunk = input_ids[:, i : i + max_len]
        with torch.no_grad():
            out = model(chunk, labels=chunk)
            neg_log_likelihood = out.loss.item() * chunk.numel()
        nlls.append(neg_log_likelihood)
    total_nll = sum(nlls)
    ppl = math.exp(total_nll / n_tokens)
    return ppl, n_tokens

def save_txt(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# ---------------- Main ----------------
def main():
    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    shuffled = shuffle_words(PARAGRAPH)
    ppl_orig, n_orig = compute_ppl(tokenizer, model, PARAGRAPH)
    ppl_shuf, n_shuf = compute_ppl(tokenizer, model, shuffled)

    report = []
    report.append("# Perplexity Analysis\n")
    report.append("Original paragraph:\n")
    report.append(textwrap.fill(PARAGRAPH, 100) + "\n")
    report.append(f"Tokens: {n_orig}, PPL: {ppl_orig:.2f}\n\n")
    report.append("Shuffled paragraph:\n")
    report.append(textwrap.fill(shuffled, 100) + "\n")
    report.append(f"Tokens: {n_shuf}, PPL: {ppl_shuf:.2f}\n\n")
    diff = ppl_shuf - ppl_orig
    ratio = ppl_shuf / ppl_orig if ppl_orig > 0 else float('inf')
    report.append(f"ΔPPL = {diff:.2f} (shuffle/original = {ratio:.2f}x)\n")
    report.append("Comment: Shuffling breaks syntax/semantics, so PPL increases.\n")

    save_txt(os.path.join(OUT_DIR, "perplexity.txt"), "\n".join(report))
    print(f"Saved results to {os.path.join(OUT_DIR, 'perplexity.txt')}")

if __name__ == "__main__":
    main()
