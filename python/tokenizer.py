import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("models/Llama-3.2-1B", use_fast=False)

text = sys.argv[1]

tokens = tokenizer.encode(text, add_special_tokens=False)

print(" ".join(map(str, tokens)))
