from llama.lora_model import load_lora_model
import torch
import os

import logging
logging.getLogger("datasets.builder").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)

from gptq.runner import run
from gptq.utils import get_args, DATASET_LIST

bits_path='rock/decapoda-research/llama-7b-hf-4b.pt.bits'
qbits = torch.load(bits_path) if os.path.exists(bits_path) else None

model = load_lora_model(f="lora-alpaca-4/checkpoint-20/pytorch_model.bin", qbits=qbits, max_lora_layers=500)
model.eval()

def rock(
    base_model="decapoda-research/llama-7b-hf",
    bb=8,
    model=None,
):
    args_ = get_args()
    args_.model = base_model
    args_.load = "lora"
    args_.bits = bb
    args_.save = ""
    args_.benchmark = 1024
    args_.perplexity = True
    args_.max_length = 64
    args_.verbose = False

    model = run(args_, model=model)
    # num_params = model.num_parameters()
    # print(f"compression rate:{num_params/125239296*100:.2f}%")
    if hasattr(model, "qbits"):
        mean_ = sum(model.qbits.values()) / len(model.qbits)
        print(f"final quant bits:{mean_:.2f}")
    args_.save = ""

    for i in DATASET_LIST:
        args_.dataset = i
        run(args_, model=model)
    return args_


if __name__ == "__main__":
    rock(model=model)
