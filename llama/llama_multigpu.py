import argparse
import json
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from tqdm import tqdm
from pathlib import Path
import hiq
import os
from llama import ModelArgs, Tokenizer, Transformer, LLaMA

NUM_SHARDS = {
    "7B": 1,
    "13B": 2,
    "30B": 4,
    "65B": 8,
}

class LLaMAInference:
    def __init__(self, state_dict_dir, model_size, device_map="auto", **kwargs):

        state_dict = os.path.join(state_dict_dir, model_size, "state_dict.pt")
        params_file = os.path.join(state_dict_dir, model_size, "params.json")
        tokenizer_path = os.path.join(state_dict_dir, "tokenizer.model")
        params = hiq.read_file(params_file, as_json=True)

        model_args = dict(
            max_seq_len=2048,
            max_batch_size=1,
            **params
        )
        model_args.update(kwargs)
        model_args = ModelArgs(**model_args)

        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = self.tokenizer.n_words

        with init_empty_weights():
            torch.set_default_tensor_type(torch.HalfTensor)
            model = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)

        self.model = load_checkpoint_and_dispatch(
            model,
            state_dict,
            device_map=device_map,
            no_split_module_classes=["TransformerBlock"]
        )

        self.generator = LLaMA(self.model, self.tokenizer)

    def generate(self, texts, temperature=0.8, top_p=0.95, max_length=256, stop_ids=None, stop_words=None):
        results = self.generator.generate(
            texts,
            max_gen_len=max_length,
            temperature=temperature,
            top_p=top_p,
            stop_ids=stop_ids,
            stop_words=stop_words
        )
        return results

def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--state_dict_dir", type=str, default="/llama_data/7B")
    parser.add_argument(
        "--model_size",
        choices=NUM_SHARDS.keys(),
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    i = LLaMAInference(args.state_dict_dir, args.model_size)
    results = i.generate(["The meaning of life is"])
    for result in results:
        print("🦙LLaMA:", result.strip())
  
  
    results = i.generate(["Question: why apple drops from the tree when it is ripe?\nAnswer:"],
                          stop_words=["Question"])
    for result in results:
        print("🦙LLaMA:", result.strip())


