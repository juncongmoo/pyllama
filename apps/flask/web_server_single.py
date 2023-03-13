import torch
import json

from pathlib import Path
from typing import List

from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import torch.distributed as dist

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="/llama_data/7B")
    parser.add_argument(
        "--tokenizer_path", type=str, default="/llama_data/tokenizer.model"
    )
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_batch_size", type=int, default=1)
    return parser.parse_args()


app = FastAPI()


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)
    return generator


def init_generator(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 512,
    max_batch_size: int = 1,
):
    local_rank, world_size = 0, 1
    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    return generator


if __name__ == "__main__":
    args = get_args()
    generator = init_generator(
        args.ckpt_dir,
        args.tokenizer_path,
        args.max_seq_len,
        args.max_batch_size,
    )

    class Config(BaseModel):
        prompts: List[str]
        max_gen_len: int
        temperature: float = 0.8
        top_p: float = 0.95

    @app.post("/llama/")
    def generate(config: Config):
        if len(config.prompts) > args.max_batch_size:
            return {"error": "too much prompts."}
        for prompt in config.prompts:
            if len(prompt) + config.max_gen_len > args.max_seq_len:
                return {"error": "max_gen_len too large."}
        results = generator.generate(
            config.prompts,
            max_gen_len=config.max_gen_len,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        return {"responses": results}

    uvicorn.run(app, host="0.0.0.0", port=8080)
