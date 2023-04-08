import torch, transformers

from llama.hf import LLaMATokenizer
from llama.hf.utils import get_llama
from gptq import load_quant, avoid_tensor_modified
from llama.hf import LLaMAForCausalLM, LLaMAConfig, LLaMATokenizer

def get_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="decapoda-research/llama-7b-hf", help="llama model to load")
    parser.add_argument(
        "--bits",
        type=int,
        default=16,
        choices=[2, 3, 4, 8, 16],
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument("--load", type=str, default="", help="Load quantized model.")
    parser.add_argument("--text", type=str, help="input text")
    parser.add_argument(
        "--min_length",
        type=int,
        default=10,
        help="The minimum length of the sequence to be generated.",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=1024,
        help="The  maximum length of the input sequence that LLaMA can process.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="The maximum length of the output sequence to be generated.",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="The value used to module the next token probabilities.",
    )
    parser.add_argument(
        "--cuda", type=str, default="cuda:0", help="GPU device string, eg cuda:0."
    )
    args = parser.parse_args()
    return args


def run(args=None):
    args = args or get_args()
    if args.load:
        config = LLaMAConfig.from_pretrained(args.model)
        avoid_tensor_modified()
        transformers.modeling_utils._init_weights = False
        torch.set_default_dtype(torch.half)
        model_ori = LLaMAForCausalLM(config)
        torch.set_default_dtype(torch.float)
        print(model_ori)
        print("*"*80)
        #import pudb; pu.db
        model = load_quant(model_ori, args.load, args.bits, ['lm_head'],seqlen=1024, for_infer=True, dev=torch.device('cuda:0'), verbose=1)
        #model = load_quant(args.model, args.load, args.bits, args.seqlen)
    else:
        model = get_llama(args.model)
        model.eval()
    if args.cuda.startswith("cuda"):
        dev = torch.device(args.cuda)
    else:
        dev = torch.device("cpu")

    model.to(dev)
    tokenizer = LLaMATokenizer.from_pretrained(args.model)
    input_ids = tokenizer.encode(args.text, return_tensors="pt").to(dev)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=args.min_length,
            max_length=args.max_length,
            top_p=args.top_p,
            temperature=args.temperature,
        )
    print("*"*80)
    print("🦙:", tokenizer.decode([el.item() for el in generated_ids[0]]))


if __name__ == "__main__":
    run()
