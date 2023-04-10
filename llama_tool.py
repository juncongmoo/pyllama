"""
👉python llama_tool.py
**************************************** bits=8 ****************************************
Number of parameters: 6738415616
Median: 0.04436063766479492
PPL: 4.304362773895264
Max memory(MiB): 13372.7216796875
Number of parameters: 6738415616
Median: 0.04503834247589111
PPL: 7.157315254211426
Max memory(MiB): 13372.7216796875
Number of parameters: 6738415616
Median: 0.044960856437683105
PPL: 8.733440399169922
Max memory(MiB): 13372.7216796875
"""
import logging
logging.getLogger("datasets.builder").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)

from gptq.runner import run
from gptq.utils import get_args, DATASET_LIST


def rock(
    base_model="decapoda-research/llama-7b-hf",
    output_dir="/home/ubuntu/fuheng/pyllama/rock",
    bb=8,
):
    args_ = get_args()
    args_.model = base_model
    args_.load = "hf"
    args_.bits = bb
    args_.save = f'{output_dir}/{base_model}-{args_.bits}b.pt'
    args_.benchmark = 1024
    args_.perplexity = True
    args_.max_length = 64
    args_.verbose = False

    model = run(args_)
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
    for b in range(2, 8):
        print("*" * 40, f"bits={b}", "*" * 40)
        rock(bb=b)
