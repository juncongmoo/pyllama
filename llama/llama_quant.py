import logging
import time

import torch
import transformers
from gptq import get_loaders
from gptq.quant_utils import (benchmark, decoder_multigpu, eval_model,
                              get_quantizer, load_quant, model_pack)
from gptq.utils import (DATASET_LIST, avoid_tensor_modified, get_args,
                        get_model, print_model)
from tqdm import tqdm
from transformers import AutoTokenizer

from llama.hf import LLaMAConfig, LLaMAForCausalLM, LLaMATokenizer
from llama.lora_model import load_lora_model

logging.getLogger("datasets.builder").setLevel(logging.ERROR)

from gptq import QuantLinear

def run(args):
    if args.load == "lora":
        model = load_lora_model(f=args.model_path, bits=args.bits, max_lora_layers=5, new_class=QuantLinear)
        model.eval()
    elif args.load == "hf":
        model = get_model(LLaMAForCausalLM, args.model)
        model.seqlen = model.config.max_sequence_length
        model.eval()
    else:
        avoid_tensor_modified()
        transformers.modeling_utils._init_weights = False
        if args.load == "p":
            config = LLaMAConfig.from_pretrained(args.model)
            torch.set_default_dtype(torch.half)
            model = LLaMAForCausalLM(config)
            torch.set_default_dtype(torch.float)
            saved_state_dict = torch.load(args.model_path)
            model.load_state_dict(saved_state_dict)
        elif args.load == "q":
            config = LLaMAConfig.from_pretrained(args.model)
            torch.set_default_dtype(torch.half)
            model = LLaMAForCausalLM(config)
            torch.set_default_dtype(torch.float)
            skip_layers = ["lm_head"]
            model = load_quant(
                model,
                args.model_path,
                args.bits,
                skip_layers,
                model.config.max_sequence_length,
                new_class=QuantLinear,
            )
        else:
            raise ValueError(f"Wrong Argument load: {args.load}!")

    print_model(model, show_buffer=True)

    print(f"Number of parameters: {model.num_parameters()}")
    dev = (
        torch.device(args.cuda) if args.cuda.startswith("cuda") else torch.device("cpu")
    )
    if args.chatbot:
        model.to(dev)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        context = (
            "A chat between a curious human and a knowledgeable artificial intelligence assistant.\n"
            "human: Hello! What can you do?\n"
            "robot: As an AI assistant, I can answer questions and chat with you.\n"
            "human: What is the name of the tallest mountain in the world?\n"
            "robot: Everest.\n"
        )
        context = ""

        while True:
            inp = input("👶: ")
            if not inp or inp.lower() in [
                "exit",
                "bye",
                "good bye",
                "bye bye",
                "q",
                "quit",
            ]:
                print("exit...")
                break
            context += inp
            with torch.no_grad():
                input_ids = tokenizer.encode(context, return_tensors="pt").to(dev)
                t0 = time.monotonic()
                generated_ids = model.generate(
                    input_ids,
                    do_sample=True,
                    min_length=args.min_length,
                    max_length=args.max_length,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                t1 = time.monotonic()
            oids = [el.item() for el in generated_ids[0]]
            ans = tokenizer.decode(oids)
            ans = ans.replace("</s>", "")
            ans = ans[len(context) :].strip()
            t2 = time.monotonic()
            print(
                "🤖:",
                ans,
                "\nlatency: generation -",
                t1 - t0,
                "s, en/decoding -",
                t2 - t1,
                "s",
            )
        return

    tokenizer = LLaMATokenizer.from_pretrained(args.model, add_eos_token=True)
    if args.eval:
        for dataset in tqdm(DATASET_LIST, desc="Processing datasets..."):
            data_loader, val_loader, _ = get_loaders(
                dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=model.seqlen,
                tokenizer=tokenizer,
            )
            tail_layers, head_layers = ("embed_tokens",), ("norm",)
            ppl = eval_model(
                model, model.model, val_loader, tail_layers, head_layers, args, dev
            )
            tqdm.write(f"Dataset: {dataset}, PPL: {ppl:.2f}")
            tqdm.set_description(f"Processing {dataset}")
        return

    data_loader, val_loader, _ = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )

    if args.benchmark:
        gpus = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            tail = ("embed_tokens",)
            head = ("norm",)
            decoder_multigpu(model, model.model, gpus, tail, head)
        else:
            model = model.to(dev)
        input_ids = next(iter(data_loader))[0][:, : args.benchmark]
        benchmark(model, model.model, input_ids, check_perplexity=args.perplexity)
        return

    if args.text:
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
        print("*" * 80)
        print("🦙:", tokenizer.decode([el.item() for el in generated_ids[0]]))
        return

    layers_to_move = ["embed_tokens", "norm"]
    layer_kws = ["attention_mask"]
    if (args.sparsity or args.prunen) and not args.gmp and args.mode == "p":
        quantizers = get_quantizer(
            model,
            model.model,
            model.model.layers,
            "model.layers",
            data_loader,
            args,
            layers_to_move,
            layer_kws,
            dev,
        )
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if "fc2" in n:
                break
    elif args.bits < 16 and not args.nearest and args.mode == "q":
        quantizers = get_quantizer(
            model,
            model.model,
            model.model.layers,
            "model.layers",
            data_loader,
            args,
            layers_to_move,
            layer_kws,
            dev,
        )
    else:
        raise ValueError("Wrong Argument!")

    if args.save:
        model = model.cpu()
        print_model(model, show_buffer=True)
        model_pack(model, quantizers, args.bits, torch.device("cpu"))
        torch.save(model.state_dict(), args.save)
        model.save_pretrained(args.save + ".pretrained")


if __name__ == "__main__":
    args_ = get_args()
    run(args_)
