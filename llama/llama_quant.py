import time

import torch, transformers
import torch.nn as nn

from gptq import (
    GPTQ,
    Quantizer,
    find_layers,
    make_quant,
    QuantLinear,
    get_loaders,
    quantize,
)
from gptq.utils import (
    get_model,
    Catcher,
    model_pack,
    load_quant,
    benchmark,
    move_to_device,
    decoder_multigpu,
    get_quantizer,
    get_args,
    get_perplexity,
    DATASET_LIST
)

from llama.hf import LLaMAForCausalLM, LLaMATokenizer, LLaMAConfig
from llama.hf.utils import avoid_tensor_modified, get_llama



@torch.no_grad()
def llama_eval(model, testenc, args, dev):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(args.wbits, perchannel=True, sym=False, mse=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


def run(args=None):
    dev = (
        torch.device(args.cuda) if args.cuda.startswith("cuda") else torch.device("cpu")
    )
    if args.load=='p':
        config = LLaMAConfig.from_pretrained(args.model)
        torch.set_default_dtype(torch.half)
        model = LLaMAForCausalLM(config)
        torch.set_default_dtype(torch.float)
        saved_state_dict = torch.load(args.local_model)
        model.load_state_dict(saved_state_dict)
    elif args.load=='q':
        config = LLaMAConfig.from_pretrained(args.model)
        avoid_tensor_modified()
        transformers.modeling_utils._init_weights = False
        torch.set_default_dtype(torch.half)
        model = LLaMAForCausalLM(config)
        torch.set_default_dtype(torch.float)
        skip_layers = ["lm_head"]
        #import pudb; pu.db
        model = load_quant(model, args.local_model, args.wbits, skip_layers, model.config.max_sequence_length, dev=dev)
    else: # args.load=='hf'
        model = get_model(LLaMAForCausalLM, args.model)
        model.seqlen = model.config.max_sequence_length
        model.eval()

    print(f"Number of parameters: {model.num_parameters()}")
    for i in range(31):
        print(model.model.layers[i].self_attn.q_proj.bias)

    for name, param in model.named_parameters():
        print(name, param.shape)
        
    if args.eval:
        for dataset in ["wikitext2", "ptb", "c4"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen, tokenizer=tokenizer
            )
            print(dataset)
            llama_eval(model, testloader, args, dev)
    
    tokenizer = LLaMATokenizer.from_pretrained(
        args.model, add_eos_token=True
    )
    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
        tokenizer=tokenizer
    )
    
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
        print("*"*80)
        print("🦙:", tokenizer.decode([el.item() for el in generated_ids[0]]))
        return

    layers_to_move = ["embed_tokens", "norm"]
    layer_kws = ["attention_mask"]
    if (args.sparsity or args.prunen) and not args.gmp and args.mode == 'p':
        quantizers = get_quantizer(
            model,
            model.model,
            model.model.layers,
            "model.layers",
            dataloader,
            args,
            layers_to_move,
            layer_kws,
            dev,
        )
        """
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'fc2' in n:
                break
        """
    elif args.wbits < 16 and not args.nearest and args.mode == 'q':
        quantizers = get_quantizer(
            model,
            model.model,
            model.model.layers,
            "model.layers",
            dataloader,
            args,
            layers_to_move,
            layer_kws,
            dev,
        )
    else:
        print("wrong argument")
        return

    if args.benchmark:
        gpus = [torch.device("cuda:%d" % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llama_multigpu(model, gpus)
        else:
            model = model.to(dev)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, : args.benchmark]
            run_benchmark(model, input_ids, check=args.check)

    if args.save:
        model = model.cpu()
        """for name, param in model.named_parameters():
            param.data = param.data.to(dev)
        for name, buffer in model.named_buffers():
            buffer.data = buffer.data.to(dev)"""
        
        for name, param in model.named_parameters():
            print(name, param.data.shape)
        print("*"*80)
        for name, buffer in model.named_buffers():
            print(name, buffer.data.shape)
        model_pack(model, quantizers, args.wbits)
        import pudb; pu.db
        d = model.state_dict()
        torch.save(d, args.save)



if __name__ == "__main__":
    args = get_args()
    run(args)

