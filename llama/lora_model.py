import torch
import torch.nn as nn
import transformers
from gptq import avoid_tensor_modified, load_quant
from gptq.quant import QuantLinear
from peft import LoraConfig, get_peft_model
from llama.hf import LLaMAConfig, LLaMAForCausalLM

MICRO_BATCH_SIZE = 16
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 1  # we don't need 3 tbh
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


def prepare(
    model,
    output_embedding_layer_name="lm_head",
    use_gradient_checkpointing=True,
    layer_norm_names=("layer_norm",),
):
    for name, param in model.named_parameters():
        param.requires_grad = False
        if param.ndim == 1 and any(
            layer_norm_name in name for layer_norm_name in layer_norm_names
        ):
            param.data = param.data.to(torch.float32)
    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model.gradient_checkpointing_enable()

    if hasattr(model, output_embedding_layer_name):
        output_embedding_layer = getattr(model, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):
            def forward(self, z_):
                return super().forward(z_.to(input_dtype)).to(torch.float32)

        setattr(
            model,
            output_embedding_layer_name,
            CastOutputToFloat(output_embedding_layer),
        )
    return model


def load_lora_model(
    f="lora-alpaca/checkpoint-1620/pytorch_model.bin",
    bits=2,
    max_lora_layers=5,
    dev=torch.device("cuda:0"),
    new_class=QuantLinear,
):
    hf_model_name = "decapoda-research/llama-7b-hf"
    config = LLaMAConfig.from_pretrained(hf_model_name)
    avoid_tensor_modified()
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model_ori = LLaMAForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = load_quant(
        model_ori,
        None,
        bits,
        ["lm_head"],
        seqlen=1024,
        for_infer=True,
        dev=dev,
        verbose=1,
        new_class=new_class,
    )
    model = prepare(model)
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["k_proj", "down_proj", "up_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        bits=bits,
        max_lora_layers=max_lora_layers,
    )
    model = get_peft_model(model, config)
    model.load_state_dict(torch.load(f))
    m = model.base_model.model
    for layer in m.model.layers:
        if not hasattr(layer.self_attn.k_proj, "lora_A"):
            break
        layer.self_attn.k_proj.lora_A.weight = nn.Parameter(
            layer.self_attn.k_proj.lora_A.weight.half()
        )
        layer.self_attn.k_proj.lora_B.weight = nn.Parameter(
            layer.self_attn.k_proj.lora_B.weight.half()
        )

        layer.mlp.down_proj.lora_A.weight = nn.Parameter(
            layer.mlp.down_proj.lora_A.weight.half()
        )
        layer.mlp.down_proj.lora_B.weight = nn.Parameter(
            layer.mlp.down_proj.lora_B.weight.half()
        )

        if not hasattr(layer.mlp.up_proj, "lora_A"):
            break
        layer.mlp.up_proj.lora_A.weight = nn.Parameter(
            layer.mlp.up_proj.lora_A.weight.half()
        )
        layer.mlp.up_proj.lora_B.weight = nn.Parameter(
            layer.mlp.up_proj.lora_B.weight.half()
        )
    return m
