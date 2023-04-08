import logging


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import transformers
from datasets import load_dataset
from gptq import avoid_tensor_modified, load_quant
from hiq.vis import print_model
from peft import LoraConfig, get_peft_model
from llama.hf import LLaMAConfig, LLaMAForCausalLM, LLaMATokenizer

# optimized for RTX 4090. for larger GPUs, increase some of these?
MICRO_BATCH_SIZE = 16  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 1  # we don't need 3 tbh
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

logging.getLogger("datasets.builder").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)


def prepare_model_for_int4_training(
    model,
    output_embedding_layer_name="lm_head",
    use_gradient_checkpointing=True,
    layer_norm_names=["layer_norm"],
):
    r"""
    This method wrapps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32
        2- making output embedding layer require grads
        3- Add the upcasting of the lm
        head to fp32
    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_8bit = True
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False
        if 1:
            # cast layer norm in fp32 for stability for 8bit models
            if param.ndim == 1 and any(
                layer_norm_name in name for layer_norm_name in layer_norm_names
            ):
                param.data = param.data.to(torch.float32)
    if loaded_in_8bit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    if hasattr(model, output_embedding_layer_name):
        output_embedding_layer = getattr(model, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):
            r"""
            Manually cast to the expected dtype of the lm_head as sometimes there is a final layer norm that is casted
            in fp32
            """

            def forward(self, z_):
                return super().forward(z_.to(input_dtype)).to(torch.float32)

        setattr(
            model,
            output_embedding_layer_name,
            CastOutputToFloat(output_embedding_layer),
        )
    return model


hf_model_name = "decapoda-research/llama-7b-hf"

config = LLaMAConfig.from_pretrained(hf_model_name)
avoid_tensor_modified()
transformers.modeling_utils._init_weights = False
torch.set_default_dtype(torch.half)
model_ori = LLaMAForCausalLM(config)
torch.set_default_dtype(torch.float)
print_model(model_ori, show_buffer=True)
print("*" * 80)
# import pudb; pu.db
model = load_quant(
    model_ori,
    "pyllama-7B2b.2.0.0+cu118.pt",
    2,
    ["lm_head"],
    seqlen=1024,
    for_infer=True,
    dev=torch.device("cuda:0"),
    verbose=1,
)


# model.is_loaded_in_8bit = True
# model._is_int8_training_enabled = True

# print(model)
# exit(0)

"""model = LLaMAForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)"""
tokenizer = LLaMATokenizer.from_pretrained(hf_model_name, add_eos_token=True)

model = prepare_model_for_int4_training(model)

print_model(model_ori, show_buffer=True)
# import pudb; pu.db
print("o" * 80)
print(model.model.layers[0].self_attn.q_proj.scales)
print(model.model.layers[0].self_attn.q_proj.zeros)
print(model.model.layers[0].self_attn.q_proj.bias)
print("o" * 80)

# ["q_proj", "v_proj", "k_proj", "gate_proj", "down_proj", "up_proj"],
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["k_proj", "down_proj", "up_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    bits=2,
    max_lora_layers=5,
)

# import pudb; pu.db
model = get_peft_model(model, config)


tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
data = load_dataset("json", data_files="dataset/alpaca_data.json")

model.print_trainable_parameters()


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


def tokenize(prompt):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }


data = data.shuffle().map(lambda x: tokenize(generate_prompt(x)))

"""
val_set_size=2000
train_val = data["train"].train_test_split(
    test_size=val_set_size, shuffle=True, seed=42
)
train_data = (
    train_val["train"].shuffle().map(generate_and_tokenize_prompt)
)
val_data = (
    train_val["test"].shuffle().map(generate_and_tokenize_prompt)
)
"""

device = torch.device("cuda:0")

for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to(device)

for name, buffer in model.named_buffers():
    buffer.data = buffer.data.to(device)

print_model(model_ori, show_buffer=True)

training_args = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=2,  # GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=2,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=2,
    logging_dir="./logs",
    optim="adamw_torch",
    output_dir="lora-alpaca",
    save_total_limit=3,
    evaluation_strategy="no",
    save_strategy="steps",
    eval_steps=None,
    save_steps=20,
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train(resume_from_checkpoint=False)

model.save_pretrained("lora-alpaca")
