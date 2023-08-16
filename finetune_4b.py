import logging

logging.getLogger("datasets.builder").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import transformers
from datasets import load_dataset
from gptq import avoid_tensor_modified, load_quant
from gptq.runner_utils import get_model_tag, load_mixed_quant
from gptq.runner import run, Runner
from gptq.utils import DATASET_LIST, get_args, get_model, print_model
from hiq.vis import print_model
from peft import LoraConfig, get_peft_model
from llama.hf import LLaMAConfig, LLaMAForCausalLM, LLaMATokenizer

# optimized for RTX 4090. for larger GPUs, increase some of these?
MICRO_BATCH_SIZE = 10  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 256
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 1  # we don't need 3 tbh
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


from typing import Tuple, Optional, Callable

import torch
from torch.optim.optimizer import Optimizer

# functions


def exists(val):
    return val is not None


# update functions


def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # stepweight decay

    p.data.mul_(1 - lr * wd)

    # weight update

    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1).sign_()
    p.add_(update, alpha=-lr)

    # decay the momentum running average coefficient

    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)


# class


class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_triton: bool = False,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)

        super().__init__(params, defaults)

        self.update_fn = update_fn

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group["params"]):
                grad, lr, wd, beta1, beta2, state = (
                    p.grad,
                    group["lr"],
                    group["weight_decay"],
                    *group["betas"],
                    self.state[p],
                )

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                self.update_fn(p, grad, exp_avg, lr, wd, beta1, beta2)

        return loss




def prepare_model_for_int4_training(
    model,
    output_embedding_layer_name="lm_head",
    use_gradient_checkpointing=True,
    layer_norm_names=("layer_norm",),
):
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False
        if 1:
            # cast layer norm in fp32 for stability for 8bit models
            if param.ndim == 1 and any(
                layer_norm_name in name for layer_norm_name in layer_norm_names
            ):
                param.data = param.data.to(torch.float32)
    if use_gradient_checkpointing:
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


args = get_args()
ru = Runner(args)
data = ru.load_custom_data("dataset/alpaca_data_cleaned.json")
ru.run()
model = ru.model

model = prepare_model_for_int4_training(model)

# print_model(model_ori, show_buffer=True)
# import pudb; pu.db
print("o" * 80)
print(model.model.layers[0].self_attn.q_proj.scales)
print(model.model.layers[0].self_attn.q_proj.shift)
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
    bits=args.bits,
    max_lora_layers=500,
)

# import pudb; pu.db
model = get_peft_model(model, config)

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
    result = ru.tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }


data = ru.data.shuffle().map(lambda x: tokenize(generate_prompt(x)))

device = torch.device("cuda:0")

for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to(device)
for name, buffer in model.named_buffers():
    buffer.data = buffer.data.to(device)

# print_model(model_ori, show_buffer=True)
optimizer = Lion(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
)

training_args = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    num_train_epochs=1,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=2,
    logging_dir="./logs-20230409",
    optim="adamw_torch",
    #optimizer=optimizer,
    output_dir=f"lora-alpaca-{args.bits}",
    save_total_limit=5,
    evaluation_strategy="no",
    save_strategy="steps",
    eval_steps=None,
    save_steps=20,
    report_to=None,
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(
        ru.tokenizer, pad_to_multiple_of=8, return_tensors="pt", mlm=False
    ),
)
model.config.use_cache = False
trainer.train(resume_from_checkpoint=False)
model.save_pretrained(f"lora-alpaca-{args.bits}bit")
