import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from llama.hf import LLaMATokenizer, LLaMAForCausalLM, LLaMAConfig
from datasets import Dataset
import json

# It's not pretty down there, sorry

MODEL_PATH = 'models/llama-7b-hf'
MODEL_PATH = 'decapoda-research/llama-7b-hf'

def get_llama(model):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = LLaMAForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

model = get_llama(MODEL_PATH)
model.eval()
model.to(torch.device('cuda:0'))

config = LLaMAConfig()

tokenizer = LLaMATokenizer.from_pretrained(
    MODEL_PATH, 
    max_length=512, 
    truncation=True, 
    return_overflowing_tokens=True, 
    config=config
)

tokenizer.pad_token = config.pad_token_id

def tokenize_function(allEntries):
    return tokenizer(allEntries['text'], padding='max_length', truncation=True, max_length=512)

def to_texts(entry):
    # From https://github.com/tatsu-lab/stanford_alpaca#data-release
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.


{entry['instruction']}


{entry['input']}

### Response:
{entry['output']}"""

with open('dataset/alpaca_data.json', 'r') as f:
    data = json.load(f)

texts = list(map(to_texts, data))
dataset = Dataset.from_dict({"text": texts})
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split the tokenized dataset into training and evaluation datasets
split = tokenized_dataset.train_test_split(test_size=0.2)
train_tokenized_dataset = split['train']
eval_tokenized_dataset = split['test']

# Print the sizes of the datasets
print("Full Train dataset size:", len(train_tokenized_dataset))
print("Full Eval dataset size:", len(eval_tokenized_dataset))

# Make small datasets for testing
small_train_dataset = train_tokenized_dataset.shuffle(seed=42).select(range(1000))
small_eval_dataset = eval_tokenized_dataset.shuffle(seed=46).select(range(1000))

print("Small Train dataset size:", len(small_train_dataset))
print("Small Eval dataset size:", len(small_eval_dataset))

# https://github.com/Tencent/TencentPretrain/blob/33dbf6635eabf9efa92dacd1bad6a2d03143fa47/models/deepspeed_config.json#L4
ds_config_dict = {
  "gradient_accumulation_steps": 'auto',
  "train_micro_batch_size_per_gpu": 'auto',
  "steps_per_print": 100,
  "optimizer": {
    "type": "Adam",
    "params": {
        # "lr": 1e-4,
        # "weight_decay": 1e-2
        # Match what's in https://github.com/tatsu-lab/stanford_alpaca#fine-tuning
        "lr": 2e-5,
        "weight_decay": 1
    }
  },
  "flops_profiler": {
    "enabled": False,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 3,
    "detailed": True
  },
  "fp16": {
    "enabled": True,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 1,
    "cpu_offload": True,
    "contiguous_gradients": True,
    "overlap_comm": True,
    "reduce_scatter": False,
    "reduce_bucket_size": 5e7,
    "allgather_bucket_size": 5e7
  },
  "activation_checkpointing": {
    "partition_activations": False,
    "contiguous_memory_optimization": False,
    "cpu_checkpointing": False
  },
  "wall_clock_breakdown": False,
  "zero_allow_untested_optimizer": True,
}

# https://github.com/tatsu-lab/stanford_alpaca#fine-tuning
training_args = TrainingArguments(
    output_dir="output", 
    learning_rate=2e-5,
    num_train_epochs=1, # 3
    weight_decay=1,
    do_eval=True,
    evaluation_strategy="epoch",
    fp16=True,
    deepspeed=ds_config_dict,
    save_strategy="steps",
    save_steps=250, # 500
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=small_train_dataset, # Change to train_tokenized_dataset for full training
    eval_dataset=small_eval_dataset,   # Change to eval_tokenized_dataset for full training
    compute_metrics=compute_metrics,
)

trainer.train()
