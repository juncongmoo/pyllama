## Generate Quantized Model

```
python -m llama.llama_quant decapoda-research/llama-30b-hf c4 --load hf --mode q --bits 4 --save pyllama-30B4b.2.0.0+cu118.pt
```

## Benchmarking

### 7B

```
python -m llama.llama_quant decapoda-research/llama-7b-hf c4 --load q --model_path pyllama-7B2b.2.0.0+cu118.pt --bits 2 --benchmark 1024 --max_length 64 --check
Number of parameters: 262410240
Median: 0.035959482192993164
PPL: 1580.4105224609375
Max memory(MiB): 2580.2841796875

python -m llama.llama_quant decapoda-research/llama-7b-hf c4 --load q --model_path pyllama-7B3b.2.0.0+cu118.pt --bits 3 --benchmark 1024 --max_length 64 --check
Median: 0.036728501319885254
PPL: 10.22502326965332
Max memory(MiB): 3352.2841796875

ubuntu@research:~/pyllama$ python -m llama.llama_quant decapoda-research/llama-7b-hf c4 --load q --model_path pyllama-7B4b.2.0.0+cu118.pt --bits 4 --benchmark 1024 --max_length 64 --check
🌳 LLaMAForCausalLM
├── model(LLaMAModel)
│   ├── embed_tokens(Embedding) weight[32000,4096](fp16)
│   ├── layers(ModuleList)
│   │   └── +0-31(LLaMADecoderLayer)
│   │       ├── self_attn(LLaMAAttention)
│   │       ├── mlp(LLaMAMLP)
│   │       └── +input_layernorm,post_attention_layernorm(RMSNorm) weight[4096](fp16)
│   └── norm(RMSNorm) weight[4096](fp16)
└── lm_head(Linear) weight[32000,4096](fp16)
Number of parameters: 262410240
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
Median: 0.03691422939300537
PPL: 5.136333465576172
Max memory(MiB): 4172.2841796875
...
Median: 0.045873045921325684
PPL: 5.137877464294434
Max memory(MiB): 4172.2841796875
```

### 4bit

After finetuning, the performance is worse in the original dataset.

After 80 steps:

```
Number of parameters: 6758076416
Processing C4 Samples: 100%|█████████| 128/128 [00:06<00:00, 19.03it/s]
Benchmarking: 100%|████████████████| 1024/1024 [01:03<00:00, 16.15it/s]
Median: 0.06077373027801514
PPL: 12.594114303588867
Max memory(MiB): 2989.69189453125
Number of parameters: 6758076416
Processing C4 Samples: 100%|█████████| 128/128 [00:06<00:00, 19.42it/s]
Benchmarking: 100%|████████████████| 1024/1024 [01:02<00:00, 16.39it/s]
Median: 0.05998396873474121
PPL: 12.597537994384766
Max memory(MiB): 2989.7041015625
Number of parameters: 6758076416
Processing Wikitext2 Samples: 100%|█████████████| 128/128 [00:00<00:00, 33944.80it/s]
Benchmarking: 100%|████████████████| 1024/1024 [01:03<00:00, 16.24it/s]
Median: 0.0605393648147583
PPL: 8.711161613464355
Max memory(MiB): 2989.7041015625
Number of parameters: 6758076416
Processing PTB Samples: 100%|█████| 128/128 [00:00<00:00, 34307.04it/s]
Benchmarking: 100%|████████████████| 1024/1024 [01:04<00:00, 15.99it/s]
Median: 0.061057090759277344
PPL: 9.273012161254883
Max memory(MiB): 2989.7041015625
```

After 100 steps:

```
Number of parameters: 6758076416
Processing C4 Samples: 100%|█████████| 128/128 [00:02<00:00, 54.37it/s]
Benchmarking: 100%|████████████████| 1024/1024 [00:49<00:00, 20.89it/s]
Median: 0.047434449195861816
PPL: 12.870625495910645
Max memory(MiB): 2989.69189453125
Number of parameters: 6758076416
Processing C4 Samples: 100%|█████████| 128/128 [00:02<00:00, 55.65it/s]
Benchmarking: 100%|████████████████| 1024/1024 [00:50<00:00, 20.33it/s]
Median: 0.04917097091674805
PPL: 12.872316360473633
Max memory(MiB): 2989.69189453125
Number of parameters: 6758076416
Processing Wikitext2 Samples: 100%|███████| 128/128 [00:00<00:00, 52852.03it/s]
Benchmarking: 100%|████████████████| 1024/1024 [00:50<00:00, 20.30it/s]
Median: 0.04886960983276367
PPL: 9.596293449401855
Max memory(MiB): 2989.69189453125
Number of parameters: 6758076416
Processing PTB Samples: 100%|█████| 128/128 [00:00<00:00, 55496.27it/s]
Benchmarking: 100%|████████████████| 1024/1024 [00:50<00:00, 20.31it/s]
Median: 0.04884529113769531
PPL: 9.792820930480957
Max memory(MiB): 2989.69189453125
```


After 60 steps:

```
Number of parameters: 6758076416
Processing C4 Samples: 100%|█████████| 128/128 [00:02<00:00, 53.99it/s]
Benchmarking: 100%|████████████████| 1024/1024 [00:49<00:00, 20.75it/s]
Median: 0.04757583141326904
PPL: 12.434136390686035
Max memory(MiB): 2989.69189453125
Number of parameters: 6758076416
Processing C4 Samples: 100%|█████████| 128/128 [00:02<00:00, 55.74it/s]
Benchmarking: 100%|████████████████| 1024/1024 [00:48<00:00, 20.93it/s]
Median: 0.04744601249694824
PPL: 12.43647289276123
Max memory(MiB): 2989.69189453125
Number of parameters: 6758076416
Processing Wikitext2 Samples: 100%|█████| 128/128 [00:00<00:00, 52624.08it/s]
Benchmarking: 100%|████████████████| 1024/1024 [00:50<00:00, 20.21it/s]
Median: 0.04935407638549805
PPL: 8.407084465026855
Max memory(MiB): 2989.69189453125
Number of parameters: 6758076416
Processing PTB Samples: 100%|█████| 128/128 [00:00<00:00, 54621.11it/s]
Benchmarking: 100%|████████████████| 1024/1024 [00:51<00:00, 20.04it/s]
Median: 0.05000042915344238
PPL: 9.228442192077637
Max memory(MiB): 2989.69189453125
```

### 13B

```
python -m llama.llama_quant decapoda-research/llama-13b-hf c4 --load q --model_path pyllama-13B2b.2.0.0+cu118.pt --bits 2 --benchmark 1024 --max_length 64 --perplexity
🌳 LLaMAForCausalLM<trainable_params:328094720,all_params:13015864320,percentage:2.52073%>
├── LLaMAModel(model)
│   ├── Embedding(embed_tokens)|weight[32000,5120]<f16>
│   ├── ModuleList(layers)
│   │   └── 💠 LLaMADecoderLayer(0-39)<🦜:10240,317204480x40>
│   │       ┣━━ LLaMAAttention(self_attn)
│   │       ┃   ┣━━ 💠 QuantLinear(q_proj,k_proj,v_proj,o_proj)<🦜:0,26214400x4>|qweight[320,5120]<i32>❄️|shift[5120,1]❄️|scales[5120,1]❄️|bias[5120]❄️
│   │       ┃   ┗━━ RotaryEmbedding(rotary_emb)|inv_freq[64]❄️
│   │       ┣━━ LLaMAMLP(mlp)
│   │       ┃   ┣━━ 💠 QuantLinear(gate_proj,up_proj)<🦜:0,70778880x2>|qweight[320,13824]<i32>❄️|shift[13824,1]❄️|scales[13824,1]❄️|bias[13824]❄️
│   │       ┃   ┗━━ QuantLinear(down_proj)|qweight[864,5120]<i32>❄️|shift[5120,1]❄️|scales[5120,1]❄️|bias[5120]❄️
│   │       ┗━━ 💠 RMSNorm(input_layernorm,post_attention_layernorm)<🦜:5120x2>|weight[5120]<f16>
│   └── RMSNorm(norm)|weight[5120]<f16>
└── Linear(lm_head)|weight[32000,5120]<f16>
Number of parameters: 13015864320
Median: 0.04523146152496338
PPL: 215.22183227539062
Max memory(MiB): 4483.3935546875

python -m llama.llama_quant --model decapoda-research/llama-13b-hf c4 --load q --model_path pyllama-13B3b.2.0.0+cu118.pt --bits 3 --benchmark 1024 --max_length 64 --perplexity
🌳 LLaMAForCausalLM<trainable_params:328094720,all_params:13015864320,percentage:2.52073%>
├── LLaMAModel(model)
│   ├── Embedding(embed_tokens)|weight[32000,5120]<f16>
│   ├── ModuleList(layers)
│   │   └── 💠 LLaMADecoderLayer(0-39)<🦜:10240,317204480x40>
│   │       ┣━━ LLaMAAttention(self_attn)
│   │       ┃   ┣━━ 💠 QuantLinear(q_proj,k_proj,v_proj,o_proj)<🦜:0,26214400x4>|qweight[480,5120]<i32>❄️|shift[5120,1]❄️|scales[5120,1]❄️|bias[5120]❄️
│   │       ┃   ┗━━ RotaryEmbedding(rotary_emb)|inv_freq[64]❄️
│   │       ┣━━ LLaMAMLP(mlp)
│   │       ┃   ┣━━ 💠 QuantLinear(gate_proj,up_proj)<🦜:0,70778880x2>|qweight[480,13824]<i32>❄️|shift[13824,1]❄️|scales[13824,1]❄️|bias[13824]❄️
│   │       ┃   ┗━━ QuantLinear(down_proj)|qweight[1296,5120]<i32>❄️|shift[5120,1]❄️|scales[5120,1]❄️|bias[5120]❄️
│   │       ┗━━ 💠 RMSNorm(input_layernorm,post_attention_layernorm)<🦜:5120x2>|weight[5120]<f16>
│   └── RMSNorm(norm)|weight[5120]<f16>
└── Linear(lm_head)|weight[32000,5120]<f16>
Number of parameters: 13015864320
Median: 0.045313119888305664
PPL: 4.71307897567749
Max memory(MiB): 6078.3935546875

python -m llama.llama_quant --model decapoda-research/llama-13b-hf c4 --load q --model_path pyllama-13B4b.2.0.0+cu118.pt --bits 4 --benchmark 1024 --max_length 64 --perplexity
🌳 LLaMAForCausalLM<trainable_params:328094720,all_params:13015864320,percentage:2.52073%>
├── LLaMAModel(model)
│   ├── Embedding(embed_tokens)|weight[32000,5120]<f16>
│   ├── ModuleList(layers)
│   │   └── 💠 LLaMADecoderLayer(0-39)<🦜:10240,317204480x40>
│   │       ┣━━ LLaMAAttention(self_attn)
│   │       ┃   ┣━━ 💠 QuantLinear(q_proj,k_proj,v_proj,o_proj)<🦜:0,26214400x4>|qweight[640,5120]<i32>❄️|shift[5120,1]❄️|scales[5120,1]❄️|bias[5120]❄️
│   │       ┃   ┗━━ RotaryEmbedding(rotary_emb)|inv_freq[64]❄️
│   │       ┣━━ LLaMAMLP(mlp)
│   │       ┃   ┣━━ 💠 QuantLinear(gate_proj,up_proj)<🦜:0,70778880x2>|qweight[640,13824]<i32>❄️|shift[13824,1]❄️|scales[13824,1]❄️|bias[13824]❄️
│   │       ┃   ┗━━ QuantLinear(down_proj)|qweight[1728,5120]<i32>❄️|shift[5120,1]❄️|scales[5120,1]❄️|bias[5120]❄️
│   │       ┗━━ 💠 RMSNorm(input_layernorm,post_attention_layernorm)<🦜:5120x2>|weight[5120]<f16>
│   └── RMSNorm(norm)|weight[5120]<f16>
└── Linear(lm_head)|weight[32000,5120]<f16>
Number of parameters: 13015864320
Median: 0.04819202423095703
PPL: 4.254908084869385
Max memory(MiB): 3773.23486328125
```

### 65B

```

🅰️  python -m llama.llama_quant decapoda-research/llama-65b-hf c4 --load q --model_path pyllama-65B2b.2.0.0+cu118.pt --bits 2 --benchmark 1024 --max_length 64 --perplexity
🌳 LLaMAForCausalLM<trainable_params:65285660672,all_params:525606912,percentage:0.80509%>
├── LLaMAModel(model)
│   ├── Embedding(embed_tokens)|weight[32000,8192]<f16>
│   ├── ModuleList(layers)
│   │   └── 💠 LLaMADecoderLayer(0-79)<🦜:16384,809517056x80>
│   │       ┣━━ LLaMAAttention(self_attn)
│   │       ┃   ┣━━ 💠 QuantLinear(q_proj,k_proj,v_proj,o_proj)<🦜:0,67108864x4>|qweight[512,8192]<i32>❄️|shift[8192,1]❄️|scales[8192,1]❄️|bias[8192]❄️
│   │       ┃   ┗━━ RotaryEmbedding(rotary_emb)|inv_freq[64]❄️
│   │       ┣━━ LLaMAMLP(mlp)
│   │       ┃   ┣━━ 💠 QuantLinear(gate_proj,up_proj)<🦜:0,180355072x2>|qweight[512,22016]<i32>❄️|shift[22016,1]❄️|scales[22016,1]❄️|bias[22016]❄️
│   │       ┃   ┗━━ QuantLinear(down_proj)|qweight[1376,8192]<i32>❄️|shift[8192,1]❄️|scales[8192,1]❄️|bias[8192]❄️
│   │       ┗━━ 💠 RMSNorm(input_layernorm,post_attention_layernorm)<🦜:8192x2>|weight[8192]<f16>
│   └── RMSNorm(norm)|weight[8192]<f16>
└── Linear(lm_head)|weight[32000,8192]<f16>
Median: 0.0913853645324707
PPL: 14.639993667602539
Max memory(MiB): 19328.5654296875

＞python -m pudb -m llama.llama_quant decapoda-research/llama-65b-hf c4 --load q --model_path pyllama-65B3b.2.0.0+cu118.pt --bits 3 --benchmark 1024 --max_length 64 --perplexity
🌳 LLaMAForCausalLM<trainable_params:525606912,all_params:65285660672,percentage:0.80509%>
├── LLaMAModel(model)
│   ├── Embedding(embed_tokens)|weight[32000,8192]<f16>
│   ├── ModuleList(layers)
│   │   └── 💠 LLaMADecoderLayer(0-79)<🦜:16384,809517056x80>
│   │       ┣━━ LLaMAAttention(self_attn)
│   │       ┃   ┣━━ 💠 QuantLinear(q_proj,k_proj,v_proj,o_proj)<🦜:0,67108864x4>|qweight[768,8192]<i32>❄️|shift[8192,1]❄️|scales[8192,1]❄️|bias[8192]❄️
│   │       ┃   ┗━━ RotaryEmbedding(rotary_emb)|inv_freq[64]❄️
│   │       ┣━━ LLaMAMLP(mlp)
│   │       ┃   ┣━━ 💠 QuantLinear(gate_proj,up_proj)<🦜:0,180355072x2>|qweight[768,22016]<i32>❄️|shift[22016,1]❄️|scales[22016,1]❄️|bias[22016]❄️
│   │       ┃   ┗━━ QuantLinear(down_proj)|qweight[2064,8192]<i32>❄️|shift[8192,1]❄️|scales[8192,1]❄️|bias[8192]❄️
│   │       ┗━━ 💠 RMSNorm(input_layernorm,post_attention_layernorm)<🦜:8192x2>|weight[8192]<f16>
│   └── RMSNorm(norm)|weight[8192]<f16>
└── Linear(lm_head)|weight[32000,8192]<f16>
Median: 0.12354862689971924
PPL: 4.009398460388184
Max memory(MiB): 13408.31298828125
```


```
> python -m llama.llama_quant decapoda-research/llama-30b-hf c4 --load q --model_path pyllama-30B4b.2.0.0+cu118.pt --bits 4 --benchmark 1024 --max_length 64 --perplexity
🌳 LLaMAForCausalLM<trainable_params:32528943616,all params:426789376,percentage:1.31203%>
├── LLaMAModel(model)
│   ├── Embedding(embed_tokens)|weight[32000,6656]<f16>
│   ├── ModuleList(layers)
│   │   └── 💠 LLaMADecoderLayer(0-59)<P:13312,535049216x60>
│   │       ┣━━ LLaMAAttention(self_attn)
│   │       ┃   ┣━━ 💠 QuantLinear(q_proj,k_proj,v_proj,o_proj)<P:0,44302336x4>|qweight[832,6656]<i32>❄️|shift[6656,1]❄️|scales[6656,1]❄️|bias[6656]❄️
│   │       ┃   ┗━━ RotaryEmbedding(rotary_emb)|inv_freq[64]❄️
│   │       ┣━━ LLaMAMLP(mlp)
│   │       ┃   ┣━━ 💠 QuantLinear(gate_proj,up_proj)<P:0,119275520x2>|qweight[832,17920]<i32>❄️|shift[17920,1]❄️|scales[17920,1]❄️|bias[17920]❄️
│   │       ┃   ┗━━ QuantLinear(down_proj)|qweight[2240,6656]<i32>❄️|shift[6656,1]❄️|scales[6656,1]❄️|bias[6656]❄️
│   │       ┗━━ 💠 RMSNorm(input_layernorm,post_attention_layernorm)<P:6656x2>|weight[6656]<f16>
│   └── RMSNorm(norm)|weight[6656]<f16>
└── Linear(lm_head)|weight[32000,6656]<f16>
Number of parameters: 32528943616
Median: 0.07418131828308105
PPL: 3.8093886375427246
Max memory(MiB): 18006.224609375
```