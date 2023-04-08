## Generate Quantized Model

```
python -m llama.llama_quant decapoda-research/llama-30b-hf c4 --load hf --mode q --bits 4 --save pyllama-30B4b.2.0.0+cu118.pt
```

## Benchmarking

```
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
Benchmarking ...
100%|████████████████████████████████████████████████████| 1024/1024 [00:38<00:00, 26.87it/s]
Median: 0.03691422939300537
PPL: 5.136333465576172
Max memory(MiB): 4172.2841796875
...
Median: 0.045873045921325684
PPL: 5.137877464294434
Max memory(MiB): 4172.2841796875
```

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
```


```
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
