```
ubuntu@research:~/pyllama$ python -m llama.llama_quant decapoda-research/llama-7b-hf c4 --load q --local_model pyllama-7B4b.2.0.0+cu118.pt --wbits 4 --benchmark 1024 --max_length 64 --check
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
python -m llama.llama_quant decapoda-research/llama-7b-hf c4 --load q --local_model pyllama-7B2b.2.0.0+cu118.pt --wbits 2 --benchmark 1024 --max_length 64 --check
Number of parameters: 262410240
Median: 0.035959482192993164
PPL: 1580.4105224609375
Max memory(MiB): 2580.2841796875

python -m llama.llama_quant decapoda-research/llama-7b-hf c4 --load q --local_model pyllama-7B3b.2.0.0+cu118.pt --wbits 3 --benchmark 1024 --max_length 64 --check
Median: 0.036728501319885254
PPL: 10.22502326965332
Max memory(MiB): 3352.2841796875

＞CUDA_VISIBLE_DEVICES=1 python -m llama.llama_quant decapoda-research/llama-65b-hf c4 --load q --local_model pyllama-65B2b.torch20.pt --wbits 2 --benchmark 1024 --max_length 64 --perplexity
🌳 LLaMAForCausalLM
├── model(LLaMAModel)
│   ├── embed_tokens(Embedding) weight[32000,8192](fp16)
│   ├── layers(ModuleList)
│   │   └── +0-79(LLaMADecoderLayer)
│   │       ├── self_attn(LLaMAAttention)
│   │       │   ├── +q_proj,k_proj,v_proj,o_proj(QuantLinear) qweight[512,8192](i32)❄️  shift[8192,1]❄️  scales[8192,1]❄️  bias[8192]❄️
│   │       │   └── rotary_emb(RotaryEmbedding) inv_freq[64]❄️
│   │       ├── mlp(LLaMAMLP)
│   │       │   ├── +gate_proj,up_proj(QuantLinear) qweight[512,22016](i32)❄️  shift[22016,1]❄️  scales[22016,1]❄️  bias[22016]❄️
│   │       │   └── down_proj(QuantLinear) qweight[1376,8192](i32)❄️  shift[8192,1]❄️  scales[8192,1]❄️  bias[8192]❄️
│   │       └── +input_layernorm,post_attention_layernorm(RMSNorm) weight[8192](fp16)
│   └── norm(RMSNorm) weight[8192](fp16)
└── lm_head(Linear) weight[32000,8192](fp16)
Number of parameters: 65285660672
Median: 0.08878755569458008
PPL: 32758.173828125
Max memory(MiB): 19328.5654296875
```
