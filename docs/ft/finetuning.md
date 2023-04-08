

After Lora Finetuning:

```
python -m llama.llama_quant decapoda-research/llama-7b-hf c4 --load lora --bits 2 --benchmark 1024 --max_length 64 --check
🌳 LLaMAForCausalLM
├── model(LLaMAModel)
│   ├── embed_tokens(Embedding) weight[32000,4096](fp16)(cuda:0)❄️
│   ├── layers(ModuleList)
│   │   ├── 0(LLaMADecoderLayer)
│   │   │   ├── self_attn(LLaMAAttention)
│   │   │   │   ├── +q_proj,v_proj,o_proj(QuantLinear) qweight[256,4096](i32)(cuda:0)❄️  zeros[4096,1](cuda:0)❄️  scales[4096,1](cuda:0)❄️  bias[4096](cuda:0)❄️
│   │   │   │   ├── k_proj(MergedQuantLinear) qweight[256,4096](i32)(cuda:0)❄️  zeros[4096,1](cuda:0)❄️  scales[4096,1](cuda:0)❄️  bias[4096](cuda:0)❄️
│   │   │   │   │   ├── lora_A(Linear) weight[64,4096](fp16)(cuda:0)
│   │   │   │   │   └── lora_B(Linear) weight[4096,64](fp16)(cuda:0)
│   │   │   │   └── rotary_emb(RotaryEmbedding) inv_freq[64](cuda:0)❄️
│   │   │   ├── mlp(LLaMAMLP)
│   │   │   │   ├── gate_proj(QuantLinear) qweight[256,11008](i32)(cuda:0)❄️  zeros[11008,1](cuda:0)❄️  scales[11008,1](cuda:0)❄️  bias[11008](cuda:0)❄️
│   │   │   │   ├── down_proj(MergedQuantLinear) qweight[688,4096](i32)(cuda:0)❄️  zeros[4096,1](cuda:0)❄️  scales[4096,1](cuda:0)❄️  bias[4096](cuda:0)❄️
│   │   │   │   │   ├── lora_A(Linear) weight[64,11008](fp16)(cuda:0)
│   │   │   │   │   └── lora_B(Linear) weight[4096,64](fp16)(cuda:0)
│   │   │   │   └── up_proj(MergedQuantLinear) qweight[256,11008](i32)(cuda:0)❄️  zeros[11008,1](cuda:0)❄️  scales[11008,1](cuda:0)❄️  bias[11008](cuda:0)❄️
│   │   │   │       ├── lora_A(Linear) weight[64,4096](fp16)(cuda:0)
│   │   │   │       └── lora_B(Linear) weight[11008,64](fp16)(cuda:0)
│   │   │   └── +input_layernorm,post_attention_layernorm(RMSNorm) weight[4096](fp16)(cuda:0)❄️
│   │   ├── 1(LLaMADecoderLayer)
│   │   │   ├── self_attn(LLaMAAttention)
│   │   │   │   ├── +q_proj,v_proj,o_proj(QuantLinear) qweight[256,4096](i32)(cuda:0)❄️  zeros[4096,1](cuda:0)❄️  scales[4096,1](cuda:0)❄️  bias[4096](cuda:0)❄️
│   │   │   │   ├── k_proj(MergedQuantLinear) qweight[256,4096](i32)(cuda:0)❄️  zeros[4096,1](cuda:0)❄️  scales[4096,1](cuda:0)❄️  bias[4096](cuda:0)❄️
│   │   │   │   │   ├── lora_A(Linear) weight[64,4096](fp16)(cuda:0)
│   │   │   │   │   └── lora_B(Linear) weight[4096,64](fp16)(cuda:0)
│   │   │   │   └── rotary_emb(RotaryEmbedding) inv_freq[64](cuda:0)❄️
│   │   │   ├── mlp(LLaMAMLP)
│   │   │   │   ├── +gate_proj,up_proj(QuantLinear) qweight[256,11008](i32)(cuda:0)❄️  zeros[11008,1](cuda:0)❄️  scales[11008,1](cuda:0)❄️  bias[11008](cuda:0)❄️
│   │   │   │   └── down_proj(MergedQuantLinear) qweight[688,4096](i32)(cuda:0)❄️  zeros[4096,1](cuda:0)❄️  scales[4096,1](cuda:0)❄️  bias[4096](cuda:0)❄️
│   │   │   │       ├── lora_A(Linear) weight[64,11008](fp16)(cuda:0)
│   │   │   │       └── lora_B(Linear) weight[4096,64](fp16)(cuda:0)
│   │   │   └── +input_layernorm,post_attention_layernorm(RMSNorm) weight[4096](fp16)(cuda:0)❄️
│   │   └── +2-31(LLaMADecoderLayer)
│   │       ├── self_attn(LLaMAAttention)
│   │       │   ├── +q_proj,k_proj,v_proj,o_proj(QuantLinear) qweight[256,4096](i32)(cuda:0)❄️  zeros[4096,1](cuda:0)❄️  scales[4096,1](cuda:0)❄️  bias[4096](cuda:0)❄️
│   │       │   └── rotary_emb(RotaryEmbedding) inv_freq[64](cuda:0)❄️
│   │       ├── mlp(LLaMAMLP)
│   │       │   ├── +gate_proj,up_proj(QuantLinear) qweight[256,11008](i32)(cuda:0)❄️  zeros[11008,1](cuda:0)❄️  scales[11008,1](cuda:0)❄️  bias[11008](cuda:0)❄️
│   │       │   └── down_proj(QuantLinear) qweight[688,4096](i32)(cuda:0)❄️  zeros[4096,1](cuda:0)❄️  scales[4096,1](cuda:0)❄️  bias[4096](cuda:0)❄️
│   │       └── +input_layernorm,post_attention_layernorm(RMSNorm) weight[4096](fp16)(cuda:0)❄️
│   └── norm(RMSNorm) weight[4096](fp16)(cuda:0)❄️
└── lm_head(CastOutputToFloat)
    └── 0(Linear) weight[32000,4096](fp16)(cuda:0)❄️
Number of parameters: 6742364160
Processing C4 Samples: 100%|██████████████████████████████████████████| 128/128 [00:06<00:00, 19.31it/s]
Benchmarking: 100%|███████████████████████████████████████████████████| 1024/1024 [00:50<00:00, 20.45it/s]
Median: 0.04806506633758545
PPL: 2063.745849609375
Max memory(MiB): 2587.87646484375
```
