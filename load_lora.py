from transformers import AutoTokenizer, AutoModelForCausalLM

# instantiate tokenizer and model
model = AutoModelForCausalLM.from_pretrained('pyllama_quant/2bits')
print(model.model.layers[0].self_attn.k_proj.lora_A.weight[0][:10])
