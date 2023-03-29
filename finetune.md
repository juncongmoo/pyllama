# Finetune LLaMA With Ease

## Fine-tuning with Naive Pipeline Parallel

For *fully* fine-tuning (larger) models, we can use (a very naively implemented version of) pipeline parallelism. This is preferable for larger models that won't fit on a single GPU.

```bash
python finetune_pp.py \
    --model_path $CKPT_DIR \
    --dataset_path /home/wfh/workspace/minimal-llama/tokenized_dataset \
    --save_dir fine_tune_dir \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_interval 2000 \
    --num_train_steps 20000
```



