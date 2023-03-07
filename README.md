# 🦙 LLaMA - Run LLM in A Single GPU


> 📢 `pyllama` is a hacked version of `LLaMA` based on original Facebook's implementation but more convenient to run in a Single consumer grade GPU.


## Setup

In a conda env with pytorch / cuda available, run
```
pip install pyllama
```

> 🐏 If you have installed llama library from other sources, please uninstall the previous llama library and use `pip install pyllama -U` to install the latest version.

## Single GPU Inference

Set the environment variables `CKPT_DIR` as your llamm model folder, for example `/llama_data/7B`, and `TOKENIZER_PATH` as your tokenizer's path, such as `/llama_data/tokenizer.model`.

And then run the following command:

```bash
python inference.py --ckpt_dir $CKPT_DIR --tokenizer_path $TOKENIZER_PATH
```

The following is an example of LLaMA running in a 8GB single GPU.

![LLaMA Inference](https://raw.githubusercontent.com/juncongmoo/pyllama/main/docs/llama_inference.png)


### Tips

- To load KV cache in CPU, run `export KV_CAHCHE_IN_GPU=0` in the shell.

- To profile CPU/GPU/Latency, run:

```bash
python inference_driver.py --ckpt_dir $CKPT_DIR --tokenizer_path $TOKENIZER_PATH
```

A sample result is like:

![LLaMA Inference](https://raw.githubusercontent.com/juncongmoo/pyllama/main/docs/llama_profiling.png)


## Multiple GPU Inference

The provided `example.py` can be run on a single or multi-gpu node with `torchrun` and will output completions for two pre-defined prompts. Using `TARGET_FOLDER` as defined in `download.sh`:

```bash
torchrun --nproc_per_node MP example.py --ckpt_dir $TARGET_FOLDER/model_size --tokenizer_path $TARGET_FOLDER/tokenizer.model
```

Different models require different MP values:

|  Model | MP |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 30B    | 4  |
| 65B    | 8  |


## Download

In order to download the checkpoints and tokenizer, fill this [google form](https://forms.gle/jk851eBVbX1m5TAv5)

Once your request is approved, you will receive links to download the tokenizer and model files.
Edit the `download.sh` script with the signed url provided in the email to download the model weights and tokenizer.

### Model Card

See [MODEL_CARD.md](https://github.com/juncongmoo/pyllama/blob/main/MODEL_CARD.md)

### License

See the [LICENSE](https://github.com/juncongmoo/pyllama/blob/main/LICENSE) file.
