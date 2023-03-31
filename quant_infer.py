import hiq, time
from hiq.memory import total_gpu_memory_mb, get_memory_mb
import platform


def main():

    try:
        wsl = 'microsoft' in platform.uname()[2].lower()
    except:
        wsl = False

    driver = hiq.HiQLatency(
        hiq_table_or_path=[
            ["llama.llama_infer", "", "run", "run_quant"],
            ["llama.llama_infer", "LLaMATokenizer", "from_pretrained", "from_pretrained"],
            ["llama.hf", "LLaMATokenizer", "encode", "encode"],
            ["llama.llama_infer", "", "load_quant", "load_quant"],
            ["llama.hf.modeling_llama", "LLaMAForCausalLM", "generate", "generate"]
        ],
        metric_funcs=[time.time, get_memory_mb] + ([total_gpu_memory_mb] if not wsl else []),  # WSL does not contain nvidia-smi
        # extra_metrics={hiq.ExtraMetrics.ARGS},
    )

    args = hiq.mod("llama.llama_infer").get_args()
    hiq.mod("llama.llama_infer").run(args)
    print("*" * 30, ("GPU/" if not wsl else "") + "CPU/Latency Profiling", "*" * 30)
    if wsl:
        print('(WSL does not contain nvidia-smi, GPU profiling is disabled)')
    driver.show()


if __name__ == "__main__":
    main()
