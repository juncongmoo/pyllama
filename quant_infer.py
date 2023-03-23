import hiq, time
from hiq.memory import total_gpu_memory_mb, get_memory_mb


def main():
    driver = hiq.HiQLatency(
        hiq_table_or_path=[
          ["llama.llama_infer", "", "run", "run_quant"],
          ["llama.llama_infer", "LLaMATokenizer", "from_pretrained", "from_pretrained"],
          ["llama.hf", "LLaMATokenizer", "encode", "encode"],
          ["llama.llama_infer", "", "load_quant", "load_quant"],
          ["llama.hf.modeling_llama","LLaMAForCausalLM","generate","generate"]
        ],
        metric_funcs=[time.time, total_gpu_memory_mb, get_memory_mb],
        # extra_metrics={hiq.ExtraMetrics.ARGS},
    )

    args = hiq.mod("llama.llama_infer").get_args()
    hiq.mod("llama.llama_infer").run(args)
    print("*" * 30, "GPU/CPU/Latency Profiling", "*" * 30)
    driver.show()


if __name__ == "__main__":
    main()




