import hiq, time
from hiq.memory import total_gpu_memory_mb, get_memory_mb


def run_main():
    driver = hiq.HiQLatency(
        hiq_table_or_path=[
            ["inference", "", "load", "load_llama"],
            ["llama.generation", "LLaMA", "generate", "generate"],
            # ["llama.model_single", "Transformer", "forward", "forward"],
        ],
        metric_funcs=[time.time, total_gpu_memory_mb, get_memory_mb],
        # extra_metrics={hiq.ExtraMetrics.ARGS},
    )

    args = hiq.mod("inference").get_args()
    hiq.mod("inference").run(args.ckpt_dir, args.tokenizer_path)
    print("*" * 30, "GPU/CPU/Latency Profiling", "*" * 30)
    driver.show()


if __name__ == "__main__":
    run_main()
