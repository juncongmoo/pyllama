from .generation import LLaMA


def pyllama_env(x, default=None) -> bool:
    import os, ast
    t = os.environ.get(x, default)
    if isinstance(t, str) and t:
        try:
            return bool(ast.literal_eval(t))
        except:
            return True
    return bool(t)

if pyllama_env("PYLLAMA_META_MP"):
    from .model_parallel import ModelArgs, Transformer
else:
    from .model_single import ModelArgs, Transformer
from .tokenizer import Tokenizer

__version__ = "0.0.2"
