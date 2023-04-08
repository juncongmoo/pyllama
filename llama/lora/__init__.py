import os
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import (
    AlignDevicesHook,
    add_hook_to_module,
    remove_hook_from_submodules,
)
from accelerate.utils import get_balanced_memory
from gptq import QuantLinear
from huggingface_hub import hf_hub_download
from transformers.utils import PushToHubMixin

WEIGHTS_NAME = "lora_adapter_model.bin"


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError("bias")


def get_sd(model, state_dict=None):
    if state_dict is None:
        state_dict = model.state_dict()
    # to_return = lora_state_dict(model, bias=model.pconfig.bias)
    # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
    # to directly with the state dict which is necessary when using DeepSpeed or FSDP
    bias = model.pconfig.bias
    if bias == "none":
        to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}
    elif bias == "all":
        to_return = {
            k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k
        }
    elif bias == "lora_only":
        to_return = {}
        for k in state_dict:
            if "lora_" in k:
                to_return[k] = state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]
    else:
        raise NotImplementedError

    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                to_return[key] = value
    return to_return


def set_sd(model, peft_model_state_dict):
    model.load_state_dict(peft_model_state_dict, strict=False)
    return model


class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


@dataclass
class LoraConfig(object):
    base_model_name_or_path: str = field(
        default=None, metadata={"help": "The name of the base model to use."}
    )
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    bits: int = field(default=4, metadata={"help": "quantization bits number/width"})
    max_lora_layers: int = field(
        default=10, metadata={"help": "max number of lora layers"}
    )
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    merge_weights: bool = field(
        default=False,
        metadata={"help": "Merge weights of the original model and the Lora model"},
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={
            "help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"
        },
    )
    enable_lora: Optional[List[bool]] = field(
        default=None, metadata={"help": "Used with `lora.MergedLinear`."}
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )


class LoraModel(torch.nn.Module):
    def __init__(self, config, model, bits=None):
        super().__init__()
        self.pconfig = config
        self.model = model
        self.bits = bits or config.bits
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.pconfig.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.pconfig.r,
            "lora_alpha": self.pconfig.lora_alpha,
            "lora_dropout": self.pconfig.lora_dropout,
            "fan_in_fan_out": self.pconfig.fan_in_fan_out,
            "merge_weights": (
                self.pconfig.merge_weights or self.pconfig.inference_mode
            )
            and not is_hf_device_map_available,
        }
        count = 0
        for key, _ in self.model.named_modules():
            if isinstance(self.pconfig.target_modules, str):
                target_module_found = re.fullmatch(self.pconfig.target_modules, key)
            else:
                target_module_found = any(
                    key.endswith(target_key)
                    for target_key in self.pconfig.target_modules
                )

            if target_module_found and count < self.pconfig.max_lora_layers:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self.__get_submodules(key)
                kwargs.update({"enable_lora": [True], "device": target.qweight.device})
                new_module = MergedQuantLinear(
                    (target.qweight.shape[0] * 32) // self.bits,
                    target.bias.shape[0],
                    self.bits,
                    **kwargs,
                )
                new_module.scales = target.scales
                new_module.zeros = target.zeros
                new_module.bias = target.bias
                self._replace_module(parent, target_name, new_module, target)
                count += 1
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.pconfig.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def __get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.qweight = old_module.qweight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.qweight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.qweight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


class MergedQuantLinear(QuantLinear, LoraLayer):
    # Lora implemented in a dense layer with QuantLinear
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits=4,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        QuantLinear.__init__(self, bits, in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        LoraLayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=False,
        )
        # if out_features % len(enable_lora) != 0:
        #    raise ValueError("The length of enable_lora must divide out_features")
        dev = kwargs["device"] if "device" in kwargs else torch.device("cpu")
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False, dtype=torch.float32)
            self.lora_B = nn.Linear(r, out_features, bias=False, dtype=torch.float32)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.qweight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # nn.init.kaiming_uniform_(self.lora_A.weight, mode='fan_in', nonlinearity='relu', a=0.01414)
            nn.init.xavier_uniform_(self.lora_A.weight)
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor):
        # import pudb; pu.db
        result = super().forward(x)
        if self.r > 0:
            drp = self.lora_dropout(x)
            tmp_a = self.lora_A(drp)
            tmp_b = self.lora_B(tmp_a)
            output = tmp_b * self.scaling
            result += output
        if torch.any(torch.isnan(result)):
            print("🔥")
        return result


class HiQModel(PushToHubMixin, torch.nn.Module):
    def __init__(self, model, pconfig: LoraConfig):
        super().__init__()
        self.pconfig = pconfig
        self.base_model = model
        self.config = self.base_model.config
        self.modules_to_save = None
        self.base_model = LoraModel(pconfig, model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise ValueError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
        os.makedirs(save_directory, exist_ok=True)

        # save only the trainable weights
        output_state_dict = get_sd(self, kwargs.get("state_dict", None))
        torch.save(output_state_dict, os.path.join(save_directory, WEIGHTS_NAME))

        # save the config and change the inference mode to `True`
        if self.pconfig.base_model_name_or_path is None:
            self.pconfig.base_model_name_or_path = (
                self.base_model.model.__dict__.get("name_or_path", None)
            )
        inference_mode = self.pconfig.inference_mode
        self.pconfig.inference_mode = True
        self.pconfig.save_pretrained(save_directory)
        self.pconfig.inference_mode = inference_mode

    @classmethod
    def from_pretrained(cls, model, model_id, **kwargs):
        # load the config
        config = LoraConfig.from_pretrained(model_id)

        if getattr(model, "hf_device_map", None) is not None:
            remove_hook_from_submodules(model)

        model = LQModel(model, config)

        # load weights if any
        if os.path.exists(os.path.join(model_id, WEIGHTS_NAME)):
            filename = os.path.join(model_id, WEIGHTS_NAME)
        else:
            try:
                filename = hf_hub_download(model_id, WEIGHTS_NAME)
            except:  # noqa
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {WEIGHTS_NAME} is present at {model_id}."
                )

        adapters_weights = torch.load(
            filename,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        # load the weights into the model
        model = set_sd(model, adapters_weights)
        return model

    def print_trainable_parameters(self):
        trainable_params, all_param, pct = model_parameters_stats(self)
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {pct}"
        )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_model, name)

    def forward(self, *args, **kwargs):
        return self.get_base_model()(*args, **kwargs)

    def get_base_model(self):
        return self.base_model.model


class LQModel(HiQModel):
    def __init__(self, model, pconfig: LoraConfig):
        super().__init__(model, pconfig)
        self.base_model_prepare_inputs_for_generation = (
            self.base_model.prepare_inputs_for_generation
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def generate(self, **kwargs):
        self.base_model.prepare_inputs_for_generation = (
            self.prepare_inputs_for_generation
        )
        try:
            outputs = self.base_model.generate(**kwargs)

        except:
            self.base_model.prepare_inputs_for_generation = (
                self.base_model_prepare_inputs_for_generation
            )
            raise
        else:
            self.base_model.prepare_inputs_for_generation = (
                self.base_model_prepare_inputs_for_generation
            )
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model_prepare_inputs_for_generation(*args, **kwargs)


def model_parameters_stats(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return trainable_params, all_param, 100 * trainable_params / all_param
