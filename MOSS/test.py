import torch
from transformers import AutoProcessor
from typing import Any, Callable
import numpy as np
import copy
from pathlib import Path

from transformers.audio_utils import load_audio
from typing import Optional, TypeVar, get_type_hints
from dataclasses import dataclass
from collections import OrderedDict
import os
from transformers import GenerationMixin
from transformers.configuration_utils import PreTrainedConfig
from collections.abc import Iterator
from torch import nn
from transformers.modeling_utils import EmbeddingAccessMixin, ModuleUtilsMixin, PushToHubMixin, PeftAdapterMixin, DistributedMixin, KernelConfig, LoadStateDictConfig, _get_resolved_checkpoint_files, _get_dtype, local_torch_dtype, ContextManagers, AttentionInterface\
,get_torch_context_manager_or_global_device, _is_on_hf_mount, _load_parameter_into_model
from transformers.generation import CompileConfig, GenerationConfig
from transformers.utils.output_capturing import _CAN_RECORD_REGISTRY, OutputRecorder
from transformers.utils.loading_report import LoadStateDictInfo, log_state_dict_report
from transformers import initialization as init
from transformers.quantizers import HfQuantizer
from torch.utils.checkpoint import checkpoint
from functools import partial, wraps
import inspect
from huggingface_hub import is_offline_mode, split_torch_state_dict_into_shards
from transformers.integrations.peft import maybe_load_adapters
from transformers.integrations.accelerate import check_and_set_device_map, get_device
from transformers.quantizers.auto import get_hf_quantizer
from transformers.monkey_patching import apply_patches, patch_output_recorders
from transformers.integrations import PeftAdapterMixin, deepspeed_config, hub_kernels, is_deepspeed_zero3_enabled, is_fsdp_enabled
from transformers.integrations.hub_kernels import allow_all_hub_kernels, is_kernel, kernelize
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
from transformers.integrations.finegrained_fp8 import ALL_FP8_EXPERTS_FUNCTIONS
from transformers.conversion_mapping import get_model_conversion_mapping
import sys, re
from transformers.loss.loss_utils import LOSS_MAPPING
from safetensors import safe_open
from transformers.core_model_loading import convert_and_load_state_dict_in_model
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.utils.quantization_config import QuantizationMethod

SpecificPreTrainedModelType = TypeVar("SpecificPreTrainedModelType", bound="PreTrainedModel")
_is_ds_init_called = False
ALL_ATTENTION_FUNCTIONS: AttentionInterface = AttentionInterface()

class PreTrainedModel(nn.Module, EmbeddingAccessMixin, ModuleUtilsMixin, PushToHubMixin, PeftAdapterMixin, DistributedMixin):

    # General model properties
    config_class: type[PreTrainedConfig] | None = None
    generation_config_class: type[GenerationConfig] = GenerationConfig  # default, used with `GenerationMixin`
    _auto_class = None
    base_model_prefix: str = ""
    _is_stateful: bool = False
    model_tags: list[str] | None = None

    # Input-related properties
    main_input_name: str = "input_ids"
    # Attributes used mainly in multimodal LLMs, though all models contain a valid field for these
    # Possible values are: text, image, video, audio and time
    input_modalities: str | list[str] = "text"

    # Device-map related properties
    _no_split_modules: set[str] | list[str] | None = None
    _skip_keys_device_placement: set[str] | list[str] | None = None

    # Specific dtype upcasting
    # `_keep_in_fp32_modules` will upcast to fp32 only if the requested dtype is fp16
    # `_keep_in_fp32_modules_strict` will upcast to fp32 independently if the requested dtype is fp16 or bf16
    _keep_in_fp32_modules: set[str] | list[str] | None = None
    _keep_in_fp32_modules_strict: set[str] | list[str] | None = None

    # Loading-specific properties
    # A dictionary `{"target": "source"}` of checkpoint keys that are potentially tied to one another
    _tied_weights_keys: dict[str, str] = None
    # A list of `re` patterns describing keys to ignore if they are missing from checkpoints to avoid warnings
    _keys_to_ignore_on_load_missing: set[str] | list[str] | None = None
    # A list of `re` patterns describing keys to ignore if they are unexpected in the checkpoints to avoid warnings
    _keys_to_ignore_on_load_unexpected: set[str] | list[str] | None = None
    # A list of keys to ignore when saving the model
    _keys_to_ignore_on_save: set[str] | list[str] | None = None

    # Attention interfaces support properties
    _supports_sdpa: bool = False
    _supports_flash_attn: bool = False
    _supports_flex_attn: bool = False
    # Model's compatible flash kernels (e.g., "kernels-community/flash-mla") defaulting to the first in the list
    _compatible_flash_implementations: list[str] | None = None

    # Set to `False` by models that can never run under context parallelism, whatever their config
    # (attention sinks, for instance, which SDPA cannot express). Models whose *config* rules it out are
    # handled by `supports_context_parallel` below, so this stays `True` for almost everything.
    _supports_context_parallel: bool = True

    # Advanced functionalities support
    supports_gradient_checkpointing: bool = False
    _can_compile_fullgraph: bool = False
    # This flag signal that the model can be used as an efficient backend in TGI and vLLM
    # In practice, it means that they support attention (mask) interface functions, fully pass the kwargs
    # through all modules up to the Attention layer, can slice logits with Tensor, and have a default TP plan
    _supports_attention_backend: bool = False
    # A mapping describing what outputs can be captured by `capture_outputs` decorator during the forward pass
    _can_record_outputs: dict | None = None

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__()
        self.config = config # todo needed?

    def post_init(self):
        self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=False)
        self.init_weights()

    # todo are these even called?
    @classmethod
    def can_generate(cls) -> bool: return True
    def get_experts_implementation(self) -> dict[str, str | None]: pass

    @torch.no_grad()
    def _init_weights(self, module):
        rope_init_fn_with_self = {
            "axial": getattr(module, "compute_axial_rope_parameters", None),
            "default": getattr(module, "compute_default_rope_parameters", None),
            **ROPE_INIT_FUNCTIONS,
        }
        rope_fn = rope_init_fn_with_self[module.rope_type]
        buffer_value, _ = rope_fn(module.config)
        init.copy_(module.inv_freq, buffer_value)
        init.copy_(module.original_inv_freq, buffer_value)

    def _initialize_weights(self, module, is_custom_code: bool = False):
        """
        Initialize the weights if they are not already initialized.
        """
        if getattr(module, "_is_hf_initialized", False):
            return

        # This check is for remote code that does NOT use either `torch.init` or `transformers.initialization` in `_init_weights`
        # which allow to check the flag directly on param. As they don't and write the params in-place, params would be reinitialized
        # otherwise
        if (
            is_custom_code
            and all(getattr(param, "_is_hf_initialized", False) for param in module.parameters(recurse=False))
            and all(
                getattr(buffer, "_is_hf_initialized", False)
                for buffer in module.buffers(recurse=False)
                if buffer is not None
            )
        ):
            module._is_hf_initialized = True
            return

        self._init_weights(module)
        module._is_hf_initialized = True

    @torch.no_grad()
    @init.guard_torch_init_functions()
    def initialize_weights(self):
        if not hasattr(torch.nn.Module, "smart_apply"):
            # This function is equivalent to `torch.nn.Module.apply`, except that it dynamically adjust the function
            # to apply as we go down the graph
            def smart_apply(module: nn.Module, fn: Callable[[nn.Module, bool], None], is_custom_code: bool):
                for child in module.children():
                    # We found a sub-model: recursively dispatch its own init function now!
                    if isinstance(child, PreTrainedModel):
                        smart_apply(child, child._initialize_weights, is_custom_code)
                    else:
                        smart_apply(child, fn, is_custom_code)
                fn(module, is_custom_code)
                return module

            setattr(torch.nn.Module, "smart_apply", smart_apply)

        # Let the magic happen with this simple call
        smart_apply_fn = getattr(self, "smart_apply")
        # `getattr(self, ...)` returns a bound method, so `self` is already provided as the receiver.
        smart_apply_fn(self._initialize_weights, self.is_custom_code())

    def get_expanded_tied_weights_keys(self, all_submodels: bool = False) -> dict:
        return {} if self._tied_weights_keys is None else self._tied_weights_keys.copy()

    def tie_weights(self, missing_keys: set[str] | None = None, recompute_mapping: bool = True):
        tied_keys = self.all_tied_weights_keys
        tied_keys = list(tied_keys.items())
        for _, (target_param_name, source_param_name) in enumerate(tied_keys):
            source_param = self.get_parameter_or_buffer(source_param_name)
            parent_name, name = target_param_name.rsplit(".", 1)
            parent = self.get_submodule(parent_name)
            setattr(parent, name, source_param)
            if missing_keys is not None: missing_keys.discard(target_param_name)

    def init_weights(self):
        """
        Initialize and tie the weights if needed. If using a custom `PreTrainedModel`, you need to implement any
        initialization logic in `_init_weights`.
        """
        # If we are initializing on meta device, there is no point in trying to run inits
        if get_torch_context_manager_or_global_device() != torch.device("meta"):
            # Initialize weights
            self.initialize_weights()
        # Tie weights needs to be called here, but it can use the pre-computed `all_tied_weights_keys`
        self.tie_weights(recompute_mapping=False)

    @wraps(torch.nn.Module.to)
    def to(self, *args, **kwargs): return super().to(*args, **kwargs)

    def half(self, *args): return super().half(*args)

    def float(self, *args): return super().float(*args)

    @classmethod
    def get_init_context(
        cls, dtype: torch.dtype, is_quantized: bool, _is_ds_init_called: bool, allow_all_kernels: bool | None
    ):
        # Need to instantiate with correct dtype
        init_contexts = [local_torch_dtype(dtype, cls.__name__), init.no_tie_weights(), apply_patches()]
        # Needed as we cannot forward the `allow_all_kernels` arg in the model's __init__
        if allow_all_kernels:
            init_contexts.append(allow_all_hub_kernels())
        if is_deepspeed_zero3_enabled():
            import deepspeed

            # We cannot initialize the model on meta device with deepspeed when not quantized
            if not is_quantized and not _is_ds_init_called:
                logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
                init_contexts.extend(
                    [
                        init.no_init_weights(),
                        deepspeed.zero.Init(config_dict_or_path=deepspeed_config()),
                        set_zero3_state(),
                    ]
                )
            elif is_quantized:
                init_contexts.extend([torch.device("meta"), set_quantized_state()])
        else:
            # meta_device_safe_creation_ops patches torch.linspace to default to CPU
            # so that custom models calling .item() during __init__ (e.g. drop-path
            # schedules) don't crash on meta tensors.
            init_contexts.extend([torch.device("meta"), init.meta_device_safe_creation_ops()])

        return init_contexts

    def _get_dtype_plan(self, dtype: torch.dtype) -> dict:
        """Create the dtype_plan describing modules/parameters that should use the `keep_in_fp32` flag."""
        dtype_plan = {}

        # The _keep_in_fp32_modules flag is only used to avoid bf16 -> fp16 casting precision issues. It was introduced
        # in case of force loading a model that should stay in bf16 in fp16
        # See https://github.com/huggingface/transformers/issues/20287 for details.
        if self._keep_in_fp32_modules is not None and dtype == torch.float16:
            dtype_plan.update(dict.fromkeys(self._keep_in_fp32_modules, torch.float32))

        # The _keep_in_fp32_modules_strict was introduced to always force upcast to fp32, for both fp16 and bf16
        if self._keep_in_fp32_modules_strict is not None and dtype in (torch.float16, torch.bfloat16):
            dtype_plan.update(dict.fromkeys(self._keep_in_fp32_modules_strict, torch.float32))

        return dtype_plan

    def set_use_kernels(self, use_kernels, kernel_config: KernelConfig | None = None, mode: "Mode | None" = None): self._use_kernels = False

    @classmethod
    def from_pretrained(
        cls: type[SpecificPreTrainedModelType],
        pretrained_model_name_or_path: str | os.PathLike | None,
        *model_args,
        config: PreTrainedConfig | str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool | None = None,
        weights_only: bool = True,
        fusion_config: dict[str, bool | dict[str, Any]] | None = None,
        disable_mmap: bool | None = None,
        **kwargs,
    ) -> SpecificPreTrainedModelType:
        state_dict = kwargs.pop("state_dict", None)
        proxies = kwargs.pop("proxies", None)
        tqdm_class = kwargs.pop("tqdm_class", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        dtype = kwargs.pop("dtype", None)
        torch_dtype = kwargs.pop("torch_dtype", None)  # kept for BC
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_buffers = kwargs.pop("offload_buffers", False)
        quantization_config = kwargs.pop("quantization_config", None)
        subfolder = kwargs.pop("subfolder", "")
        kwargs.pop("_commit_hash", None)  # BC: not used anymore, `revision` is resolved to a commit hash instead
        variant = kwargs.pop("variant", None)
        adapter_kwargs = (kwargs.pop("adapter_kwargs", {}) or {}).copy()
        adapter_name = kwargs.pop("adapter_name", "default")
        generation_config = kwargs.pop("generation_config", None)
        gguf_file = kwargs.pop("gguf_file", None)
        distributed_config: DistributedConfig = kwargs.pop("distributed_config", None)
        device_mesh = kwargs.pop("device_mesh", None)
        tp_plan = kwargs.pop("tp_plan", None)
        tp_size = kwargs.pop("tp_size", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        allow_all_kernels = kwargs.pop("allow_all_kernels", False)
        use_kernels = kwargs.pop("use_kernels", False)
        kernel_config = kwargs.pop("kernel_config", None)
        key_mapping = kwargs.pop("key_mapping", None)

        # Not used anymore -- remove them from the kwargs
        for name in ["mirror", "_fast_init", "low_cpu_mem_usage", "from_tf", "from_flax", "offload_state_dict"]:
            _ = kwargs.pop(name, None)

        # For BC on torch_dtype argument
        if torch_dtype is not None:
            dtype = dtype if dtype is not None else torch_dtype
        if dtype is None:
            dtype = "auto"

        if is_offline_mode() and not local_files_only:
            local_files_only = True

        # Resolve the revision once and for all: config, weights, generation config and adapters are then all loaded
        # from the exact same repository state, without any further call to the Hub to revalidate a mutable revision.
        requested_revision = revision

        download_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "local_files_only": local_files_only,
            "token": token,
            "revision": revision,
            "subfolder": subfolder,
        }


        has_standalone_tp_args = tp_plan is not None or tp_size is not None

        if distributed_config is not None:
            distributed_config, device_map, device_mesh = cls.prepare_distribute_model(
                distributed_config, device_map=device_map
            )


        if adapter_kwargs is None:
            adapter_kwargs = {}

        adapter_repo_id = pretrained_model_name_or_path
        _adapter_model_path, pretrained_model_name_or_path, adapter_kwargs = maybe_load_adapters(
            pretrained_model_name_or_path,
            download_kwargs,
            **adapter_kwargs,
        )
        device_map = check_and_set_device_map(device_map)  # warn, error and fix the device map

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        # Load config if we don't provide a configuration
        if not isinstance(config, PreTrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config_class = cls.config_class
            if config_class is None:
                raise ValueError(
                    f"{cls.__name__} does not define `config_class`; pass an explicit config to `from_pretrained`."
                )
            config, model_kwargs = config_class.from_pretrained(
                config_path,
                return_unused_kwargs=True,
                gguf_file=gguf_file,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **download_kwargs,
                **kwargs,
            )
            if "gguf_file" in model_kwargs:
                model_kwargs.pop("gguf_file")
        else:
            config = copy.deepcopy(config)
            model_kwargs = kwargs

        if distributed_config is not None:
            config.distributed_config = distributed_config

        # Because some composite configs call super().__init__ before instantiating the sub-configs, we need this call
        # to correctly redispatch recursively if the kwarg is provided
        if "attn_implementation" in kwargs:
            config._attn_implementation = kwargs.pop("attn_implementation")

        if "experts_implementation" in kwargs:
            config._experts_implementation = kwargs.pop("experts_implementation")

        hf_quantizer, config, device_map = get_hf_quantizer(
            config, quantization_config, device_map, weights_only, user_agent, gguf_file=gguf_file
        )

        if kernel_config is not None and not use_kernels:
            logger.warning_once(
                "A kernel_config was provided but use_kernels is False; setting use_kernels=True automatically. To suppress this warning, explicitly set use_kernels to True."
            )
            use_kernels = True

        checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            variant=variant,
            gguf_file=gguf_file,
            use_safetensors=use_safetensors,
            download_kwargs=download_kwargs,
            user_agent=user_agent,
            is_remote_code=cls.is_remote_code(),
            transformers_explicit_filename=getattr(config, "transformers_weights", None),
            tqdm_class=tqdm_class,
        )

        is_quantized = hf_quantizer is not None

        if gguf_file:
            # Read before the dtype is settled: a GGUF's own float type is what `dtype="auto"` resolves to.
            hf_quantizer.read_header(checkpoint_files[0])

        # Find the correct dtype based on current state
        config, dtype = _get_dtype(
            dtype, checkpoint_files, config, sharded_metadata, state_dict, weights_only, hf_quantizer
        )

        config.name_or_path = pretrained_model_name_or_path

        # Overwrite `config.fusion_config` if it is provided.
        if fusion_config is not None:
            config.fusion_config = copy.deepcopy(fusion_config)

        # Register fusion patches
        fusion_config = getattr(config, "fusion_config", None)
        if fusion_config is not None:
            from .fusion_mapping import register_fusion_patches

            register_fusion_patches(cls, config, fusion_config)

        # Kernel patches: single-layer replacement (stateful __init__) then fusions.
        if kernel_config is not None and use_kernels:
            from .integrations.hub_kernels import register_kernel_replacements_and_fusions

            # For remote kernels, we need to apply the context manager
            allow_all_kernels_context = [allow_all_hub_kernels()] if allow_all_kernels else []
            with ContextManagers(allow_all_kernels_context):
                register_kernel_replacements_and_fusions(cls, config, kernel_config)

        model_init_context = cls.get_init_context(dtype, is_quantized, _is_ds_init_called, allow_all_kernels)

        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        with ContextManagers(model_init_context):
            model = cls(config, *model_args, **model_kwargs)
            patch_output_recorders(model)

            if hf_quantizer is not None:  # replace module with quantized modules (does not touch weights)
                hf_quantizer.preprocess_model(
                    model=model,
                    dtype=dtype,
                    device_map=device_map,
                    checkpoint_files=checkpoint_files,
                    use_kernels=use_kernels,
                )

        if gguf_file:
            state_dict = hf_quantizer.get_state_dict(checkpoint_files[0], model)

        # Create the dtype_plan to potentially use the `keep_in_fp32` flags (this needs to be called on the already
        # instantiated model, as the flags can be modified by instances sometimes)
        dtype_plan = model._get_dtype_plan(dtype)

        # Obtain the weight conversion mapping for this model if any are registered and apply to all submodels recursively
        weight_conversions = get_model_conversion_mapping(model, key_mapping, hf_quantizer)

        if distributed_config is not None:
            model = cls.maybe_distribute_model(model, distributed_config, device_mesh)

        # Finalize model weight initialization
        load_config = LoadStateDictConfig(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            sharded_metadata=sharded_metadata,
            device_map=device_map,
            disk_offload_folder=offload_folder,
            offload_buffers=offload_buffers,
            dtype=dtype,
            dtype_plan=dtype_plan,
            hf_quantizer=hf_quantizer,
            device_mesh=device_mesh,
            weights_only=weights_only,
            weight_mapping=weight_conversions,
            use_safetensors=use_safetensors,
            download_kwargs=download_kwargs,
            disable_mmap=disable_mmap,
        )
        loading_info, disk_offload_index = cls._load_pretrained_model(model, state_dict, checkpoint_files, load_config)
        loading_info = cls._finalize_model_loading(model, load_config, loading_info)
        model.eval()  # Set model in evaluation mode to deactivate Dropout modules by default
        model.set_use_kernels(use_kernels, kernel_config)


        model.adjust_generation_fn(
            generation_config,
            from_auto_class,
            from_pipeline,
            pretrained_model_name_or_path,
            **download_kwargs,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        if hf_quantizer is not None:
            model.hf_quantizer = hf_quantizer
            hf_quantizer.postprocess_model(
                model
            )  # usually a no-op but sometimes needed, e.g to remove the quant config when dequantizing

        if _adapter_model_path is not None:
            if token is not None:
                adapter_kwargs["token"] = token
            loading_info = model.load_adapter(
                _adapter_model_path,
                adapter_name=adapter_name,
                load_config=load_config,
                adapter_kwargs=adapter_kwargs,
            )

        if output_loading_info:
            return model, loading_info.to_dict()
        return model

    @staticmethod
    def _load_pretrained_model(
        model: "PreTrainedModel",
        state_dict: dict | None,
        checkpoint_files: list[str] | None,
        load_config: LoadStateDictConfig,
        expected_keys: list[str] | None = None,
    ) -> tuple[LoadStateDictInfo, dict]:
        """Perform the actual loading of some checkpoints into a `model`, by reading them from disk and dispatching them accordingly."""
        hf_quantizer = load_config.hf_quantizer
        is_quantized = load_config.is_quantized
        is_hqq_or_quark = hf_quantizer is not None and hf_quantizer.quantization_config.quant_method in {
            QuantizationMethod.HQQ,
            QuantizationMethod.QUARK,
        }

        # Model's definition arriving here is final (TP hooks added, quantized layers replaces)
        expected_keys = list(model.state_dict().keys()) if expected_keys is None else expected_keys

        # This offload index if for params explicitly on the "disk" in the device_map
        disk_offload_index = None

        all_pointer = set()
        if state_dict is not None:
            merged_state_dict = state_dict
        elif checkpoint_files is not None and checkpoint_files[0].endswith(".safetensors") and state_dict is None:
            merged_state_dict = {}
            for file in checkpoint_files:
                if load_config.disable_mmap or _is_on_hf_mount(file):
                    with open(file, "rb") as _fh:
                        merged_state_dict.update(_safe_load_bytes(_fh.read()))
                    continue
                is_mps = load_config.device_map is not None and any(
                    (d.type if isinstance(d, torch.device) else d) == "mps"
                    for d in load_config.device_map.values()
                )
                # Use pread on MPS (mmap incompatible) and Windows (mmap reserves
                # copy-on-write commit charge for the entire file, exhausting memory
                # for large multi-shard checkpoints).
                if is_mps:
                    backend, device = "pread", "mps"
                elif sys.platform == "win32":
                    backend, device = "pread", "cpu"
                else:
                    backend, device = "mmap", "cpu"
                file_pointer = safe_open(file, framework="pt", device=device, backend=backend)
                all_pointer.add(file_pointer)
                for k in file_pointer.keys():
                    merged_state_dict[k] = file_pointer.get_slice(k)  # don't materialize yet
        # Checkpoints are .bin
        elif checkpoint_files is not None:
            merged_state_dict = {}
            for ckpt_file in checkpoint_files:
                merged_state_dict.update(load_state_dict(ckpt_file, disable_mmap=load_config.disable_mmap))
        else:
            raise ValueError("Neither a state dict nor checkpoint files were found.")

        loading_info, disk_offload_index = convert_and_load_state_dict_in_model(
            model=model,
            state_dict=merged_state_dict,
            load_config=load_config,
            disk_offload_index=disk_offload_index,
        )

        # finally close all opened file pointers
        for k in all_pointer:
            k.__exit__(None, None, None)

        return loading_info, disk_offload_index

    @staticmethod
    def _finalize_model_loading(
        model, load_config: LoadStateDictConfig, loading_info: LoadStateDictInfo
    ) -> LoadStateDictInfo:
        model.mark_tied_weights_as_initialized(loading_info)
        model._move_missing_keys_from_meta_to_device(
            loading_info.missing_and_mismatched(),
            load_config.device_map,
            load_config.device_mesh,
            load_config.hf_quantizer,
        )
        model.initialize_weights()
        model.tie_weights(missing_keys=loading_info.missing_keys, recompute_mapping=False)

        # Adjust missing and unexpected keys
        model._adjust_missing_and_unexpected_keys(loading_info)

        return loading_info

    def retrieve_modules_from_names(self, names, add_prefix=False, remove_prefix=False):
        module_keys = {".".join(key.split(".")[:-1]) for key in names}

        # torch.nn.ParameterList is a special case where two parameter keywords
        # are appended to the module name, *e.g.* bert.special_embeddings.0
        module_keys = module_keys.union(
            {".".join(key.split(".")[:-2]) for key in names if len(key) > 0 and key[-1].isdigit()}
        )

        retrieved_modules = []
        # retrieve all modules that has at least one missing weight name
        for name, module in self.named_modules():
            if remove_prefix:
                _prefix = f"{self.base_model_prefix}."
                name = name.removeprefix(_prefix)
            elif add_prefix:
                name = ".".join([self.base_model_prefix, name]) if len(name) > 0 else self.base_model_prefix

            if name in module_keys:
                retrieved_modules.append(module)

        return retrieved_modules

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoModel"):
        """
        Register this class with a given auto class. This should only be used for custom models as the ones in the
        library are already mapped with an auto class.



        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoModel"`):
                The auto class to register this new model with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    def warn_if_padding_and_no_attention_mask(self, input_ids, attention_mask):
        """
        Shows a one-time warning if the input_ids appear to contain padding and no attention mask was given.
        """

        # Skip the check during tracing.
        if is_tracing(input_ids):
            return

        if (attention_mask is not None) or (self.config.pad_token_id is None):
            return

        # Check only the first and last input IDs to reduce overhead.
        if self.config.pad_token_id in input_ids[:, [-1, 0]]:
            warn_string = (
                "We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See "
                "https://huggingface.co/docs/transformers/troubleshooting"
                "#incorrect-output-when-padding-tokens-arent-masked."
            )

            # If the pad token is equal to either BOS, EOS, or SEP, we do not know whether the user should use an
            # attention_mask or not. In this case, we should still show a warning because this is a rare case.
            # NOTE: `sep_token_id` is not used in all models and it can be absent in the config
            sep_token_id = getattr(self.config, "sep_token_id", None)
            if (
                (self.config.bos_token_id is not None and self.config.bos_token_id == self.config.pad_token_id)
                or (self.config.eos_token_id is not None and self.config.eos_token_id == self.config.pad_token_id)
                or (sep_token_id is not None and sep_token_id == self.config.pad_token_id)
            ):
                warn_string += (
                    f"\nYou may ignore this warning if your `pad_token_id` ({self.config.pad_token_id}) is identical "
                    f"to the `bos_token_id` ({self.config.bos_token_id}), `eos_token_id` ({self.config.eos_token_id}), "
                    f"or the `sep_token_id` ({sep_token_id}), and your input is not padded."
                )

            logger.warning_once(warn_string)

    @property
    def use_kernels(self) -> bool:
        return getattr(self, "_use_kernels", False)

    @use_kernels.setter
    def use_kernels(self, value: bool) -> None:
        # Avoid re-kernelizing if already enabled
        if bool(value) and getattr(self, "_use_kernels", False):
            return

        if value:
            self.set_use_kernels(True)
        else:
            if getattr(self, "_use_kernels", False):
                logger.warning_once(
                    "Disabling kernels at runtime is a no-op as there is no 'unkernelize' routine; keeping current kernels active."
                )
            self._use_kernels = False

    def _default_compile_config(self) -> CompileConfig:
        """Build the default `CompileConfig` for `get_compiled_call`.

        Inductor + `reduce-overhead` (the `CompileConfig` defaults) target CUDA.
        torch_tpu registers its own TorchDynamo backend named `"tpu"`; route
        `device.type == "tpu"` through it with static shapes to match the
        common StaticCache + fixed-prefill usage."""
        if self.device.type == "tpu":
            return CompileConfig(backend="tpu", dynamic=False, mode="default")
        return CompileConfig()

    def get_compiled_call(self, compile_config: CompileConfig | None) -> Callable:
        """Return a `torch.compile`'d version of `self.__call__`. This is useful to dynamically choose between
        non-compiled/compiled `forward` during inference, especially to switch between prefill (where we don't
        want to use compiled version to avoid recomputing the graph with new shapes) and iterative decoding
        (where we want the speed-ups of compiled version with static shapes)."""
        # Only reset it if not present or different from previous config
        if "llama4" in self.config.model_type:  # TODO try to enable for FULL COMPILE HYBRID CACHE SUPPORT
            return self.__call__
        compile_config = compile_config or self._default_compile_config()
        default_config = getattr(self.generation_config, "compile_config", None) or self._default_compile_config()
        if (
            not hasattr(self, "_compiled_call")
            or getattr(self, "_last_compile_config", default_config) != compile_config
        ):
            self._last_compile_config = compile_config
            self._compiled_call = torch.compile(self.__call__, **compile_config.to_dict())
        return self._compiled_call

    @classmethod
    def is_backend_compatible(cls):
        return cls._supports_attention_backend

    def _move_missing_keys_from_meta_to_device(
        self,
        missing_keys: list[str],
        device_map: dict | None,
        device_mesh: "DeviceMeshLike | None",
        hf_quantizer: HfQuantizer | None,
    ) -> None:
        """Move missing params/buffers off meta to their target device.

        Loaded weights are handled earlier in `convert_and_load_state_dict_in_model`
        via `DtensorShardOperation` and `set_param_for_module`. This only
        materializes keys that were not loaded (or mismatched) so
        `_initialize_missing_keys` can run proper init on them.
        """
        is_quantized = hf_quantizer is not None
        # This is the only case where we do not initialize the model on meta device, so we don't have to do anything here
        if is_deepspeed_zero3_enabled() and not is_quantized:
            return

        # In this case we need to move everything back
        if is_fsdp_enabled() and not is_local_dist_rank_0() and not is_quantized:
            for key, param in self.named_parameters():
                value = torch.zeros_like(param, device="cpu")
                _load_parameter_into_model(self, key, value)
            for key, buffer in self.named_buffers():
                value = torch.zeros_like(buffer, device="cpu")
                _load_parameter_into_model(self, key, value)
            return

        # The tied weight keys are in the "missing" usually, but they should not be moved (they will be tied anyway)
        # This is especially important because if they are moved, they will lose the `_is_hf_initialized` flag, and they
        # will be re-initialized for nothing (which can be quite long)
        for key in missing_keys - self.all_tied_weights_keys.keys():
            param = self.get_parameter_or_buffer(key)
            param_device = get_device(device_map, key, valid_torch_device=True)
            value = torch.empty_like(param, device=param_device)
            # For TP, we may need to shard the param
            if is_dtensor(param):
                local = torch.empty(param._local_tensor.shape, dtype=param.dtype, device=param_device)
                value = torch.nn.Parameter(
                    _dtensor_from_local_like(local, param),
                    requires_grad=param.requires_grad,
                )
            _load_parameter_into_model(self, key, value)
        # We need to move back non-persistent buffers as well, as they are not part of loaded weights anyway
        for key, buffer in self.named_non_persistent_buffers():
            buffer_device = get_device(device_map, key, valid_torch_device=True)
            value = torch.empty_like(buffer, device=buffer_device)
            _load_parameter_into_model(self, key, value)

    def _adjust_missing_and_unexpected_keys(self, loading_info: LoadStateDictInfo) -> None:
        """Adjust the `missing_keys` and `unexpected_keys` based on current model's exception rules, to avoid
        raising unneeded warnings/errors. This is performed in-place.
        """
        # Old checkpoints may have keys for rotary_emb.inv_freq for each layer, however we moved this buffer to the main model
        # (so the buffer name has changed). Remove them in such a case. This is another exception that was not added to
        # `_keys_to_ignore_on_load_unexpected` as it touches many models -> we add it manually to the existing patterns
        has_inv_freq_buffers = any(buffer.endswith("rotary_emb.inv_freq") for buffer, _ in self.named_buffers())
        additional_unexpected_patterns = {r"rotary_emb\.inv_freq"} if has_inv_freq_buffers else set()
        # Same idea for `position_ids`: used to be a persistent buffer, now `persistent=False` in most models.
        has_position_ids_buffers = any(buffer.endswith("position_ids") for buffer, _ in self.named_buffers())
        if has_position_ids_buffers:
            additional_unexpected_patterns.add(r"(^|\.)position_ids$")

        missing_patterns = self._keys_to_ignore_on_load_missing or set()
        unexpected_patterns = (self._keys_to_ignore_on_load_unexpected or set()) | additional_unexpected_patterns
        ignore_missing_regex, ignore_unexpected_regex = None, None
        if len(missing_patterns) > 0:
            ignore_missing_regex = re.compile("|".join(rf"({pattern})" for pattern in missing_patterns))
        if len(unexpected_patterns) > 0:
            ignore_unexpected_regex = re.compile("|".join(rf"({pattern})" for pattern in unexpected_patterns))

        # Clean-up missing keys
        if ignore_missing_regex is not None:
            loading_info.missing_keys = {
                key for key in loading_info.missing_keys if ignore_missing_regex.search(key) is None
            }

        # Clean-up unexpected keys
        if ignore_unexpected_regex is not None:
            loading_info.unexpected_keys = {
                key for key in loading_info.unexpected_keys if ignore_unexpected_regex.search(key) is None
            }

    def mark_tied_weights_as_initialized(self, loading_info):
        """Adds the `_is_hf_initialized` flag on parameters that will be tied, in order to avoid initializing them
        later as they will be tied (overwritten) anyway.
        This is very important as most embeddings are tied, and they are huge params (vocabularies are often 256k), so
        running inits on them is very costly."""
        for tied_param in getattr(self, "all_tied_weights_keys", {}).keys():
            param = self.get_parameter(tied_param)
            setattr(param, "_is_hf_initialized", True)

        # Some custom code models define module tying (not parameter tying) in their __init__. When modules themselves are shared,
        # weights inside both modules appear in the `state_dict` but only one will appear in the safetensors checkpoints
        # as they are inherently tied because the 2 modules are the same object. In this case, once we load a parameter
        # inside one of the 2 modules, the other will also automatically be loaded and will have the `_is_hf_initialized`
        # flag (because we call `setattr` with the loaded param on the module, which is the same object), but its counterpart
        # will still appear as a missing key as we never get it out of the set (because it appears in the state_dict as well).
        # So we remove it now - otherwise it's considered missing and will be wrongly reinitialized
        # Note: this is never an issue in main Transformers, as we never do module-tying, only parameter-tying, and we know
        # which params are supposed to be tied to which other params
        if self.is_custom_code():
            # Remove those that are already initialized, but appear as missing due to module tying (only if they are not known
            # tied weights, i.e. we did not explicitly mark them as initialized just above)
            loading_info.missing_keys = {
                key
                for key in loading_info.missing_keys
                if key in self.all_tied_weights_keys
                or not getattr(self.get_parameter_or_buffer(key), "_is_hf_initialized", False)
            }

    def get_parameter_or_buffer(self, target: str):
        """
        Return the parameter or buffer given by `target` if it exists, otherwise throw an error. This combines
        `get_parameter()` and `get_buffer()` in a single handy function. If the target is an `_extra_state` attribute,
        it will return the extra state provided by the module. Note that it only work if `target` is a leaf of the model.
        """
        try:
            return self.get_parameter(target)
        except AttributeError:
            pass
        try:
            return self.get_buffer(target)
        except AttributeError:
            pass
        module, param_name = get_module_from_name(self, target)
        if (
            param_name == "_extra_state"
            and getattr(module.__class__, "get_extra_state", torch.nn.Module.get_extra_state)
            is not torch.nn.Module.get_extra_state
        ):
            return module.get_extra_state()

        raise AttributeError(f"`{target}` is neither a parameter, buffer, nor extra state.")

    def named_non_persistent_buffers(
        self, recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Similar to `named_buffers`, but only yield non-persistent ones. It is handy as it's not perfectly straightforward
        to know if they are persistent or not"""
        for name, tensor in self.named_buffers(recurse=recurse, remove_duplicate=remove_duplicate):
            # We have to grab the parent here, as the attribute `_non_persistent_buffers_set` is on the immediate
            # parent only
            parent, buf_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = self.get_submodule(parent)
            if buf_name in parent._non_persistent_buffers_set:
                yield name, tensor

    def train(self, mode: bool = True):
        changed_mode = self.training != mode
        out = super().train(mode)
        # Avoid recasting kernels if not necessary
        if self.use_kernels and changed_mode:
            self.set_use_kernels(True)
        return out

    def eval(self):
        return self.train(False)

    @classmethod
    def is_remote_code(cls) -> bool:
        """Return whether the current model is custom code, i.e. code loaded from the hub, or class that we just registered
        via `register_for_auto_class`."""
        return cls._auto_class is not None

    @classmethod
    def is_custom_code(cls) -> bool:
        """Return whether the current model is custom code, i.e. either code loaded from the hub, or defined in any user-specific
        module/session."""
        return cls.is_remote_code() or not cls.__module__.startswith("transformers.")

_LazyAutoMappingValue = tuple[type[Any] | None, type[Any] | None]
_T = TypeVar("_T")

class _LazyAutoMapping(OrderedDict[Any, _LazyAutoMappingValue]):
    """
    A mapping config to object (model or tokenizer for instance) that will load keys and values when it is accessed.

    Args:
        - config_mapping: The map model type to config class
        - model_mapping: The map model type to model (or tokenizer) class
    """

    def __init__(self, config_mapping, model_mapping) -> None:
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        self._model_mapping = model_mapping
        self._model_mapping._model_mapping = self
        self._extra_content = {}
        self._modules = {}

    def __len__(self) -> int:
        common_keys = set(self._config_mapping.keys()).intersection(self._model_mapping.keys())
        return len(common_keys) + len(self._extra_content)

    def __getitem__(self, key: Any) -> _LazyAutoMappingValue:
        if key in self._extra_content:
            return self._extra_content[key]
        model_type = self._reverse_config_mapping[key.__name__]
        if model_type in self._model_mapping:
            model_name = self._model_mapping[model_type]
            return self._load_attr_from_module(model_type, model_name)

        # Maybe there was several model types associated with this config.
        model_types = [k for k, v in self._config_mapping.items() if v == key.__name__]
        for mtype in model_types:
            if mtype in self._model_mapping:
                model_name = self._model_mapping[mtype]
                return self._load_attr_from_module(mtype, model_name)
        raise KeyError(key)

    def _load_attr_from_module(self, model_type, attr):
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
        return getattribute_from_module(self._modules[module_name], attr)

    def keys(self):
        mapping_keys = [
            self._load_attr_from_module(key, name)
            for key, name in self._config_mapping.items()
            if key in self._model_mapping
        ]
        return mapping_keys + list(self._extra_content.keys())

    def get(self, key: Any, default: _T) -> _LazyAutoMappingValue | _T:
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __bool__(self) -> bool:
        return bool(self.keys())

    def values(self) -> list[_LazyAutoMappingValue]:
        mapping_values = [
            self._load_attr_from_module(key, name)
            for key, name in self._model_mapping.items()
            if key in self._config_mapping
        ]
        return mapping_values + list(self._extra_content.values())

    def items(self) -> list[tuple[Any, _LazyAutoMappingValue]]:
        mapping_items = [
            (
                self._load_attr_from_module(key, self._config_mapping[key]),
                self._load_attr_from_module(key, self._model_mapping[key]),
            )
            for key in self._model_mapping
            if key in self._config_mapping
        ]
        return mapping_items + list(self._extra_content.items())

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item: type) -> bool:
        if item in self._extra_content:
            return True
        if not hasattr(item, "__name__") or item.__name__ not in self._reverse_config_mapping:
            return False
        model_type = self._reverse_config_mapping[item.__name__]
        return model_type in self._model_mapping

    def register(self, key: Any | str, value: _LazyAutoMappingValue, exist_ok=False) -> None:
        """
        Register a new model in this mapping.
        """
        if hasattr(key, "__name__") and key.__name__ in self._reverse_config_mapping:
            model_type = self._reverse_config_mapping[key.__name__]
            if model_type in self._model_mapping and not exist_ok:
                raise ValueError(f"'{key}' is already used by a Transformers model.")

        # Some remote code may simply register a new custom model/processor/..., while using a native Transformers config. In such
        # cases, we should skip registering, as we will otherwise always remap the native config to the custom model/processor/... in
        # the same session, even if `trust_remote_code=False` is specified by the user (in which case we should use the native
        # Transformers model/processor/... corresponding to the config)
        # This is because remote/native is indistinguisable from the config class only in such cases, as they both use the same class - then
        # `from_pretrained`/`from_config` are responsible to grab the correct class depending on whether `trust_remote_code` is True/False
        if getattr(key, "__module__", "").startswith("transformers."):
            return

        # Register the new mapping (this will always take precedence in __getattr__ and __contains__ compared to base mapping)
        self._extra_content[key] = value

    def __reduce__(self):
        return (
            self.__class__._from_pickle,
            (self._config_mapping, self._model_mapping, dict(self._extra_content)),
        )

    @classmethod
    def _from_pickle(cls, config_mapping, model_mapping, extra_content):
        obj = cls(config_mapping, model_mapping)
        obj._extra_content = extra_content
        return obj

CONFIG_MAPPING_NAMES = OrderedDict(
    [
        ("afmoe", "AfmoeConfig"),
        ("aimv2", "Aimv2Config"),
        ("aimv2_text_model", "Aimv2TextConfig"),
        ("aimv2_vision_model", "Aimv2VisionConfig"),
        ("albert", "AlbertConfig"),
        ("align", "AlignConfig"),
        ("align_text_model", "AlignTextConfig"),
        ("align_vision_model", "AlignVisionConfig"),
        ("altclip", "AltCLIPConfig"),
        ("altclip_text_model", "AltCLIPTextConfig"),
        ("altclip_vision_model", "AltCLIPVisionConfig"),
        ("apertus", "ApertusConfig"),
        ("arcee", "ArceeConfig"),
        ("aria", "AriaConfig"),
        ("aria_text", "AriaTextConfig"),
        ("audio-spectrogram-transformer", "ASTConfig"),
        ("audioflamingo3", "AudioFlamingo3Config"),
        ("audioflamingo3_encoder", "AudioFlamingo3EncoderConfig"),
        ("autoformer", "AutoformerConfig"),
        ("axk1", "AXK1Config"),
        ("axk2", "AXK2Config"),
        ("aya_vision", "AyaVisionConfig"),
        ("bamba", "BambaConfig"),
        ("bark", "BarkConfig"),
        ("bart", "BartConfig"),
        ("beit", "BeitConfig"),
        ("bert", "BertConfig"),
        ("bert-generation", "BertGenerationConfig"),
        ("big_bird", "BigBirdConfig"),
        ("bigbird_pegasus", "BigBirdPegasusConfig"),
        ("biogpt", "BioGptConfig"),
        ("bit", "BitConfig"),
        ("bitnet", "BitNetConfig"),
        ("blenderbot", "BlenderbotConfig"),
        ("blenderbot-small", "BlenderbotSmallConfig"),
        ("blip", "BlipConfig"),
        ("blip-2", "Blip2Config"),
        ("blip_2_qformer", "Blip2QFormerConfig"),
        ("blip_2_vision_model", "Blip2VisionConfig"),
        ("blip_text_model", "BlipTextConfig"),
        ("blip_vision_model", "BlipVisionConfig"),
        ("bloom", "BloomConfig"),
        ("blt", "BltConfig"),
        ("blt_global_transformer", "BltGlobalTransformerConfig"),
        ("blt_local_decoder", "BltLocalDecoderConfig"),
        ("blt_local_encoder", "BltLocalEncoderConfig"),
        ("blt_patcher", "BltPatcherConfig"),
        ("bridgetower", "BridgeTowerConfig"),
        ("bridgetower_text_model", "BridgeTowerTextConfig"),
        ("bridgetower_vision_model", "BridgeTowerVisionConfig"),
        ("bros", "BrosConfig"),
        ("camembert", "CamembertConfig"),
        ("canary", "CanaryConfig"),
        ("canary_decoder", "CanaryDecoderConfig"),
        ("canine", "CanineConfig"),
        ("chameleon", "ChameleonConfig"),
        ("chameleon_vqgan", "ChameleonVQVAEConfig"),
        ("chinese_clip", "ChineseCLIPConfig"),
        ("chinese_clip_text_model", "ChineseCLIPTextConfig"),
        ("chinese_clip_vision_model", "ChineseCLIPVisionConfig"),
        ("chmv2", "CHMv2Config"),
        ("clap", "ClapConfig"),
        ("clap_audio_model", "ClapAudioConfig"),
        ("clap_text_model", "ClapTextConfig"),
        ("clip", "CLIPConfig"),
        ("clip_text_model", "CLIPTextConfig"),
        ("clip_vision_model", "CLIPVisionConfig"),
        ("clipseg", "CLIPSegConfig"),
        ("clipseg_text_model", "CLIPSegTextConfig"),
        ("clipseg_vision_model", "CLIPSegVisionConfig"),
        ("clvp", "ClvpConfig"),
        ("clvp_decoder", "ClvpDecoderConfig"),
        ("clvp_encoder", "ClvpEncoderConfig"),
        ("codegen", "CodeGenConfig"),
        ("cohere", "CohereConfig"),
        ("cohere2", "Cohere2Config"),
        ("cohere2_moe", "Cohere2MoeConfig"),
        ("cohere2_vision", "Cohere2VisionConfig"),
        ("cohere_asr", "CohereAsrConfig"),
        ("cohere_compass", "CohereCompassConfig"),
        ("cohere_compass_text", "CohereCompassTextConfig"),
        ("cohere_compass_vision", "CohereCompassVisionConfig"),
        ("colmodernvbert", "ColModernVBertConfig"),
        ("colpali", "ColPaliConfig"),
        ("colqwen2", "ColQwen2Config"),
        ("conditional_detr", "ConditionalDetrConfig"),
        ("convbert", "ConvBertConfig"),
        ("convnext", "ConvNextConfig"),
        ("convnextv2", "ConvNextV2Config"),
        ("cosmos3_edge", "Cosmos3EdgeConfig"),
        ("cosmos3_edge_text", "Cosmos3EdgeTextConfig"),
        ("cosmos3_edge_vision", "Cosmos3EdgeVisionConfig"),
        ("cosmos3_omni", "Cosmos3OmniConfig"),
        ("cpmant", "CpmAntConfig"),
        ("csm", "CsmConfig"),
        ("csm_depth_decoder_model", "CsmDepthDecoderConfig"),
        ("ctrl", "CTRLConfig"),
        ("cvt", "CvtConfig"),
        ("cwm", "CwmConfig"),
        ("d_fine", "DFineConfig"),
        ("dab-detr", "DabDetrConfig"),
        ("dac", "DacConfig"),
        ("data2vec-audio", "Data2VecAudioConfig"),
        ("data2vec-text", "Data2VecTextConfig"),
        ("data2vec-vision", "Data2VecVisionConfig"),
        ("dbrx", "DbrxConfig"),
        ("deberta", "DebertaConfig"),
        ("deberta-v2", "DebertaV2Config"),
        ("decision_transformer", "DecisionTransformerConfig"),
        ("deepseek_ocr2", "DeepseekOcr2Config"),
        ("deepseek_ocr2_encoder", "DeepseekOcr2VisionEncoderConfig"),
        ("deepseek_ocr2_sam_vision_model", "DeepseekOcr2SamVisionConfig"),
        ("deepseek_ocr2_text", "DeepseekOcr2TextConfig"),
        ("deepseek_ocr2_vision", "DeepseekOcr2VisionConfig"),
        ("deepseek_v2", "DeepseekV2Config"),
        ("deepseek_v3", "DeepseekV3Config"),
        ("deepseek_v32", "DeepseekV32Config"),
        ("deepseek_v4", "DeepseekV4Config"),
        ("deepseek_vl", "DeepseekVLConfig"),
        ("deepseek_vl_hybrid", "DeepseekVLHybridConfig"),
        ("deformable_detr", "DeformableDetrConfig"),
        ("deimv2", "Deimv2Config"),
        ("deit", "DeiTConfig"),
        ("depth_anything", "DepthAnythingConfig"),
        ("depth_pro", "DepthProConfig"),
        ("detr", "DetrConfig"),
        ("dia", "DiaConfig"),
        ("dia_decoder", "DiaDecoderConfig"),
        ("dia_encoder", "DiaEncoderConfig"),
        ("diffllama", "DiffLlamaConfig"),
        ("diffusion_gemma", "DiffusionGemmaConfig"),
        ("diffusion_gemma_text", "DiffusionGemmaTextConfig"),
        ("dinat", "DinatConfig"),
        ("dinov2", "Dinov2Config"),
        ("dinov2_with_registers", "Dinov2WithRegistersConfig"),
        ("dinov3_convnext", "DINOv3ConvNextConfig"),
        ("dinov3_vit", "DINOv3ViTConfig"),
        ("distilbert", "DistilBertConfig"),
        ("doge", "DogeConfig"),
        ("donut-swin", "DonutSwinConfig"),
        ("dots1", "Dots1Config"),
        ("dpr", "DPRConfig"),
        ("dpt", "DPTConfig"),
        ("edgetam", "EdgeTamConfig"),
        ("edgetam_video", "EdgeTamVideoConfig"),
        ("edgetam_vision_model", "EdgeTamVisionConfig"),
        ("efficientloftr", "EfficientLoFTRConfig"),
        ("efficientnet", "EfficientNetConfig"),
        ("electra", "ElectraConfig"),
        ("emu3", "Emu3Config"),
        ("emu3_text_model", "Emu3TextConfig"),
        ("emu3_vqgan", "Emu3VQVAEConfig"),
        ("encodec", "EncodecConfig"),
        ("encoder-decoder", "EncoderDecoderConfig"),
        ("eomt", "EomtConfig"),
        ("eomt_dinov3", "EomtDinov3Config"),
        ("ernie", "ErnieConfig"),
        ("ernie4_5", "Ernie4_5Config"),
        ("ernie4_5_moe", "Ernie4_5_MoeConfig"),
        ("ernie4_5_vl_moe", "Ernie4_5_VLMoeConfig"),
        ("ernie4_5_vl_moe_text", "Ernie4_5_VLMoeTextConfig"),
        ("ernie4_5_vl_moe_vision", "Ernie4_5_VLMoeVisionConfig"),
        ("esm", "EsmConfig"),
        ("esmc", "EsmcConfig"),
        ("esmfold2", "EsmFold2Config"),
        ("eurobert", "EuroBertConfig"),
        ("evolla", "EvollaConfig"),
        ("exaone4", "Exaone4Config"),
        ("exaone4_5", "Exaone4_5_Config"),
        ("exaone4_5_vision", "Exaone4_5_VisionConfig"),
        ("exaone_moe", "ExaoneMoeConfig"),
        ("falcon", "FalconConfig"),
        ("falcon_h1", "FalconH1Config"),
        ("falcon_mamba", "FalconMambaConfig"),
        ("fast_vlm", "FastVlmConfig"),
        ("fastspeech2_conformer", "FastSpeech2ConformerConfig"),
        ("fastspeech2_conformer_hifigan", "FastSpeech2ConformerHifiGanConfig"),
        ("fastspeech2_conformer_with_hifigan", "FastSpeech2ConformerWithHifiGanConfig"),
        ("flaubert", "FlaubertConfig"),
        ("flava", "FlavaConfig"),
        ("flava_image_model", "FlavaImageConfig"),
        ("flava_multimodal_model", "FlavaMultimodalConfig"),
        ("flava_text_model", "FlavaTextConfig"),
        ("flex_olmo", "FlexOlmoConfig"),
        ("florence2", "Florence2Config"),
        ("florence_vision", "Florence2VisionConfig"),
        ("fnet", "FNetConfig"),
        ("focalnet", "FocalNetConfig"),
        ("fsmt", "FSMTConfig"),
        ("fun_asr_nano", "FunAsrNanoConfig"),
        ("fun_asr_nano_encoder", "FunAsrNanoEncoderConfig"),
        ("funnel", "FunnelConfig"),
        ("fuyu", "FuyuConfig"),
        ("gemma", "GemmaConfig"),
        ("gemma2", "Gemma2Config"),
        ("gemma3", "Gemma3Config"),
        ("gemma3_text", "Gemma3TextConfig"),
        ("gemma3n", "Gemma3nConfig"),
        ("gemma3n_audio", "Gemma3nAudioConfig"),
        ("gemma3n_text", "Gemma3nTextConfig"),
        ("gemma3n_vision", "Gemma3nVisionConfig"),
        ("gemma4", "Gemma4Config"),
        ("gemma4_assistant", "Gemma4AssistantConfig"),
        ("gemma4_audio", "Gemma4AudioConfig"),
        ("gemma4_text", "Gemma4TextConfig"),
        ("gemma4_unified", "Gemma4UnifiedConfig"),
        ("gemma4_unified_assistant", "Gemma4UnifiedAssistantConfig"),
        ("gemma4_unified_audio", "Gemma4UnifiedAudioConfig"),
        ("gemma4_unified_text", "Gemma4UnifiedTextConfig"),
        ("gemma4_unified_vision", "Gemma4UnifiedVisionConfig"),
        ("gemma4_vision", "Gemma4VisionConfig"),
        ("git", "GitConfig"),
        ("git_vision_model", "GitVisionConfig"),
        ("glm", "GlmConfig"),
        ("glm4", "Glm4Config"),
        ("glm46v", "Glm46VConfig"),
        ("glm4_moe", "Glm4MoeConfig"),
        ("glm4_moe_lite", "Glm4MoeLiteConfig"),
        ("glm4v", "Glm4vConfig"),
        ("glm4v_moe", "Glm4vMoeConfig"),
        ("glm4v_moe_text", "Glm4vMoeTextConfig"),
        ("glm4v_moe_vision", "Glm4vMoeVisionConfig"),
        ("glm4v_text", "Glm4vTextConfig"),
        ("glm4v_vision", "Glm4vVisionConfig"),
        ("glm5_next", "Glm5NextConfig"),
        ("glm5_next_text", "Glm5NextTextConfig"),
        ("glm5_next_vision", "Glm5NextVisionConfig"),
        ("glm_image", "GlmImageConfig"),
        ("glm_image_text", "GlmImageTextConfig"),
        ("glm_image_vision", "GlmImageVisionConfig"),
        ("glm_image_vqmodel", "GlmImageVQVAEConfig"),
        ("glm_moe_dsa", "GlmMoeDsaConfig"),
        ("glm_ocr", "GlmOcrConfig"),
        ("glm_ocr_text", "GlmOcrTextConfig"),
        ("glm_ocr_vision", "GlmOcrVisionConfig"),
        ("glmasr", "GlmAsrConfig"),
        ("glmasr_encoder", "GlmAsrEncoderConfig"),
        ("glmga", "GlmgaConfig"),
        ("glpn", "GLPNConfig"),
        ("got_ocr2", "GotOcr2Config"),
        ("gpt2", "GPT2Config"),
        ("gpt_bigcode", "GPTBigCodeConfig"),
        ("gpt_neo", "GPTNeoConfig"),
        ("gpt_neox", "GPTNeoXConfig"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseConfig"),
        ("gpt_oss", "GptOssConfig"),
        ("gptj", "GPTJConfig"),
        ("granite", "GraniteConfig"),
        ("granite4_vision", "Granite4VisionConfig"),
        ("granite4_vision_text", "Granite4VisionTextConfig"),
        ("granite_speech", "GraniteSpeechConfig"),
        ("granite_speech5_ctc", "GraniteSpeech5CTCConfig"),
        ("granite_speech5_encoder", "GraniteSpeech5EncoderConfig"),
        ("granite_speech_encoder", "GraniteSpeechEncoderConfig"),
        ("granite_speech_plus", "GraniteSpeechPlusConfig"),
        ("granite_speech_plus_encoder", "GraniteSpeechPlusEncoderConfig"),
        ("granite_swa", "GraniteSWAConfig"),
        ("granitemoe", "GraniteMoeConfig"),
        ("granitemoe_swa", "GraniteMoeSWAConfig"),
        ("granitemoehybrid", "GraniteMoeHybridConfig"),
        ("granitemoeshared", "GraniteMoeSharedConfig"),
        ("grounding-dino", "GroundingDinoConfig"),
        ("groupvit", "GroupViTConfig"),
        ("groupvit_text_model", "GroupViTTextConfig"),
        ("groupvit_vision_model", "GroupViTVisionConfig"),
        ("helium", "HeliumConfig"),
        ("hgnet_v2", "HGNetV2Config"),
        ("hiera", "HieraConfig"),
        ("higgs_audio_v2", "HiggsAudioV2Config"),
        ("higgs_audio_v2_tokenizer", "HiggsAudioV2TokenizerConfig"),
        ("hrm_text", "HrmTextConfig"),
        ("hubert", "HubertConfig"),
        ("hunyuan_v1_dense", "HunYuanDenseV1Config"),
        ("hunyuan_v1_moe", "HunYuanMoEV1Config"),
        ("hunyuan_vl", "HunYuanVLConfig"),
        ("hunyuan_vl_text", "HunYuanVLTextConfig"),
        ("hunyuan_vl_vision", "HunYuanVLVisionConfig"),
        ("hy_v3", "HYV3Config"),
        ("hy_v4", "HYV4Config"),
        ("hyperclovax", "HyperCLOVAXConfig"),
        ("hyperclovax_vision_v2", "HyperCLOVAXVisionV2Config"),
        ("ibert", "IBertConfig"),
        ("idefics", "IdeficsConfig"),
        ("idefics2", "Idefics2Config"),
        ("idefics2_perceiver", "Idefics2PerceiverConfig"),
        ("idefics2_vision", "Idefics2VisionConfig"),
        ("idefics3", "Idefics3Config"),
        ("idefics3_vision", "Idefics3VisionConfig"),
        ("idefics_perciever", "IdeficsPerceiverConfig"),
        ("idefics_vision", "IdeficsVisionConfig"),
        ("ijepa", "IJepaConfig"),
        ("imagegpt", "ImageGPTConfig"),
        ("informer", "InformerConfig"),
        ("inkling_audio", "InklingAudioConfig"),
        ("inkling_mm_model", "InklingConfig"),
        ("inkling_text", "InklingTextConfig"),
        ("inkling_vision", "InklingVisionConfig"),
        ("instructblip", "InstructBlipConfig"),
        ("instructblip_qformer", "InstructBlipQFormerConfig"),
        ("instructblip_vision_model", "InstructBlipVisionConfig"),
        ("instructblipvideo", "InstructBlipVideoConfig"),
        ("instructblipvideo_qformer", "InstructBlipVideoQFormerConfig"),
        ("instructblipvideo_vision_model", "InstructBlipVideoVisionConfig"),
        ("internvl", "InternVLConfig"),
        ("internvl_vision", "InternVLVisionConfig"),
        ("jais2", "Jais2Config"),
        ("jamba", "JambaConfig"),
        ("janus", "JanusConfig"),
        ("janus_vision_model", "JanusVisionConfig"),
        ("janus_vqgan", "JanusVQVAEConfig"),
        ("jetmoe", "JetMoeConfig"),
        ("jina_embeddings_v3", "JinaEmbeddingsV3Config"),
        ("kimi_k25", "Kimi_K25Config"),
        ("kimi_k25_vision", "Kimi_K25VisionConfig"),
        ("kimi_linear", "KimiLinearConfig"),
        ("kosmos-2", "Kosmos2Config"),
        ("kosmos-2.5", "Kosmos2_5Config"),
        ("kosmos_2_5_text_model", "Kosmos2_5TextConfig"),
        ("kosmos_2_5_vision_model", "Kosmos2_5VisionConfig"),
        ("kosmos_2_text_model", "Kosmos2TextConfig"),
        ("kosmos_2_vision_model", "Kosmos2VisionConfig"),
        ("kyutai_speech_to_text", "KyutaiSpeechToTextConfig"),
        ("laguna", "LagunaConfig"),
        ("lasr_ctc", "LasrCTCConfig"),
        ("lasr_encoder", "LasrEncoderConfig"),
        ("layoutlm", "LayoutLMConfig"),
        ("layoutlmv2", "LayoutLMv2Config"),
        ("layoutlmv3", "LayoutLMv3Config"),
        ("layoutxlm", "LayoutXLMConfig"),
        ("led", "LEDConfig"),
        ("levit", "LevitConfig"),
        ("lfm2", "Lfm2Config"),
        ("lfm2_moe", "Lfm2MoeConfig"),
        ("lfm2_vl", "Lfm2VlConfig"),
        ("lightglue", "LightGlueConfig"),
        ("lighton_ocr", "LightOnOcrConfig"),
        ("lilt", "LiltConfig"),
        ("llama", "LlamaConfig"),
        ("llama4", "Llama4Config"),
        ("llama4_text", "Llama4TextConfig"),
        ("llama4_vision_model", "Llama4VisionConfig"),
        ("llava", "LlavaConfig"),
        ("llava_next", "LlavaNextConfig"),
        ("llava_next_video", "LlavaNextVideoConfig"),
        ("llava_onevision", "LlavaOnevisionConfig"),
        ("longcat_flash", "LongcatFlashConfig"),
        ("longformer", "LongformerConfig"),
        ("longt5", "LongT5Config"),
        ("luke", "LukeConfig"),
        ("lw_detr", "LwDetrConfig"),
        ("lw_detr_vit", "LwDetrViTConfig"),
        ("lxmert", "LxmertConfig"),
        ("m2m_100", "M2M100Config"),
        ("mamba", "MambaConfig"),
        ("mamba2", "Mamba2Config"),
        ("marian", "MarianConfig"),
        ("markuplm", "MarkupLMConfig"),
        ("mask2former", "Mask2FormerConfig"),
        ("maskformer", "MaskFormerConfig"),
        ("maskformer-swin", "MaskFormerSwinConfig"),
        ("mbart", "MBartConfig"),
        ("megatron-bert", "MegatronBertConfig"),
        ("mellum", "MellumConfig"),
        ("metaclip_2", "MetaClip2Config"),
        ("metaclip_2_text_model", "MetaClip2TextConfig"),
        ("metaclip_2_vision_model", "MetaClip2VisionConfig"),
        ("mgp-str", "MgpstrConfig"),
        ("mimi", "MimiConfig"),
        ("mimo_v2_flash", "MiMoV2FlashConfig"),
        ("minicpm3", "MiniCPM3Config"),
        ("minicpmv4_6", "MiniCPMV4_6Config"),
        ("minicpmv4_6_vision", "MiniCPMV4_6VisionConfig"),
        ("minicpmv4_7", "MiniCPMV4_7Config"),
        ("minicpmv4_7_vision", "MiniCPMV4_7VisionConfig"),
        ("minimax", "MiniMaxConfig"),
        ("minimax_m2", "MiniMaxM2Config"),
        ("minimax_m3_vl", "MiniMaxM3VLConfig"),
        ("minimax_m3_vl_text", "MiniMaxM3VLTextConfig"),
        ("minimax_m3_vl_vision", "MiniMaxM3VLVisionConfig"),
        ("ministral", "MinistralConfig"),
        ("ministral3", "Ministral3Config"),
        ("mistral", "MistralConfig"),
        ("mistral3", "Mistral3Config"),
        ("mistral4", "Mistral4Config"),
        ("mixtral", "MixtralConfig"),
        ("mlcd_vision_model", "MLCDVisionConfig"),
        ("mllama", "MllamaConfig"),
        ("mllama_text_model", "MllamaTextConfig"),
        ("mllama_vision_model", "MllamaVisionConfig"),
        ("mm-grounding-dino", "MMGroundingDinoConfig"),
        ("mobilebert", "MobileBertConfig"),
        ("mobilenet_v1", "MobileNetV1Config"),
        ("mobilenet_v2", "MobileNetV2Config"),
        ("mobilevit", "MobileViTConfig"),
        ("mobilevitv2", "MobileViTV2Config"),
        ("modernbert", "ModernBertConfig"),
        ("modernbert-decoder", "ModernBertDecoderConfig"),
        ("modernvbert", "ModernVBertConfig"),
        ("moonshine", "MoonshineConfig"),
        ("moonshine_streaming", "MoonshineStreamingConfig"),
        ("moonshine_streaming_encoder", "MoonshineStreamingEncoderConfig"),
        ("moshi", "MoshiConfig"),
        ("moshi_depth", "MoshiDepthConfig"),
        ("mpnet", "MPNetConfig"),
        ("mpt", "MptConfig"),
        ("mra", "MraConfig"),
        ("mt5", "MT5Config"),
        ("muse_glimmer", "MuseGlimmerConfig"),
        ("muse_glimmer_assistant", "MuseGlimmerAssistantConfig"),
        ("muse_glimmer_text", "MuseGlimmerTextConfig"),
        ("muse_glimmer_vision", "MuseGlimmerVisionConfig"),
        ("musicflamingo", "MusicFlamingoConfig"),
        ("musicgen", "MusicgenConfig"),
        ("musicgen_decoder", "MusicgenDecoderConfig"),
        ("musicgen_melody", "MusicgenMelodyConfig"),
        ("musicgen_melody_decoder", "MusicgenMelodyDecoderConfig"),
        ("mvp", "MvpConfig"),
        ("nanochat", "NanoChatConfig"),
        ("nemotron", "NemotronConfig"),
        ("nemotron3_5_asr", "Nemotron3_5AsrConfig"),
        ("nemotron3_diarization", "Nemotron3DiarizationConfig"),
        ("nemotron3_diarization_audio", "Nemotron3DiarizationAudioConfig"),
        ("nemotron_asr_streaming", "NemotronAsrStreamingConfig"),
        ("nemotron_asr_streaming_encoder", "NemotronAsrStreamingEncoderConfig"),
        ("nemotron_h", "NemotronHConfig"),
        ("nemotron_h_omni", "NemotronH_Omni_Reasoning_V3_Config"),
        ("neomme", "NeoMMEConfig"),
        ("neucodec", "NeuCodecConfig"),
        ("nllb-moe", "NllbMoeConfig"),
        ("nomic_bert", "NomicBertConfig"),
        ("nougat", "NougatConfig"),
        ("nystromformer", "NystromformerConfig"),
        ("olmo", "OlmoConfig"),
        ("olmo2", "Olmo2Config"),
        ("olmo3", "Olmo3Config"),
        ("olmo_hybrid", "OlmoHybridConfig"),
        ("olmoe", "OlmoeConfig"),
        ("omdet-turbo", "OmDetTurboConfig"),
        ("oneformer", "OneFormerConfig"),
        ("openai-gpt", "OpenAIGPTConfig"),
        ("openai_privacy_filter", "OpenAIPrivacyFilterConfig"),
        ("opt", "OPTConfig"),
        ("ovis2", "Ovis2Config"),
        ("owlv2", "Owlv2Config"),
        ("owlv2_text_model", "Owlv2TextConfig"),
        ("owlv2_vision_model", "Owlv2VisionConfig"),
        ("owlvit", "OwlViTConfig"),
        ("owlvit_text_model", "OwlViTTextConfig"),
        ("owlvit_vision_model", "OwlViTVisionConfig"),
        ("paddleocr_vl", "PaddleOCRVLConfig"),
        ("paddleocr_vl_text", "PaddleOCRTextConfig"),
        ("paddleocr_vl_vision", "PaddleOCRVisionConfig"),
        ("paligemma", "PaliGemmaConfig"),
        ("parakeet_ctc", "ParakeetCTCConfig"),
        ("parakeet_encoder", "ParakeetEncoderConfig"),
        ("parakeet_rnnt", "ParakeetRNNTConfig"),
        ("patchtsmixer", "PatchTSMixerConfig"),
        ("patchtst", "PatchTSTConfig"),
        ("pe_audio", "PeAudioConfig"),
        ("pe_audio_encoder", "PeAudioEncoderConfig"),
        ("pe_audio_video", "PeAudioVideoConfig"),
        ("pe_audio_video_encoder", "PeAudioVideoEncoderConfig"),
        ("pe_video", "PeVideoConfig"),
        ("pe_video_encoder", "PeVideoEncoderConfig"),
        ("pegasus", "PegasusConfig"),
        ("pegasus_x", "PegasusXConfig"),
        ("perceiver", "PerceiverConfig"),
        ("perception_lm", "PerceptionLMConfig"),
        ("persimmon", "PersimmonConfig"),
        ("phi", "PhiConfig"),
        ("phi3", "Phi3Config"),
        ("phi4_multimodal", "Phi4MultimodalConfig"),
        ("phi4_multimodal_audio", "Phi4MultimodalAudioConfig"),
        ("phi4_multimodal_vision", "Phi4MultimodalVisionConfig"),
        ("phimoe", "PhimoeConfig"),
        ("pi0", "PI0Config"),
        ("pix2struct", "Pix2StructConfig"),
        ("pix2struct_text_model", "Pix2StructTextConfig"),
        ("pix2struct_vision_model", "Pix2StructVisionConfig"),
        ("pixio", "PixioConfig"),
        ("pixtral", "PixtralVisionConfig"),
        ("plbart", "PLBartConfig"),
        ("poolformer", "PoolFormerConfig"),
        ("pop2piano", "Pop2PianoConfig"),
        ("pp_chart2table", "PPChart2TableConfig"),
        ("pp_doclayout_v2", "PPDocLayoutV2Config"),
        ("pp_doclayout_v3", "PPDocLayoutV3Config"),
        ("pp_formulanet", "PPFormulaNetConfig"),
        ("pp_lcnet", "PPLCNetConfig"),
        ("pp_lcnet_v3", "PPLCNetV3Config"),
        ("pp_lcnet_v4", "PPLCNetV4Config"),
        ("pp_ocrv5_mobile_det", "PPOCRV5MobileDetConfig"),
        ("pp_ocrv5_mobile_rec", "PPOCRV5MobileRecConfig"),
        ("pp_ocrv5_server_det", "PPOCRV5ServerDetConfig"),
        ("pp_ocrv5_server_rec", "PPOCRV5ServerRecConfig"),
        ("pp_ocrv6_medium_det", "PPOCRV6MediumDetConfig"),
        ("pp_ocrv6_small_det", "PPOCRV6SmallDetConfig"),
        ("pp_ocrv6_small_rec", "PPOCRV6SmallRecConfig"),
        ("pp_ocrv6_tiny_rec", "PPOCRV6TinyRecConfig"),
        ("prompt_depth_anything", "PromptDepthAnythingConfig"),
        ("prophetnet", "ProphetNetConfig"),
        ("pvt", "PvtConfig"),
        ("pvt_v2", "PvtV2Config"),
        ("qianfan_ocr", "QianfanOCRConfig"),
        ("qianfan_ocr_vision", "QianfanOCRVisionConfig"),
        ("qwen2", "Qwen2Config"),
        ("qwen2_5_omni", "Qwen2_5OmniConfig"),
        ("qwen2_5_omni_audio_encoder", "Qwen2_5OmniAudioEncoderConfig"),
        ("qwen2_5_omni_bigvgan", "Qwen2_5OmniBigVGANConfig"),
        ("qwen2_5_omni_dit", "Qwen2_5OmniDiTConfig"),
        ("qwen2_5_omni_talker", "Qwen2_5OmniTalkerConfig"),
        ("qwen2_5_omni_text", "Qwen2_5OmniTextConfig"),
        ("qwen2_5_omni_thinker", "Qwen2_5OmniThinkerConfig"),
        ("qwen2_5_omni_token2wav", "Qwen2_5OmniToken2WavConfig"),
        ("qwen2_5_omni_vision_encoder", "Qwen2_5OmniVisionEncoderConfig"),
        ("qwen2_5_vl", "Qwen2_5_VLConfig"),
        ("qwen2_5_vl_text", "Qwen2_5_VLTextConfig"),
        ("qwen2_5_vl_vision", "Qwen2_5_VLVisionConfig"),
        ("qwen2_audio", "Qwen2AudioConfig"),
        ("qwen2_audio_encoder", "Qwen2AudioEncoderConfig"),
        ("qwen2_moe", "Qwen2MoeConfig"),
        ("qwen2_vl", "Qwen2VLConfig"),
        ("qwen2_vl_text", "Qwen2VLTextConfig"),
        ("qwen2_vl_vision", "Qwen2VLVisionConfig"),
        ("qwen3", "Qwen3Config"),
        ("qwen3_5", "Qwen3_5Config"),
        ("qwen3_5_moe", "Qwen3_5MoeConfig"),
        ("qwen3_5_moe_text", "Qwen3_5MoeTextConfig"),
        ("qwen3_5_moe_vision", "Qwen3_5MoeVisionConfig"),
        ("qwen3_5_text", "Qwen3_5TextConfig"),
        ("qwen3_5_vision", "Qwen3_5VisionConfig"),
        ("qwen3_asr", "Qwen3ASRConfig"),
        ("qwen3_asr_encoder", "Qwen3ASREncoderConfig"),
        ("qwen3_moe", "Qwen3MoeConfig"),
        ("qwen3_next", "Qwen3NextConfig"),
        ("qwen3_omni_moe", "Qwen3OmniMoeConfig"),
        ("qwen3_omni_moe_audio_encoder", "Qwen3OmniMoeAudioEncoderConfig"),
        ("qwen3_omni_moe_talker_code_predictor", "Qwen3OmniMoeTalkerCodePredictorConfig"),
        ("qwen3_omni_moe_talker_text", "Qwen3OmniMoeTalkerTextConfig"),
        ("qwen3_omni_moe_text", "Qwen3OmniMoeTextConfig"),
        ("qwen3_omni_moe_thinker", "Qwen3OmniMoeThinkerConfig"),
        ("qwen3_omni_moe_vision_encoder", "Qwen3OmniMoeVisionEncoderConfig"),
        ("qwen3_vl", "Qwen3VLConfig"),
        ("qwen3_vl_moe", "Qwen3VLMoeConfig"),
        ("qwen3_vl_moe_text", "Qwen3VLMoeTextConfig"),
        ("qwen3_vl_moe_vision", "Qwen3VLMoeVisionConfig"),
        ("qwen3_vl_text", "Qwen3VLTextConfig"),
        ("qwen3_vl_vision", "Qwen3VLVisionConfig"),
        ("qwen4_exp", "Qwen4ExpConfig"),
        ("qwen4_exp_text", "Qwen4ExpTextConfig"),
        ("qwen4_exp_vision", "Qwen4ExpVisionConfig"),
        ("radio", "RadioConfig"),
        ("rag", "RagConfig"),
        ("recurrent_gemma", "RecurrentGemmaConfig"),
        ("reformer", "ReformerConfig"),
        ("regnet", "RegNetConfig"),
        ("rembert", "RemBertConfig"),
        ("resnet", "ResNetConfig"),
        ("rf_detr", "RfDetrConfig"),
        ("rf_detr_dinov2", "RfDetrDinov2Config"),
        ("roberta", "RobertaConfig"),
        ("roberta-prelayernorm", "RobertaPreLayerNormConfig"),
        ("roc_bert", "RoCBertConfig"),
        ("roformer", "RoFormerConfig"),
        ("rt_detr", "RTDetrConfig"),
        ("rt_detr_resnet", "RTDetrResNetConfig"),
        ("rt_detr_v2", "RTDetrV2Config"),
        ("rwkv", "RwkvConfig"),
        ("sam", "SamConfig"),
        ("sam2", "Sam2Config"),
        ("sam2_hiera_det_model", "Sam2HieraDetConfig"),
        ("sam2_video", "Sam2VideoConfig"),
        ("sam2_vision_model", "Sam2VisionConfig"),
        ("sam3", "Sam3Config"),
        ("sam3_detr_decoder", "Sam3DETRDecoderConfig"),
        ("sam3_detr_encoder", "Sam3DETREncoderConfig"),
        ("sam3_geometry_encoder", "Sam3GeometryEncoderConfig"),
        ("sam3_lite_text", "Sam3LiteTextConfig"),
        ("sam3_lite_text_detr_decoder", "Sam3LiteTextDETRDecoderConfig"),
        ("sam3_lite_text_detr_encoder", "Sam3LiteTextDETREncoderConfig"),
        ("sam3_lite_text_geometry_encoder", "Sam3LiteTextGeometryEncoderConfig"),
        ("sam3_lite_text_mask_decoder", "Sam3LiteTextMaskDecoderConfig"),
        ("sam3_lite_text_text_model", "Sam3LiteTextTextConfig"),
        ("sam3_mask_decoder", "Sam3MaskDecoderConfig"),
        ("sam3_tracker", "Sam3TrackerConfig"),
        ("sam3_tracker_video", "Sam3TrackerVideoConfig"),
        ("sam3_video", "Sam3VideoConfig"),
        ("sam3_vision_model", "Sam3VisionConfig"),
        ("sam3_vit_model", "Sam3ViTConfig"),
        ("sam_hq", "SamHQConfig"),
        ("sam_hq_vision_model", "SamHQVisionConfig"),
        ("sam_vision_model", "SamVisionConfig"),
        ("sapiens2", "Sapiens2Config"),
        ("sapiens2_head", "Sapiens2HeadConfig"),
        ("seamless_m4t", "SeamlessM4TConfig"),
        ("seamless_m4t_v2", "SeamlessM4Tv2Config"),
        ("seed_oss", "SeedOssConfig"),
        ("segformer", "SegformerConfig"),
        ("seggpt", "SegGptConfig"),
        ("sew", "SEWConfig"),
        ("sew-d", "SEWDConfig"),
        ("shieldgemma2", "ShieldGemma2Config"),
        ("siglip", "SiglipConfig"),
        ("siglip2", "Siglip2Config"),
        ("siglip2_text_model", "Siglip2TextConfig"),
        ("siglip2_vision_model", "Siglip2VisionConfig"),
        ("siglip_text_model", "SiglipTextConfig"),
        ("siglip_vision_model", "SiglipVisionConfig"),
        ("slanet", "SLANetConfig"),
        ("slanext", "SLANeXtConfig"),
        ("smollm3", "SmolLM3Config"),
        ("smolvlm", "SmolVLMConfig"),
        ("smolvlm_vision", "SmolVLMVisionConfig"),
        ("solar_open", "SolarOpenConfig"),
        ("speech-encoder-decoder", "SpeechEncoderDecoderConfig"),
        ("speech_to_text", "Speech2TextConfig"),
        ("speecht5", "SpeechT5Config"),
        ("speecht5_hifigan", "SpeechT5HifiGanConfig"),
        ("splinter", "SplinterConfig"),
        ("squeezebert", "SqueezeBertConfig"),
        ("stablelm", "StableLmConfig"),
        ("starcoder2", "Starcoder2Config"),
        ("step3p5", "Step3p7TextConfig"),
        ("step3p5_vision", "Step3p7VisionConfig"),
        ("step3p7", "Step3p7Config"),
        ("superglue", "SuperGlueConfig"),
        ("superpoint", "SuperPointConfig"),
        ("swiftformer", "SwiftFormerConfig"),
        ("swin", "SwinConfig"),
        ("swin2sr", "Swin2SRConfig"),
        ("swinv2", "Swinv2Config"),
        ("switch_transformers", "SwitchTransformersConfig"),
        ("t5", "T5Config"),
        ("t5_gemma_module", "T5GemmaModuleConfig"),
        ("t5gemma", "T5GemmaConfig"),
        ("t5gemma2", "T5Gemma2Config"),
        ("t5gemma2_decoder", "T5Gemma2DecoderConfig"),
        ("t5gemma2_encoder", "T5Gemma2EncoderConfig"),
        ("t5gemma2_text", "T5Gemma2TextConfig"),
        ("table-transformer", "TableTransformerConfig"),
        ("tapas", "TapasConfig"),
        ("textnet", "TextNetConfig"),
        ("time_series_transformer", "TimeSeriesTransformerConfig"),
        ("timesfm", "TimesFmConfig"),
        ("timesfm2_5", "TimesFm2_5Config"),
        ("timesformer", "TimesformerConfig"),
        ("timm_backbone", "TimmBackboneConfig"),
        ("timm_wrapper", "TimmWrapperConfig"),
        ("tipsv2", "Tipsv2Config"),
        ("tipsv2_dpt", "Tipsv2DptConfig"),
        ("tipsv2_text_model", "Tipsv2TextConfig"),
        ("tipsv2_vision_model", "Tipsv2VisionConfig"),
        ("trocr", "TrOCRConfig"),
        ("tvp", "TvpConfig"),
        ("udop", "UdopConfig"),
        ("umt5", "UMT5Config"),
        ("unispeech", "UniSpeechConfig"),
        ("unispeech-sat", "UniSpeechSatConfig"),
        ("univnet", "UnivNetConfig"),
        ("upernet", "UperNetConfig"),
        ("uvdoc", "UVDocConfig"),
        ("uvdoc_backbone", "UVDocBackboneConfig"),
        ("vaultgemma", "VaultGemmaConfig"),
        ("vibevoice", "VibeVoiceConfig"),
        ("vibevoice_acoustic_tokenizer", "VibeVoiceAcousticTokenizerConfig"),
        ("vibevoice_asr", "VibeVoiceAsrConfig"),
        ("video_llama_3", "VideoLlama3Config"),
        ("video_llama_3_vision", "VideoLlama3VisionConfig"),
        ("video_llava", "VideoLlavaConfig"),
        ("videomae", "VideoMAEConfig"),
        ("videomt", "VideomtConfig"),
        ("videoprism", "VideoPrismConfig"),
        ("videoprism_text_model", "VideoPrismTextConfig"),
        ("videoprism_vision_model", "VideoPrismVisionConfig"),
        ("vilt", "ViltConfig"),
        ("vipllava", "VipLlavaConfig"),
        ("vision-encoder-decoder", "VisionEncoderDecoderConfig"),
        ("vision-text-dual-encoder", "VisionTextDualEncoderConfig"),
        ("visual_bert", "VisualBertConfig"),
        ("vit", "ViTConfig"),
        ("vit_mae", "ViTMAEConfig"),
        ("vit_msn", "ViTMSNConfig"),
        ("vitdet", "VitDetConfig"),
        ("vitmatte", "VitMatteConfig"),
        ("vitpose", "VitPoseConfig"),
        ("vitpose_backbone", "VitPoseBackboneConfig"),
        ("vits", "VitsConfig"),
        ("vivit", "VivitConfig"),
        ("vjepa2", "VJEPA2Config"),
        ("voxtral", "VoxtralConfig"),
        ("voxtral_encoder", "VoxtralEncoderConfig"),
        ("voxtral_realtime", "VoxtralRealtimeConfig"),
        ("voxtral_realtime_encoder", "VoxtralRealtimeEncoderConfig"),
        ("voxtral_realtime_text", "VoxtralRealtimeTextConfig"),
        ("wav2vec2", "Wav2Vec2Config"),
        ("wav2vec2-bert", "Wav2Vec2BertConfig"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerConfig"),
        ("wavlm", "WavLMConfig"),
        ("whisper", "WhisperConfig"),
        ("xclip", "XCLIPConfig"),
        ("xclip_text_model", "XCLIPTextConfig"),
        ("xclip_vision_model", "XCLIPVisionConfig"),
        ("xcodec", "XcodecConfig"),
        ("xcodec2", "Xcodec2Config"),
        ("xglm", "XGLMConfig"),
        ("xlm", "XLMConfig"),
        ("xlm-roberta", "XLMRobertaConfig"),
        ("xlm-roberta-xl", "XLMRobertaXLConfig"),
        ("xlnet", "XLNetConfig"),
        ("xlstm", "xLSTMConfig"),
        ("xmod", "XmodConfig"),
        ("yolos", "YolosConfig"),
        ("yoso", "YosoConfig"),
        ("youtu", "YoutuConfig"),
        ("zamba", "ZambaConfig"),
        ("zamba2", "Zamba2Config"),
        ("zaya", "ZayaConfig"),
        ("zoedepth", "ZoeDepthConfig"),
    ]
)

CONFIG_MAPPING_NAMES.update(
    {
        "EvollaModel": "EvollaConfig",
        "mlcd": "MLCDVisionConfig",
        "parakeet_tdt": "ParakeetTDTConfig",
        "vibevoice_acoustic_tokenizer_decoder": "VibeVoiceAcousticTokenizerDecoderConfig",
        "vibevoice_acoustic_tokenizer_encoder": "VibeVoiceAcousticTokenizerEncoderConfig",
    }
)

CONFIG_MAPPING_NAMES = OrderedDict(**{"gpt-sw3": "GPT2Config"}, **CONFIG_MAPPING_NAMES)

class Qwen3Config(PreTrainedConfig):
    model_type = "qwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen3`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 32
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters = None
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window: int | None = 4096
    max_window_layers: int = 28
    layer_types: list[str] | None = None
    attention_dropout: float | int = 0.0
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        super().__post_init__(**kwargs)

class WhisperConfig(PreTrainedConfig):
    def __post_init__(self, **kwargs): pass

    model_type = "whisper"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_key_value_heads": "encoder_attention_heads",
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "encoder_layers",
    }

    vocab_size: int = 51865
    num_mel_bins: int = 80
    encoder_layers: int = 4
    encoder_attention_heads: int = 6
    decoder_layers: int = 4
    decoder_attention_heads: int = 6
    decoder_ffn_dim: int = 1536
    encoder_ffn_dim: int = 1536
    encoder_layerdrop: float | int = 0.0
    decoder_layerdrop: float | int = 0.0
    decoder_start_token_id: int = 50257
    use_cache: bool = True
    is_encoder_decoder: bool = True
    activation_function: str = "gelu"
    d_model: int = 384
    dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    init_std: float = 0.02
    scale_embedding: bool = False
    max_source_positions: int = 1500
    max_target_positions: int = 448
    pad_token_id: int | None = 50256
    bos_token_id: int | None = 50256
    eos_token_id: int | list[int] | None = 50256
    suppress_tokens: list | None = None
    begin_suppress_tokens: list[int] | tuple[int, ...] | None = (220, 50256)
    use_weighted_layer_sum: bool = False
    classifier_proj_size: int = 256
    apply_spec_augment: bool = False
    mask_time_prob: float | int = 0.05
    mask_time_length: int = 10
    mask_time_min_masks: int = 2
    mask_feature_prob: float | int = 0.0
    mask_feature_length: int = 10
    mask_feature_min_masks: int = 0
    median_filter_width: int = 7
    tie_word_embeddings: bool = True

class MossTranscribeDiarizeConfig(PreTrainedConfig):
    model_type = "moss_transcribe_diarize"
    sub_configs = {"text_config": Qwen3Config, "audio_config": WhisperConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        audio_config=None,
        audio_token_id: int = 151671,
        audio_merge_size: int = 4,
        adaptor_input_dim: int | None = None,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        text_config = Qwen3Config()
        text_config.attention_bias = False
        text_config.attention_dropout = 0.0
        text_config.bos_token_id = None
        text_config.eos_token_id = None
        text_config.head_dim = 128
        text_config.hidden_act = "silu"
        text_config.hidden_size = 1024
        text_config.initializer_range = 0.02
        text_config.intermediate_size = 3072
        text_config.layer_types = [
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention",
"full_attention"
]
        text_config.max_position_embeddings = 131072
        text_config.max_window_layers = 28
        text_config.model_type = "qwen3"
        text_config.num_attention_heads = 16
        text_config.num_hidden_layers = 28
        text_config.num_key_value_heads = 8
        text_config.pad_token_id = 151643
        text_config.rms_norm_eps = 1e-6
        text_config.rope_parameters = {
"rope_theta": 1000000,
"rope_type": "default"
}
        text_config.sliding_window = None
        text_config.tie_word_embeddings = True
        text_config.transformers_version = "5.17.0"
        text_config.use_cache = True
        text_config.use_sliding_window = False
        text_config.vocab_size = 151936

        audio_config = WhisperConfig()
        audio_config.num_mel_bins=80
        audio_config.d_model=1024
        audio_config.encoder_layers=24
        audio_config.encoder_attention_heads=16
        audio_config.encoder_ffn_dim=4096
        audio_config.max_source_positions=1500
        audio_config.dropout=0.0
        audio_config.attention_dropout=0.0
        audio_config.activation_dropout=0.0
        audio_config.activation_function="gelu"
        audio_config.encoder_layerdrop=0.0
        audio_config.scale_embedding=False

        text_config.tie_word_embeddings = tie_word_embeddings

        self.text_config = text_config
        self.audio_config = audio_config
        self.audio_token_id = audio_token_id
        self.audio_merge_size = audio_merge_size
        self.adaptor_input_dim = adaptor_input_dim or audio_config.d_model * audio_merge_size
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
from transformers.models.whisper.modeling_whisper import WhisperPreTrainedModel, WhisperEncoderLayer, WhisperAttention
from transformers.utils.generic import merge_with_config_defaults, split_attention_implementation, is_flash_attention_requested
from transformers.utils.output_capturing import capture_outputs
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.modeling_outputs import BaseModelOutput
import math

class WhisperEncoder(WhisperPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: WhisperConfig
    """

    _can_record_outputs = {
        "hidden_states": WhisperEncoderLayer,
        "attentions": WhisperAttention,
    }
    input_modalities = ("audio",)

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)

        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_features,
        attention_mask=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        all_positions = torch.arange(self.embed_positions.num_embeddings, device=inputs_embeds.device)

        hidden_states = inputs_embeds + self.embed_positions(all_positions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        for idx, encoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if not to_drop:
                hidden_states = encoder_layer(
                    hidden_states,
                    None,
                    **kwargs,
                )

        hidden_states = self.layer_norm(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


class VQAdaptor(nn.Module):
    """Projects merged Whisper features to LM hidden dim.

    ``Linear(in → hidden) → SiLU → Linear(hidden → hidden) → LayerNorm``
    """

    def __init__(self, input_dim: int, hidden_size: int, norm_eps: float = 1e-6):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size, eps=norm_eps, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.layers(x)

class MossTranscribeDiarizeModel(PreTrainedModel):
    base_model_prefix = "model"
    def __init__(self, config: MossTranscribeDiarizeConfig):
        super().__init__(config)

        self.language_model: nn.Module = Qwen3Model(config.text_config)
        self.whisper_encoder: nn.Module = WhisperEncoder(config.audio_config)
        self.vq_adaptor: VQAdaptor = VQAdaptor(
            input_dim=config.adaptor_input_dim,
            hidden_size=config.text_config.hidden_size,
            norm_eps=config.text_config.rms_norm_eps,
        )
        self.post_init()

    # ---- 4x time merge ---------------------------------------------------

    def time_merge(self, features: torch.Tensor) -> torch.Tensor:
        """``(B, T, D) -> (B, T//M, D*M)`` where M is ``audio_merge_size``."""
        B, T, D = features.shape
        merge_size = int(self.config.audio_merge_size)
        T_trim = (T // merge_size) * merge_size
        return features[:, :T_trim, :].reshape(B, T_trim // merge_size, D * merge_size)

    # ---- audio feature extraction -----------------------------------------

    def get_audio_features(
        self,
        input_features: torch.Tensor,
        audio_feature_lengths: torch.LongTensor,
        audio_chunk_mapping: Optional[torch.LongTensor] = None,
    ) -> list[torch.Tensor]:
        device = next(self.whisper_encoder.parameters()).device
        encoder_dtype = next(self.whisper_encoder.parameters()).dtype
        input_features = input_features.to(device=device, dtype=encoder_dtype)
        audio_feature_lengths = audio_feature_lengths.to(device=device)

        whisper_features = self.whisper_encoder(input_features, return_dict=True).last_hidden_state

        chunk_mapping = (
            audio_chunk_mapping.to(device=device)
            if audio_chunk_mapping is not None
            else torch.zeros(input_features.shape[0], dtype=torch.long, device=device)
        )

        num_audios = int(chunk_mapping.max().item()) + 1 if chunk_mapping.numel() else 0
        per_audio_chunks = [[] for _ in range(num_audios)]
        for chunk_idx, token_len in enumerate(audio_feature_lengths.tolist()):
            sample_idx = int(chunk_mapping[chunk_idx].item())
            per_audio_chunks[sample_idx].append(
                whisper_features[chunk_idx : chunk_idx + 1, : int(token_len) * 4]
            )

        adapted = []
        for parts in per_audio_chunks:
            feat = torch.cat(parts, dim=1)
            feat = feat.to(self.dtype)
            merged = self.time_merge(feat)
            adapted.append(self.vq_adaptor(merged))
        return adapted

    # ---- inject audio into text embeddings --------------------------------

    def get_placeholder_mask(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: torch.FloatTensor,
        audio_features: torch.Tensor,
    ) -> torch.BoolTensor:
        special_audio_mask = input_ids.to(device=inputs_embeds.device) == self.config.audio_token_id
        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        return special_audio_mask

    def inject_audio_features(
        self,
        input_ids,
        inputs_embeds,
        input_features,
        audio_feature_lengths,
        audio_chunk_mapping,
    ):
        """Replace audio placeholder positions with projected audio features."""
        if input_features is None:
            return inputs_embeds
        audio_features = self.get_audio_features(
            input_features=input_features,
            audio_feature_lengths=audio_feature_lengths,
            audio_chunk_mapping=audio_chunk_mapping,
        )
        audio_embeds = torch.cat([f.squeeze(0) for f in audio_features], dim=0)
        audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds, audio_embeds)
        return inputs_embeds.masked_scatter(audio_mask, audio_embeds)

    # ---- forward ----------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_features: Optional[torch.FloatTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        audio_chunk_mapping: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        inputs_embeds = self.inject_audio_features(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            input_features=input_features,
            audio_feature_lengths=audio_feature_lengths,
            audio_chunk_mapping=audio_chunk_mapping,
        )
        outputs = self.language_model(
            input_ids=None, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds,
            use_cache=use_cache, **kwargs,
        )
        return outputs


class MossTranscribeDiarizeForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = MossTranscribeDiarizeConfig
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: MossTranscribeDiarizeConfig):
        super().__init__(config)
        self.model = MossTranscribeDiarizeModel(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def tie_weights(self, *args, **kwargs):
        result = super().tie_weights(*args, **kwargs)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight
        return result

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_features: Optional[torch.FloatTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        audio_chunk_mapping: Optional[torch.LongTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            input_features=input_features,
            audio_feature_lengths=audio_feature_lengths,
            audio_chunk_mapping=audio_chunk_mapping,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        return CausalLMOutputWithPast(
            loss=None, logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        input_features=None,
        audio_feature_lengths=None,
        audio_chunk_mapping=None,
        is_first_iteration=False,
        use_cache=True,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask, inputs_embeds=inputs_embeds,
            is_first_iteration=is_first_iteration, use_cache=use_cache, **kwargs,
        )
        if input_features is not None and (is_first_iteration or not use_cache):
            model_inputs["input_features"] = input_features
            model_inputs["audio_feature_lengths"] = audio_feature_lengths
            model_inputs["audio_chunk_mapping"] = audio_chunk_mapping
        return model_inputs

class _BaseAutoModelClass:
    _model_mapping = _LazyAutoMapping(CONFIG_MAPPING_NAMES, OrderedDict([]))
    def __init__(self, *args, **kwargs) -> None: pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], *model_args, **kwargs):
        kwargs["_from_auto"] = True
        adapter_kwargs = None

        kwargs["adapter_kwargs"] = adapter_kwargs
        
        return MossTranscribeDiarizeForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, *model_args, config=None, **kwargs)

from dataclasses import fields

class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __init_subclass__(cls) -> None:
        """Register subclasses as pytree nodes.

        This is necessary to synchronize gradients when using `torch.nn.parallel.DistributedDataParallel` with
        `static_graph=True` with modules that output `ModelOutput` subclasses.
        """
        _register_model_output_pytree_node(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _register_model_output_pytree_node(type(self))

        # Subclasses of ModelOutput must use the @dataclass decorator
        # This check is done in __init__ because the @dataclass decorator operates after __init_subclass__
        # issubclass() would return True for issubclass(ModelOutput, ModelOutput) when False is needed
        # Just need to check that the current class is not ModelOutput
        is_modeloutput_subclass = self.__class__ != ModelOutput

        if is_modeloutput_subclass and not is_dataclass(self):
            raise TypeError(
                f"{self.__module__}.{self.__class__.__name__} is not a dataclass."
                " This is a subclass of ModelOutput and so must use the @dataclass decorator."
            )

    def __post_init__(self):
        """Check the ModelOutput dataclass.

        Only occurs if @dataclass decorator has been used.
        """
        _register_model_output_pytree_node(type(self))
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(self.__dict__.get(field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                # reset first field to None and remove it from the internal dictionary
                setattr(self, class_fields[0].name, None)
                super().__delitem__(class_fields[0].name)
                for idx, element in enumerate(iterator):
                    if not isinstance(element, (list, tuple)) or len(element) != 2 or not isinstance(element[0], str):
                        if idx == 0:
                            # If we do not have an iterator of key/values, set it as attribute
                            self[class_fields[0].name] = first_field
                        else:
                            # If we have a mixed iterator, raise an error
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = self.__dict__.get(field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        field_names = {field.name for field in fields(self)}
        if name in field_names and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return callable, args, *remaining

    def to_tuple(self) -> tuple:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())

_registered_model_output_types: set[type[Any]] = set()
def _model_output_flatten(output: ModelOutput) -> tuple[list[Any], list[str]]:
    return list(output.values()), list(output.keys())
from functools import partial
from collections.abc import Callable, Iterable

def _model_output_unflatten(
    values: Iterable[Any],
    context: list[str],
    output_type: type[ModelOutput] | None = None,
) -> ModelOutput:
    return output_type(**dict(zip(context, values)))


def _register_model_output_pytree_node(output_type: type[ModelOutput]) -> None:
    import torch

    # AMD CI runs PyTorch 2.8.0+rocm which does not support tracing `set.__contains__`
    # through TorchDynamo. Skip registration during compilation since the pytree node
    # is already registered from the preceding eager run.
    if torch.compiler.is_compiling():
        return
    if output_type in _registered_model_output_types:
        return

    import torch.utils._pytree as torch_pytree

    torch_pytree.register_pytree_node(
        output_type,
        _model_output_flatten,
        partial(_model_output_unflatten, output_type=output_type),
        serialized_type_name=f"{output_type.__module__}.{output_type.__name__}",
        flatten_with_keys_fn=torch_pytree._dict_flatten_with_keys,
    )
    _registered_model_output_types.add(output_type)

@dataclass
class CausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Any = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None

# todo just use needed entry...

DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)
TokenCallback = Callable[[int], None]
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".flv", ".wmv"}

@dataclass(slots=True, frozen=True)
class TranscriptSegment:
    start: float
    end: float
    speaker: str
    text: str

def _is_timestamp_char(ch: str) -> bool: return ("0" <= ch <= "9") or ch == "."

def _is_speaker_char(ch: str) -> bool: return ch == "S" or ("0" <= ch <= "9")

def _parse_timestamp(chars: list[str]) -> float | None:
    if not chars:
        return None

    dot_count = 0
    digit_count = 0
    for ch in chars:
        if "0" <= ch <= "9":
            digit_count += 1
        elif ch == ".":
            dot_count += 1
            if dot_count > 1:
                return None
        else:
            return None

    if digit_count == 0:
        return None
    return float("".join(chars))

def _parse_speaker(chars: list[str]) -> str | None:
    if len(chars) < 2 or chars[0] != "S":
        return None
    for ch in chars[1:]:
        if not ("0" <= ch <= "9"):
            return None
    return "".join(chars)

class TranscriptStreamParser:
    """Streaming parser for compact MOSS transcript output.

    Expected segment format:

        [start][Sxx]text[end]

    The parser deliberately avoids regular expressions. It scans characters
    once, keeps only the active token/text buffers, and emits a segment after
    an end timestamp is confirmed by the next segment start or by ``close()``.
    """

    _SEEK_START = 0
    _READ_START = 1
    _EXPECT_SPEAKER_OPEN = 2
    _READ_SPEAKER = 3
    _READ_TEXT = 4
    _READ_END = 5
    _AFTER_END = 6

    def __init__(self, *, strip_text: bool = True, skip_empty: bool = True):
        self.strip_text = strip_text
        self.skip_empty = skip_empty
        self._state = self._SEEK_START
        self._token: list[str] = []
        self._text: list[str] = []
        self._pending_after_end: list[str] = []
        self._start: float | None = None
        self._end: float | None = None
        self._end_token = ""
        self._speaker: str | None = None

    def reset(self) -> None:
        self._state = self._SEEK_START
        self._token.clear()
        self._text.clear()
        self._pending_after_end.clear()
        self._start = None
        self._end = None
        self._end_token = ""
        self._speaker = None

    def feed(self, chunk: str) -> list[TranscriptSegment]:
        """Consume a text chunk and return any newly completed segments."""
        segments: list[TranscriptSegment] = []
        self.feed_into(chunk, segments.append)
        return segments

    def feed_into(self, chunk: str, emit: Callable[[TranscriptSegment], None]) -> None:
        """Consume a text chunk and send completed segments to ``emit``."""
        if not isinstance(chunk, str):
            raise TranscriptParseError(f"chunk must be str, got {type(chunk).__name__}")

        for ch in chunk:
            state = self._state
            if state == self._SEEK_START:
                self._seek_start(ch)
            elif state == self._READ_START:
                self._read_start(ch)
            elif state == self._EXPECT_SPEAKER_OPEN:
                self._expect_speaker_open(ch)
            elif state == self._READ_SPEAKER:
                self._read_speaker(ch)
            elif state == self._READ_TEXT:
                self._read_text(ch)
            elif state == self._READ_END:
                self._read_end(ch, emit)
            elif state == self._AFTER_END:
                self._after_end(ch, emit)

    def close(self) -> list[TranscriptSegment]:
        """Finish the stream and return a final segment if one is complete."""
        segments: list[TranscriptSegment] = []
        self.close_into(segments.append)
        return segments

    def close_into(self, emit: Callable[[TranscriptSegment], None]) -> None:
        """Finish the stream and send a final complete segment to ``emit``."""
        if self._state == self._AFTER_END:
            self._emit_segment(emit)
        self.reset()

    def _seek_start(self, ch: str) -> None:
        if ch == "[":
            self._token.clear()
            self._state = self._READ_START

    def _read_start(self, ch: str) -> None:
        if ch == "]":
            start = _parse_timestamp(self._token)
            if start is None:
                self.reset()
                return
            self._start = start
            self._state = self._EXPECT_SPEAKER_OPEN
            self._token.clear()
            return

        if _is_timestamp_char(ch):
            self._token.append(ch)
            if len(self._token) <= 32:
                return

        self.reset()
        if ch == "[":
            self._state = self._READ_START

    def _expect_speaker_open(self, ch: str) -> None:
        if ch == "[":
            self._token.clear()
            self._state = self._READ_SPEAKER
        elif not ch.isspace():
            self.reset()

    def _read_speaker(self, ch: str) -> None:
        if ch == "]":
            speaker = _parse_speaker(self._token)
            if speaker is None:
                self.reset()
                return
            self._speaker = speaker
            self._text.clear()
            self._state = self._READ_TEXT
            self._token.clear()
            return

        if _is_speaker_char(ch):
            self._token.append(ch)
            if len(self._token) <= 16:
                return

        self.reset()
        if ch == "[":
            self._state = self._READ_START

    def _read_text(self, ch: str) -> None:
        if ch == "[":
            self._token.clear()
            self._state = self._READ_END
        else:
            self._text.append(ch)

    def _read_end(self, ch: str, emit: Callable[[TranscriptSegment], None]) -> None:
        if ch == "]":
            end = _parse_timestamp(self._token)
            if end is not None and self._start is not None and end >= self._start:
                self._end = end
                self._end_token = "".join(self._token)
                self._pending_after_end.clear()
                self._state = self._AFTER_END
            else:
                self._text.append("[")
                self._text.extend(self._token)
                self._text.append("]")
                self._state = self._READ_TEXT
            self._token.clear()
            return

        if _is_timestamp_char(ch):
            self._token.append(ch)
            if len(self._token) <= 32:
                return

        self._text.append("[")
        self._text.extend(self._token)
        self._text.append(ch)
        self._token.clear()
        self._state = self._READ_TEXT

    def _after_end(self, ch: str, emit: Callable[[TranscriptSegment], None]) -> None:
        if ch == "[":
            self._emit_segment(emit)
            self._token.clear()
            self._state = self._READ_START
            return

        if ch.isspace():
            self._pending_after_end.append(ch)
            return

        self._text.append("[")
        self._text.append(self._end_token)
        self._text.append("]")
        self._text.extend(self._pending_after_end)
        self._text.append(ch)
        self._pending_after_end.clear()
        self._end = None
        self._end_token = ""
        self._state = self._READ_TEXT

    def _emit_segment(self, emit: Callable[[TranscriptSegment], None]) -> None:
        if self._start is None or self._end is None or self._speaker is None:
            self.reset()
            return

        text = "".join(self._text)
        if self.strip_text:
            text = text.strip()
        if text or not self.skip_empty:
            emit(
                TranscriptSegment(
                    start=self._start,
                    end=self._end,
                    speaker=self._speaker,
                    text=text,
                )
            )

        self._token.clear()
        self._text.clear()
        self._pending_after_end.clear()
        self._start = None
        self._end = None
        self._end_token = ""
        self._speaker = None
        self._state = self._SEEK_START

def build_transcription_messages(audio_path: str | Path, prompt: str = DEFAULT_PROMPT) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(audio_path)},
                {"type": "text", "text": prompt.strip() or DEFAULT_PROMPT},
            ],
        }
    ]

def _token_count(value) -> int:
    if hasattr(value, "numel"):
        return int(value.numel())
    if isinstance(value, (list, tuple)):
        return sum(_token_count(item) for item in value)
    return 1

class ProgressStreamer:
    """Count generated tokens from ``generate(streamer=...)`` without decoding text."""

    def __init__(self, callback: TokenCallback):
        self.callback = callback
        self.generated_tokens = 0
        self._seen_prompt = False

    def put(self, value):
        token_count = _token_count(value)
        if not self._seen_prompt:
            self._seen_prompt = True
            return
        self.generated_tokens += token_count
        self.callback(self.generated_tokens)

    def end(self):
        return None

def load_audio_av(audio: str, sampling_rate: int) -> np.ndarray:
    """Decode an audio stream from a media container with PyAV."""
    try:
        import av
    except ImportError as exc:
        raise ImportError("Install `av` to decode audio from video containers.") from exc

    chunks: list[np.ndarray] = []
    with av.open(audio) as container:
        stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if stream is None:
            raise ValueError(f"No audio stream found in {audio!r}.")

        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
        for frame in container.decode(stream):
            frames = resampler.resample(frame)
            if frames is None:
                continue
            if not isinstance(frames, list):
                frames = [frames]
            for resampled in frames:
                chunks.append(resampled.to_ndarray().reshape(-1))

        frames = resampler.resample(None)
        if frames is not None:
            if not isinstance(frames, list):
                frames = [frames]
            for resampled in frames:
                chunks.append(resampled.to_ndarray().reshape(-1))

    if not chunks:
        raise ValueError(f"No decodable audio samples found in {audio!r}.")
    return (np.concatenate(chunks).astype(np.float32) / 32768.0).astype(np.float32, copy=False)

def load_audio_item(audio: str | np.ndarray, sampling_rate: int) -> np.ndarray:
    return load_audio(audio, sampling_rate=sampling_rate)
    
def process_audio_info(messages: list[dict[str, Any]], sampling_rate: int):
    """Load audio items from chat messages in the same order as the template."""
    audios = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            continue
        for item in content:
            if item.get("type") != "audio":
                continue
            audio = item.get("audio") or item.get("audio_url") or item.get("url") or item.get("path")
            if audio is None:
                raise ValueError("Audio content must include audio, audio_url, url, or path.")
            audios.append(load_audio_item(audio, sampling_rate=sampling_rate))
    return audios

def prepare_inputs(processor, messages, *, max_length: int = 131072, device: torch.device | None = None):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios = process_audio_info(messages, sampling_rate=processor.feature_extractor.sampling_rate)
    audio_kwargs = {"device": str(device)} if device is not None and device.type == "cuda" else {}
    return processor(
        text=text,
        audio=audios,
        max_length=max_length,
        audio_kwargs=audio_kwargs,
        return_tensors="pt",
    )

def generate_transcription(
    model,
    processor,
    messages,
    *,
    max_length: int = 131072,
    max_new_tokens: int | None = None,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    input_callback: Callable[[int], None] | None = None,
    token_callback: TokenCallback | None = None,
) -> dict[str, Any]:
    device = device or next(model.parameters()).device
    dtype = dtype or next(model.parameters()).dtype
    context = (
        torch.amp.autocast("cuda", dtype=dtype)
        if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        else torch.no_grad()
    )
    with context:
        inputs = prepare_inputs(processor, messages, max_length=max_length, device=device).to(device)

    prompt_len = int(inputs["attention_mask"][0].sum().item())
    if input_callback is not None:
        input_callback(prompt_len)
    generation_config = copy.deepcopy(model.generation_config)
    if max_new_tokens is not None:
        generation_config.max_new_tokens = max_new_tokens
    generation_config.do_sample = do_sample
    if do_sample and temperature is not None:
        generation_config.temperature = temperature
    if do_sample and top_p is not None:
        generation_config.top_p = top_p
    if do_sample and top_k is not None:
        generation_config.top_k = top_k
    streamer = ProgressStreamer(token_callback) if token_callback is not None else None
    generate_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "input_features": inputs["input_features"],
        "audio_feature_lengths": inputs["audio_feature_lengths"],
        "audio_chunk_mapping": inputs["audio_chunk_mapping"],
        "generation_config": generation_config,
    }
    if streamer is not None:
        generate_kwargs["streamer"] = streamer

    with torch.inference_mode(), (
        torch.amp.autocast("cuda", dtype=dtype)
        if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        else torch.no_grad()
    ):
        try:
            outputs = model.generate(**generate_kwargs)
        except TypeError as exc:
            if streamer is None or "streamer" not in str(exc):
                raise
            generate_kwargs.pop("streamer", None)
            outputs = model.generate(**generate_kwargs)

    generated_ids = outputs[0][prompt_len:]
    text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "prompt_len": prompt_len,
        "generated_tokens": int(generated_ids.numel()),
    }

def parse_transcript(text: str, **parser_kwargs) -> list[TranscriptSegment]:
    parser = TranscriptStreamParser(**parser_kwargs)
    segments = parser.feed(text)
    segments.extend(parser.close())
    return segments

device = torch.device("cpu")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
model = _BaseAutoModelClass.from_pretrained("OpenMOSS-Team/MOSS-Transcribe-Diarize").to(dtype=dtype).to(device).eval()

model_id = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
audio_path = "MOSS/output.wav" # 10 mins for now

processor = AutoProcessor.from_pretrained(model_id)

messages = build_transcription_messages(audio_path)
result = generate_transcription(
    model,
    processor,
    messages,
    max_new_tokens=2048, # was 2048, this should reduce ram use?
    do_sample=False,
    device=device,
    dtype=dtype,
)

print("rory unparsed result =",result,"\n\n\n")

excepted = "[9.26][S01]哈喽大家好，我是小俊[10.73][11.02][S01]今天我们的嘉宾是Google DeepMind研究员姚舜宇[14.31][14.63][S01]硅谷有两个很有名的姚舜宇，一个之前在OpenAI跳槽去了腾讯出任腾讯的首席科学家[21.51][21.88][S01]他之前也来过我们节目[23.31][23.62][S01]那今天我邀请的是另一位姚舜宇，他此前在Anthropic，现在在Google DeepMind[29.55][29.88][S01]我们从近期一系列的模型巨变开始聊起[33.75][34.29][S01]那接下来就是我对舜宇的访谈[36.98][37.06][S02]Anthropic作为一个公司来说，它能够实行这种就是比较top down的机制，是一个很独特的事[43.27][43.61][S01]这对于其他模型公司很难吗[45.01][45.38][S02]很难，比如说OpenAI就干不了，就是Gemini也比较难，大公司和和startup它打法本来就不一样，因为startup重要的是make bet[56.47][56.84][S02]就是我得我得赌一件事[58.67][59.15][S02]是我觉得大家现在就是每个人都是冲浪的人，本质上是那个浪，而不是你那个冲浪的人，因为AI这个事本来也不太需要脑子[67.69][68.94][S01]不太需要脑子[69.91][69.74][S02]真的不太需要脑子[70.79][70.81][S01]需要什么[71.42][71.93][S02]我觉得这个这个行业就是最重要的特质就是靠谱，就是做事细，然后对自己做的事负责任，这是最重要的特质[81.92][87.89][S01]硅谷不是有两个姚舜宇吗？你要不要先给大家介绍一下你自己，然后给大家科普一下两个姚舜宇的区别[94.52][94.94][S02]啊可以，对，就是呃我叫姚舜宇，然后显然也有一个跟我呃几乎同名的朋友，然后呃我们俩主要履历也有一些overlap，所以说可能看起来非常的难以区分。对，然后[109.31][109.78][S02]呃我是我以前是做呃学物理的，然后我本科的时候在呃清华啊那时候做宁态理论，然后后来去斯坦福啊做呃理论高能物理，然后和量子信息黑洞相关的一些方面。[124.11][124.49][S02]然后呃离开斯坦福之后，去呃伯克利短暂的待了两个星期的postdoc过后，然后就离离职了，去了Anthropic，然后在Anthropic待了一年，[137.01][137.38][S02]啊去年九月底十月初的时候呃加入了Gemini。[141.01][141.47][S02]对，然后呃如果大家非要区分的话，我觉得最大的区分就是那个舜宇他一开始就是一直都是做CS，就是计算机相关的。然后我其实呃从某种意义上来说是个半道出家。对，就是我之前是做理论物理为主的。对。[157.41][157.88][S01]你们是不是好朋友？[159.11][159.25][S01]你们好像大学就认识，而且是一级的对吧？[161.58][161.75][S02]对[161.95][161.92][S01]他是一个什么样的人？你是一个什么样的人？你评价一下他，你也评价一下自己。[165.51][166.52][S02]对对对，我们本科就认识，因为我们本科是一级的，然后在清华，但他一开始就是学计算机的嘛，所以他在那个姚班就是计算机科学实验班，然后呃我是学物理，所以我在机科班。[175.69][175.98][S02]对，然后呃后来他去了普林，我去斯坦福，然后这可能也是另一个有点令人费解的点，就是好像这个普世世界里觉得斯坦福应该是学计算机的人该去的地方，然后觉得普林斯顿是学物理人该去的地方，但我俩然后反过来，[191.61][192.38][S02]所以说也可能产生了一些费解的事情。[194.89][194.89][S02]对，然后我俩其实也还真的挺不一样，我觉得他是一个比我有趣的多的人。[200.01][200.45][S02]我觉得我我从他身上也是在过去也是能学习到了一些和我很不一样的点。比如说他可能花了很多时间去思考，比如在AI方面，他花了很多时间去思考，就是人和AI的交互呀，然后包括一些产品上的事情。然后我觉得其实对我来说呃是一个很不一样的朋友，然后我也从他那学到了很多东西。[220.94][221.68][S01]你们之前在硅谷的时候多久见一次面？[223.51][223.51][S01]你们现在是不是还频繁打电话多频繁？[225.67][226.23][S02]呃，我们在硅谷的时候见面确实挺频繁的，可能每每几个星期吧，但是好像见面主要是为了凑一块玩。[236.34][236.98][S01]玩啥[237.45][238.04][S02]就是真的就是纯玩，就是可能出去散散步，扯扯有的没的，然后可能有时候吃个饭打个牌啊之类的。[247.11][247.11][S02]对，[247.38][247.38][S02]对，然后他回去之后，其实我们也也是也还是经常会打电话。[251.91][252.32][S01]最近一次电话聊啥了？好像就是前一两个星期[254.95][255.34][S02]啊你怎么知道的？呃，可能就是会过几个月，然后然后就catchup一下呃大家最近的近况吧。[264.24][264.24][S02]对。[264.55][265.01][S01]他是不是多次想把你拉过去？[266.71][267.24][S02]啊[268.55][270.93][S02]可能有这个意思吧，但是但是我觉得不关键不关键。[274.51][275.49][S01]你为什么不去？[276.12][276.12][S02]我觉得对我自己来说，我呃没想清楚吧。嗯，我觉得呃多半是我自己的原因，然后呃我也没有去任何[285.11][285.98][S02]呃中国的地方，然后我觉得主要原因是因为呃在去年的九月或者八九月这个时候，我觉得呃那时候我离开离开Anthropic，然后离开之后决定要去哪的时候，[298.99][298.99][S02]最大的动机是呃我想学一些不一样的东西。[302.51][303.21][S02]呃，对我来说我可能就没有去考虑，[306.01][306.01][S02]呃，没有没有更着重的去考虑说能够我去领导一个项目，或者领导一个project之类的。我更多的是是那个时候更多的是优先去学习一些东西，所以那时候选择去了Gemini。[318.01][319.57][S01]我发现你们两个老被放在一起比较和讨论，对你来说是困扰更多还是享受更多？[324.06][324.29][S02]啊，我没什么感觉，然后因因为我这个人也不太关注社交媒体，所以我其实真的没什么感觉。[333.61][334.65][S01]嗯[334.92][335.80][S01]因为那个舜宇他之前在去年的时候说AI进入了the second half，进入下半场，这个成为了一个非常有名的观点。你觉得今天的AI在一个什么样的时期？你能给它一个定义吗？[347.41"
assert result["text"] == excepted

changes = []
speakers = []
parsed =  parse_transcript(result["text"])
# get changes
speaker = -1 # init
for i in range(len(parsed)):
  if parsed[i].speaker != speaker:
    speaker = parsed[i].speaker
    changes.append(parsed[i].start)
    speakers.append(int(parsed[i].speaker.replace("S","")))

for segment in parsed:
    print(segment.start, segment.end, segment.speaker, segment.text)

print("rory changes =", changes)
print("rory speakers =",speakers)
