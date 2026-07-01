import torch
import comfy
import comfy.model_management as mm
import folder_paths
import types


def log_gpu_memory(tag=""):
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        free = total - reserved
        print(f"[MultiGPU] {tag} GPU-{i}: 已用 {allocated:.2f}GB | 预留 {reserved:.2f}GB | 空闲 {free:.2f}GB | 总计 {total:.2f}GB")


def is_video_model(diffusion):
    module_names = [m.__class__.__name__ for m in diffusion.modules()]
    unique = set(module_names)
    video_markers = [
        "AVTransformerBlock", "AVBlock", "LTXTransformerBlock",
        "WanTransformerBlock", "WanBlock", "WanAttentionBlock",
        "HunyuanVideoBlock", "HYVideoBlock", "VideoBlock",
        "TemporalBlock", "SpaceTimeBlock", "TemporalAttentionBlock",
        "CogVideoXBlock", "CogVideoBlock", "MochiBlock",
    ]
    matched = [m for m in video_markers if m in unique]
    if matched:
        print(f"[MultiGPU] 检测到视频模型标记: {matched}")
        return True
    return False


def move_tensors_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_tensors_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        moved = [move_tensors_to_device(v, device) for v in obj]
        return type(obj)(moved)
    return obj


def patch_patcher_for_multigpu(patcher):
    patcher.model_loaded = types.MethodType(lambda self: True, patcher)
    patcher.loaded_size = types.MethodType(lambda self: 0, patcher)

    def load_wrapper(self, device_to=None, *args, **kwargs):
        return self.model
    patcher.load = types.MethodType(load_wrapper, patcher)

    def unload_wrapper(self, *args, **kwargs):
        return self.model
    patcher.unload_model = types.MethodType(unload_wrapper, patcher)

    original_patch = patcher.patch_model
    def patch_wrapper(self, device_to=None, *args, **kwargs):
        if getattr(self, '_multigpu_dispatched', False):
            return original_patch(device_to=torch.device("cpu"), *args, **kwargs)
        return original_patch(device_to, *args, **kwargs)
    patcher.patch_model = types.MethodType(patch_wrapper, patcher)

    original_unpatch = patcher.unpatch_model
    def unpatch_wrapper(self, device_to=None, *args, **kwargs):
        if getattr(self, '_multigpu_dispatched', False):
            return original_unpatch(device_to=torch.device("cpu"), *args, **kwargs)
        return original_unpatch(device_to, *args, **kwargs)
    patcher.unpatch_model = types.MethodType(unpatch_wrapper, patcher)

    return patcher


def dispatch_lumina2(diffusion, memory_ratio):
    blocks = getattr(diffusion, 'transformer_blocks', None)
    if blocks is None:
        print("[MultiGPU] 未找到 transformer_blocks，无法分发")
        return False

    n = len(blocks)
    split = max(1, n // 2)

    for i in range(split):
        blocks[i].to("cuda:0")
    for i in range(split, n):
        blocks[i].to("cuda:1")

    if hasattr(diffusion, 'final_layer'):
        diffusion.final_layer.to("cuda:1")

    original_forward = diffusion.forward
    diffusion._original_forward = original_forward

    def forward(x, t, context=None, **kwargs):
        x = x.to("cuda:0")
        if isinstance(t, torch.Tensor):
            t = t.to("cuda:0")
        ctx = context.to("cuda:0") if context is not None else None

        x = diffusion.x_embedder(x)
        t_emb = diffusion.t_embedder(t)
        if ctx is not None and hasattr(diffusion, 'context_embedder'):
            ctx = diffusion.context_embedder(ctx)

        for i in range(split):
            x = blocks[i](x, t_emb, ctx, **kwargs)

        x = x.to("cuda:1")
        t_emb = t_emb.to("cuda:1")
        if ctx is not None:
            ctx = ctx.to("cuda:1")
        kwargs = move_tensors_to_device(kwargs, "cuda:1")

        for i in range(split, n):
            x = blocks[i](x, t_emb, ctx, **kwargs)

        x = diffusion.final_layer(x, t_emb)
        x = x.to("cuda:0")
        return x

    diffusion.forward = forward
    print(f"[MultiGPU] Lumina2: {split}/{n} blocks → cuda:0, {n-split}/{n} blocks → cuda:1")
    return True


def dispatch_model_manual(model_patcher, memory_ratio=0.45):
    if torch.cuda.device_count() < 2:
        print(f"[MultiGPU] 仅 {torch.cuda.device_count()} 张卡，跳过")
        return model_patcher

    diffusion = getattr(model_patcher.model, "diffusion_model", None)
    if diffusion is None:
        print("[MultiGPU] 未找到 diffusion_model")
        return model_patcher

    if is_video_model(diffusion):
        print("[MultiGPU] 视频模型不支持，回退单卡")
        return model_patcher

    print("[MultiGPU] 加载模型到 cuda:0 并注入 manual_cast...")
    model_patcher.load(device_to=torch.device("cuda:0"))
    log_gpu_memory("加载后 ")

    blocks = getattr(diffusion, 'transformer_blocks', None)
    has_double = hasattr(diffusion, 'double_blocks')
    has_single = hasattr(diffusion, 'single_blocks')

    dispatched = False

    if blocks is not None and len(blocks) > 0 and not has_double and not has_single:
        print("[MultiGPU] 识别为 Lumina2/PixArt 风格单流架构")
        dispatched = dispatch_lumina2(diffusion, memory_ratio)
    elif has_double or has_single:
        print("[MultiGPU] Flux 双流架构暂不支持多卡，回退单卡")
    else:
        print("[MultiGPU] 未识别可分发架构，回退单卡")

    if dispatched:
        model_patcher._multigpu_dispatched = True
        patch_patcher_for_multigpu(model_patcher)
        log_gpu_memory("分发后 ")

    return model_patcher


class UNETLoaderMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "选择扩散模型"}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2", "fp16", "bf16"], {"tooltip": "bf16推荐"}),
                "memory_ratio": ("FLOAT", {"default": 0.45, "min": 0.1, "max": 0.95, "step": 0.05, "tooltip": "显存比例，越小拆分越激进"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders/MultiGPU"

    def load_unet(self, unet_name, weight_dtype, memory_ratio):
        ckpt_path = folder_paths.get_full_path("diffusion_models", unet_name)

        original_device = mm.get_torch_device
        mm.get_torch_device = lambda: torch.device("cpu")

        try:
            model_options = {}
            if weight_dtype != "default":
                dtype_map = {
                    "fp8_e4m3fn": torch.float8_e4m3fn,
                    "fp8_e5m2": torch.float8_e5m2,
                    "fp16": torch.float16,
                    "bf16": torch.bfloat16,
                }
                if weight_dtype in dtype_map:
                    model_options["dtype"] = dtype_map[weight_dtype]

            print(f"[MultiGPU] 开始加载: {unet_name}")
            model = comfy.sd.load_diffusion_model(ckpt_path, model_options=model_options)
        finally:
            mm.get_torch_device = original_device

        return (dispatch_model_manual(model, memory_ratio),)


class CheckpointLoaderMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "选择检查点"}),
                "memory_ratio": ("FLOAT", {"default": 0.45, "min": 0.1, "max": 0.95, "step": 0.05, "tooltip": "显存比例"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders/MultiGPU"

    def load_checkpoint(self, ckpt_name, memory_ratio):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        original_device = mm.get_torch_device
        mm.get_torch_device = lambda: torch.device("cpu")

        try:
            print(f"[MultiGPU] 开始加载检查点: {ckpt_name}")
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )
        finally:
            mm.get_torch_device = original_device

        model_patcher, clip, vae = out[:3]
        return (dispatch_model_manual(model_patcher, memory_ratio), clip, vae)


class ApplyLoRAMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "输入MODEL"}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "选择LoRA"}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01, "tooltip": "强度"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_lora"
    CATEGORY = "loaders/MultiGPU"

    def apply_lora(self, model, lora_name, strength_model):
        if strength_model == 0:
            return (model,)

        if getattr(model, '_multigpu_dispatched', False):
            diffusion = model.model.diffusion_model
            if hasattr(diffusion, '_original_forward'):
                diffusion.forward = diffusion._original_forward
            diffusion.to("cpu")
            torch.cuda.empty_cache()
            print("[MultiGPU] 模型已收回 CPU")
            model._multigpu_dispatched = False

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        original_device = mm.get_torch_device
        mm.get_torch_device = lambda: torch.device("cpu")
        try:
            model.patch_model(device_to=torch.device("cpu"))
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map={})
            loaded_lora = comfy.lora.load_lora(lora, key_map)
            for key in loaded_lora:
                model.add_patches({key: loaded_lora[key]}, strength_model)
            print(f"[MultiGPU] LoRA 合并: {lora_name} @ {strength_model}")
        finally:
            mm.get_torch_device = original_device

        return (dispatch_model_manual(model, 0.45),)


NODE_CLASS_MAPPINGS = {
    "UNETLoaderMultiGPU": UNETLoaderMultiGPU,
    "CheckpointLoaderMultiGPU": CheckpointLoaderMultiGPU,
    "ApplyLoRAMultiGPU": ApplyLoRAMultiGPU,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UNETLoaderMultiGPU": "Load Diffusion Model (MultiGPU)",
    "CheckpointLoaderMultiGPU": "Load Checkpoint (MultiGPU)",
    "ApplyLoRAMultiGPU": "Apply LoRA (MultiGPU)",
}
