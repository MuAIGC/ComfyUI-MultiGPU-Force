import torch
import comfy
import comfy.model_management as mm
import comfy.model_base
import folder_paths
from accelerate import infer_auto_device_map, dispatch_model
import types
import gc

def log_gpu_memory(tag=""):
    """打印每张卡的显存占用情况"""
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        free = total - reserved
        print(f"[MultiGPU] {tag} GPU-{i}: 已用 {allocated:.2f}GB | 预留 {reserved:.2f}GB | 空闲 {free:.2f}GB | 总计 {total:.2f}GB")

def is_video_model(diffusion):
    """检测是否为视频模型（LTX/Wan/HunyuanVideo等），视频模型不支持多卡分发"""
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

def patch_patcher_for_multigpu(patcher):
    """阻止 ComfyUI 搬运模型，保持多卡 dispatch 状态"""
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

def get_no_split_modules(diffusion):
    """根据模型内部模块自动推断 no_split 类名"""
    module_names = [m.__class__.__name__ for m in diffusion.modules()]
    unique = set(module_names)

    if "DoubleStreamBlock" in unique or "SingleStreamBlock" in unique:
        return "flux", ["DoubleStreamBlock", "SingleStreamBlock", "LastLayer", "EmbedND"]
    elif "JointTransformerBlock" in unique:
        return "sd3", ["JointTransformerBlock", "MMDiTBlock"]
    elif "SpatialTransformer" in unique or "CrossAttention" in unique:
        return "sdxl", ["SpatialTransformer", "ResnetBlock", "TimestepBlock", "AttentionBlock"]
    elif "ResBlock" in unique or "ResnetBlock" in unique:
        return "sd15", ["ResnetBlock", "SpatialTransformer", "TimestepBlock"]
    else:
        return "lumina2", ["TransformerBlock", "Attention", "RMSNorm"]

def dispatch_diffusion_model(model_patcher, memory_ratio=0.45):
    """把 ModelPatcher 内部的 diffusion_model dispatch 到多卡"""
    if torch.cuda.device_count() < 2:
        print(f"[MultiGPU] 仅检测到 {torch.cuda.device_count()} 张卡，跳过多卡分发")
        return model_patcher

    diffusion = getattr(model_patcher.model, "diffusion_model", None)
    if diffusion is None:
        print("[MultiGPU] 未找到 diffusion_model，跳过")
        return model_patcher

    # 视频模型直接跳过，不支持
    if is_video_model(diffusion):
        print("[MultiGPU] 视频模型不支持多卡分发，自动回退到单卡加载")
        return model_patcher

    family, no_split = get_no_split_modules(diffusion)
    print(f"[MultiGPU] 模型架构: {family}")
    log_gpu_memory("加载前 ")

    # 关键：降低每卡可用显存，强制 accelerate 切分
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory
        max_memory[i] = int(total * memory_ratio)
    max_memory["cpu"] = 200 * 1024**3

    try:
        diffusion.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

        inferred = infer_auto_device_map(
            diffusion,
            max_memory=max_memory,
            no_split_module_classes=no_split
        )
        devices_used = set(v for v in inferred.values() if isinstance(v, int))
        print(f"[MultiGPU] 分发策略: {inferred}")
        print(f"[MultiGPU] 实际使用显卡: {devices_used}")

        if len(devices_used) < 2:
            print(f"[MultiGPU] 警告: accelerate 仍将模型放在单卡上，建议降低 memory_ratio（当前 {memory_ratio}）")

        dispatch_model(diffusion, device_map=inferred)
        print("[MultiGPU] 模型分发完成")
        log_gpu_memory("分发后 ")

    except Exception as e:
        print(f"[MultiGPU] 分发失败: {e}")
        return model_patcher

    model_patcher._multigpu_dispatched = True
    patch_patcher_for_multigpu(model_patcher)
    return model_patcher

class UNETLoaderMultiGPU:
    """
    多卡显存合并加载器 —— 扩散模型加载节点。
    将单个大模型自动拆分到多张显卡上运行，解决单卡显存不足问题。
    支持 Flux、SDXL、SD3、Lumina2 等图像模型。
    不支持视频模型（LTX/Wan/HunyuanVideo/CogVideoX 等），会自动回退到单卡。
    使用方式：替换原生的 UNETLoader，后续节点正常连接即可。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), {
                    "tooltip": "选择要加载的扩散模型（UNet/DiT）。支持 Flux/SDXL/SD3/Lumina2 等图像模型。视频模型会自动回退单卡。"
                }),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2", "fp16", "bf16"], {
                    "tooltip": "模型权重精度。\n• default: 跟随模型原始精度\n• bf16: 推荐，显存省一半，速度正常\n• fp16: 更快但可能精度损失\n• fp8: 显存最小，需显卡支持"
                }),
                "memory_ratio": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.1,
                    "max": 0.95,
                    "step": 0.05,
                    "tooltip": "每块显卡可用于加载模型的显存比例（强制拆分阈值）。\n\n推荐值：\n• 0.45（推荐）: 24GB双卡强制拆分11GB+模型\n• 0.30（保守）: 显存紧张，需预留空间给LoRA/ControlNet\n• 0.80（宽松）: 尽量单卡加载，仅显存不足时才拆分\n\n原理：告诉 accelerate 每卡只有这么多显存可用，模型超限时自动拆分到其他卡。"
                }),
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

            print(f"[MultiGPU] 开始加载模型: {unet_name}")
            model = comfy.sd.load_diffusion_model(ckpt_path, model_options=model_options)
        finally:
            mm.get_torch_device = original_device

        return (dispatch_diffusion_model(model, memory_ratio),)

class CheckpointLoaderMultiGPU:
    """
    多卡显存合并加载器 —— 完整检查点加载节点。
    将 Checkpoint 中的 UNet 部分自动拆分到多卡，CLIP 和 VAE 仍放在 cuda:0。
    不支持视频模型，会自动回退到单卡。
    使用方式：替换原生的 CheckpointLoader，后续节点正常连接。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {
                    "tooltip": "选择要加载的完整检查点（含 UNet+CLIP+VAE）。视频模型会自动回退单卡。"
                }),
                "memory_ratio": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.1,
                    "max": 0.95,
                    "step": 0.05,
                    "tooltip": "每块显卡可用于加载模型的显存比例（强制拆分阈值）。\n\n推荐值：\n• 0.45（推荐）: 24GB双卡强制拆分11GB+模型\n• 0.30（保守）: 显存紧张，需预留空间给LoRA/ControlNet\n• 0.80（宽松）: 尽量单卡加载，仅显存不足时才拆分"
                }),
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
        return (dispatch_diffusion_model(model_patcher, memory_ratio), clip, vae)

class ApplyLoRAMultiGPU:
    """
    多卡 LoRA 加载节点。
    在 CPU 上合并 LoRA 权重后重新分发到多卡。
    注意：合并过程中模型会短暂回到 CPU，确保系统内存充足。
    使用方式：接在 UNETLoaderMultiGPU 之后，输出 MODEL 接 KSampler。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "输入来自 MultiGPU 加载器的 MODEL"
                }),
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "选择要加载的 LoRA 文件"
                }),
                "strength_model": ("FLOAT", {
                    "default": 1.0,
                    "min": -20.0,
                    "max": 20.0,
                    "step": 0.01,
                    "tooltip": "LoRA 对模型的影响强度。1.0=完整效果，0=无效果，负值=反向"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_lora"
    CATEGORY = "loaders/MultiGPU"

    def apply_lora(self, model, lora_name, strength_model):
        if strength_model == 0:
            return (model,)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        diffusion = getattr(model.model, "diffusion_model", None)
        if diffusion is not None and getattr(model, '_multigpu_dispatched', False):
            print("[MultiGPU] 收回模型到 CPU 准备 LoRA 合并...")
            for name, param in diffusion.named_parameters():
                param.data = param.data.to("cpu")
            for name, buf in diffusion.named_buffers():
                buf.data = buf.data.to("cpu")
            torch.cuda.empty_cache()
            log_gpu_memory("收回后 ")

        original_device = mm.get_torch_device
        mm.get_torch_device = lambda: torch.device("cpu")
        try:
            model.patch_model(device_to=torch.device("cpu"))
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map={})
            loaded_lora = comfy.lora.load_lora(lora, key_map)
            for key in loaded_lora:
                model.add_patches({key: loaded_lora[key]}, strength_model)
            print(f"[MultiGPU] LoRA 合并完成: {lora_name} @ {strength_model}")
        finally:
            mm.get_torch_device = original_device

        if getattr(model, '_multigpu_dispatched', False):
            model = dispatch_diffusion_model(model, memory_ratio=0.45)

        return (model,)

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
