ComfyUI-MultiGPU-Force
ComfyUI 多卡显存合并插件。将单个大模型自动拆分到多张显卡上运行，解决单卡显存不足问题。
> ⚠️ **仅限图像模型**：支持 Flux / SDXL / SD3 / Lumina2 等。视频模型（LTX / Wan / HunyuanVideo / CogVideoX 等）会自动回退到单卡加载。
---
安装
```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/yourname/ComfyUI-MultiGPU-Force.git
cd ComfyUI-MultiGPU-Force
pip install -r requirements.txt
```
重启 ComfyUI。
---
节点列表
节点	说明
Load Diffusion Model (MultiGPU)	替换原生 `UNETLoader`，加载时自动拆分模型到多卡
Load Checkpoint (MultiGPU)	替换原生 `CheckpointLoader`，仅拆分 UNet，CLIP/VAE 仍在 cuda:0
Apply LoRA (MultiGPU)	替换原生 `Load LoRA`，在 CPU 上合并后重新分发到多卡
---
使用方式
基础用法（UNET 分离加载）
```
[Load Diffusion Model (MultiGPU)] → [ModelSamplingFlux/ModelSamplingLumina2] → [KSampler]
         ↓ MODEL
[CLIPLoader] → [CLIPTextEncode] → [KSampler]
         ↓ CLIP
[VAELoader] → [VAEDecode] → [SaveImage]
         ↓ VAE
```
完整检查点加载
```
[Load Checkpoint (MultiGPU)] → [ModelSamplingFlux] → [KSampler]
         ↓ MODEL    ↓ CLIP    ↓ VAE
```
带 LoRA
```
[Load Diffusion Model (MultiGPU)] → [Apply LoRA (MultiGPU)] → [KSampler]
```
---
参数说明
weight_dtype（权重精度）
选项	说明
`default`	跟随模型原始精度
`bf16`	推荐，显存省一半，速度正常
`fp16`	更快但可能精度损失
`fp8_e4m3fn`	显存最小，需显卡支持 FP8
`fp8_e5m2`	同上，另一种 FP8 格式
memory_ratio（显存比例 / 强制拆分阈值）
控制每块显卡可用于加载模型的显存比例。值越小，越容易触发多卡拆分。
值	场景	说明
0.45	推荐	24GB 双卡强制拆分 11GB+ 模型
0.30	保守	显存紧张，需预留空间给 LoRA / ControlNet / 高分辨率
0.80	宽松	尽量单卡加载，仅显存不足时才拆分
原理：告诉 `accelerate` 每卡只有这么多显存可用，模型超限时自动拆分到其他卡。
---
日志示例
```
[MultiGPU] 开始加载模型: z_image_turbo_bf16.safetensors
[MultiGPU] 模型架构: lumina2
[MultiGPU] 加载前  GPU-0: 已用 0.12GB | 预留 0.15GB | 空闲 23.16GB | 总计 24.00GB
[MultiGPU] 加载前  GPU-1: 已用 0.08GB | 预留 0.10GB | 空闲 23.90GB | 总计 24.00GB
[MultiGPU] 分发策略: {'transformer_blocks.0': 0, 'transformer_blocks.1': 0, ...}
[MultiGPU] 实际使用显卡: {0, 1}
[MultiGPU] 模型分发完成
[MultiGPU] 分发后  GPU-0: 已用 5.83GB | 预留 6.10GB | 空闲 17.90GB | 总计 24.00GB
[MultiGPU] 分发后  GPU-1: 已用 5.20GB | 预留 5.45GB | 空闲 18.55GB | 总计 24.00GB
```
---
视频模型回退
视频模型（LTX / Wan / HunyuanVideo / CogVideoX / Mochi）在 `forward` 中有大量跨层就地操作和跨层共享状态，`accelerate` 的自动切分无法处理，会导致 `RuntimeError: Expected all tensors to be on the same device`。插件会自动检测并回退到单卡加载：
```
[MultiGPU] 检测到视频模型标记: ['AVTransformerBlock']
[MultiGPU] 视频模型不支持多卡分发，自动回退到单卡加载
```
---
注意事项
LoRA 合并时模型会短暂回到 CPU，确保系统内存 ≥ 模型大小 × 2
多次叠加 LoRA 建议一次合并完，不要链式调用多个 `Apply LoRA (MultiGPU)`
采样速度不会提升，甚至可能略降（PCIe 通信开销）。目的是显存扩容，不是加速
不支持显存池化（16GB + 24GB ≠ 40GB），accelerate 按层切分，不是均匀分配
NVLink 对推理无帮助，accelerate 的跨卡搬运走 PCIe，和 NVLink 无关
---
依赖
`accelerate >= 0.30.0`
ComfyUI >= 0.26.0
PyTorch >= 2.0.0
多卡环境（`torch.cuda.device_count() >= 2`）
---
License
MIT
