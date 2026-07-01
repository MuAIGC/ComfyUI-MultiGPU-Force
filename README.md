
# ComfyUI-MultiGPU-Force

<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI-0.26%2B-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/CUDA-Multi--GPU-green?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
</p>

**将单个大模型自动拆分到多张显卡，突破单卡显存瓶颈。**

> ⚠️ **仅限图像模型**：支持 Flux / SDXL / SD3 / Lumina2 等。  
> 视频模型（LTX / Wan / HunyuanVideo / CogVideoX / Mochi）在 `forward` 中存在大量跨层就地操作，`accelerate` 无法处理，会自动回退到单卡加载。

---

## 功能

| 特性 | 状态 |
|------|------|
| ✅ 图像模型多卡显存合并 | Flux / SDXL / SD3 / Lumina2 |
| ✅ 自动模型架构识别 | 无需手动配置 |
| ✅ 实时显存日志 | 每步打印各卡占用 |
| ✅ LoRA 多卡支持 | CPU 合并后重新分发 |
| ✅ 视频模型自动回退 | 检测到即单卡加载，不报错 |
| ✅ 中文 Tooltip | 鼠标悬停参数即见说明 |

---

## 安装

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/yourname/ComfyUI-MultiGPU-Force.git
cd ComfyUI-MultiGPU-Force
pip install -r requirements.txt
```

重启 ComfyUI。

**依赖**：`accelerate >= 0.30.0`

---

## 节点

### 1. Load Diffusion Model (MultiGPU)
替换原生 `UNETLoader`，加载时自动拆分模型到多卡。

```
┌─────────────────────────────┐
│ Load Diffusion Model (MultiGPU) │
│  unet_name: xxx.safetensors │
│  weight_dtype: bf16         │
│  memory_ratio: 0.45         │
└──────────────┬──────────────┘
               │ MODEL
               ▼
        [ModelSamplingFlux]
               │
               ▼
          [KSampler]
```

### 2. Load Checkpoint (MultiGPU)
替换原生 `CheckpointLoader`，仅拆分 UNet，CLIP / VAE 仍在 `cuda:0`。

```
┌─────────────────────────────┐
│ Load Checkpoint (MultiGPU)  │
│  ckpt_name: xxx.safetensors │
│  memory_ratio: 0.45         │
└──────┬────────┬──────┬─────┘
       │ MODEL  │ CLIP │ VAE
       ▼        ▼      ▼
   [KSampler] [TE] [VAEDecode]
```

### 3. Apply LoRA (MultiGPU)
替换原生 `Load LoRA`，在 CPU 上合并权重后重新分发到多卡。

```
[Load Diffusion Model (MultiGPU)] ──► [Apply LoRA (MultiGPU)] ──► [KSampler]
               │ MODEL                        │ MODEL
               ▼                              ▼
         [ModelSampling]                [ModelSampling]
```

---

## 参数

### `weight_dtype` — 权重精度

| 选项 | 显存 | 速度 | 建议 |
|------|------|------|------|
| `default` | 原始 | 原始 | 不确定时选 |
| **`bf16`** | **-50%** | 正常 | **推荐** |
| `fp16` | -50% | 更快 | 可能精度损失 |
| `fp8_e4m3fn` | -75% | 正常 | 需显卡支持 FP8 |
| `fp8_e5m2` | -75% | 正常 | 需显卡支持 FP8 |

### `memory_ratio` — 显存比例 / 强制拆分阈值

控制每卡可用于加载模型的显存比例。**值越小，越容易触发多卡拆分。**

| 值 | 场景 | 效果 |
|----|------|------|
| **`0.45`** | **推荐** | 24GB 双卡强制拆分 11GB+ 模型 |
| `0.30` | 保守 | 预留空间给 LoRA / ControlNet / 高分辨率 |
| `0.80` | 宽松 | 尽量单卡加载，仅显存不足时才拆分 |

**原理**：告诉 `accelerate` 每卡只有这么多显存可用，模型超限时自动拆分到其他卡。

---

## 日志示例

```text
[INFO] got prompt
[MultiGPU] 开始加载模型: z_image_turbo_bf16.safetensors
[MultiGPU] 模型架构: lumina2
[MultiGPU] 加载前  GPU-0: 已用 0.12GB | 预留 0.15GB | 空闲 23.16GB | 总计 24.00GB
[MultiGPU] 加载前  GPU-1: 已用 0.08GB | 预留 0.10GB | 空闲 23.90GB | 总计 24.00GB
[MultiGPU] 分发策略: {'transformer_blocks.0': 0, 'transformer_blocks.1': 0, ...}
[MultiGPU] 实际使用显卡: {0, 1}
[MultiGPU] 模型分发完成
[MultiGPU] 分发后  GPU-0: 已用 5.83GB | 预留 6.10GB | 空闲 17.90GB | 总计 24.00GB
[MultiGPU] 分发后  GPU-1: 已用 5.20GB | 预留 5.45GB | 空闲 18.55GB | 总计 24.00GB
[INFO] model weight dtype torch.bfloat16, manual cast: None
[INFO] model_type FLOW
  25%|██▌       | 2/8 [00:12<00:36,  6.0s/it]
```

---

## 视频模型回退

视频模型在 `forward` 中有大量跨层就地操作（如 `addcmul_`、`copy_`）和跨层共享状态，`accelerate` 的层间切分无法处理，会导致：

```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1!
```

插件会自动检测视频模型标记并回退到单卡：

```text
[MultiGPU] 检测到视频模型标记: ['AVTransformerBlock']
[MultiGPU] 视频模型不支持多卡分发，自动回退到单卡加载
```

---

## 注意事项

1. **LoRA 合并时模型会短暂回到 CPU**，确保系统内存 ≥ 模型大小 × 2
2. **多次叠加 LoRA 建议一次合并完**，不要链式调用多个 `Apply LoRA (MultiGPU)`
3. **采样速度不会提升**，甚至可能略降（PCIe 通信开销）。目的是**显存扩容**，不是加速
4. **不支持显存池化**（16GB + 24GB ≠ 40GB），`accelerate` 按层切分，不是均匀分配
5. **NVLink 对推理无帮助**，跨卡搬运走 PCIe，和 NVLink 无关

---

## License

MIT
```
