# nclassinference

用于 `xuanpluspro/nclassbenchmark` 已训练模型的独立语义分割推理与论文结果可视化。支持二分类和多分类，统一类别颜色、输入图片、结果目录和拼图方式，不包含训练代码。

模型按 checkpoint 的 `config` 恢复，严格加载权重；不重新下载 ImageNet 预训练文件，不创建优化器。训练完成的权重放在 `weights/`，也可直接引用其他仓库的 `best.pth`。

## 支持的模型

与源仓库 SMP 0.5.0 的十组配置一致（见 `experiments.json`）：

| 架构 | 编码器 |
| --- | --- |
| UNet、UNet++、MAnet、FPN、PSPNet、DeepLabV3+ | ResNet34 |
| UNet、DeepLabV3+ | MobileNetV2 |
| SegFormer | MiT-B0、MiT-B2 |

`plant-seg-lab` / `plant-seg-ablation` 的自定义结构不是 SMP 模型，不能仅改 `arch` 后加载。本仓库提供 `import_masks.py`，可以把它们输出的预测掩膜接入同一套拼图，见下文。这里不声称已实现这些自定义模型的独立权重加载器。

## 安装（GPUFree）

```bash
cd /root/gpufree-data
git clone https://github.com/xuanpluspro/nclassinference.git
cd nclassinference
```

保留镜像自带的 GPU 版 torch / torchvision。按你习惯逐个安装，其余依赖由 uv 解析：

```bash
uv pip install --python "$(command -v python)" numpy -i https://mirror.sjtu.edu.cn/pypi/web/simple
uv pip install --python "$(command -v python)" Pillow -i https://mirror.sjtu.edu.cn/pypi/web/simple
uv pip install --python "$(command -v python)" segmentation-models-pytorch==0.5.0 -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

如果原训练环境有版本记录，优先使用相同 torch、torchvision、SMP 版本。SMP 安装会解析其依赖，请先留意 uv 的变更清单。推理需要 CUDA；不会自动回退 CPU。拼图、导入掩膜及轻量测试不需要 CUDA，也不导入 torch。

## 先用一张图片检查

以下路径是示例：把权重路径改成你实际训练输出中的 `best.pth`。

```bash
python -u predict.py \
  --weights /root/gpufree-data/nclassbenchmark/outputs/unet_resnet34/best.pth \
  --input /root/gpufree-data/nclassbenchmark/data/GWFSSnclass/JPEGImages \
  --split /root/gpufree-data/nclassbenchmark/data/GWFSSnclass/ImageSets/test.txt \
  --output outputs/gwfss/unet_resnet34 \
  --mode resize --max-images 1 --trust-checkpoint
```

确认输出正常后，去掉 `--max-images 1` 并加上 `--resume`，处理其余图片。

`--trust-checkpoint` 仅用于自己训练、可信来源的文件：原训练 checkpoint 包含 NumPy RNG 等对象，可能需要 pickle 完整反序列化。默认采用 `weights_only=True`，失败时不会自动启用不受限加载。

模型配置完整时无需填写架构、编码器、类别数。若手头只有纯 state_dict，传 `--model-config /path/model.json`，至少包含：

```json
{
  "arch": "Unet",
  "encoder": "resnet34",
  "num_classes": 2,
  "class_names": ["background", "leaf"],
  "size": 512,
  "background_index": 0,
  "ignore_index": 255
}
```

配置不能与已有 checkpoint 的架构、编码器、类别数、类别名称冲突。不猜测自定义结构。

## 三种推理模式

| 参数 | 行为 | 适用场景 |
| --- | --- | --- |
| `--mode resize`（默认） | PIL 双线性缩放至训练尺寸；预测 logits 双线性映射回原尺寸后分类 | 训练预处理一致的对比展示，显存需求较低 |
| `--mode full` | 保留原尺寸；右侧/底部补到 32 倍数；裁去补边 | 整图跨域推理，较占显存 |
| `--mode tile --tile-size 512 --overlap 128` | 原分辨率切块；重叠区前景/各类 softmax 概率等权平均后 argmax | 大图细节保留；结果可能存在切块边界差异 |

三种模式均输出原图宽高的掩膜。默认每次一张、FP32；可主动加 `--amp` 降低显存。无缩图/切块/CPU 的自动兜底；显存不足会停止并保留已完成结果。归一化与训练一致：RGB、ImageNet mean/std。补边是归一化后的零值；不自动旋转 EXIF 方向。

不同模式改变输入尺度与上下文，不能当作等价的评估协议。`resize` 在原尺寸 logits 上分类，和原训练评估“先把 GT 最近邻缩放至 512、再算指标”的边界处理不同，本仓库指标不应直接覆盖源仓库测试表。

## 输出都在同一目录

以 `outputs/gwfss/unet_resnet34/` 为例：

```text
masks/图片名.png              原尺寸类别索引；0..C-1，非 RGB 染色图
colors/图片名_color.png       统一颜色的分割图
overlays/图片名_overlay.jpg   原图叠加图
per_image.csv                每图时间、显存峰值、各类别像素面积
metrics.json                 仅在提供 --gt 时生成
meta/classes.json            类别名称、索引与配色
meta/run.json                权重哈希、推理配置、完成记录
meta/console.log             日志
```

默认保存以上三类图，可用 `--no-save-color` 关闭彩色图。索引掩膜始终保存，供后续拼图及定量分析。图片名不带重复扩展名；子目录结构保留。相同路径主干的不同扩展名会报错，避免覆盖。

- `--resume`：核对权重 SHA256、配置和输入文件大小/修改时间，跳过输出完整的已完成图片。
- `--overwrite`：重新计算并替换同名输出，重建本次统计；保留其他无关文件。替换不会进回收站。
- 两者不能同时使用，不支持多个进程同时写同一个结果目录。
- `seconds` 是预处理、传输、前向、输出回传和分类耗时；不含图片读取、权重加载、保存图片。不是纯模型 forward FPS。
- `peak_gib` 是 PyTorch 本张峰值 allocated 显存，包含常驻模型，不等于 nvitop 中整个进程的显存。

## 有 GT 时计算指标

给上面的预测命令追加：

```bash
--gt /root/gpufree-data/nclassbenchmark/data/GWFSSnclass/SegmentationClass
```

GT 必须是原图同尺寸、单通道或 P 模式的索引 PNG；不接受 RGB 彩色掩膜，也不会自动推断颜色到类别的映射。忽略标签沿用 checkpoint 的 `ignore_index`，通常为 255。输出全局混淆矩阵、含背景 mIoU、去背景 mIoU、宏平均 Dice、Pixel Accuracy，以及各类 IoU/Dice/Precision/Recall。无并集的类别不参与 mIoU 平均，以 null 保存；无有效 GT 像素时指标为 null。

这里类别面积是图像像素面积，不是物理面积。没有 GT 的黑麦草图片只生成预测与面积统计，不生成准确率。

## 批量推理多个模型

复制 `experiments_inference.example.json`，填写实际权重路径。名称是输出子目录名；相对权重路径以实验 JSON 所在目录为基准。不同数据集/类别体系要分开运行。

```bash
python -u batch_predict.py \
  --experiments experiments_inference.example.json \
  --input /path/to/JPEGImages \
  --split /path/to/ImageSets/test.txt \
  --output outputs/comparison \
  --mode resize --trust-checkpoint
```

各模型依次在独立进程运行，退出后释放显存，不同时加载十个模型。第一次可加 `--max-images 1`；检查后去掉并加 `--resume`。某模型失败时停止，修复后可续跑。`experiments.json` 是源仓库模型清单，`experiments_inference.example.json` 才是带权重路径的批量推理配置。

## 生成论文对比图

```bash
python compare.py \
  --input /path/to/JPEGImages \
  --split /path/to/ImageSets/test.txt \
  --gt /path/to/SegmentationClass \
  --runs UNet=outputs/comparison/unet_resnet34 DeepLabV3Plus=outputs/comparison/deeplabv3plus_resnet34 SegFormer=outputs/comparison/segformer_b0 \
  --output outputs/figures \
  --view color --panel-size 512 --columns 4
```

每张图片生成一幅“Original、GT、各模型”的拼图，另存 `legend.png` 和 `comparison.json`。可省略 `--gt`，用 `--view overlay` 查看叠加效果；默认 PNG 便于论文排版，可改 `--format jpg` 节约空间。`--panel-size` 只影响展示大小，不改变保存的预测掩膜。类别颜色图采用最近邻显示缩放；原图保持宽高比，不拉伸。

`--crop X Y W H` 对所有面板裁同一处原图区域，适合比较细叶、边缘等局部细节。`--max-images 5` 可先看少量图。中文标题可用 `--font /path/to/ChineseFont.ttf`，默认字体主要适合英文标签。标题过长会提示调整。

拼图前检查类别名称、顺序、颜色以及掩膜尺寸；缺失预测会直接报错，不拿空白图代替。类别语义不同（如 GWFSS 的穗/茎/叶与 RiceSeg 的六类）不能直接拼成同一分类任务的比较。

## 接入自定义模型或消融结果

先在对应训练/推理仓库恢复正确的模型结构并导出预测掩膜，然后：

```bash
python import_masks.py \
  --input /path/to/JPEGImages \
  --masks /path/to/ablation_predicted_masks \
  --class-config outputs/comparison/unet_resnet34/meta/classes.json \
  --output outputs/comparison/ablation_without_module
```

输出可直接加到 `compare.py --runs WithoutModule=outputs/comparison/ablation_without_module`。导入只规范文件和颜色，不重新运行模型，也不把一个模型的类别语义转换成另一种。

`plant-seg-inference` 可通过 `--save-mask --no-largest-component` 导出未做最大连通域处理的预测。若其二分类掩膜是 0/255，导入时显式加 `--binary-255`；文件主干有 `_mask` 则再加 `--suffix _mask`，以实际文件名为准。多分类索引中的 255 通常代表 ignore，不得启用二分类转换。

公平展示时，请固定输入图片、类别含义、推理尺度、精度及后处理设置，并记录外部模型的设置。默认本仓库不做最大连通域、去小区域或形态学处理。训练集图片可用于展示，但不能替代独立测试集结果。

## 来源与验证范围

模型/预处理/配置依据：[nclassbenchmark](https://github.com/xuanpluspro/nclassbenchmark) 固定提交 `e9091677da7f5866566110c4dd29bcfd357f778c`。使用安装包 `segmentation-models-pytorch==0.5.0` 构建模型，没有复制其模型源码；相关依赖许可由原项目保留。两个 dataset config 和模型清单沿用源仓库；GWFSS 的数值类别含义仍需用户核实，不按论文表格顺序猜测。

14 项轻量测试覆盖文件名、索引掩膜、配色、GT 指标、切块覆盖、拼图和配置检查。未运行 CPU 训练，也未在本环境安装 torch 或进行真实 checkpoint/CUDA 推理；请先按上文跑一张图验证 GPU 环境与权重兼容性。

```bash
python -m unittest discover -s tests -v
```
