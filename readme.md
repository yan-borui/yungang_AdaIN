# 云冈风格迁移项目

一个基于AdaIN（自适应实例归一化）的图像风格迁移系统，专门用于将云冈石窟的艺术风格应用于任意图像。

## 📋 项目概述

本项目使用深度学习技术，通过AdaIN算法实现实时风格迁移，可以将云冈石窟独特的艺术风格应用到用户上传的任何图片上。项目提供了Web界面、命令行工具和Python API三种使用方式。

## ✨ 主要特性

- **实时风格迁移**：基于AdaIN算法，实现快速风格转换
- **多种使用方式**：支持Web界面、命令行和Python API
- **强度可调**：支持0.0-1.0的风格强度调整
- **批量处理**：支持对目录中的所有图片进行批量风格化
- **模型训练**：提供完整的训练脚本，支持自定义风格训练
- **响应式Web界面**：直观易用的Web应用界面

## 📁 项目结构

```
yungang_AdaIN/
├── app.py                    # Flask Web应用
├── net.py                    # 网络模型定义
├── yungang_adain.py          # 云冈风格迁移类
├── function.py               # AdaIN核心算法函数
├── train.py                  # 模型训练脚本
├── sampler.py                # 数据采样器
├── apply_yungang.py          # 命令行工具
├── run.py                    # 一键启动脚本
├── requirements.txt          # Python依赖包
├── models/                   # 模型文件目录
│   ├── vgg_normalised.pth    # VGG预训练模型
│   └── yungang_decoder.pth   # 云冈风格解码器
├── yungang_styles/           # 云冈风格图片目录
├── static/                   # Web静态资源
│   └── uploads/              # 上传文件目录
└── templates/                # Web模板目录
    └── index.html            # 主页面模板
```

## 🚀 快速开始

### 环境要求

- Python 3.7+
- PyTorch 1.9.0+
- CUDA 11.0+（如果使用GPU）
- **Git LFS**（用于管理模型文件）

### 安装步骤

1. **克隆项目**
```bash
git lfs install  # 确保已安装 Git LFS
git clone https://github.com/yan-borui/yungang_AdaIN
cd yungang_AdaIN
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **准备风格图片**
   - 将云冈石窟风格图片放入 `yungang_styles/` 目录

4. **创建必要目录**
```bash
mkdir -p models yungang_styles static/uploads
```

## 🎮 使用方法

### 方式一：Web应用（推荐）

1. **一键启动**
```bash
python run.py
```

2. 启动后，在浏览器中访问：`http://localhost:5000`（事实上会自动打开）

3. **使用步骤**：
   - 上传要处理的图片
   - 选择云冈风格（可预览）
   - 调整风格强度（0.0-1.0）
   - 点击"应用风格"按钮
   - 下载处理后的图片

### 方式二：命令行工具

```bash
# 处理单张图片
python apply_yungang.py --input input.jpg --output output.jpg --alpha 0.8

# 批量处理目录中的所有图片
python apply_yungang.py --input images/ --output results/ --batch --alpha 0.8

# 指定特定风格图片
python apply_yungang.py --input input.jpg --output output.jpg --style_index 0
```

### 方式三：Python API

```python
from yungang_adain import YungangStyleTransfer

# 初始化
yungang = YungangStyleTransfer(
    style_dir="yungang_styles",
    model_path="models/yungang_decoder.pth"
)

# 应用风格
result = yungang.apply_filter(
    content_path="input.jpg",
    output_path="output.jpg",
    alpha=0.8,          # 风格强度
    random_style=True   # 随机选择风格
)

# 批量处理
processed = yungang.batch_process(
    content_dir="input_images/",
    output_dir="output_images/",
    alpha=0.8
)
```

## 🏋️ 模型训练

### 准备数据

1. **内容图片**：包含各种场景的通用图片
2. **风格图片**：云冈石窟相关图片

### 开始训练

```bash
# 训练云冈风格解码器
python train.py \
    --content_dir path/to/content_images \
    --style_dir yungang_styles \
    --save_dir experiments \
    --log_dir logs \
    --max_iter 160000 \
    --batch_size 8 \
    --style_weight 10.0 \
    --content_weight 1.0
```

### 训练参数说明

- `--content_dir`: 内容图片目录
- `--style_dir`: 风格图片目录
- `--save_dir`: 模型保存目录
- `--log_dir`: 日志保存目录
- `--max_iter`: 最大训练迭代次数
- `--batch_size`: 批次大小
- `--style_weight`: 风格损失权重
- `--content_weight`: 内容损失权重

## ⚙️ 配置选项

### Web应用配置（app.py）
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 上传文件大小限制
app.config['UPLOAD_FOLDER'] = 'static/uploads'       # 上传文件目录
app.config['MODEL_PATH'] = 'models/yungang_decoder.pth'  # 模型路径
app.config['STYLE_DIR'] = 'yungang_styles'           # 风格图片目录
```

### 风格迁移配置（YungangStyleTransfer）
```python
__init__(
    style_dir="yungang_styles",   # 风格图片目录
    device=None,                  # 计算设备（自动选择）
    model_path="models/yungang_decoder.pth"  # 模型路径
)
```

## 📊 技术细节

### 核心算法
- **AdaIN (Adaptive Instance Normalization)**: 自适应实例归一化，在不改变内容结构的前提下匹配风格统计特征
- **VGG-19编码器**: 用于提取图像深度特征
- **轻量解码器**: 将AdaIN后的特征解码回图像空间

### 损失函数
- **内容损失**: MSE损失，保持内容结构
- **风格损失**: 特征统计匹配损失（均值和方差）

### 性能优化
- 支持GPU加速
- 内存高效处理
- 批量处理支持

## 🐛 常见问题

### Q1: 运行报错 "ModuleNotFoundError"
**A**: 确保已安装所有依赖：
```bash
pip install -r requirements.txt
```

### Q2: 风格化效果不理想
**A**: 尝试：
1. 调整alpha参数（0.6-0.9通常效果最佳）
2. 准备更多样化的风格图片
3. 重新训练模型

### Q3: 处理速度慢
**A**:
- 确保使用GPU（如有）
- 减小输入图片尺寸
- 关闭其他占用GPU的程序

## 🔧 故障排除

### 内存不足
- 减小批次大小
- 降低输入分辨率
- 使用CPU模式

### 模型加载失败
```python
# 检查模型文件路径
print(os.path.exists('models/vgg_normalised.pth'))
print(os.path.exists('models/yungang_decoder.pth'))
```

### Web应用无法启动
```bash
# 检查端口占用
netstat -ano | findstr :5000

# 更改端口
app.run(host='0.0.0.0', port=5001, debug=True)
```

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📝 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢 [pytorch-AdaIN原作者](https://github.com/naoto0804/pytorch-AdaIN)的工作
- 感谢云冈石窟提供艺术灵感

---

**注意**：本项目仅用于学习和研究目的，商业使用请确保遵守相关法律法规和版权要求。
