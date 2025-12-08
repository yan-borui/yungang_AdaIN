# -*- coding: utf-8 -*-
import os
import io
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# 导入项目中的模块
from net import vgg, decoder as net_decoder

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_PATH'] = 'models/yungang_decoder.pth'
app.config['STYLE_DIR'] = 'yungang_styles'

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs(app.config['STYLE_DIR'], exist_ok=True)

# 全局变量存储模型
device = None
vgg_model = None
decoder_model = None
style_images = []


def load_models():
    """加载模型"""
    global device, vgg_model, decoder_model, style_images

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    try:
        # 加载VGG模型
        vgg_model = vgg
        vgg_model.load_state_dict(torch.load('models/vgg_normalised.pth', map_location=device))

        # 加载Decoder模型
        decoder_model = net_decoder
        decoder_model.load_state_dict(torch.load(app.config['MODEL_PATH'], map_location=device))

        # 截取VGG的前31层（到relu4_1）
        vgg_model = nn.Sequential(*list(vgg_model.children())[:31])

        vgg_model.to(device)
        decoder_model.to(device)
        vgg_model.eval()
        decoder_model.eval()

        print("模型加载成功")

        # 加载风格图片
        load_style_images()

    except Exception as e:
        print(f"模型加载失败: {e}")
        raise


def load_style_images():
    """加载风格图片"""
    global style_images
    style_images = []

    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_dir = Path(app.config['STYLE_DIR'])
    if style_dir.exists():
        for img_file in style_dir.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                try:
                    img = Image.open(img_file).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0)
                    style_images.append({
                        'name': img_file.stem,
                        'tensor': img_tensor,
                        'preview': get_image_preview(img)
                    })
                except Exception as e:
                    print(f"加载风格图片 {img_file} 失败: {e}")

    print(f"加载了 {len(style_images)} 张风格图片")


def get_image_preview(img):
    """生成图片预览（base64编码）"""
    img.thumbnail((100, 100))
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def calc_mean_std(feat, eps=1e-5):
    """计算特征的均值和标准差"""
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """AdaIN核心算法"""
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def preprocess_image(image, size=512):
    """预处理图片"""
    original_size = image.size
    scale = size / max(original_size)
    new_size = (int(original_size[0] * scale), int(original_size[1] * scale))

    transform = transforms.Compose([
        transforms.Resize(new_size),
        transforms.ToTensor()
    ])

    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, original_size


def transfer_style(content_tensor, style_tensor, alpha=0.8):
    """执行风格迁移"""
    with torch.no_grad():
        # 调整风格图片大小以匹配内容
        style_tensor_resized = F.interpolate(
            style_tensor.to(device),
            size=content_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # 提取特征
        content_feat = vgg_model(content_tensor.to(device))
        style_feat = vgg_model(style_tensor_resized)

        # AdaIN
        target_feat = adaptive_instance_normalization(content_feat, style_feat)

        # 插值控制风格强度
        target_feat = alpha * target_feat + (1 - alpha) * content_feat

        # 解码生成图片
        output = decoder_model(target_feat)

        # 裁剪到有效范围 [0, 1]
        output = torch.clamp(output, 0, 1)

        # 转换为PIL图像
        output_image = transforms.ToPILImage()(output.squeeze(0).cpu())

    return output_image


@app.route('/')
def index():
    """主页"""
    return render_template('index.html',
                           style_images=style_images,
                           device=str(device))


@app.route('/api/transfer', methods=['POST'])
def style_transfer():
    """风格迁移API"""
    try:
        # 检查文件
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图片'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400

        # 获取参数
        alpha = float(request.form.get('alpha', 0.8))
        style_index = int(request.form.get('style_index', 0))

        if not (0 <= alpha <= 1):
            return jsonify({'error': 'alpha参数必须在0-1之间'}), 400

        if style_index < 0 or style_index >= len(style_images):
            return jsonify({'error': '无效的风格索引'}), 400

        # 读取图片
        image = Image.open(file.stream).convert('RGB')

        # 预处理
        content_tensor, original_size = preprocess_image(image)

        # 获取风格图片
        style_tensor = style_images[style_index]['tensor']

        # 执行风格迁移
        result_image = transfer_style(content_tensor, style_tensor, alpha)

        # 恢复原始尺寸
        result_image = result_image.resize(original_size, Image.Resampling.LANCZOS)

        # 转换为base64返回
        buffered = io.BytesIO()
        result_image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{img_str}'
        })

    except Exception as e:
        print(f"处理错误: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/styles')
def get_styles():
    """获取所有风格图片"""
    styles_data = []
    for i, style in enumerate(style_images):
        styles_data.append({
            'index': i,
            'name': style['name'],
            'preview': style['preview']
        })
    return jsonify(styles_data)


@app.route('/api/health')
def health_check():
    """健康检查"""
    models_loaded = vgg_model is not None and decoder_model is not None
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'device': str(device),
        'style_count': len(style_images),
        'models_loaded': models_loaded
    })


if __name__ == '__main__':
    # 加载模型
    print("正在加载模型...")
    load_models()

    # 启动Flask应用
    print("启动Web服务器...")
    app.run(host='0.0.0.0', port=5000, debug=True)