# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import random


class YungangStyleTransfer:
    """
    云冈风格滤镜 - 基于AdaIN实现
    输入任意图片，输出云冈风格图片
    """

    def __init__(self, style_dir="yungang_styles", device=None, model_path="models/yungang_decoder.pth"):
        """
        初始化云冈风格滤镜

        Args:
            style_dir: 云冈风格图片目录
            device: 使用的设备 ('cuda' 或 'cpu')
            model_path: 模型路径
        """
        self.style_dir = style_dir
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.vgg, self.decoder = self.load_models(model_path)
        self.vgg.to(self.device)
        self.decoder.to(self.device)
        self.vgg.eval()
        self.decoder.eval()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(512),  # 调整到512x512
            transforms.ToTensor()
        ])

        # 加载所有云冈风格图片
        self.style_images = self.load_style_images()
        print(f"云冈风格滤镜初始化完成，加载了 {len(self.style_images)} 张风格图片")

    def load_models(self, model_path):
        """加载VGG和decoder模型"""
        # 导入模型定义
        import sys
        sys.path.append('.')  # 添加当前目录到路径

        from net import vgg, decoder

        # 加载VGG
        vgg_model = vgg
        vgg_model.load_state_dict(torch.load('models/vgg_normalised.pth'))

        # 加载Decoder
        decoder_model = decoder
        decoder_model.load_state_dict(torch.load(model_path))

        # 截取VGG的前31层（到relu4_1）
        vgg_model = nn.Sequential(*list(vgg_model.children())[:31])

        return vgg_model, decoder_model

    def load_style_images(self):
        """加载云冈风格图片"""
        style_images = []
        if os.path.exists(self.style_dir):
            for file in os.listdir(self.style_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    try:
                        img_path = os.path.join(self.style_dir, file)
                        img = Image.open(img_path).convert('RGB')
                        img_tensor = self.transform(img).unsqueeze(0)
                        style_images.append(img_tensor)
                    except Exception as e:
                        print(f"加载风格图片 {file} 失败: {e}")

        if not style_images:
            print(f"风格目录 '{self.style_dir}' 中没有找到图片")
            print(f"请将云冈风格图片放入 '{self.style_dir}' 目录")

        return style_images

    def preprocess_image(self, image_path, size=512):
        """预处理图片"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        elif isinstance(image_path, Image.Image):
            image = image_path.convert('RGB')
        else:
            raise ValueError("输入必须是图片路径或PIL.Image对象")

        # 保持宽高比调整大小
        original_size = image.size
        scale = size / max(original_size)
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))

        transform = transforms.Compose([
            transforms.Resize(new_size),
            transforms.ToTensor()
        ])

        image_tensor = transform(image).unsqueeze(0)
        return image_tensor, original_size

    def postprocess_image(self, tensor, original_size=None):
        """后处理：tensor转PIL图片"""
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = transforms.ToPILImage()(image)

        # 如果需要，恢复到原始大小
        if original_size:
            image = image.resize(original_size, Image.Resampling.LANCZOS)

        return image

    def adaptive_instance_normalization(self, content_feat, style_feat):
        """AdaIN核心算法"""
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def calc_mean_std(self, feat, eps=1e-5):
        """计算特征的均值和标准差"""
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def transfer_style(self, content_tensor, style_tensor, alpha=0.8):
        """执行风格迁移"""
        with torch.no_grad():
            # 提取特征
            content_feat = self.vgg(content_tensor.to(self.device))
            style_feat = self.vgg(style_tensor.to(self.device))

            # AdaIN
            target_feat = self.adaptive_instance_normalization(content_feat, style_feat)

            # 插值控制风格强度
            target_feat = alpha * target_feat + (1 - alpha) * content_feat

            # 解码生成图片
            output = self.decoder(target_feat)

            # 裁剪到有效范围 [0, 1]
            output = torch.clamp(output, 0, 1)

        return output

    def apply_filter(self, content_path, output_path=None, alpha=0.8, random_style=True, style_index=None):
        """
        应用云冈风格滤镜

        Args:
            content_path: 内容图片路径或PIL.Image对象
            output_path: 输出图片路径（如为None则返回PIL.Image）
            alpha: 风格化强度 (0.0-1.0)
            random_style: 是否随机选择风格
            style_index: 指定风格图片索引

        Returns:
            PIL.Image对象（如果output_path为None）
        """
        if not self.style_images:
            raise ValueError("没有可用的风格图片，请先添加云冈风格图片到风格目录")

        # 1. 预处理内容图片
        content_tensor, original_size = self.preprocess_image(content_path)

        # 2. 选择风格图片
        if style_index is not None and 0 <= style_index < len(self.style_images):
            style_tensor = self.style_images[style_index]
        elif random_style:
            style_tensor = random.choice(self.style_images)
        else:
            # 默认使用第一张
            style_tensor = self.style_images[0]

        # 3. 调整风格图片大小以匹配内容
        style_tensor_resized = F.interpolate(
            style_tensor,
            size=content_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # 4. 执行风格迁移
        output_tensor = self.transfer_style(content_tensor, style_tensor_resized, alpha)

        # 5. 后处理并保存/返回
        output_image = self.postprocess_image(output_tensor, original_size)

        if output_path:
            output_image.save(output_path)
            print(f"云冈风格图片已保存: {output_path}")
            return output_path
        else:
            return output_image

    def batch_process(self, content_dir, output_dir, alpha=0.8):
        """批量处理目录中的所有图片"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp')
        processed = 0

        for file in os.listdir(content_dir):
            if file.lower().endswith(supported_formats):
                try:
                    content_path = os.path.join(content_dir, file)
                    output_path = os.path.join(output_dir, f"yungang_{file}")

                    self.apply_filter(content_path, output_path, alpha)
                    processed += 1
                except Exception as e:
                    print(f"处理 {file} 失败: {e}")

        print(f"批量处理完成，共处理 {processed} 张图片")
        return processed
