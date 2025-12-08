# apply_yungang.py
# coding=utf-8
import argparse
import os
from pathlib import Path
from yungang_adain import YungangStyleTransfer


def main():
    parser = argparse.ArgumentParser(description="应用云冈风格滤镜")
    parser.add_argument('--input', type=str, required=True,
                        help='输入图片或目录路径')
    parser.add_argument('--output', type=str, default='./output',
                        help='输出目录或图片路径')
    parser.add_argument('--model', type=str, default='models/yungang_decoder.pth',
                        help='decoder模型路径')
    parser.add_argument('--style_dir', type=str, default='yungang_styles',
                        help='风格图片目录')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='风格化强度 (0.0-1.0)')
    parser.add_argument('--style_index', type=int, default=None,
                        help='指定风格图片索引（默认随机）')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda/cpu)')
    parser.add_argument('--batch', action='store_true',
                        help='批量处理整个目录')

    args = parser.parse_args()

    # 创建输出目录
    if args.batch:
        os.makedirs(args.output, exist_ok=True)

    # 创建风格迁移器
    yungang = YungangStyleTransfer(
        style_dir=args.style_dir,
        device=args.device,
        model_path=args.model
    )

    print(f"云冈风格滤镜已加载，使用模型: {args.model}")
    print(f"风格图片数量: {len(yungang.style_images)}")

    if args.batch:
        # 批量处理
        print(f"批量处理: {args.input} -> {args.output}")
        count = yungang.batch_process(
            content_dir=args.input,
            output_dir=args.output,
            alpha=args.alpha
        )
        print(f"处理完成，共 {count} 张图片")
    else:
        # 单张处理
        print(f"处理单张图片: {args.input}")
        output_path = args.output

        # 如果输出是目录，自动生成文件名
        if os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)
            input_name = Path(args.input).stem
            output_path = os.path.join(output_path, f"yungang_{input_name}.jpg")

        result = yungang.apply_filter(
            content_path=args.input,
            output_path=output_path,
            alpha=args.alpha,
            random_style=(args.style_index is None),
            style_index=args.style_index
        )
        print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
