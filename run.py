# coding=utf-8
# run.py - 一键启动脚本
import os
import sys
import codecs
import subprocess
import webbrowser
import time
import threading
import requests

if sys.platform == 'win32':
    # Windows下设置控制台编码为UTF-8
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)


def check_requirements():
    """检查依赖"""
    try:
        import flask
        import torch
        import torchvision
        import PIL
        print("所有依赖已安装")
        return True
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("正在安装依赖...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("依赖安装完成")
            return True
        except:
            print("依赖安装失败")
            return False


def check_directories():
    """检查目录"""
    directories = ['models', 'yungang_styles', 'static/uploads']
    for dir in directories:
        if not os.path.exists(dir):
            os.makedirs(dir)
            print(f"创建目录: {dir}")

    # 检查模型文件
    required_models = ['models/vgg_normalised.pth', 'models/yungang_decoder.pth']
    missing_models = []
    for model in required_models:
        if not os.path.exists(model):
            missing_models.append(model)

    if missing_models:
        print("\n缺少以下模型文件:")
        for model in missing_models:
            print(f"  - {model}")
        print("\n请下载模型文件并放入相应目录")
        print("可以从以下地址下载:")
        print("1. vgg_normalised.pth: https://github.com/naoto0804/pytorch-AdaIN/raw/master/models/vgg_normalised.pth")
        print("2. yungang_decoder.pth: 训练自己的模型或使用预训练模型")
        return False

    return True


def open_browser_when_ready():
    """当Flask应用就绪时打开浏览器"""
    max_retries = 30  # 最多尝试30次
    retry_interval = 1  # 每次间隔1秒

    for i in range(max_retries):
        try:
            # 尝试连接Flask服务
            response = requests.get('http://localhost:5000', timeout=2)
            if response.status_code < 500:  # 只要不是服务器错误
                print(f"Flask应用已就绪 (尝试 {i + 1} 次)")
                webbrowser.open('http://localhost:5000')
                return
        except requests.RequestException:
            pass  # 服务还没启动，继续等待

        time.sleep(retry_interval)

    print("警告: Flask应用启动超时，请手动访问 http://localhost:5000")


def main():
    print("=" * 50)
    print("云冈风格迁移 Web 应用")
    print("=" * 50)

    # 检查依赖
    if not check_requirements():
        return

    # 检查目录
    if not check_directories():
        return

    # 检查风格图片
    style_files = os.listdir('yungang_styles')
    if not any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) for f in style_files):
        print("\nyungang_styles 目录中没有找到风格图片")
        print("请将云冈风格图片放入该目录")

    print("\n" + "=" * 50)
    print("启动Web应用...")
    print("将在浏览器中打开 http://localhost:5000")
    print("按 Ctrl+C 停止服务")
    print("=" * 50 + "\n")

    flask_process = None

    # 启动Flask应用
    try:
        # 创建并启动Flask进程
        flask_process = subprocess.Popen([sys.executable, 'app.py'])

        # 在新线程中等待并打开浏览器
        browser_thread = threading.Thread(target=open_browser_when_ready)
        browser_thread.daemon = True  # 设置为守护线程
        browser_thread.start()

        # 等待Flask进程结束
        flask_process.wait()

    except KeyboardInterrupt:
        print("\n\n正在停止服务...")
        if flask_process:  # 检查是否为None
            flask_process.terminate()
            flask_process.wait(timeout=5)  # 等待进程结束
        print("服务已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        if flask_process:  # 检查是否为None
            flask_process.terminate()


if __name__ == "__main__":
    main()
