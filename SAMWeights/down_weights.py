# download_fixed.py
import urllib.request
import os
import time
import sys


def download_with_progress(url, filename):
    """带进度显示的下载函数"""

    def progress_callback(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        if percent % 10 == 0:  # 每10%显示一次
            print(
                f"  下载进度: {percent}% ({count * block_size / (1024 * 1024):.1f}MB / {total_size / (1024 * 1024):.1f}MB)")

    print(f"开始下载: {filename}")
    print(f"来自: {url}")

    try:
        urllib.request.urlretrieve(url, filename, progress_callback)
        print(f"✅ 下载完成: {filename}")
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False


def main():
    print("=" * 60)
    print("SAM 预训练权重下载工具 (修复版)")
    print("=" * 60)

    # 权重统一保存到本脚本所在目录（SAMWeights/），不受运行时工作目录影响
    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)

    # 明确的权重文件信息
    weights_info = {
        'vit_b': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
            'filename': os.path.join(save_dir, 'sam_vit_b_01ec64.pth'),
            'size_mb': 376
        },
        'vit_l': {  # 确保这个键名是 vit_l
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
            'filename': os.path.join(save_dir, 'sam_vit_l_0b3195.pth'),
            'size_mb': 1250
        },
        'vit_h': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            'filename': os.path.join(save_dir, 'sam_vit_h_4b8939.pth'),
            'size_mb': 2560
        }
    }

    print("\n可用模型:")
    for model_name, info in weights_info.items():
        print(f"  - {model_name}: {info['size_mb']} MB")

    print("\n建议:")
    print("  vit_b (376MB) - 适合测试，速度最快")
    print("  vit_l (1.25GB) - 平衡效果和速度")
    print("  vit_h (2.56GB) - 效果最好，但最慢")

    # 选择要下载的模型
    while True:
        choice = input("\n请输入要下载的模型名称 (vit_b/vit_l/vit_h)，直接回车默认下载 vit_b: ").strip().lower()

        if not choice:  # 直接回车
            choice = 'vit_b'
            break

        if choice in weights_info:
            break
        else:
            print(f"❌ 错误: 不支持的模型 '{choice}'")
            print(f"请从 {list(weights_info.keys())} 中选择")

    info = weights_info[choice]

    # 检查文件是否已存在
    if os.path.exists(info['filename']):
        file_size = os.path.getsize(info['filename']) / (1024 * 1024)  # MB
        print(f"\n⚠️  文件已存在: {info['filename']}")
        print(f"   当前大小: {file_size:.1f} MB")

        if file_size >= info['size_mb'] * 0.9:  # 如果文件大小接近预期
            print("✅ 文件看起来是完整的")
            overwrite = input("是否重新下载？(y/N): ").strip().lower()
            if overwrite != 'y':
                print("跳过下载。")
                return
        else:
            print("⚠️  文件大小异常，可能是未下载完整")
            overwrite = input("是否重新下载？(Y/n): ").strip().lower()
            if overwrite == 'n':
                print("使用现有文件，但可能会出错。")
                return

    print(f"\n正在下载 {choice}...")
    print(f"文件大小: {info['size_mb']} MB")
    print(f"保存路径: {info['filename']}")
    print("下载可能需要几分钟，请耐心等待...")
    print("如果下载失败，可以重新运行此脚本。")

    # 开始下载
    start_time = time.time()
    success = download_with_progress(info['url'], info['filename'])

    if success:
        end_time = time.time()
        file_size = os.path.getsize(info['filename']) / (1024 * 1024)  # MB
        elapsed = end_time - start_time

        print(f"\n✅ 下载成功!")
        print(f"   文件保存到: {os.path.abspath(info['filename'])}")
        print(f"   文件大小: {file_size:.1f} MB")
        print(f"   耗时: {elapsed:.1f} 秒")

        if elapsed > 0:
            speed = file_size / elapsed
            print(f"   平均速度: {speed:.1f} MB/秒")

        # 检查文件大小是否合理
        if abs(file_size - info['size_mb']) < 10:  # 允许10MB误差
            print("✅ 文件大小正常")
        else:
            print(f"⚠️  文件大小可能有异常 (预期约{info['size_mb']}MB)")
    else:
        print("\n❌ 下载失败，请尝试：")
        print("1. 检查网络连接")
        print("2. 稍后重试")
        print("3. 手动下载（见下面链接）")
        print("\n手动下载链接：")
        print(f"  {info['url']}")


if __name__ == "__main__":
    main()