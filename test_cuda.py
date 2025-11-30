"""
测试PyTorch CUDA支持
"""
import torch
import sys

print("=" * 60)
print("PyTorch CUDA检查")
print("=" * 60)
print(f"PyTorch版本: {torch.__version__}")
print(f"Python版本: {sys.version.split()[0]}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ CUDA版本: {torch.version.cuda}")
    if hasattr(torch.backends.cudnn, 'version'):
        print(f"✓ cuDNN版本: {torch.backends.cudnn.version()}")
    print(f"✓ GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    内存: {props.total_memory / 1024**3:.2f} GB")
        print(f"    计算能力: {props.major}.{props.minor}")
    
    # 测试GPU计算
    try:
        print("\n测试GPU计算...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"✓ GPU计算测试成功")
        print(f"  结果形状: {z.shape}")
        print(f"  结果设备: {z.device}")
    except Exception as e:
        print(f"✗ GPU计算测试失败: {e}")
else:
    print("\n⚠ CUDA不可用！")
    print("\n可能的原因：")
    print("1. 未安装GPU版本的PyTorch（当前可能是CPU版本）")
    print("2. 没有NVIDIA GPU")
    print("3. NVIDIA驱动未安装或版本过旧")
    
    # 检查是否有NVIDIA GPU
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("\n✓ 检测到NVIDIA GPU和驱动")
            print("  但PyTorch无法使用CUDA，可能是PyTorch版本问题")
            print("\n解决方案：")
            print("  pip3 uninstall torch torchvision torchaudio -y")
            print("  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        else:
            print("\n✗ 无法运行nvidia-smi，可能没有NVIDIA GPU或驱动未安装")
    except:
        print("\n? 无法检查nvidia-smi")

print("=" * 60)


