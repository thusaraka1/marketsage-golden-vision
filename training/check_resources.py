# 🔍 System Resource Checker for ML Training
# Run this to check if your PC can handle local training

import sys
import os
import platform

print("=" * 60)
print("🖥️  SYSTEM RESOURCE CHECK")
print("=" * 60)

# 1. Python Version
print(f"\n📌 Python: {sys.version}")

# 2. OS Info
print(f"📌 OS: {platform.system()} {platform.release()}")
print(f"📌 Machine: {platform.machine()}")

# 3. CPU Info
import multiprocessing
print(f"\n🔧 CPU Cores: {multiprocessing.cpu_count()}")

# 4. RAM
try:
    import psutil
    ram = psutil.virtual_memory()
    print(f"🔧 RAM Total: {ram.total / (1024**3):.1f} GB")
    print(f"🔧 RAM Available: {ram.available / (1024**3):.1f} GB")
    print(f"🔧 RAM Used: {ram.percent}%")
except ImportError:
    print("⚠️ Install psutil for RAM info: pip install psutil")

# 5. GPU Check - TensorFlow
print("\n" + "=" * 60)
print("🎮 GPU CHECK (TensorFlow)")
print("=" * 60)

try:
    import tensorflow as tf
    print(f"✅ TensorFlow Version: {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU Found: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        
        # GPU Memory
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU Memory Growth: Enabled")
        except:
            pass
            
        # Quick GPU test
        print("\n🔬 Running GPU test...")
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
        print("✅ GPU Computation: Working!")
        
    else:
        print("❌ No GPU detected by TensorFlow")
        print("   Possible reasons:")
        print("   - NVIDIA GPU not installed")
        print("   - CUDA not installed")
        print("   - cuDNN not installed")
        print("   - TensorFlow-GPU not installed")
        
except ImportError:
    print("❌ TensorFlow not installed")
    print("   Install: pip install tensorflow")
except Exception as e:
    print(f"❌ TensorFlow GPU Error: {e}")

# 6. GPU Check - PyTorch (alternative)
print("\n" + "=" * 60)
print("🎮 GPU CHECK (PyTorch)")
print("=" * 60)

try:
    import torch
    print(f"✅ PyTorch Version: {torch.__version__}")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print(f"✅ GPU Count: {torch.cuda.device_count()}")
        print(f"✅ GPU Name: {torch.cuda.get_device_name(0)}")
        
        # GPU Memory
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU Memory: {gpu_mem:.1f} GB")
        
except ImportError:
    print("⚠️ PyTorch not installed (optional)")
except Exception as e:
    print(f"⚠️ PyTorch check failed: {e}")

# 7. NVIDIA Driver Check
print("\n" + "=" * 60)
print("🔧 NVIDIA DRIVER CHECK")
print("=" * 60)

import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        # Parse key info
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Driver Version' in line or 'CUDA Version' in line:
                print(f"✅ {line.strip()}")
            if 'MiB' in line and '/' in line:
                print(f"📊 {line.strip()}")
    else:
        print("❌ nvidia-smi failed")
except FileNotFoundError:
    print("❌ nvidia-smi not found - NVIDIA driver not installed")
except Exception as e:
    print(f"❌ Error: {e}")

# 8. Summary
print("\n" + "=" * 60)
print("📋 VERDICT")
print("=" * 60)

can_train = False
try:
    import tensorflow as tf
    if tf.config.list_physical_devices('GPU'):
        can_train = True
        print("✅ Your PC CAN train ML models locally with GPU!")
        print("   Recommended batch size: 32-64")
        print("   Expected training time: 2-5 minutes per model")
    else:
        print("⚠️ No GPU found. Training will use CPU (slower)")
        print("   CPU training is ~10x slower than GPU")
        print("   Recommended: Use Google Colab for faster training")
except:
    print("❌ TensorFlow not properly set up")
    print("   Please install: pip install tensorflow")

print("=" * 60)
