#!/bin/bash
# RTX 40xx GPU训练环境设置脚本

echo "🔧 Setting up RTX 40xx environment for training..."

# 设置CUDA架构支持
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 清理之前的编译缓存
rm -rf ~/.cache/torch_extensions/py38_cu113/gymtorch/

echo "✅ Environment variables set:"
echo "   TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "   CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo "   PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

echo "🚀 Ready for training!"