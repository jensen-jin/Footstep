#!/bin/bash
# RTX 4060优化训练脚本

echo "🚀 Starting Humanoid Robot Training for RTX 4060"
echo "================================================"

# 激活环境
if command -v conda &> /dev/null; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate footstep
    echo "✅ Environment activated: footstep"
else
    echo "⚠️  Conda not found, using current environment"
fi

# 方案1: 高性能CPU训练（推荐）
echo ""
echo "🔥 Option 1: High-Performance CPU Training (RECOMMENDED)"
echo "-------------------------------------------------------"
echo "Command: python gym/scripts/train.py --task=humanoid_controller --num_envs=512 --max_iterations=3000 --headless --experiment_name=cpu_production --seed=42 --sim_device=cpu --rl_device=cpu"
echo ""

# 方案2: GPU训练（需要解决CUDA架构问题）
echo "⚡ Option 2: GPU Training (May have CUDA architecture issues)"
echo "------------------------------------------------------------"
echo "First set environment variables:"
echo "export TORCH_CUDA_ARCH_LIST=\"7.5;8.0;8.6;8.9\""
echo "export CUDA_LAUNCH_BLOCKING=1"
echo "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512"
echo ""
echo "Then run:"
echo "python gym/scripts/train.py --task=humanoid_controller --num_envs=512 --max_iterations=3000 --headless --experiment_name=gpu_training --seed=42"
echo ""

# 询问用户选择
echo "Choose your training mode:"
echo "1) CPU Training (Stable, works perfectly)"
echo "2) GPU Training (May have issues, but faster if works)"
echo "3) Test both modes"
echo ""
read -p "Enter your choice (1/2/3): " choice

case $choice in
    1)
        echo "🔄 Starting CPU Training..."
        python gym/scripts/train.py \
            --task=humanoid_controller \
            --num_envs=512 \
            --max_iterations=3000 \
            --headless \
            --experiment_name=cpu_production_$(date +%Y%m%d_%H%M%S) \
            --seed=42 \
            --sim_device=cpu \
            --rl_device=cpu
        ;;
    2)
        echo "🔄 Setting up GPU environment and starting GPU training..."
        export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"
        export CUDA_LAUNCH_BLOCKING=1
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
        
        python gym/scripts/train.py \
            --task=humanoid_controller \
            --num_envs=512 \
            --max_iterations=3000 \
            --headless \
            --experiment_name=gpu_training_$(date +%Y%m%d_%H%M%S) \
            --seed=42
        ;;
    3)
        echo "🧪 Testing CPU mode first..."
        python gym/scripts/train.py \
            --task=humanoid_controller \
            --num_envs=64 \
            --max_iterations=2 \
            --headless \
            --experiment_name=cpu_test \
            --seed=42 \
            --sim_device=cpu \
            --rl_device=cpu
        
        if [ $? -eq 0 ]; then
            echo "✅ CPU test passed!"
            echo ""
            echo "🧪 Testing GPU mode..."
            export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"
            export CUDA_LAUNCH_BLOCKING=1
            export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
            
            timeout 60s python gym/scripts/train.py \
                --task=humanoid_controller \
                --num_envs=64 \
                --max_iterations=1 \
                --headless \
                --experiment_name=gpu_test \
                --seed=42
            
            if [ $? -eq 0 ]; then
                echo "✅ GPU test passed! You can use both CPU and GPU modes."
            else
                echo "❌ GPU test failed. Use CPU mode for training."
            fi
        else
            echo "❌ CPU test failed. Please check your environment."
        fi
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "🎉 Training completed! Check gym/logs/ for results."
echo "To play the trained model, run:"
echo "python gym/scripts/play.py --task=humanoid_controller"