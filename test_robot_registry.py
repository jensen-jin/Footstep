#!/usr/bin/env python3
"""
Test script for the robot registry system.

This script demonstrates and validates the robot registry functionality.
"""

import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from gym.envs.robots import RobotRegistry, RobotConfigFactory
from gym.envs.robots.unified_trainer import UnifiedTrainer


def test_robot_registry():
    """Test basic robot registry functionality."""
    print("🧪 Testing Robot Registry System")
    print("=" * 50)
    
    # Test 1: List available robots
    print("\n1️⃣ Available Robots:")
    robots = RobotRegistry.list_robots()
    for robot in robots:
        print(f"   ✅ {robot}")
    
    # Test 2: List by category
    print("\n2️⃣ Robots by Category:")
    categories = RobotRegistry.list_categories()
    for category in categories:
        robots_in_category = RobotRegistry.list_robots(category)
        print(f"   {category}: {robots_in_category}")
    
    # Test 3: Get robot details
    print("\n3️⃣ Robot Details:")
    test_robot = "mit_humanoid_fixed_arms"
    try:
        robot_spec = RobotRegistry.get_robot(test_robot)
        print(f"   Robot: {robot_spec.name}")
        print(f"   Category: {robot_spec.category}")
        print(f"   Actuators: {robot_spec.num_actuators}")
        print(f"   Description: {robot_spec.description}")
        print(f"   URDF: {robot_spec.urdf_path}")
        print(f"   Joint names: {robot_spec.get_joint_names()[:3]}... (showing first 3)")
    except Exception as e:
        print(f"   ❌ Error: {e}")


def test_config_factory():
    """Test robot configuration factory."""
    print("\n🏭 Testing Robot Configuration Factory")
    print("=" * 50)
    
    test_cases = [
        ("mit_humanoid_fixed_arms", "limp"),
        ("mit_humanoid_fixed_arms", "ipc3d"),
        ("cartpole", "ipc3d"),
    ]
    
    for robot_name, guidance_model in test_cases:
        print(f"\n🔧 Creating config: {robot_name} + {guidance_model}")
        try:
            config = RobotConfigFactory.create_config(
                robot_name=robot_name,
                guidance_model=guidance_model
            )
            
            print(f"   ✅ Config created: {type(config).__name__}")
            print(f"   📁 URDF: {getattr(config.asset, 'file', 'N/A')}")
            print(f"   ⚙️  Actuators: {getattr(config.env, 'num_actuators', 'N/A')}")
            print(f"   🎯 Guidance: {getattr(config, 'guidance', 'N/A')}")
            
            if hasattr(config, 'guidance'):
                guidance = config.guidance
                print(f"      Model type: {guidance.model_type}")
                if guidance.model_type == 'limp':
                    print(f"      COM height: {guidance.limp_com_height}")
                elif guidance.model_type == 'ipc3d':
                    print(f"      Pole length: {guidance.ipc3d_pole_length}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")


def test_unified_trainer():
    """Test unified trainer functionality."""
    print("\n🎓 Testing Unified Trainer")
    print("=" * 50)
    
    trainer = UnifiedTrainer()
    
    # Test 1: List available robots
    print("\n1️⃣ Available robots via trainer:")
    trainer.list_available_robots()
    
    # Test 2: Get robot info
    print("\n2️⃣ Robot information:")
    trainer.get_robot_info("mit_humanoid_fixed_arms")
    
    # Test 3: Train single robot (dry run)
    print("\n3️⃣ Single robot training (dry run):")
    try:
        config = trainer.train_single_robot(
            robot_name="mit_humanoid_fixed_arms",
            guidance_model="ipc3d",
            experiment_name="test_experiment"
        )
        print(f"   ✅ Training setup successful")
    except Exception as e:
        print(f"   ❌ Training setup failed: {e}")


def test_config_overrides():
    """Test configuration overrides."""
    print("\n⚙️  Testing Configuration Overrides")
    print("=" * 50)
    
    try:
        config = RobotConfigFactory.create_config(
            robot_name="mit_humanoid_fixed_arms",
            guidance_model="ipc3d",
            env={'num_envs': 2048, 'episode_length_s': 15},
            terrain={'mesh_type': 'heightfield'},
            rewards={'weights': {'base_height': 3.0}}
        )
        
        print("   ✅ Config with overrides created")
        print(f"   🌍 Num envs: {config.env.num_envs}")
        print(f"   ⏱️  Episode length: {config.env.episode_length_s}")
        print(f"   🏔️  Terrain type: {config.terrain.mesh_type}")
        print(f"   🎯 Base height weight: {config.rewards.weights.base_height}")
        
    except Exception as e:
        print(f"   ❌ Override test failed: {e}")


def test_comparison_study():
    """Test comparison study functionality."""
    print("\n📊 Testing Comparison Study")
    print("=" * 50)
    
    trainer = UnifiedTrainer()
    
    # Small comparison study
    try:
        results = trainer.run_comparison_study(
            robots=["mit_humanoid_fixed_arms"],
            guidance_models=["limp", "ipc3d"],
            base_experiment_name="test_comparison"
        )
        
        print("   ✅ Comparison study completed")
        print(f"   📊 Results: {len(results)} configurations tested")
        
        for config_name, result in results.items():
            status = result['status']
            emoji = "✅" if status == 'success' else "❌"
            print(f"   {emoji} {config_name}: {status}")
            
    except Exception as e:
        print(f"   ❌ Comparison study failed: {e}")


def main():
    """Run all tests."""
    print("🚀 Robot Registry System Test Suite")
    print("=" * 60)
    
    try:
        test_robot_registry()
        test_config_factory()
        test_unified_trainer()
        test_config_overrides()
        test_comparison_study()
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed!")
        print("\n💡 Next steps:")
        print("   • Integrate with actual training pipeline")
        print("   • Add IPC3D algorithm implementation")
        print("   • Test with real robot training")
        
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()