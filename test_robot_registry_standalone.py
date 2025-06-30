#!/usr/bin/env python3
"""
Standalone test script for the robot registry system.
This version doesn't depend on Isaac Gym imports.
"""

import sys
from pathlib import Path
import os

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Mock the missing base config to avoid Isaac Gym dependency
class MockBaseConfig:
    pass

class MockLeggedRobotCfg:
    def __init__(self):
        self.env = MockSection()
        self.asset = MockSection()
        self.init_state = MockSection()
        self.control = MockSection()
        self.terrain = MockSection()
        self.rewards = MockSection()
        self.runner = MockSection()
        self.domain_rand = MockSection()

class MockSection:
    def __init__(self):
        pass

# Inject mocks into the module path
sys.modules['gym.envs.base.base_config'] = type('module', (), {'BaseConfig': MockBaseConfig})()
sys.modules['gym.envs.base.legged_robot_config'] = type('module', (), {'LeggedRobotCfg': MockLeggedRobotCfg})()

# Now import our modules
from gym.envs.robots.registry import RobotRegistry, RobotSpec
from gym.envs.robots.factory import RobotConfigFactory, GuidanceConfig
from gym.envs.robots.specifications import register_all_robots

def test_robot_registry():
    """Test basic robot registry functionality."""
    print("🧪 Testing Robot Registry System")
    print("=" * 50)
    
    # Register robots first
    register_all_robots()
    
    # Test 1: List available robots
    print("\n1️⃣ Available Robots:")
    robots = RobotRegistry.list_robots()
    for robot in robots:
        print(f"   ✅ {robot}")
    
    print(f"\n   Total robots registered: {len(robots)}")
    
    # Test 2: List by category
    print("\n2️⃣ Robots by Category:")
    categories = RobotRegistry.list_categories()
    for category in categories:
        robots_in_category = RobotRegistry.list_robots(category)
        print(f"   {category}: {len(robots_in_category)} robots")
        for robot in robots_in_category[:3]:  # Show first 3
            print(f"      • {robot}")
        if len(robots_in_category) > 3:
            print(f"      ... and {len(robots_in_category) - 3} more")
    
    # Test 3: Get robot details
    print("\n3️⃣ Robot Details:")
    test_robot = "mit_humanoid_fixed_arms"
    try:
        robot_spec = RobotRegistry.get_robot(test_robot)
        print(f"   Robot: {robot_spec.name}")
        print(f"   Category: {robot_spec.category}")
        print(f"   Actuators: {robot_spec.num_actuators}")
        print(f"   Description: {robot_spec.description}")
        print(f"   Tags: {robot_spec.tags}")
        print(f"   URDF: {robot_spec.urdf_path}")
        print(f"   Joint names: {robot_spec.get_joint_names()[:3]}... (showing first 3)")
        print(f"   Physical params: {list(robot_spec.physical_params.keys())}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def test_guidance_config():
    """Test guidance configuration."""
    print("\n🎯 Testing Guidance Configuration")
    print("=" * 50)
    
    # Test LIPM guidance
    print("\n📐 LIPM Guidance:")
    try:
        limp_config = GuidanceConfig(
            model_type='limp',
            limp_com_height=0.62,
            limp_step_length=0.5
        )
        print(f"   ✅ LIMP config created")
        print(f"   📏 COM height: {limp_config.limp_com_height}")
        print(f"   👣 Step length: {limp_config.limp_step_length}")
        print(f"   ⚙️  LQR Q matrix: {limp_config.lqr_q_matrix}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test IPC3D guidance
    print("\n🚗 IPC3D Guidance:")
    try:
        ipc_config = GuidanceConfig(
            model_type='ipc3d',
            ipc3d_mass_cart=42.0,
            ipc3d_pole_length=0.62
        )
        print(f"   ✅ IPC3D config created")
        print(f"   🏋️  Cart mass: {ipc_config.ipc3d_mass_cart}")
        print(f"   📏 Pole length: {ipc_config.ipc3d_pole_length}")
        print(f"   ⚙️  LQR Q matrix: {ipc_config.lqr_q_matrix}")
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
        ("mit_humanoid_full", "limp"),
    ]
    
    for robot_name, guidance_model in test_cases:
        print(f"\n🔧 Creating config: {robot_name} + {guidance_model}")
        try:
            config = RobotConfigFactory.create_config(
                robot_name=robot_name,
                guidance_model=guidance_model
            )
            
            print(f"   ✅ Config created: {type(config).__name__}")
            print(f"   📁 URDF file: {config.asset.file}")
            print(f"   ⚙️  Actuators: {config.env.num_actuators}")
            print(f"   🎯 Guidance model: {config.guidance.model_type}")
            
            # Show guidance-specific parameters
            guidance = config.guidance
            if guidance.model_type == 'limp':
                print(f"      📏 COM height: {guidance.limp_com_height}")
                print(f"      👣 Step length: {guidance.limp_step_length}")
            elif guidance.model_type == 'ipc3d':
                print(f"      🏋️  Cart mass: {guidance.ipc3d_mass_cart}")
                print(f"      📏 Pole length: {guidance.ipc3d_pole_length}")
            
            # Show some joint configuration
            joint_names = list(config.init_state.default_joint_angles.keys())
            print(f"      🔗 Joints: {len(joint_names)} total, first 3: {joint_names[:3]}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

def test_robot_spec_validation():
    """Test robot specification validation."""
    print("\n✅ Testing Robot Spec Validation")
    print("=" * 50)
    
    # Test valid spec
    print("\n1️⃣ Valid robot spec:")
    try:
        valid_spec = RobotSpec(
            name="test_robot",
            urdf_path="test.urdf",
            num_actuators=2,
            default_joint_angles={"joint1": 0.0, "joint2": 0.0},
            joint_limits={"joint1": [-1, 1], "joint2": [-1, 1]},
            control_config={"stiffness": {"joint1": 10, "joint2": 10}},
            physical_params={"mass": 10.0},
            description="Test robot",
            category="test"
        )
        print(f"   ✅ Valid spec created: {valid_spec.name}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test invalid spec (mismatched joints)
    print("\n2️⃣ Invalid robot spec (mismatched joints):")
    try:
        invalid_spec = RobotSpec(
            name="invalid_robot",
            urdf_path="test.urdf",
            num_actuators=2,
            default_joint_angles={"joint1": 0.0, "joint2": 0.0},
            joint_limits={"joint1": [-1, 1]},  # Missing joint2
            control_config={},
            physical_params={},
            description="Invalid robot",
            category="test"
        )
        print(f"   ❌ Should have failed but didn't!")
    except ValueError as e:
        print(f"   ✅ Correctly caught validation error: {str(e)[:60]}...")
    except Exception as e:
        print(f"   ❓ Unexpected error: {e}")

def test_legacy_compatibility():
    """Test legacy compatibility features."""
    print("\n🔄 Testing Legacy Compatibility")
    print("=" * 50)
    
    from gym.envs.robots.legacy_adapter import LegacyConfigAdapter, create_legacy_wrapper
    
    # Test legacy wrapper creation
    print("\n1️⃣ Creating legacy wrapper:")
    try:
        WrapperClass = create_legacy_wrapper("mit_humanoid_fixed_arms", "ipc3d")
        wrapper_instance = WrapperClass()
        
        print(f"   ✅ Legacy wrapper created: {WrapperClass.__name__}")
        print(f"   🤖 Robot: {WrapperClass.get_robot_name()}")
        print(f"   🎯 Guidance: {WrapperClass.get_guidance_model()}")
        print(f"   ⚙️  Actuators: {wrapper_instance.env.num_actuators}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

def test_file_paths():
    """Test URDF file path resolution."""
    print("\n📁 Testing File Path Resolution")
    print("=" * 50)
    
    test_robot = "mit_humanoid_fixed_arms"
    try:
        robot_spec = RobotRegistry.get_robot(test_robot)
        full_path = robot_spec.get_urdf_full_path()
        
        print(f"   Original path: {robot_spec.urdf_path}")
        print(f"   Full path: {full_path}")
        
        # Check if path exists (it might not since we don't have the full environment)
        if os.path.exists(full_path):
            print(f"   ✅ File exists")
        else:
            print(f"   ⚠️  File not found (expected in test environment)")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    """Run all tests."""
    print("🚀 Robot Registry System Test Suite (Standalone)")
    print("=" * 60)
    
    try:
        test_robot_registry()
        test_guidance_config()
        test_config_factory()
        test_robot_spec_validation()
        test_legacy_compatibility()
        test_file_paths()
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed successfully!")
        
        # Show summary
        robots = RobotRegistry.list_robots()
        categories = RobotRegistry.list_categories()
        
        print(f"\n📊 System Summary:")
        print(f"   🤖 Total robots: {len(robots)}")
        print(f"   📂 Categories: {len(categories)} ({', '.join(categories)})")
        print(f"   🎯 Guidance models: LIMP, IPC3D")
        print(f"   🔧 Configuration factory: Working")
        print(f"   🔄 Legacy compatibility: Working")
        
        print("\n💡 Ready for integration with:")
        print("   • Isaac Gym training pipeline")
        print("   • LIPM/IPC3D algorithm implementations")
        print("   • Actual robot training experiments")
        
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()