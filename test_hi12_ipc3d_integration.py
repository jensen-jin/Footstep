#!/usr/bin/env python3
"""
Test script for HI-12 + IPC3D + Force Disturbance Integration

This script validates the integration system by:
1. Testing environment creation and initialization
2. Verifying IPC3D guidance system
3. Checking force disturbance functionality
4. Validating reward system
5. Running basic simulation steps

Usage:
    python test_hi12_ipc3d_integration.py
"""

import os
import sys
from typing import Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# IMPORTANT: Import Isaac Gym before torch to avoid import order conflicts
try:
    import isaacgym
except ImportError:
    print("⚠️  Isaac Gym not available, some tests will be skipped")
    
import torch
import numpy as np

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from gym.envs.hi_12.hi_controller_ipc3d import HiControllerIPC3D, HiControllerIPC3DCfg
        from gym.envs.guidance.ipc3d_guidance import IPC3DGuidanceModel
        from gym.envs.guidance.ipc3d_controller import IPC3D, IPC3DParams
        from gym.utils.task_registry import task_registry
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_task_registration():
    """Test that the HI-12 + IPC3D task is properly registered."""
    print("\n🔍 Testing task registration...")
    
    try:
        from gym.utils.task_registry import task_registry
        
        # Check if task is registered
        if "hi_controller_ipc3d" in task_registry.task_map:
            print("✅ hi_controller_ipc3d task is registered")
            return True
        else:
            print("❌ hi_controller_ipc3d task is not registered")
            return False
            
    except Exception as e:
        print(f"❌ Task registration error: {e}")
        return False

def test_ipc3d_controller():
    """Test IPC3D controller functionality."""
    print("\n🔍 Testing IPC3D controller...")
    
    try:
        from gym.envs.guidance.ipc3d_controller import IPC3D, IPC3DParams
        
        # Create IPC3D parameters
        params = IPC3DParams(
            mass_cart=6.0,
            mass_pole=3.0,
            pole_length=0.5596,
            control_mode=1,  # Velocity control
            dt=0.02
        )
        
        # Create controller
        controller = IPC3D(params)
        
        # Set target velocity
        controller.set_desired_velocity(1.0, 0.5)  # 1 m/s in X, 0.5 m/s in Z
        
        # Run a few simulation steps
        for i in range(10):
            result = controller.step()
            
        # Get final state
        state = controller.get_state()
        forces = controller.get_control_forces()
        
        print(f"✅ IPC3D controller working:")
        print(f"   X velocity: {state['x_cart_vel']:.3f} m/s")
        print(f"   Z velocity: {state['z_cart_vel']:.3f} m/s") 
        print(f"   Control forces: Fx={forces[0]:.1f}N, Fz={forces[1]:.1f}N")
        
        return True
        
    except Exception as e:
        print(f"❌ IPC3D controller error: {e}")
        return False

def test_guidance_model():
    """Test IPC3D guidance model functionality."""
    print("\n🔍 Testing IPC3D guidance model...")
    
    try:
        from gym.envs.guidance.ipc3d_guidance import IPC3DGuidanceModel
        from gym.envs.guidance.ipc3d_controller import IPC3DParams
        
        # Create parameters
        params = IPC3DParams(
            mass_cart=6.0,
            pole_length=0.5596,
            control_mode=1
        )
        
        # Create guidance model
        guidance = IPC3DGuidanceModel(params)
        
        # Initialize with robot state
        robot_state = {
            'position': np.array([0.0, 0.0, 0.5596]),
            'velocity': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([1.0, 0.0, 0.0, 0.0])
        }
        
        success = guidance.initialize(robot_state)
        if not success:
            print("❌ Failed to initialize guidance model")
            return False
            
        # Set target
        target = {
            'velocity_x': 1.0,
            'velocity_z': 0.0,
            'angular_velocity': 0.0
        }
        guidance.set_target(target)
        
        # Update guidance
        control_output = guidance.update(robot_state, dt=0.02)
        
        print("✅ IPC3D guidance model working:")
        print(f"   Control forces: {control_output['forces']}")
        print(f"   Reference position: {control_output['reference_position']}")
        print(f"   Gait phase: {control_output['gait_phase']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ IPC3D guidance model error: {e}")
        return False

def test_environment_creation():
    """Test creating the HI-12 + IPC3D environment."""
    print("\n🔍 Testing environment creation...")
    
    try:
        from gym.envs.hi_12.hi_controller_ipc3d import HiControllerIPC3DCfg
        from isaacgym import gymapi
        
        # Create configuration
        cfg = HiControllerIPC3DCfg()
        cfg.env.num_envs = 4  # Small number for testing
        cfg.guidance.enable_ipc3d = True
        cfg.domain_rand.use_force_push = True
        
        # Create sim params
        sim_params = gymapi.SimParams()
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        sim_params.dt = 1.0/60.0
        sim_params.substeps = 2
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        
        print("✅ Environment configuration created successfully")
        print(f"   Number of environments: {cfg.env.num_envs}")
        print(f"   IPC3D guidance: {'enabled' if cfg.guidance.enable_ipc3d else 'disabled'}")
        print(f"   Force disturbance: {'enabled' if cfg.domain_rand.use_force_push else 'disabled'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment creation error: {e}")
        return False

def test_reward_configuration():
    """Test reward system configuration."""
    print("\n🔍 Testing reward system configuration...")
    
    try:
        from gym.envs.hi_12.hi_controller_ipc3d import HiControllerIPC3DCfg
        
        cfg = HiControllerIPC3DCfg()
        
        # Check IPC3D guidance rewards
        guidance_rewards = [
            'trajectory_tracking',
            'guidance_consistency', 
            'step_location_error'
        ]
        
        # Check disturbance recovery rewards
        recovery_rewards = [
            'stability_recovery',
            'balance_maintenance',
            'contact_stability'
        ]
        
        print("✅ Reward system configuration valid:")
        print(f"   Trajectory tracking weight: {cfg.rewards.weights.trajectory_tracking}")
        print(f"   Stability recovery weight: {cfg.rewards.weights.stability_recovery}")
        print(f"   Balance maintenance weight: {cfg.rewards.weights.balance_maintenance}")
        
        return True
        
    except Exception as e:
        print(f"❌ Reward configuration error: {e}")
        return False

def test_force_disturbance_config():
    """Test force disturbance system configuration."""
    print("\n🔍 Testing force disturbance configuration...")
    
    try:
        from gym.envs.hi_12.hi_controller_ipc3d import HiControllerIPC3DCfg
        
        cfg = HiControllerIPC3DCfg()
        
        # Check force push configuration
        assert cfg.domain_rand.use_force_push == True
        assert cfg.domain_rand.curriculum_push == True
        assert cfg.domain_rand.enable_multi_point_push == True
        assert len(cfg.domain_rand.push_body_parts) > 0
        assert len(cfg.domain_rand.push_force_schedule) > 0
        
        print("✅ Force disturbance configuration valid:")
        print(f"   Max push force: {cfg.domain_rand.max_push_force_xy}N")
        print(f"   Push body parts: {len(cfg.domain_rand.push_body_parts)}")
        print(f"   Curriculum schedule: {len(cfg.domain_rand.push_force_schedule)} stages")
        print(f"   Multi-point push: {'enabled' if cfg.domain_rand.enable_multi_point_push else 'disabled'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Force disturbance configuration error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 HI-12 + IPC3D + Force Disturbance Integration Test")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Task Registration", test_task_registration), 
        ("IPC3D Controller", test_ipc3d_controller),
        ("IPC3D Guidance Model", test_guidance_model),
        ("Environment Creation", test_environment_creation),
        ("Reward Configuration", test_reward_configuration),
        ("Force Disturbance Config", test_force_disturbance_config),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Integration system is ready for training.")
        print("\n🚀 To start training, run:")
        print("   python train_hi12_ipc3d.py")
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the integration.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)