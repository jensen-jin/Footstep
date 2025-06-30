#!/usr/bin/env python3
"""
Test script for IPC3D implementation.

This script tests the IPC3D controller and guidance model implementation.
"""

import sys
import numpy as np
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import IPC3D components
from gym.envs.guidance.ipc3d_controller import IPC3D, IPC3DParams, IPCView
from gym.envs.guidance.ipc3d_guidance import IPC3DGuidanceModel
from gym.envs.guidance.guidance_factory import GuidanceModelFactory


def test_ipc_controller():
    """Test basic IPC controller functionality."""
    print("🧪 Testing IPC Controller")
    print("=" * 40)
    
    # Create single-axis IPC controller
    controller = IPCView(
        M=40.0,    # Cart mass (kg)
        m=5.0,     # Pole mass (kg) 
        b=0.1,     # Damping
        I=0.1,     # Inertia
        g=9.81,    # Gravity
        l=0.6,     # Pole length
        dt=0.02,   # Time step
        q=10.0,    # LQR weight
        qd=1.0     # LQR velocity weight
    )
    
    print(f"✅ IPC controller created")
    print(f"   Control mode: {controller.mode} (1=velocity)")
    print(f"   Pole length: {controller.l:.2f} m")
    print(f"   Max force: {controller.max_force:.0f} N")
    
    # Set target velocity
    target_velocity = 1.0  # m/s
    controller.set_param(target_velocity)
    print(f"   Target velocity: {target_velocity} m/s")
    
    # Run simulation
    print(f"\n🏃 Running simulation...")
    results = []
    
    for i in range(100):
        ddtheta = controller.one_step()
        
        if i % 20 == 0:
            state = controller.x
            control = controller.U[0]
            print(f"   Step {i:3d}: pos={state[0]:.3f}m, vel={state[2]:.3f}m/s, F={control:.1f}N")
            
        results.append({
            'step': i,
            'position': controller.x[0],
            'velocity': controller.x[2],
            'control': controller.U[0]
        })
    
    # Check convergence
    final_velocity = controller.x[2]
    velocity_error = abs(final_velocity - target_velocity)
    
    print(f"\n📊 Results:")
    print(f"   Target velocity: {target_velocity:.3f} m/s")
    print(f"   Final velocity: {final_velocity:.3f} m/s")
    print(f"   Velocity error: {velocity_error:.3f} m/s")
    
    success = velocity_error < 0.1
    print(f"   {'✅ PASS' if success else '❌ FAIL'}")
    
    return success


def test_ipc3d_controller():
    """Test 3D IPC controller."""
    print(f"\n🚗 Testing IPC3D Controller")
    print("=" * 40)
    
    # Create IPC3D parameters
    params = IPC3DParams(
        mass_cart=42.0,
        mass_pole=5.0,
        pole_length=0.62,
        inertia=0.1,
        gravity=9.81,
        damping=0.1,
        dt=0.02,
        max_force=500.0,
        q_position=10.0,
        q_velocity=1.0,
        r_control=0.1,
        control_mode=1  # Velocity control
    )
    
    # Create controller
    controller = IPC3D(params)
    print(f"✅ IPC3D controller created")
    
    # Set target velocities
    target_x = 1.0  # m/s forward
    target_z = 0.5  # m/s lateral
    controller.set_desired_velocity(target_x, target_z)
    print(f"   Target: vx={target_x:.1f}, vz={target_z:.1f} m/s")
    
    # Run simulation
    print(f"\n🏃 Running 3D simulation...")
    
    for i in range(100):
        result = controller.step()
        
        if i % 20 == 0:
            state = controller.get_state()
            fx, fz = controller.get_control_forces()
            
            print(f"   Step {i:3d}: vx={state['x_cart_vel']:.3f}, vz={state['z_cart_vel']:.3f} m/s")
            print(f"            Fx={fx:.1f}, Fz={fz:.1f} N")
    
    # Check final state
    final_state = controller.get_state()
    vx_error = abs(final_state['x_cart_vel'] - target_x)
    vz_error = abs(final_state['z_cart_vel'] - target_z)
    
    print(f"\n📊 3D Results:")
    print(f"   Target: vx={target_x:.3f}, vz={target_z:.3f} m/s")
    print(f"   Final:  vx={final_state['x_cart_vel']:.3f}, vz={final_state['z_cart_vel']:.3f} m/s")
    print(f"   Error:  ex={vx_error:.3f}, ez={vz_error:.3f} m/s")
    
    success = vx_error < 0.1 and vz_error < 0.1
    print(f"   {'✅ PASS' if success else '❌ FAIL'}")
    
    return success


def test_ipc3d_guidance_model():
    """Test IPC3D guidance model."""
    print(f"\n🎯 Testing IPC3D Guidance Model")
    print("=" * 40)
    
    # Create parameters
    params = IPC3DParams(
        mass_cart=42.0,
        mass_pole=5.0,
        pole_length=0.62,
        dt=0.02
    )
    
    # Create guidance model
    guidance = IPC3DGuidanceModel(params)
    print(f"✅ IPC3D guidance model created")
    
    # Initialize with robot state
    robot_state = {
        'position': [0.0, 0.0, 0.0],
        'velocity': [0.0, 0.0, 0.0],
        'orientation': [1.0, 0.0, 0.0, 0.0]
    }
    
    success = guidance.initialize(robot_state)
    print(f"   Initialization: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    # Set target
    target = {
        'velocity_x': 1.5,
        'velocity_z': 0.3,
        'step_length': 0.6
    }
    guidance.set_target(target)
    print(f"   Target set: vx={target['velocity_x']}, vz={target['velocity_z']} m/s")
    
    # Update guidance model
    print(f"\n🔄 Testing guidance updates...")
    
    for i in range(50):
        # Simulate robot state evolution
        robot_state['position'][0] += robot_state['velocity'][0] * guidance.dt
        robot_state['position'][2] += robot_state['velocity'][2] * guidance.dt
        
        # Update guidance
        control_output = guidance.update(robot_state)
        
        # Apply simple integration (for testing)
        forces = control_output['forces']
        robot_state['velocity'][0] += forces[0] * guidance.dt / 50.0  # Simplified dynamics
        robot_state['velocity'][2] += forces[2] * guidance.dt / 50.0
        
        if i % 10 == 0:
            forces = control_output['forces']
            ref_vel = control_output['reference_velocity']
            print(f"   Step {i:2d}: vel=({robot_state['velocity'][0]:.2f}, {robot_state['velocity'][2]:.2f})")
            print(f"           force=({forces[0]:.1f}, {forces[2]:.1f}) N")
    
    # Generate trajectory  
    print(f"\n📈 Testing trajectory generation...")
    
    trajectory = guidance.generate_trajectory(duration=2.0, target_params=target)
    print(f"   Trajectory generated: {trajectory.num_steps} steps")
    print(f"   Duration: {trajectory.timestamps[-1]:.2f} s")
    print(f"   Valid: {'✅ YES' if trajectory.is_valid else '❌ NO'}")
    
    # Test control reference
    ref_t1 = guidance.get_control_reference(1.0)
    print(f"   Reference at t=1.0s: pos=({ref_t1['position'][0]:.2f}, {ref_t1['position'][2]:.2f})")
    
    return success and trajectory.is_valid


def test_guidance_factory():
    """Test guidance model factory."""
    print(f"\n🏭 Testing Guidance Model Factory")
    print("=" * 40)
    
    # Test available models
    available = GuidanceModelFactory.list_available_models()
    print(f"   Available models: {available}")
    
    # Test config-based creation
    config = {
        'model_type': 'ipc3d',
        'robot_mass': 45.0,
        'robot_height': 0.65,
        'dt': 0.02,
        'ipc3d_max_force': 600.0
    }
    
    try:
        guidance = GuidanceModelFactory.create_from_config_dict(config)
        print(f"   ✅ IPC3D model created from config")
        print(f"      Model type: {guidance.model_type}")
        
        # Test with placeholder model
        config['model_type'] = 'placeholder'
        placeholder = GuidanceModelFactory.create_from_config_dict(config)
        print(f"   ✅ Placeholder model created")
        print(f"      Model type: {placeholder.model_type}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Factory test failed: {e}")
        return False


def test_integration_with_registry():
    """Test integration with robot registry."""
    print(f"\n🤖 Testing Integration with Robot Registry")
    print("=" * 40)
    
    try:
        # Mock robot registry components
        from gym.envs.robots.factory import RobotConfigFactory, GuidanceConfig
        
        # Test guidance config creation
        guidance_config = GuidanceConfig(
            model_type='ipc3d',
            ipc3d_mass_cart=40.0,
            ipc3d_pole_length=0.6,
            lqr_q_matrix=[15.0, 2.0],
            lqr_r_matrix=[0.05]
        )
        
        print(f"   ✅ Guidance config created")
        print(f"      Model: {guidance_config.model_type}")
        print(f"      Cart mass: {guidance_config.ipc3d_mass_cart} kg")
        print(f"      LQR Q: {guidance_config.lqr_q_matrix}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 IPC3D Implementation Test Suite")
    print("=" * 60)
    
    test_results = []
    
    try:
        # Run individual tests
        test_results.append(("IPC Controller", test_ipc_controller()))
        test_results.append(("IPC3D Controller", test_ipc3d_controller())) 
        test_results.append(("IPC3D Guidance Model", test_ipc3d_guidance_model()))
        test_results.append(("Guidance Factory", test_guidance_factory()))
        test_results.append(("Registry Integration", test_integration_with_registry()))
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Test Results Summary")
        print("=" * 60)
        
        passed = 0
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name:<25} {status}")
            if result:
                passed += 1
        
        success_rate = passed / len(test_results) * 100
        print(f"\n   Success Rate: {success_rate:.1f}% ({passed}/{len(test_results)})")
        
        if passed == len(test_results):
            print("\n🎉 All tests passed! IPC3D implementation is working correctly.")
            print("\n💡 Ready for:")
            print("   • Integration with Isaac Gym training")
            print("   • Robot registry configuration") 
            print("   • Unified training experiments")
            print("   • Performance optimization")
        else:
            print(f"\n⚠️  {len(test_results) - passed} test(s) failed. Please review implementation.")
            
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()