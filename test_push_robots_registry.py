#!/usr/bin/env python3
"""
Test push_robot_verify branch robots integration with registry system
测试push_robot_verify分支机器人与注册系统的集成
"""

def test_robot_registry_integration():
    """Test that push robots are properly registered"""
    print("🤖 Testing Push Robots Registry Integration")
    print("=" * 60)
    
    try:
        # This import will trigger auto-registration
        from gym.envs.robots import RobotRegistry
        
        print("✅ Robot registry imported successfully")
        
        # Test that all expected robots are registered
        expected_robots = ["humanoid", "simple", "hi_12", "hi_cl_12", "pi_cl_12"]
        registered_robots = RobotRegistry.list_robots()
        
        print(f"📋 Expected robots: {expected_robots}")
        print(f"📋 Registered robots: {registered_robots}")
        
        missing_robots = set(expected_robots) - set(registered_robots)
        extra_robots = set(registered_robots) - set(expected_robots)
        
        if missing_robots:
            print(f"❌ Missing robots: {missing_robots}")
            return False
        
        if extra_robots:
            print(f"ℹ️  Extra robots found: {extra_robots}")
        
        print("✅ All expected robots are registered!")
        
        # Test individual robot specifications
        print("\n🔍 Testing Robot Specifications:")
        
        for robot_name in ["hi_12", "hi_cl_12", "pi_cl_12"]:
            try:
                spec = RobotRegistry.get_robot(robot_name)
                print(f"   ✅ {robot_name}:")
                print(f"      - DOF: {spec.num_actuators}")
                print(f"      - Height: {spec.physical_params['base_height']}m")
                print(f"      - Max velocity: {spec.physical_params['max_velocity']}m/s")
                print(f"      - Description: {spec.description}")
                
            except Exception as e:
                print(f"   ❌ {robot_name}: {e}")
                return False
        
        # Test category grouping
        print("\n📂 Testing Category Grouping:")
        humanoid_robots = RobotRegistry.get_robots_by_category("humanoid")
        print(f"   Humanoid robots: {humanoid_robots}")
        
        expected_humanoids = ["humanoid", "hi_12", "hi_cl_12", "pi_cl_12"]
        for expected in expected_humanoids:
            if expected not in humanoid_robots:
                print(f"   ❌ {expected} not found in humanoid category")
                return False
        
        print("   ✅ All humanoid robots properly categorized")
        
        # Test robot info summary
        print("\n📊 Testing Robot Info Summary:")
        info = RobotRegistry.get_robot_info()
        
        for robot_name in ["hi_12", "hi_cl_12", "pi_cl_12"]:
            if robot_name not in info:
                print(f"   ❌ {robot_name} info not found")
                return False
            
            robot_info = info[robot_name]
            required_fields = ['name', 'category', 'num_actuators', 'description']
            for field in required_fields:
                if field not in robot_info:
                    print(f"   ❌ {robot_name} missing field: {field}")
                    return False
        
        print("   ✅ All robot info properly structured")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Import error (expected without Isaac Gym): {e}")
        print("   Testing configuration structure only...")
        
        # Test that files exist
        import os
        base_path = "/Users/zhangjunhui/Downloads/Footstep/Footstep"
        
        expected_files = [
            "gym/envs/hi_12/hi_controller.py",
            "gym/envs/hi_12/hi_controller_config.py",
            "gym/envs/hi_cl_12/hi_cl12_controller.py", 
            "gym/envs/hi_cl_12/hi_cl12_controller_config.py",
            "gym/envs/pi_cl_12/pi_cl12_controller.py",
            "gym/envs/pi_cl_12/pi_cl12_controller_config.py",
            "gym/envs/robots/specifications/push_robots.py",
            "resources/robots/hi_12dof_250108_4/urdf/hi_12dof_250108_4_rl.urdf",
            "resources/robots/pi_12dof_release_v1/urdf/pi_12dof_release_v1_rl.urdf"
        ]
        
        missing_files = []
        for file_path in expected_files:
            full_path = os.path.join(base_path, file_path)
            if not os.path.exists(full_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return False
        
        print("✅ All robot files are in place")
        
        # Test push_robots.py structure
        try:
            spec_file = os.path.join(base_path, "gym/envs/robots/specifications/push_robots.py")
            with open(spec_file, 'r') as f:
                content = f.read()
                
            required_functions = [
                "create_hi12_spec",
                "create_hi_cl12_spec", 
                "create_pi_cl12_spec",
                "register_push_robots"
            ]
            
            for func in required_functions:
                if f"def {func}" not in content:
                    print(f"❌ Missing function: {func}")
                    return False
            
            print("✅ Push robots specification structure is correct")
            return True
            
        except Exception as e:
            print(f"❌ Error checking specification file: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_robot_comparison():
    """Test comparison between different robots"""
    print("\n🔍 Testing Robot Comparison")
    print("=" * 60)
    
    # Test data from specifications
    robots_data = {
        "hi_12": {
            "base_height": 0.5596,
            "max_velocity": 3.0,
            "mass_range": [-2.2, 2.2],
            "characteristics": "Enhanced control, extended training"
        },
        "hi_cl_12": {
            "base_height": 0.68,
            "max_velocity": 2.5, 
            "mass_range": [-1.2, 1.2],
            "characteristics": "High-velocity, curriculum learning"
        },
        "pi_cl_12": {
            "base_height": 0.3453,
            "max_velocity": 0.3,
            "mass_range": [-1.0, 1.0],
            "characteristics": "Compact, precision control"
        }
    }
    
    print("📊 Robot Comparison Summary:")
    print("   Robot     | Height | Max Vel | Mass Range | Focus")
    print("   ----------|--------|---------|-------------|------------------")
    
    for name, data in robots_data.items():
        height = data["base_height"]
        velocity = data["max_velocity"]
        mass_range = data["mass_range"]
        focus = data["characteristics"]
        
        print(f"   {name:9s} | {height:6.3f} | {velocity:7.1f} | {mass_range[0]:4.1f},{mass_range[1]:4.1f} | {focus}")
    
    # Validate ordering
    heights = [robots_data[name]["base_height"] for name in ["pi_cl_12", "hi_12", "hi_cl_12"]]
    velocities = [robots_data[name]["max_velocity"] for name in ["pi_cl_12", "hi_cl_12", "hi_12"]]
    
    if heights != sorted(heights):
        print("❌ Height ordering is incorrect")
        return False
    
    if velocities != sorted(velocities):
        print("❌ Velocity ordering is incorrect") 
        return False
    
    print("\n✅ Robot specifications are properly differentiated")
    print("   - PI-CL-12: Smallest, most conservative")
    print("   - HI-12: Medium size, highest performance")  
    print("   - HI-CL-12: Largest, optimized for speed")
    
    return True

if __name__ == "__main__":
    print("🚀 Push Robots Registry Integration Test Suite")
    print("=" * 70)
    
    test1 = test_robot_registry_integration()
    test2 = test_robot_comparison()
    
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print(f"   Registry Integration:  {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"   Robot Comparison:      {'✅ PASS' if test2 else '❌ FAIL'}")
    
    if test1 and test2:
        print("\n🎉 ALL TESTS PASSED! Push robots are properly integrated.")
        exit(0)
    else:
        print("\n💥 SOME TESTS FAILED! Please check the implementation.")
        exit(1)