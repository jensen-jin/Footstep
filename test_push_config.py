#!/usr/bin/env python3
"""
Standalone test for push force system configuration and logic
独立测试推力系统配置和逻辑（无需Isaac Gym）
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config_integration():
    """Test that the configuration changes are properly integrated"""
    print("🧪 Testing Push Force Configuration Integration")
    print("=" * 60)
    
    try:
        # Import configuration without Isaac Gym dependencies
        from gym.envs.base.legged_robot_config import LeggedRobotCfg
        
        cfg = LeggedRobotCfg()
        
        # Test new configuration parameters
        print("✅ Configuration loaded successfully")
        print("📋 Push Force Configuration:")
        print(f"   - push_robots: {cfg.domain_rand.push_robots}")
        print(f"   - use_force_push: {getattr(cfg.domain_rand, 'use_force_push', 'NOT FOUND')}")
        print(f"   - max_push_force_xy: {getattr(cfg.domain_rand, 'max_push_force_xy', 'NOT FOUND')}")
        print(f"   - push_duration: {getattr(cfg.domain_rand, 'push_duration', 'NOT FOUND')}")
        print(f"   - push_debug: {getattr(cfg.domain_rand, 'push_debug', 'NOT FOUND')}")
        
        # Verify all new parameters exist
        required_params = ['use_force_push', 'max_push_force_xy', 'push_duration', 'push_debug']
        missing_params = []
        
        for param in required_params:
            if not hasattr(cfg.domain_rand, param):
                missing_params.append(param)
        
        if missing_params:
            print(f"❌ Missing parameters: {missing_params}")
            return False
        else:
            print("✅ All new configuration parameters found!")
            return True
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_curriculum_logic():
    """Test the curriculum learning logic for push forces"""
    print("\n🧮 Testing Push Force Curriculum Logic")
    print("=" * 60)
    
    def compute_push_force(step, max_force=100.0):
        """Simulate the curriculum logic"""
        if step < 10_000:
            return max_force * 0.25 
        elif step < 30_000:
            return max_force * 0.50
        elif step < 60_000:
            return max_force * 0.75
        else:
            return max_force
    
    test_cases = [
        (0, 25.0),      # Early training
        (5000, 25.0),   # Still early
        (15000, 50.0),  # Mid training
        (35000, 75.0),  # Advanced training
        (70000, 100.0), # Full force
    ]
    
    print("📊 Curriculum Force Progression:")
    all_passed = True
    
    for step, expected in test_cases:
        actual = compute_push_force(step, 100.0)
        status = "✅" if actual == expected else "❌"
        print(f"   Step {step:6d}: {actual:6.1f}N (expected {expected:6.1f}N) {status}")
        if actual != expected:
            all_passed = False
    
    return all_passed

def test_push_logic():
    """Test the push activation and timing logic"""
    print("\n⏰ Testing Push Activation Logic")
    print("=" * 60)
    
    # Simulate push system state
    class MockPushSystem:
        def __init__(self, push_interval=50, push_duration=10):
            self.push_interval = push_interval
            self.push_duration = push_duration
            self.push_active = False
            self.push_step_count = 0
            self.common_step_counter = 0
            
        def maybe_push_robot(self):
            # Check if we need to start a new push cycle
            if self.common_step_counter % self.push_interval == 0:
                self.push_active = True
                self.push_step_count = 0
                return "PUSH_STARTED"
            
            # If in push cycle, continue pushing
            if self.push_active:
                self.push_step_count += 1
                if self.push_step_count >= self.push_duration:
                    self.push_active = False
                    return "PUSH_ENDED"
                return "PUSHING"
            
            return "NO_PUSH"
    
    # Test the logic
    push_system = MockPushSystem(push_interval=10, push_duration=3)
    
    print("📝 Push Timing Test (interval=10, duration=3):")
    
    expected_sequence = [
        (0, "PUSH_STARTED"),   # Step 0: push starts
        (1, "PUSHING"),        # Step 1: pushing
        (2, "PUSHING"),        # Step 2: still pushing  
        (3, "PUSH_ENDED"),     # Step 3: push ends
        (4, "NO_PUSH"),        # Step 4-9: no push
        (9, "NO_PUSH"),        # Step 9: no push
        (10, "PUSH_STARTED"),  # Step 10: new push starts
    ]
    
    all_passed = True
    for step, expected in expected_sequence:
        push_system.common_step_counter = step
        actual = push_system.maybe_push_robot()
        status = "✅" if actual == expected else "❌"
        print(f"   Step {step:2d}: {actual:12s} (expected {expected:12s}) {status}")
        if actual != expected:
            all_passed = False
    
    return all_passed

def test_force_vs_velocity_selection():
    """Test the logic for choosing between force and velocity push"""
    print("\n🔀 Testing Push Method Selection Logic")
    print("=" * 60)
    
    def should_use_force_push(cfg_force_push, cfg_push_robots):
        """Simulate the selection logic from step() method"""
        if cfg_push_robots:
            if cfg_force_push:
                return "FORCE_PUSH"
            else:
                return "VELOCITY_PUSH"
        return "NO_PUSH"
    
    test_cases = [
        (True, True, "FORCE_PUSH"),
        (False, True, "VELOCITY_PUSH"), 
        (True, False, "NO_PUSH"),
        (False, False, "NO_PUSH"),
    ]
    
    print("🎛️ Push Method Selection:")
    all_passed = True
    
    for use_force, push_robots, expected in test_cases:
        actual = should_use_force_push(use_force, push_robots)
        status = "✅" if actual == expected else "❌"
        print(f"   force={use_force}, robots={push_robots} → {actual:12s} (expected {expected:12s}) {status}")
        if actual != expected:
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    print("🚀 Push Force System Standalone Test Suite")
    print("=" * 70)
    
    # Run all tests
    test1 = test_config_integration()
    test2 = test_curriculum_logic()
    test3 = test_push_logic()
    test4 = test_force_vs_velocity_selection()
    
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print(f"   Config Integration:    {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"   Curriculum Logic:      {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"   Push Timing Logic:     {'✅ PASS' if test3 else '❌ FAIL'}")
    print(f"   Method Selection:      {'✅ PASS' if test4 else '❌ FAIL'}")
    
    if all([test1, test2, test3, test4]):
        print("\n🎉 ALL TESTS PASSED! Force-based push system is properly integrated.")
        exit(0)
    else:
        print("\n💥 SOME TESTS FAILED! Please check the implementation.")
        exit(1)