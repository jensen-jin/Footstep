"""
Unified training interface for robot models with guidance selection.

This module provides a high-level interface for training robots with
different guidance models (LIPM/IPC3D) in a standardized way.
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import os
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from .factory import RobotConfigFactory
from .registry import RobotRegistry


class UnifiedTrainer:
    """
    Unified trainer for robots with configurable guidance models.
    
    This class provides a standardized interface for training different
    robot models with different guidance systems.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize unified trainer.
        
        Args:
            config_path: Optional path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config() if config_path else {}
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"📋 Loaded config from: {self.config_path}")
            return config
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return {}
    
    def train_single_robot(
        self,
        robot_name: str,
        guidance_model: str = 'limp',
        experiment_name: Optional[str] = None,
        **training_kwargs
    ):
        """
        Train a single robot with specified guidance model.
        
        Args:
            robot_name: Name of robot to train
            guidance_model: Type of guidance model ('limp' or 'ipc3d')
            experiment_name: Optional experiment name for logging
            **training_kwargs: Additional training parameters
        """
        print(f"🚀 Starting training:")
        print(f"   Robot: {robot_name}")
        print(f"   Guidance: {guidance_model}")
        print(f"   Experiment: {experiment_name or 'default'}")
        
        # Create robot configuration
        try:
            config = RobotConfigFactory.create_config(
                robot_name=robot_name,
                guidance_model=guidance_model,
                **training_kwargs.get('config_overrides', {})
            )
        except Exception as e:
            print(f"❌ Failed to create config: {e}")
            return
        
        # Set experiment name
        if experiment_name:
            if hasattr(config, 'runner'):
                config.runner.experiment_name = experiment_name
                config.runner.run_name = f"{robot_name}_{guidance_model}"
        
        # Import and run training (this would normally import the actual training code)
        print("💡 Training would start here with the generated config")
        print(f"   Config type: {type(config)}")
        print(f"   Guidance config: {getattr(config, 'guidance', None)}")
        
        # TODO: Integrate with actual training pipeline
        # from gym.scripts.train import train_robot
        # train_robot(config)
        
        return config
    
    def run_comparison_study(
        self,
        robots: List[str],
        guidance_models: List[str],
        base_experiment_name: str = "comparison_study"
    ):
        """
        Run a comparison study across multiple robots and guidance models.
        
        Args:
            robots: List of robot names to test
            guidance_models: List of guidance models to test
            base_experiment_name: Base name for experiments
        """
        print(f"🔬 Starting comparison study: {base_experiment_name}")
        print(f"   Robots: {robots}")
        print(f"   Guidance models: {guidance_models}")
        
        results = {}
        
        for robot in robots:
            for guidance in guidance_models:
                experiment_name = f"{base_experiment_name}_{robot}_{guidance}"
                
                try:
                    config = self.train_single_robot(
                        robot_name=robot,
                        guidance_model=guidance,
                        experiment_name=experiment_name
                    )
                    results[f"{robot}_{guidance}"] = {
                        'status': 'success',
                        'config': config
                    }
                except Exception as e:
                    print(f"❌ Failed training {robot} with {guidance}: {e}")
                    results[f"{robot}_{guidance}"] = {
                        'status': 'failed',
                        'error': str(e)
                    }
        
        self._save_comparison_results(results, base_experiment_name)
        return results
    
    def _save_comparison_results(self, results: Dict, experiment_name: str):
        """Save comparison study results."""
        results_dir = Path("comparison_results")
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / f"{experiment_name}_results.yaml"
        
        # Convert results to serializable format
        serializable_results = {}
        for key, value in results.items():
            if value['status'] == 'success':
                serializable_results[key] = {
                    'status': value['status'],
                    'config_type': str(type(value['config']))
                }
            else:
                serializable_results[key] = value
        
        with open(results_file, 'w') as f:
            yaml.dump(serializable_results, f, default_flow_style=False)
        
        print(f"💾 Saved comparison results to: {results_file}")
    
    def run_batch_experiments(self):
        """Run batch experiments from configuration file."""
        if 'experiments' not in self.config:
            print("❌ No experiments defined in config file")
            return
        
        experiments = self.config['experiments']
        print(f"🔬 Running {len(experiments)} batch experiments")
        
        for i, exp_config in enumerate(experiments):
            print(f"\n--- Experiment {i+1}/{len(experiments)} ---")
            
            robot_name = exp_config.get('robot')
            guidance_model = exp_config.get('guidance', 'limp')
            experiment_name = exp_config.get('name', f"batch_exp_{i+1}")
            overrides = exp_config.get('overrides', {})
            
            if not robot_name:
                print(f"❌ No robot specified for experiment {i+1}")
                continue
            
            try:
                self.train_single_robot(
                    robot_name=robot_name,
                    guidance_model=guidance_model,
                    experiment_name=experiment_name,
                    config_overrides=overrides
                )
            except Exception as e:
                print(f"❌ Experiment {i+1} failed: {e}")
    
    @staticmethod
    def list_available_robots():
        """List all available robots."""
        print("🤖 Available robots:")
        robots_info = RobotConfigFactory.list_available_robots()
        for name, description in robots_info.items():
            print(f"   {name}: {description}")
    
    @staticmethod
    def get_robot_info(robot_name: str):
        """Get detailed information about a specific robot."""
        try:
            info = RobotConfigFactory.get_robot_details(robot_name)
            print(f"🔍 Robot: {robot_name}")
            for key, value in info.items():
                if key != 'name':
                    print(f"   {key}: {value}")
        except Exception as e:
            print(f"❌ Robot '{robot_name}' not found: {e}")


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line interface parser."""
    parser = argparse.ArgumentParser(
        description="Unified robot training with configurable guidance models"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a single robot')
    train_parser.add_argument('--robot', required=True, help='Robot name')
    train_parser.add_argument('--guidance', default='limp', 
                             choices=['limp', 'ipc3d'], help='Guidance model')
    train_parser.add_argument('--experiment', help='Experiment name')
    train_parser.add_argument('--config', help='YAML config file')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Run batch experiments')
    batch_parser.add_argument('--config', required=True, help='YAML config file')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Run comparison study')
    compare_parser.add_argument('--robots', nargs='+', required=True, help='Robot names')
    compare_parser.add_argument('--guidance', nargs='+', default=['limp', 'ipc3d'],
                               help='Guidance models')
    compare_parser.add_argument('--name', default='comparison', help='Study name')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available robots')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get robot information')
    info_parser.add_argument('robot', help='Robot name')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    if args.command == 'train':
        trainer = UnifiedTrainer(args.config)
        trainer.train_single_robot(
            robot_name=args.robot,
            guidance_model=args.guidance,
            experiment_name=args.experiment
        )
    
    elif args.command == 'batch':
        trainer = UnifiedTrainer(args.config)
        trainer.run_batch_experiments()
    
    elif args.command == 'compare':
        trainer = UnifiedTrainer()
        trainer.run_comparison_study(
            robots=args.robots,
            guidance_models=args.guidance,
            base_experiment_name=args.name
        )
    
    elif args.command == 'list':
        UnifiedTrainer.list_available_robots()
    
    elif args.command == 'info':
        UnifiedTrainer.get_robot_info(args.robot)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()