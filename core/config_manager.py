"""
Configuration Manager for Chest X-Ray Classification Pipeline
Handles loading, saving, and updating configuration files
"""

import yaml
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import copy


class ConfigManager:
    """Manages configuration files for the training pipeline"""

    def __init__(self, config_path: str = "config_baseline.yaml"):
        """
        Initialize config manager

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self.load_config(config_path)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def save_config(self, config: Dict[str, Any], output_path: str):
        """
        Save configuration to YAML file

        Args:
            config: Configuration dictionary
            output_path: Path to save config
        """
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def create_new_config(self, changes: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """
        Create a new configuration based on suggested changes

        Args:
            changes: Dictionary of changes to apply
            iteration: Iteration number

        Returns:
            New configuration dictionary
        """
        # Deep copy the current config
        new_config = copy.deepcopy(self.config)

        # Update metadata
        new_config['metadata']['iteration'] = iteration
        new_config['metadata']['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_config['metadata']['parent_config'] = self.config_path

        # Apply changes
        self._apply_changes(new_config, changes)

        return new_config

    def _apply_changes(self, config: Dict[str, Any], changes: Dict[str, Any], prefix: str = ""):
        """
        Recursively apply changes to configuration

        Args:
            config: Configuration dictionary to update
            changes: Changes to apply
            prefix: Key prefix for nested dictionaries
        """
        for key, value in changes.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                # Recursively update nested dictionaries
                self._apply_changes(config[key], value, f"{prefix}{key}.")
            else:
                # Update value
                config[key] = value

    def save_iteration_config(self, iteration: int, config: Optional[Dict[str, Any]] = None):
        """
        Save configuration for a specific iteration

        Args:
            iteration: Iteration number
            config: Configuration to save (uses current config if None)
        """
        if config is None:
            config = self.config

        output_path = f"config_iteration_{iteration:03d}.yaml"
        self.save_config(config, output_path)
        return output_path

    def get_config_summary(self) -> str:
        """
        Get a human-readable summary of the current configuration

        Returns:
            Configuration summary as string
        """
        summary = []
        summary.append("=" * 60)
        summary.append("CONFIGURATION SUMMARY")
        summary.append("=" * 60)

        # Metadata
        meta = self.config.get('metadata', {})
        summary.append(f"Version: {meta.get('config_version', 'N/A')}")
        summary.append(f"Iteration: {meta.get('iteration', 0)}")
        summary.append(f"Description: {meta.get('description', 'N/A')}")
        summary.append("")

        # Model
        model = self.config.get('model', {})
        summary.append("MODEL:")
        summary.append(f"  Architecture: {model.get('architecture', 'N/A')}")
        summary.append(f"  Dropout: {model.get('dropout_rate', 'N/A')}")
        summary.append("")

        # Training
        training = self.config.get('training', {})
        summary.append("TRAINING:")
        summary.append(f"  Batch Size: {training.get('batch_size', 'N/A')}")
        summary.append(f"  Learning Rate: {training.get('learning_rate', 'N/A')}")
        summary.append(f"  Epochs: {training.get('num_epochs', 'N/A')}")
        summary.append(f"  Optimizer: {training.get('optimizer', {}).get('type', 'N/A')}")
        summary.append("")

        # Loss
        loss = self.config.get('loss', {})
        summary.append("LOSS:")
        summary.append(f"  Type: {loss.get('type', 'N/A')}")
        summary.append(f"  Gamma: {loss.get('gamma', 'N/A')}")
        summary.append("")

        summary.append("=" * 60)

        return "\n".join(summary)

    def get_config_dict(self) -> Dict[str, Any]:
        """Get the current configuration dictionary"""
        return self.config

    def update_config(self, new_config: Dict[str, Any]):
        """Update the current configuration"""
        self.config = new_config

    def export_for_llm(self) -> str:
        """
        Export configuration in a format suitable for LLM analysis

        Returns:
            JSON string of configuration
        """
        return json.dumps(self.config, indent=2)

    @staticmethod
    def create_baseline_config(output_path: str = "config_baseline.yaml"):
        """
        Create a baseline configuration file
        This is useful for initializing the pipeline
        """
        # This would contain the default config structure
        # For now, we'll assume the baseline config already exists
        pass


if __name__ == "__main__":
    # Example usage
    manager = ConfigManager("config_baseline.yaml")
    print(manager.get_config_summary())

    # Example: Create a new config with changes
    changes = {
        "training": {
            "learning_rate": 0.0001,
            "batch_size": 32
        }
    }

    new_config = manager.create_new_config(changes, iteration=1)
    manager.save_config(new_config, "config_iteration_001.yaml")
    print("\nNew config created with changes:")
    print(f"  Learning Rate: {new_config['training']['learning_rate']}")
    print(f"  Batch Size: {new_config['training']['batch_size']}")