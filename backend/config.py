import yaml, pathlib, os
from typing import Dict, Any, Optional, List, Union

class ConfigManager:
    """
    Central configuration manager for the DeepResearch Pipeline.
    
    This class handles loading configuration from files, validating configuration values,
    and providing access to configuration settings and paths throughout the application.
    """
    
    # Default configuration values
    DEFAULT_CONFIG = {
        "llm": {
            "mode": "local",
            "base_url": "http://localhost:11434",
            "model_id": "mistral"
        },
        "agent": {
            "mode": "code",
            "system_prompt": "You are an expert research assistant for a private corpus."
        },
        "reformulation": {
            "enabled": True,
            "model": "mistral",
            "temperature": 0.3
        },
        "memory": {
            "max_snippets": 40,
            "track_reflections": True,
            "reflection_interval": 5,
            "reflection_timeout": 120
        },
        "slice": {
            "default_chars": 750,
            "default_mode": "auto"
        },
        "chunking": {
            "mode": "semantic",
            "semantic": {
                "embeddings_model": "intfloat/multilingual-e5-large-instruct",
                "breakpoint_threshold": 0.6
            },
            "recursive": {
                "chunk_size": 1000,
                "chunk_overlap": 200
            }
        },
        "limits": {
            "max_tool_calls": 60,
            "max_tokens_out": 4096
        },
        "export": {
            "pdf": True
        },
        "paths": {
            "config": "configs/pipeline.yaml",
            "docs": "docs",
            "indexing": {
                "root": "indexing",
                "whoosh": "indexing/whoosh",
                "faiss": "indexing/faiss.index",
                "meta": "indexing/meta.json",
                "map": "indexing/map.json",
                "graph": "indexing/navigator.gpkl"
            }
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses the default path.
        """
        self.config_path = config_path or self.DEFAULT_CONFIG["paths"]["config"]
        self.config = self._load_config()
        self._validate_config()
        self._set_environment_variables()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file and merge with defaults.
        
        Returns:
            The merged configuration dictionary.
        """
        config_path = pathlib.Path(self.config_path)
        if not config_path.exists():
            print(f"Warning: Configuration file {self.config_path} not found. Using defaults.")
            return self.DEFAULT_CONFIG.copy()
            
        try:
            user_config = yaml.safe_load(config_path.read_text())
            
            # Add paths section if not present
            if "paths" not in user_config:
                user_config["paths"] = self.DEFAULT_CONFIG["paths"]
                
            # Merge with defaults (shallow merge for now)
            merged_config = self.DEFAULT_CONFIG.copy()
            for key, value in user_config.items():
                if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
                    # For dictionaries, update rather than replace
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value
                    
            return merged_config
        except Exception as e:
            print(f"Error loading configuration: {str(e)}. Using defaults.")
            return self.DEFAULT_CONFIG.copy()
            
    def _validate_config(self) -> None:
        """
        Validate configuration values.
        
        Raises:
            ValueError: If a required configuration value is missing or invalid.
        """
        # Validate LLM configuration
        if self.config["llm"]["mode"] not in ["local", "remote"]:
            print(f"Warning: Invalid LLM mode '{self.config['llm']['mode']}'. Using 'local'.")
            self.config["llm"]["mode"] = "local"
            
        # Validate agent configuration
        if self.config["agent"]["mode"] not in ["code", "reasoning", "multi"]:
            print(f"Warning: Invalid agent mode '{self.config['agent']['mode']}'. Using 'code'.")
            self.config["agent"]["mode"] = "code"
            
        # Validate chunking configuration
        if self.config["chunking"]["mode"] not in ["semantic", "recursive"]:
            print(f"Warning: Invalid chunking mode '{self.config['chunking']['mode']}'. Using 'semantic'.")
            self.config["chunking"]["mode"] = "semantic"
            
        # Validate slice configuration
        if self.config["slice"]["default_mode"] not in ["auto", "page", "paragraph", "section"]:
            print(f"Warning: Invalid slice mode '{self.config['slice']['default_mode']}'. Using 'auto'.")
            self.config["slice"]["default_mode"] = "auto"
            
    def _set_environment_variables(self) -> None:
        """
        Set environment variables based on configuration.
        """
        if self.config["llm"]["mode"] == "local":
            os.environ["OPENAI_API_BASE"] = self.config["llm"]["base_url"]
            os.environ["LOCAL_LLM_BASE_URL"] = os.environ["OPENAI_API_BASE"]
            os.environ["OPENAI_MODEL_NAME"] = self.config["llm"]["model_id"]
            os.environ["LOCAL_LLM_MODEL_ID"] = os.environ["OPENAI_MODEL_NAME"]
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: The configuration key (dot notation supported, e.g., 'llm.mode')
            default: The default value to return if the key is not found
            
        Returns:
            The configuration value, or the default if not found
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def get_path(self, key: str) -> pathlib.Path:
        """
        Get a path from the configuration.
        
        Args:
            key: The path key (dot notation supported, e.g., 'indexing.whoosh')
            
        Returns:
            The path as a pathlib.Path object
            
        Raises:
            ValueError: If the path is not found in the configuration
        """
        path_str = self.get(f"paths.{key}")
        if path_str is None:
            raise ValueError(f"Path '{key}' not found in configuration")
            
        return pathlib.Path(path_str)
        
    def ensure_dir(self, key: str) -> pathlib.Path:
        """
        Ensure a directory exists and return its path.
        
        Args:
            key: The path key (dot notation supported, e.g., 'indexing.whoosh')
            
        Returns:
            The path as a pathlib.Path object
            
        Raises:
            ValueError: If the path is not found in the configuration
        """
        path = self.get_path(key)
        path.mkdir(parents=True, exist_ok=True)
        return path

# Create a singleton instance
CONFIG = ConfigManager()
