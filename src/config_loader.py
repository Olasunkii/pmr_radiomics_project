from pathlib import Path
import yaml


class ConfigLoader:
    """
    ConfigLoader is responsible for loading and managing project configuration
    from a YAML file.

    It provides utilities to:
    - Load configuration settings
    - Resolve relative paths into absolute paths based on project root

    This ensures that all file paths remain consistent regardless of where
    the code is executed (notebooks, scripts, pipelines).
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the ConfigLoader.

        Parameters
        ----------
        config_path : str, optional
            Relative path to the YAML configuration file from the project root.
            Default is "config/config.yaml".
        """
        # Define project root (two levels up from this file)
        self.project_root = Path(__file__).resolve().parent.parent

        # Load YAML configuration
        config_file = self.project_root / config_path
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, "r") as file:
            self.config = yaml.safe_load(file)

    def get(self, *keys):
        """
        Retrieve a configuration value using nested keys.

        Parameters
        ----------
        *keys : str
            Sequence of keys to access nested configuration values.

        Returns
        -------
        Any
            The value stored at the specified configuration path.

        Raises
        ------
        KeyError
            If any key in the sequence does not exist.
        """
        value = self.config
        for key in keys:
            if key not in value:
                raise KeyError(f"Key not found in config: {' -> '.join(keys)}")
            value = value[key]
        return value

    def get_path(self, *keys) -> Path:
        """
        Retrieve a path from the configuration and resolve it to an absolute path.

        Parameters
        ----------
        *keys : str
            Sequence of keys pointing to a path value in the config.

        Returns
        -------
        pathlib.Path
            Absolute path resolved relative to the project root.
        """
        relative_path = self.get(*keys)
        return self.project_root / relative_path