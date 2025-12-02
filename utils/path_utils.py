"""
Path and directory utilities.
"""
from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    """
    Get project root directory.
    
    Returns:
        Path to project root
    """
    # Assumes this file is in utils/, so parent is project root
    return Path(__file__).parent.parent.resolve()


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_root() -> Path:
    """Get data directory path."""
    return get_project_root() / 'data'


def get_dataset_root() -> Path:
    """Get IDLRW dataset root path."""
    return get_data_root() / 'IDLRW-DATASET'


def get_extracted_root() -> Path:
    """Get extracted landmarks root path."""
    return get_data_root() / 'extracted'


def get_features_root() -> Path:
    """Get features root path."""
    return get_project_root() / 'features'


def get_models_root() -> Path:
    """Get models root path."""
    return get_project_root() / 'models'


def get_results_root() -> Path:
    """Get results root path."""
    return get_project_root() / 'results'


def get_logs_root() -> Path:
    """Get logs root path."""
    return get_project_root() / 'logs'

