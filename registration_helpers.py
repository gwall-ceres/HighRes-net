import os
import numpy as np
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration loading function
DEFAULT_CONFIG = {
    "reference_image": "",
    "reference_mask": "",
    "template_image": "",
    "template_mask": "",
    "current_deltax": 0.0,
    "current_deltay": 0.0,
    "shift_step_x": 5.0,  # Example default shift steps
    "shift_step_y": 5.0
}

def load_config(config_path='config.json'):
    if not os.path.exists(config_path):
        logging.warning(f"Config file '{config_path}' not found. Using default settings.")
        return DEFAULT_CONFIG.copy()
    
    with open(config_path, 'r') as f:
        try:
            user_config = json.load(f)
            logging.info(f"Loaded configuration from {config_path}.")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}. Using default settings.")
            return DEFAULT_CONFIG.copy()
    
    # Merge user_config with DEFAULT_CONFIG, preferring user_config values
    config = DEFAULT_CONFIG.copy()
    config.update({k: v for k, v in user_config.items() if k in DEFAULT_CONFIG})
    
    # Optionally, warn about any unknown fields
    unknown_fields = set(user_config.keys()) - set(DEFAULT_CONFIG.keys())
    if unknown_fields:
        logging.warning(f"Unknown config fields detected and ignored: {unknown_fields}")
    
    return config

def contrast_stretch(array):
    """
    Perform contrast stretching on a NumPy array to map its values to the 0-255 range.
    """
    #logging.debug(f"Original array min: {array.min()}, max: {array.max()}")
    array = array.astype(float)
    min_val = np.min(array)
    max_val = np.max(array)

    # Avoid division by zero
    if max_val - min_val == 0:
        logging.warning("Max and min values are the same. Returning a zero array.")
        return np.zeros_like(array, dtype=np.uint8)

    # Perform contrast stretching
    stretched = (array - min_val) / (max_val - min_val) * 255.0
    #logging.debug(f"Stretched array min: {stretched.min()}, max: {stretched.max()}")

    # Clip values to the 0-255 range and convert to uint8
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)

    #logging.debug(f"Final stretched array min: {stretched.min()}, max: {stretched.max()}")
    return stretched