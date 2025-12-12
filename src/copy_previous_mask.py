"""
Selects previous existing mask M. Previous mask by default. Can select other mask.

Copies mask M and replaces mask for current image with selected previous mask L.
"""
import shutil
import os
import pathlib

def copy_previous_mask(current_mask_path: str, previous_mask_path: str):
    current_file_name = os.path.basename(current_mask_path)
    previous_file_name = os.path.basename(previous_mask_path)

    shutil.copy(previous_mask_path, current_mask_path)
    print(f"Copied previous mask {previous_file_name} to current mask {current_file_name}")

