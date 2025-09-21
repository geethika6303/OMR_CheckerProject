import os
import json
from csv import QUOTE_NONNUMERIC
from pathlib import Path
from time import time
import tempfile
import shutil

import cv2
import pandas as pd
from rich.table import Table
import streamlit as st

from src import constants
from src.defaults import CONFIG_DEFAULTS
from src.evaluation import EvaluationConfig, evaluate_concatenated_response
from src.logger import console, logger
from src.template import Template
from src.utils.file import Paths, setup_dirs_for_paths, setup_outputs_for_template
from src.utils.image import ImageUtils
from src.utils.interaction import Stats
from src.utils.parsing import get_concatenated_response, open_config_with_defaults

# Load processors
STATS = Stats()

def _ensure_path(p):
    if isinstance(p, Path):
        return p
    return Path(str(p))

def _is_filelike(obj):
    return hasattr(obj, "read") and hasattr(obj, "seek")

def _copy_template_and_assets_to_temp(template_path: Path, temp_dir_path: Path):
    dest_template = temp_dir_path / template_path.name
    shutil.copy2(template_path, dest_template)
    try:
        with open(dest_template, "r", encoding="utf-8") as f:
            tpl = json.load(f)
    except Exception:
        return dest_template

    pre_keys = ["preprocessors", "preProcessors", "preProcessors", "preProcessor", "pre_processors"]
    rel_path = None
    for key in pre_keys:
        if key in tpl and isinstance(tpl[key], list):
            for item in tpl[key]:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("Name")
                    if name and name.lower() == "croponmarkers":
                        opts = item.get("options", {})
                        rp = opts.get("relativePath") or opts.get("relative_path")
                        if rp:
                            rel_path = rp
                            break
        if rel_path:
            break

    if rel_path:
        relp = Path(rel_path)
        if not relp.is_absolute():
            candidate = template_path.parent / relp
            if candidate.exists():
                shutil.copy2(candidate, temp_dir_path / relp.name)
            else:
                for p in template_path.parent.iterdir():
                    if p.is_file() and p.name.lower() == relp.name.lower():
                        shutil.copy2(p, temp_dir_path / relp.name)
                        break
    return dest_template

def _write_filelike_to_temp_and_return_path(filelike, filename_hint: str, temp_dir_path: Path):
    name = getattr(filelike, "name", None) or filename_hint
    dest = temp_dir_path / name
    try:
        filelike.seek(0)
    except Exception:
        pass
    with open(dest, "wb") as out:
        out.write(filelike.read())
    return dest

def entry_point(input_dir, args, template_file=None):
    root_dir = _ensure_path(input_dir)
    if not root_dir.exists():
        raise Exception(f"Given input directory does not exist: '{root_dir}'")
    return process_dir(root_dir, root_dir, args, template_file=template_file)

def print_config_summary(curr_dir, omr_files, template, tuning_config, local_config_path, evaluation_config, args):
    logger.info("")
    table = Table(title="Current Configurations", show_header=False, show_lines=False)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_row("Directory Path", f"{curr_dir}")
    table.add_row("Count of Images", f"{len(omr_files)}")
    table.add_row("Set Layout Mode ", "ON" if args.get("setLayout") else "OFF")
    pre_processor_names = [pp.__class__.__name__ for pp in template.pre_processors]
    table.add_row("Markers Detection", "ON" if "CropOnMarkers" in pre_processor_names else "OFF")
    table.add_row("Auto Alignment", f"{getattr(tuning_config.alignment_params, 'auto_align', False)}")
    table.add_row("Detected Template Path", f"{template}")
    if local_config_path:
        table.add_row("Detected Local Config", f"{local_config_path}")
    if evaluation_config:
        table.add_row("Detected Evaluation Config", f"{evaluation_config}")
    table.add_row("Detected pre-processors", ", ".join(pre_processor_names))
    console.print(table, justify="center")

# ------------------------------
# Rest of processing functions stay the same, just remove any GUI windows like cv2.imshow or waitKey
# ------------------------------
