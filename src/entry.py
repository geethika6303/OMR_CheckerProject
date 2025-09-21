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
    """Return a pathlib.Path for p (which may be Path or str)."""
    if isinstance(p, Path):
        return p
    return Path(str(p))


def _is_filelike(obj):
    """Rudimentary check for file-like (Streamlit uploadedfile or open file)."""
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


def process_dir(root_dir, curr_dir, args, template_file=None, template=None, tuning_config=CONFIG_DEFAULTS, evaluation_config=None):
    root_dir = _ensure_path(root_dir)
    curr_dir = _ensure_path(curr_dir)

    local_config_path = curr_dir.joinpath(constants.CONFIG_FILENAME)
    if os.path.exists(local_config_path):
        tuning_config = open_config_with_defaults(local_config_path)

    output_dir = Path(args.get("output_dir", "outputs"), curr_dir.relative_to(root_dir))
    paths = Paths(output_dir)

    temp_template_dir = None
    if template_file is not None:
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_template_dir = Path(temp_dir_obj.name)
        if isinstance(template_file, (str, Path)):
            tpl_path = _ensure_path(template_file)
            if not tpl_path.exists():
                raise Exception(f"Template path given does not exist: {tpl_path}")
            dest_tpl = _copy_template_and_assets_to_temp(tpl_path, temp_template_dir)
            template = Template(dest_tpl, tuning_config)
        elif _is_filelike(template_file):
            dest = _write_filelike_to_temp_and_return_path(template_file, "uploaded_template", temp_template_dir)
            is_json = False
            try:
                with open(dest, "r", encoding="utf-8") as f:
                    json.load(f)
                is_json = True
            except Exception:
                is_json = False
            if is_json:
                template = Template(dest, tuning_config)
            else:
                raise Exception("template_file appears to be an image / not a template JSON. OMRChecker expects a template JSON file.")
        else:
            raise Exception("Unsupported template_file type provided.")
    else:
        local_template_path = curr_dir.joinpath(constants.TEMPLATE_FILENAME)
        if os.path.exists(local_template_path):
            template = Template(local_template_path, tuning_config)

    exts = ("*.[pP][nN][gG]", "*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]")
    omr_files = sorted([f for ext in exts for f in curr_dir.glob(ext)])

    excluded_files = []
    if template:
        for pp in template.pre_processors:
            try:
                excluded_files.extend(Path(p) for p in pp.exclude_files())
            except Exception:
                pass

    local_evaluation_path = curr_dir.joinpath(constants.EVALUATION_FILENAME)
    if not args.get("setLayout", False) and os.path.exists(local_evaluation_path):
        if not template:
            logger.warning(f"Found an evaluation file without a parent template file: {local_evaluation_path}")
        evaluation_config = EvaluationConfig(curr_dir, local_evaluation_path, template, tuning_config)
        try:
            excluded_files.extend(Path(exclude_file) for exclude_file in evaluation_config.get_exclude_files())
        except Exception:
            pass

    omr_files = [f for f in omr_files if f not in excluded_files]

    if omr_files:
        if not template:
            logger.error(f"Found images, but no template in the directory tree '{curr_dir}'. Place {constants.TEMPLATE_FILENAME} in the appropriate directory.")
            raise Exception(f"No template file found in the directory tree of {curr_dir}")

        setup_dirs_for_paths(paths)
        outputs_namespace = setup_outputs_for_template(paths, template)

        print_config_summary(curr_dir, omr_files, template, tuning_config, local_config_path, evaluation_config, args)

        if args.get("setLayout", False):
            # Streamlit version of template layout display
            for file_path in omr_files:
                file_name = file_path.name
                in_omr = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                crop_on_markers_only = [pp for pp in template.pre_processors if pp.__class__.__name__ == "CropOnMarkers"]
                template.image_instance_ops.pre_processors = crop_on_markers_only
                in_omr = template.image_instance_ops.apply_preprocessors(file_path, in_omr, template)
                template_layout = template.image_instance_ops.draw_template_layout(in_omr, template, shifted=False, border=2)
                st.image(template_layout, channels="BGR", caption=f"Template Layout: {file_name}")
        else:
            process_files(omr_files, template, tuning_config, evaluation_config, outputs_namespace)

    subdirs = [d for d in curr_dir.iterdir() if d.is_dir()]
    for d in subdirs:
        process_dir(root_dir, d, args, template_file=None, template=template, tuning_config=tuning_config, evaluation_config=evaluation_config)


def process_files(omr_files, template, tuning_config, evaluation_config, outputs_namespace):
    start_time = int(time())
    files_counter = 0
    STATS.files_not_moved = 0

    for file_path in omr_files:
        files_counter += 1
        file_name = file_path.name

        in_omr = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if in_omr is None:
            logger.error(f"Failed to read image: {file_path}")
            continue

        logger.info("")
        logger.info(f"({files_counter}) Opening image: \t'{file_path}'\tResolution: {in_omr.shape}")

        template.image_instance_ops.reset_all_save_img()
        template.image_instance_ops.append_save_img(1, in_omr)

        crop_on_markers_only = [pp for pp in template.pre_processors if pp.__class__.__name__ == "CropOnMarkers"]
        template.image_instance_ops.pre_processors = crop_on_markers_only
        in_omr = template.image_instance_ops.apply_preprocessors(file_path, in_omr, template)

        if in_omr is None:
            new_file_path = outputs_namespace.paths.errors_dir.joinpath(file_name)
            outputs_namespace.OUTPUT_SET.append([file_name] + outputs_namespace.empty_resp)
            continue

        file_id = str(file_name)
        save_dir = outputs_namespace.paths.save_marked_dir
        response_dict, final_marked, multi_marked, _ = template.image_instance_ops.read_omr_response(template, image=in_omr, name=file_id, save_dir=save_dir)
        omr_response = get_concatenated_response(response_dict, template)

        if evaluation_config:
            score = evaluate_concatenated_response(omr_response, evaluation_config, file_path, outputs_namespace.paths.evaluation_dir)
            logger.info(f"(/{files_counter}) Graded with score: {round(score,2)}\t for file: '{file_id}'")
        else:
            score = 0
            logger.info(f"(/{files_counter}) Processed file: '{file_id}'")

        # Streamlit display for marked images
        if getattr(tuning_config.outputs, "show_image_level", 0) >= 2:
            resized_img = ImageUtils.resize_util_h(final_marked, int(tuning_config.dimensions.display_height * 1.3))
            st.image(resized_img, channels="BGR", caption=f"Final Marked Bubbles: {file_id}")

        resp_array = [omr_response[k] for k in template.output_columns]
        outputs_namespace.OUTPUT_SET.append([file_name] + resp_array)

        if multi_marked == 0 or not getattr(tuning_config.outputs, "filter_out_multimarked_files", False):
            STATS.files_not_moved += 1
            new_file_path = save_dir.joinpath(file_id)
        else:
            new_file_path = outputs_namespace.paths.multi_marked_dir.joinpath(file_name)


def print_stats(start_time, files_counter, tuning_config):
    time_checking = max(1, round(time() - start_time, 2))
    log = logger.info
    log("")
    log(f"{'Total file(s) moved': <27}: {STATS.files_moved}")
    log(f"{'Total file(s) not moved': <27}: {STATS.files_not_moved}")
    log("--------------------------------")
    log(f"{'Total file(s) processed': <27}: {files_counter} ({'Sum Tallied!' if files_counter == (STATS.files_moved + STATS.files_not_moved) else 'Not Tallying!'})")

    if getattr(tuning_config.outputs, "show_image_level", 0) <= 0:
        log(f"\nFinished Checking {files_counter} file(s) in {round(time_checking, 1)} seconds i.e. ~{round(time_checking / 60, 1)} minute(s).")
        try:
            log(f"{'OMR Processing Rate': <27}: \t ~ {round(time_checking / files_counter, 2)} seconds/OMR")
            log(f"{'OMR Processing Speed': <27}: \t ~ {round((files_counter * 60) / time_checking, 2)} OMRs/minute")
        except Exception:
            pass
    else:
        log(f"\n{'Total script time': <27}: {time_checking} seconds")

    if getattr(tuning_config.outputs, "show_image_level", 0) <= 1:
        log("\nTip: To see some awesome visuals, open config.json and increase 'show_image_level'")
