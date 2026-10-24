import warnings

warnings.filterwarnings("ignore")

# GENERAL IMPORTS
import os

# this is needed to limit the number of scipy threads
# and let spikeinterface handle parallelization
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import numpy as np
from pathlib import Path
import json
import pickle
import time
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.preprocessing as spre
from spikeinterface.core.core_tools import check_json

# AIND
from aind_data_schema.core.processing import DataProcess

URL = "https://github.com/AllenNeuralDynamics/aind-ephys-preprocessing"
VERSION = "1.0-custom"  # Mark as custom version


data_folder = Path("../data/")
scratch_folder = Path("../scratch/")
results_folder = Path("../results/")

motion_presets = spre.get_motion_presets()

# define argument parser
parser = argparse.ArgumentParser(description="Preprocess AIND Neurpixels data")

# positional arguments
denoising_group = parser.add_mutually_exclusive_group()
denoising_help = "Which denoising strategy to use. Can be 'cmr' or 'destripe'"
denoising_group.add_argument("--denoising", choices=["cmr", "destripe"], help=denoising_help)
denoising_group.add_argument("static_denoising", nargs="?", default="cmr", help=denoising_help)

filter_group = parser.add_mutually_exclusive_group()
filter_help = "Which filter to use. Can be 'highpass' or 'bandpass'"
filter_group.add_argument("--filter-type", choices=["highpass", "bandpass"], help=filter_help)
filter_group.add_argument("static_filter_type", nargs="?", default="highpass", help=filter_help)

# Bandpass filter frequency arguments
parser.add_argument("--bandpass-freq-min", type=float, default=None, 
                    help="Lower frequency cutoff for bandpass filter (Hz). Only used when --filter-type bandpass")
parser.add_argument("--bandpass-freq-max", type=float, default=None,
                    help="Upper frequency cutoff for bandpass filter (Hz). Only used when --filter-type bandpass")

remove_out_channels_group = parser.add_mutually_exclusive_group()
remove_out_channels_help = "Whether to remove out channels"
remove_out_channels_group.add_argument("--no-remove-out-channels", action="store_true", help=remove_out_channels_help)
remove_out_channels_group.add_argument(
    "static_remove_out_channels", nargs="?", default="true", help=remove_out_channels_help
)

remove_bad_channels_group = parser.add_mutually_exclusive_group()
remove_bad_channels_help = "Whether to remove bad channels"
remove_bad_channels_group.add_argument("--no-remove-bad-channels", action="store_true", help=remove_bad_channels_help)
remove_bad_channels_group.add_argument(
    "static_remove_bad_channels", nargs="?", default="true", help=remove_bad_channels_help
)

max_bad_channel_fraction_group = parser.add_mutually_exclusive_group()
max_bad_channel_fraction_help = (
    "Maximum fraction of bad channels to remove. If more than this fraction, processing is skipped"
)
max_bad_channel_fraction_group.add_argument(
    "--max-bad-channel-fraction", default=0.5, help=max_bad_channel_fraction_help
)
max_bad_channel_fraction_group.add_argument(
    "static_max_bad_channel_fraction", nargs="?", default=None, help=max_bad_channel_fraction_help
)

motion_correction_group = parser.add_mutually_exclusive_group()
motion_correction_help = "How to deal with motion correction. Can be 'skip', 'compute', or 'apply'"
motion_correction_group.add_argument("--motion", choices=["skip", "compute", "apply"], help=motion_correction_help)
motion_correction_group.add_argument("static_motion", nargs="?", default="compute", help=motion_correction_help)

motion_preset_group = parser.add_mutually_exclusive_group()
motion_preset_help = (
    f"What motion preset to use. Supported presets are: {', '.join(motion_presets)}."
)
motion_preset_group.add_argument(
    "--motion-preset",
    choices=motion_presets,
    help=motion_preset_help,
)
motion_preset_group.add_argument("static_motion_preset", nargs="?", default=None, help=motion_preset_help)

t_start_group = parser.add_mutually_exclusive_group()
t_start_help = (
    "Start time of the recording in seconds (assumes recording starts at 0). "
    "This parameter is ignored in case of multi-segment or multi-block recordings."
    "Default is None (start of recording)"
)
t_start_group.add_argument("static_t_start", nargs="?", default=None, help=t_start_help)
t_start_group.add_argument("--t-start", default=None, help=t_start_help)

t_stop_group = parser.add_mutually_exclusive_group()
t_stop_help = (
    "Stop time of the recording in seconds (assumes recording starts at 0). "
    "This parameter is ignored in case of multi-segment or multi-block recordings."
    "Default is None (end of recording)"
)
t_stop_group.add_argument("static_t_stop", nargs="?", default=None, help=t_stop_help)
t_stop_group.add_argument("--t-stop", default=None, help=t_stop_help)

n_jobs_group = parser.add_mutually_exclusive_group()
n_jobs_help = (
    "Number of jobs to use for parallel processing. Default is -1 (all available cores). "
    "It can also be a float between 0 and 1 to use a fraction of available cores"
)
n_jobs_group.add_argument("static_n_jobs", nargs="?", default=None, help=n_jobs_help)
n_jobs_group.add_argument("--n-jobs", default="-1", help=n_jobs_help)

params_group = parser.add_mutually_exclusive_group()
params_file_help = "Optional json file with parameters"
params_group.add_argument("static_params_file", nargs="?", default=None, help=params_file_help)
params_group.add_argument("--params-file", default=None, help=params_file_help)
params_group.add_argument("--params-str", default=None, help="Optional json string with parameters")

# ============================================================================
# NEW: Manual bad channel arguments
# ============================================================================
bad_channel_ids_help = (
    "Comma-separated list of numeric bad channel indices to add to auto-detected bad channels. "
    "Example: --bad-channel-ids '0,5,127'"
)
parser.add_argument("--bad-channel-ids", default=None, help=bad_channel_ids_help)

bad_channels_config_help = (
    "Path to a bad_channels.conf file with session-specific bad channel mappings. "
    "Format: session_name = channel1, channel2, channel3 (numeric indices only, e.g., '0,5,127')"
)
parser.add_argument("--bad-channels-config", default=None, help=bad_channels_config_help)

# ============================================================================
# EF fixes (Emily Ferenczi, 2026-10-24) -- all off by default, so a run without these flags is unchanged
# ============================================================================
parser.add_argument(
    "--destripe-float32", action="store_true",
    help="Run bad-channel interpolation, destripe and CMR on float32 data and round back to int16 before saving. "
    "SpikeInterface < 0.104 divides by the destripe AGC gain in place in the input dtype, so int16 input sets every "
    "sample smaller than the local mean |x| to 0 (fixed upstream in SpikeInterface 0.104).",
)
parser.add_argument(
    "--destripe-pad-to-group", action="store_true",
    help="Cap destripe's n_channel_pad / n_channel_taper at each channel group's size. Without it, a group (shank) with "
    "fewer channels than n_channel_pad (60) makes destripe fail, and the pipeline falls back to CMR.",
)
parser.add_argument(
    "--destripe-strict", action="store_true",
    help="Stop with an error when destripe fails, instead of falling back to CMR.",
)


# ============================================================================
# Helper functions for manual bad channels
# ============================================================================
def parse_bad_channel_ids(bad_channel_str: str) -> List[str]:
    """
    Parse comma-separated numeric channel indices from command line.
    
    Args:
        bad_channel_str: Comma-separated string like "0,5,127"
    
    Returns:
        List of channel index strings (numeric only)
    """
    if not bad_channel_str or bad_channel_str.strip() == "":
        return []
    return [ch.strip() for ch in bad_channel_str.split(",") if ch.strip()]


def load_bad_channels_from_config(config_path: Path, session_name: str) -> List[str]:
    """
    Load numeric bad channel indices from config file for a specific session.
    
    Config file format:
        # Comments start with #
        session_name = channel1, channel2, channel3
        partial_match = channel4, channel5
    
    Note: Only numeric channel indices are supported (e.g., "0,5,127").
    
    Args:
        config_path: Path to bad_channels.conf file
        session_name: Name of the recording session
    
    Returns:
        List of channel index strings (numeric only, empty list if no match found)
    """
    if not config_path.exists():
        print(f"\t[custom] No bad channels config found at {config_path}")
        return []
    
    bad_channels = []
    
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            
            # Parse key=value pairs
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                
                # Check for exact or partial match (case-insensitive)
                if key.lower() in session_name.lower() or session_name.lower() in key.lower():
                    channels = [ch.strip() for ch in value.split(",") if ch.strip()]
                    bad_channels.extend(channels)
                    print(f"\t[custom] Found bad channels for '{key}': {channels}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_channels = []
    for ch in bad_channels:
        if ch not in seen:
            seen.add(ch)
            unique_channels.append(ch)
    
    return unique_channels


def get_channel_ids_from_names(recording, channel_names: List[str]) -> np.ndarray:
    """
    Convert numeric channel indices to channel IDs that SpikeInterface understands.
    
    Only supports numeric channel indices (e.g., "0", "5", "127" -> channel at index 0, 5, 127).
    
    Args:
        recording: SpikeInterface recording object
        channel_names: List of numeric channel indices as strings (e.g., ['0', '5', '127'])
    
    Returns:
        Array of channel IDs
    """
    all_channel_ids = recording.get_channel_ids()
    
    matched_ids = []
    for name in channel_names:
        # Only support numeric indices
        try:
            idx = int(name.strip())
            if 0 <= idx < len(all_channel_ids):
                matched_ids.append(all_channel_ids[idx])
            else:
                print(f"\t[custom] Warning: Channel index {idx} out of range (0-{len(all_channel_ids)-1})")
        except ValueError:
            print(f"\t[custom] Warning: Invalid channel ID '{name}' - only numeric indices are supported (e.g., '0', '5', '127')")
    
    return np.array(matched_ids)


def detect_multi_shank_probe(recording, recording_dict: dict, data_folder: Path) -> Tuple[bool, Optional[int]]:
    """
    Detect if a recording is from a multi-shank probe.
    
    For SpikeGLX data, checks:
    1. If 'group' property already exists (indicates grouping is set up)
    2. Probe geometry for shank information
    3. SpikeGLX metadata files for imDatPrb_type field
    
    Args:
        recording: SpikeInterface recording object
        recording_dict: Recording dictionary from job config
        data_folder: Path to data folder
    
    Returns:
        Tuple of (is_multi_shank: bool, num_shanks: Optional[int])
        If detection fails, defaults to NP2 4-shank (returns (True, 4))
    """
    num_channels = recording.get_num_channels()
    print(f"\t[custom] Detection method 1: Checking for existing 'group' property...")
    
    # Check if 'group' property already exists
    if "group" in recording.get_property_keys():
        unique_groups = np.unique(recording.get_property("group"))
        num_groups = len(unique_groups)
        if num_groups > 1:
            print(f"\t[custom] ✓ Found existing 'group' property with {num_groups} groups (multi-shank detected)")
            return True, num_groups
        else:
            print(f"\t[custom]   Found 'group' property but only {num_groups} group(s) - continuing detection")
    else:
        print(f"\t[custom]   No existing 'group' property found")
    
    # Try to detect from probe geometry
    print(f"\t[custom] Detection method 2: Checking probe geometry...")
    try:
        probe = recording.get_probe()
        if probe is not None:
            print(f"\t[custom]   Probe found, checking for shank information...")
            # Check if probe has shank_ids
            if hasattr(probe, 'get_shank_ids'):
                try:
                    shank_ids = probe.get_shank_ids()
                    if shank_ids is not None and len(shank_ids) > 1:
                        num_shanks = len(np.unique(shank_ids))
                        print(f"\t[custom] ✓ Detected {num_shanks} shanks from probe.get_shank_ids()")
                        return True, num_shanks
                    else:
                        print(f"\t[custom]   probe.get_shank_ids() returned {len(shank_ids) if shank_ids is not None else 'None'} shank(s)")
                except Exception as e:
                    print(f"\t[custom]   probe.get_shank_ids() failed: {e}")
            
            # Check if probe has shank_ids attribute directly
            if hasattr(probe, 'shank_ids') and probe.shank_ids is not None:
                shank_ids = probe.shank_ids
                unique_shanks = np.unique(shank_ids)
                if len(unique_shanks) > 1:
                    num_shanks = len(unique_shanks)
                    print(f"\t[custom] ✓ Detected {num_shanks} shanks from probe.shank_ids attribute")
                    return True, num_shanks
            
            # Check channel properties for shank information
            if probe.contact_ids is not None:
                probe_num_channels = len(probe.contact_ids)
                print(f"\t[custom]   Probe has {probe_num_channels} contacts, recording has {num_channels} channels")
                # Try to infer from probe structure
                # For NP2 4-shank: typically 1280 channels, 320 per shank
                if probe_num_channels >= 1280:
                    print(f"\t[custom] ✓ Detected high channel count ({probe_num_channels}), assuming NP2 4-shank")
                    return True, 4
                elif num_channels >= 1280:
                    print(f"\t[custom] ✓ Recording has high channel count ({num_channels}), assuming NP2 4-shank")
                    return True, 4
        else:
            print(f"\t[custom]   No probe geometry found")
    except Exception as e:
        print(f"\t[custom]   Could not detect from probe geometry: {e}")
    
    # Try to detect from SpikeGLX metadata files
    print(f"\t[custom] Detection method 3: Checking SpikeGLX metadata files...")
    try:
        # Look for .meta files in the recording path
        # SpikeGLX metadata typically in same directory as binary files
        meta_files = []
        if "kwargs" in recording_dict:
            file_path = recording_dict.get("kwargs", {}).get("file_path")
            if file_path:
                meta_path = Path(file_path).parent / f"{Path(file_path).stem}.meta"
                if meta_path.exists():
                    meta_files.append(meta_path)
                    print(f"\t[custom]   Found metadata file: {meta_path}")
        
        # Also search in data folder for .meta files
        if not meta_files:
            found_meta = list(data_folder.glob("*.meta"))
            meta_files.extend(found_meta)
            if found_meta:
                print(f"\t[custom]   Found {len(found_meta)} metadata file(s) in data folder")
        
        if not meta_files:
            print(f"\t[custom]   No metadata files found")
        else:
            for meta_file in meta_files:
                try:
                    print(f"\t[custom]   Reading metadata file: {meta_file.name}")
                    with open(meta_file, 'r') as f:
                        for line in f:
                            if 'imDatPrb_type' in line:
                                try:
                                    probe_type = int(line.split('=')[1].strip())
                                    print(f"\t[custom]   Found imDatPrb_type={probe_type}")
                                    if probe_type == 24:
                                        print(f"\t[custom] ✓ Detected NP2 4-shank from metadata (imDatPrb_type=24)")
                                        return True, 4
                                    elif probe_type == 21:
                                        print(f"\t[custom] ✓ Detected NP2 1-shank from metadata (imDatPrb_type=21)")
                                        return False, 1
                                    elif probe_type == 0:
                                        print(f"\t[custom] ✓ Detected NP1 1-shank from metadata (imDatPrb_type=0)")
                                        return False, 1
                                    else:
                                        print(f"\t[custom]   Unknown probe type: {probe_type}")
                                except (ValueError, IndexError) as e:
                                    print(f"\t[custom]   Could not parse imDatPrb_type: {e}")
                                    continue
                except Exception as e:
                    print(f"\t[custom]   Error reading {meta_file}: {e}")
                    continue
    except Exception as e:
        print(f"\t[custom]   Could not read metadata files: {e}")
    
    # Default to NP2 4-shank if detection fails (per user requirement)
    print(f"\t[custom] ⚠ All detection methods failed, defaulting to NP2 4-shank (as requested)")
    return True, 4


def setup_channel_grouping(recording, num_shanks: int) -> si.BaseRecording:
    """
    Set up channel grouping for multi-shank probes.
    
    For NP2 4-shank probes, assumes ~320 channels per shank.
    Channels are assigned to shanks based on their index.
    
    Args:
        recording: SpikeInterface recording object
        num_shanks: Number of shanks
    
    Returns:
        Recording with 'group' property set
    """
    # Check if grouping already exists
    if "group" in recording.get_property_keys():
        print(f"\t[custom] Channel grouping already exists")
        return recording
    
    # Try to use probe shank_ids if available
    try:
        probe = recording.get_probe()
        if probe is not None:
            # Check if probe has shank_ids attribute
            if hasattr(probe, 'shank_ids') and probe.shank_ids is not None:
                shank_ids = probe.shank_ids
                if len(np.unique(shank_ids)) > 1:
                    # Map contact_ids to channel_ids
                    contact_to_shank = {contact_id: shank_id for contact_id, shank_id in zip(probe.contact_ids, shank_ids)}
                    channel_ids = recording.get_channel_ids()
                    groups = [contact_to_shank.get(ch_id, 0) for ch_id in channel_ids]
                    recording = recording.set_channel_property(key="group", values=groups)
                    unique_groups = np.unique(groups)
                    print(f"\t[custom] Set up channel grouping from probe geometry: {len(unique_groups)} shanks")
                    return recording
    except Exception as e:
        print(f"\t[custom] Could not use probe shank_ids: {e}")
    
    # Fallback: assign groups based on channel index (assumes sequential shank assignment)
    channel_ids = recording.get_channel_ids()
    num_channels = len(channel_ids)
    
    # Calculate channels per shank
    channels_per_shank = num_channels // num_shanks
    
    # Assign groups based on channel index
    groups = []
    for i, ch_id in enumerate(channel_ids):
        shank_id = i // channels_per_shank
        # Ensure we don't exceed num_shanks
        shank_id = min(shank_id, num_shanks - 1)
        groups.append(shank_id)
    
    # Set the group property
    recording = recording.set_channel_property(key="group", values=groups)
    
    unique_groups = np.unique(groups)
    print(f"\t[custom] Set up channel grouping: {len(unique_groups)} shanks, ~{channels_per_shank} channels per shank")
    
    return recording


def dump_to_json_or_pickle(recording, results_folder, base_name, relative_to):
    if recording.check_serializability("json"):
        recording.dump_to_json(results_folder / f"{base_name}.json", relative_to=relative_to)
    else:
        recording.dump_to_pickle(results_folder / f"{base_name}.pkl", relative_to=relative_to)


def apply_by_group(recording, func, func_kwargs, group_property="group", func_name="function", kwargs_for_group=None):
    """
    Apply a SpikeInterface preprocessing function per channel-group (e.g., per shank),
    then aggregate back into a single recording.
    
    Args:
        recording: Recording to process
        func: Function to apply
        func_kwargs: Keyword arguments for the function
        group_property: Property name for grouping (default: "group")
        func_name: Name of the function for logging (default: "function")
        kwargs_for_group: optional callable (group_recording, func_kwargs) -> kwargs for that group (EF fixes)
    """
    func_kwargs = func_kwargs or {}
    kw = (lambda r: kwargs_for_group(r, func_kwargs)) if kwargs_for_group is not None else (lambda r: func_kwargs)

    # No grouping info -> just run normally
    if group_property not in recording.get_property_keys():
        print(f"\t[custom] Applying {func_name} without grouping (no '{group_property}' property)")
        return func(recording, **kw(recording))

    split = recording.split_by(group_property)

    # If split_by returns a dict with >1 group, process each recording separately
    # then aggregate them back together
    if isinstance(split, dict) and len(split) > 1:
        print(f"\t[custom] Applying {func_name} with grouping: {len(split)} groups")
        processed_dict = {}
        keys = sorted(split.keys(), key=str)
        for key in keys:
            num_channels = split[key].get_num_channels()
            print(f"\t[custom]   Processing group {key}: {num_channels} channels")
            processed_dict[key] = func(split[key], **kw(split[key]))
        print(f"\t[custom] ✓ {func_name} completed for all groups, aggregating channels")
        return si.aggregate_channels([processed_dict[k] for k in keys])

    print(f"\t[custom] Applying {func_name} without grouping (single group)")
    return func(recording, **kw(recording))


def pad_to_group(recording, func_kwargs):
    """EF fixes: cap destripe's channel padding / taper at the group's channel count"""
    kw = dict(func_kwargs); n = recording.get_num_channels()
    for k in ("n_channel_pad", "n_channel_taper"):
        if kw.get(k) is not None and int(kw[k]) > n:
            print(f"\t[custom]   EF fixes: {k} {kw[k]} -> {n} (group has {n} channels)")
            kw[k] = n
    return kw


def detect_bad_channels_by_group(recording, detect_kwargs, group_property="group"):
    """
    Run detect_bad_channels() per channel-group (e.g., per shank) and merge the returned
    channel_labels back to the full recording channel order.
    """
    detect_kwargs = detect_kwargs or {}

    # If no grouping, just run normally
    if group_property not in recording.get_property_keys():
        _, labels = spre.detect_bad_channels(recording, **detect_kwargs)
        return labels

    split = recording.split_by(group_property)
    if not isinstance(split, dict) or len(split) <= 1:
        _, labels = spre.detect_bad_channels(recording, **detect_kwargs)
        return labels

    # Allocate full labels array aligned to recording.channel_ids order
    full_channel_ids = list(recording.channel_ids)
    id_to_index = {ch_id: i for i, ch_id in enumerate(full_channel_ids)}
    full_labels = np.empty(len(full_channel_ids), dtype=object)

    for gid, rec_g in split.items():
        _, labels_g = spre.detect_bad_channels(rec_g, **detect_kwargs)
        for ch_id, lab in zip(rec_g.channel_ids, labels_g):
            full_labels[id_to_index[ch_id]] = lab

    return full_labels



if __name__ == "__main__":
    # ================================================================
    # CUSTOM VERSION MARKER - This confirms the custom script is running
    # ================================================================
    print("=" * 60)
    print(f"CUSTOM PREPROCESSING SCRIPT VERSION: {VERSION}")
    print("=" * 60)
    
    args = parser.parse_args()

    DENOISING_STRATEGY = args.denoising or args.static_denoising
    DESTRIPE_FLOAT32 = args.destripe_float32
    DESTRIPE_PAD_TO_GROUP = args.destripe_pad_to_group
    DESTRIPE_STRICT = args.destripe_strict
    FILTER_TYPE = args.filter_type or args.static_filter_type
    BANDPASS_FREQ_MIN = args.bandpass_freq_min
    BANDPASS_FREQ_MAX = args.bandpass_freq_max
    REMOVE_OUT_CHANNELS = False if args.no_remove_out_channels else args.static_remove_out_channels == "true"
    REMOVE_BAD_CHANNELS = False if args.no_remove_bad_channels else args.static_remove_bad_channels == "true"
    MAX_BAD_CHANNEL_FRACTION = float(args.static_max_bad_channel_fraction or args.max_bad_channel_fraction)
    motion_arg = args.motion or args.static_motion
    MOTION_PRESET = args.static_motion_preset or args.motion_preset
    COMPUTE_MOTION = True if motion_arg != "skip" else False
    APPLY_MOTION = True if motion_arg == "apply" else False
    T_START = args.static_t_start or args.t_start
    if isinstance(T_START, str) and T_START == "":
        T_START = None
    T_STOP = args.static_t_stop or args.t_stop
    if isinstance(T_STOP, str) and T_STOP == "":
        T_STOP = None

    N_JOBS = args.static_n_jobs or args.n_jobs
    N_JOBS = int(N_JOBS) if not N_JOBS.startswith("0.") else float(N_JOBS)
    PARAMS_FILE = args.static_params_file or args.params_file
    PARAMS_STR = args.params_str

    # NEW: Parse manual bad channel arguments
    MANUAL_BAD_CHANNEL_IDS = parse_bad_channel_ids(args.bad_channel_ids) if args.bad_channel_ids else []
    BAD_CHANNELS_CONFIG = Path(args.bad_channels_config) if args.bad_channels_config else None

    # Use CO_CPUS env variable if available
    N_JOBS_CO = os.getenv("CO_CPUS")
    N_JOBS = int(N_JOBS_CO) if N_JOBS_CO is not None else N_JOBS

    print(f"Running preprocessing with the following parameters:")
    print(f"\tDENOISING_STRATEGY: {DENOISING_STRATEGY}")
    print(f"\tFILTER TYPE: {FILTER_TYPE}")
    print(f"\tREMOVE_OUT_CHANNELS: {REMOVE_OUT_CHANNELS}")
    print(f"\tREMOVE_BAD_CHANNELS: {REMOVE_BAD_CHANNELS}")
    print(f"\tMAX BAD CHANNEL FRACTION: {MAX_BAD_CHANNEL_FRACTION}")
    print(f"\tCOMPUTE_MOTION: {COMPUTE_MOTION}")
    print(f"\tAPPLY_MOTION: {APPLY_MOTION}")
    print(f"\tMOTION PRESET: {MOTION_PRESET}")
    print(f"\tT_START: {T_START}")
    print(f"\tT_STOP: {T_STOP}")
    print(f"\tN_JOBS: {N_JOBS}")
    # NEW: Print manual bad channel info
    if MANUAL_BAD_CHANNEL_IDS:
        print(f"\tMANUAL_BAD_CHANNEL_IDS: {MANUAL_BAD_CHANNEL_IDS}")
    if BAD_CHANNELS_CONFIG:
        print(f"\tBAD_CHANNELS_CONFIG: {BAD_CHANNELS_CONFIG}")

    if PARAMS_FILE is not None:
        print(f"\nUsing custom parameter file: {PARAMS_FILE}")
        with open(PARAMS_FILE, "r") as f:
            processing_params = json.load(f)
    elif PARAMS_STR is not None:
        processing_params = json.loads(PARAMS_STR)
    else:
        with open("params.json", "r") as f:
            processing_params = json.load(f)

    data_process_prefix = "data_process_preprocessing"

    job_kwargs = processing_params["job_kwargs"]
    job_kwargs["n_jobs"] = N_JOBS
    si.set_global_job_kwargs(**job_kwargs)

    preprocessing_params = processing_params["preprocessing"]
    preprocessing_params["denoising_strategy"] = DENOISING_STRATEGY
    preprocessing_params["ef_fixes"] = dict(destripe_float32=DESTRIPE_FLOAT32, destripe_pad_to_group=DESTRIPE_PAD_TO_GROUP,
                                            destripe_strict=DESTRIPE_STRICT)
    preprocessing_params["remove_out_channels"] = REMOVE_OUT_CHANNELS
    preprocessing_params["remove_bad_channels"] = REMOVE_BAD_CHANNELS
    preprocessing_params["max_bad_channel_fraction"] = MAX_BAD_CHANNEL_FRACTION
    motion_params = processing_params["motion_correction"]
    motion_params["compute"] = COMPUTE_MOTION
    motion_params["apply"] = APPLY_MOTION
    if MOTION_PRESET is not None:
        motion_params["preset"] = MOTION_PRESET

    # load job files
    job_config_files = [p for p in data_folder.iterdir() if (p.suffix == ".json" or p.suffix == ".pickle" or p.suffix == ".pkl") and "job" in p.name]
    print(f"Found {len(job_config_files)} configurations")

    if len(job_config_files) > 0:
        ####### PREPROCESSING #######
        print("\n\nPREPROCESSING")
        t_preprocessing_start_all = time.perf_counter()
        preprocessing_vizualization_data = {}

        for job_config_file in job_config_files:
            datetime_start_preproc = datetime.now()
            t_preprocessing_start = time.perf_counter()
            preprocessing_notes = ""

            if job_config_file.suffix == ".json":
                with open(job_config_file, "r") as f:
                    job_config = json.load(f)
            else:
                with open(job_config_file, "rb") as f:
                    job_config = pickle.load(f)

            session_name = job_config["session_name"]
            recording_name = job_config["recording_name"]
            recording_dict = job_config["recording_dict"]
            skip_times = job_config.get("skip_times", False)
            debug = job_config.get("debug", False)

            try:
                recording = si.load_extractor(recording_dict, base_folder=data_folder)
            except:
                raise RuntimeError(
                    f"Could not find load recording {recording_name} from dict. "
                    f"Make sure mapping is correct!"
                )
            if skip_times:
                print("Resetting recording timestamps")
                recording.reset_times()

            # ================================================================
            # NEW: Auto-detect multi-shank probe and set up grouping
            # ================================================================
            print(f"\n\t[custom] === Multi-shank probe detection ===")
            is_multi_shank, num_shanks = detect_multi_shank_probe(recording, recording_dict, data_folder)
            if is_multi_shank:
                print(f"\t[custom] Multi-shank probe detected: {num_shanks} shanks")
                recording = setup_channel_grouping(recording, num_shanks)
                # Verify grouping was set up
                if "group" in recording.get_property_keys():
                    unique_groups = np.unique(recording.get_property("group"))
                    print(f"\t[custom] ✓ Channel grouping verified: {len(unique_groups)} groups found")
                    for group_id in sorted(unique_groups):
                        group_mask = recording.get_property("group") == group_id
                        num_channels_in_group = np.sum(group_mask)
                        print(f"\t[custom]   - Group {group_id}: {num_channels_in_group} channels")
                preprocessing_notes += f"\n- Multi-shank probe detected ({num_shanks} shanks), grouping enabled"
            else:
                print(f"\t[custom] Single-shank probe detected (no grouping needed)")
            print(f"\t[custom] ========================================\n")
            # ================================================================

            skip_processing = False
            vizualization_file_is_json_serializable = True

            preprocessing_vizualization_data[recording_name] = {}
            preprocessing_output_process_json = results_folder / f"{data_process_prefix}_{recording_name}.json"
            preprocessing_output_folder = results_folder / f"preprocessed_{recording_name}"
            preprocessingviz_output_filename = f"preprocessedviz_{recording_name}"
            preprocessing_output_filename = f"preprocessed_{recording_name}"
            motioncorrected_output_filename = f"motioncorrected_{recording_name}"
            binary_output_filename = f"binary_{recording_name}"

            print(f"Preprocessing recording: {session_name} - {recording_name}")

            # ================================================================
            # NEW: Load manual bad channels for this session
            # ================================================================
            session_manual_bad_channels = list(MANUAL_BAD_CHANNEL_IDS)  # Start with CLI-provided channels
            
            # Add channels from config file if provided
            if BAD_CHANNELS_CONFIG is not None:
                config_bad_channels = load_bad_channels_from_config(BAD_CHANNELS_CONFIG, session_name)
                session_manual_bad_channels.extend(config_bad_channels)
            
            # Remove duplicates
            session_manual_bad_channels = list(dict.fromkeys(session_manual_bad_channels))
            
            if session_manual_bad_channels:
                print(f"\t[custom] Manual bad channels for this session: {session_manual_bad_channels}")
                preprocessing_notes += f"\n- Manual bad channels specified: {session_manual_bad_channels}"
            # ================================================================

            if (T_START is not None or T_STOP is not None):
                if recording.get_num_segments() > 1:
                    print(f"\tRecording has multiple segments. Ignoring T_START and T_STOP")
                else:
                    if T_START is None:
                        T_START = 0
                    if T_STOP is None:
                        T_STOP = recording.get_duration()
                    T_START = float(T_START)
                    T_STOP = float(T_STOP)
                    T_STOP = min(T_STOP, recording.get_duration())
                    print(f"\tOriginal recording duration: {recording.get_duration()} -- Clipping to {T_START}-{T_STOP} s")
                    start_frame = int(T_START * recording.get_sampling_frequency())
                    end_frame = int(T_STOP * recording.get_sampling_frequency()) + 1
                    total_samples = recording.get_num_samples()
                    end_frame = min(end_frame, total_samples - 1)
                    recording = recording.frame_slice(start_frame=start_frame, end_frame=end_frame)

            print(f"\tDuration: {np.round(recording.get_total_duration(), 2)} s")

            preprocessing_vizualization_data[recording_name]["timeseries"] = dict()
            preprocessing_vizualization_data[recording_name]["timeseries"]["full"] = dict(
                raw=recording.to_dict(relative_to=data_folder, recursive=True)
            )
            if not recording.check_serializability("json"):
                vizualization_file_is_json_serializable = False
            # maybe a recording is from a different source and it doesn't need to be phase shifted
            if "inter_sample_shift" in recording.get_property_keys():
                recording_ps_full = spre.phase_shift(recording, **preprocessing_params["phase_shift"])
                preprocessing_vizualization_data[recording_name]["timeseries"]["full"].update(
                    dict(phase_shift=recording_ps_full.to_dict(relative_to=data_folder, recursive=True))
                )
            else:
                recording_ps_full = recording

            if FILTER_TYPE == "highpass":
                recording_filt_full = spre.highpass_filter(recording_ps_full, **preprocessing_params["highpass_filter"])
                preprocessing_vizualization_data[recording_name]["timeseries"]["full"].update(
                    dict(highpass=recording_filt_full.to_dict(relative_to=data_folder, recursive=True))
                )
                preprocessing_params["filter_type"] = "highpass"
            elif FILTER_TYPE == "bandpass":
                # Use command-line frequencies if provided, otherwise use params.json defaults
                bandpass_kwargs = preprocessing_params["bandpass_filter"].copy()
                if BANDPASS_FREQ_MIN is not None:
                    bandpass_kwargs["freq_min"] = BANDPASS_FREQ_MIN
                    print(f"\t[custom] Using bandpass freq_min from command line: {BANDPASS_FREQ_MIN} Hz")
                if BANDPASS_FREQ_MAX is not None:
                    bandpass_kwargs["freq_max"] = BANDPASS_FREQ_MAX
                    print(f"\t[custom] Using bandpass freq_max from command line: {BANDPASS_FREQ_MAX} Hz")
                if BANDPASS_FREQ_MIN is not None or BANDPASS_FREQ_MAX is not None:
                    print(f"\t[custom] Bandpass filter range: {bandpass_kwargs.get('freq_min', 'default')}-{bandpass_kwargs.get('freq_max', 'default')} Hz")
                
                recording_filt_full = spre.bandpass_filter(recording_ps_full, **bandpass_kwargs)
                preprocessing_vizualization_data[recording_name]["timeseries"]["full"].update(
                    dict(bandpass=recording_filt_full.to_dict(relative_to=data_folder, recursive=True))
                )
                preprocessing_params["filter_type"] = "bandpass"
            else:
                raise ValueError(f"Filter type {FILTER_TYPE} not recognized")

            if recording.get_total_duration() < preprocessing_params["min_preprocessing_duration"] and not debug:
                print(f"\tRecording is too short ({recording.get_total_duration()}s). Skipping further processing")
                preprocessing_notes += (
                    f"\n- Recording is too short ({recording.get_total_duration()}s). Skipping further processing\n"
                )
                channel_labels = None
                skip_processing = True
            else:
                # IBL bad channel detection
                # _, channel_labels = spre.detect_bad_channels(
                #     recording_filt_full, **preprocessing_params["detect_bad_channels"]
                # )
                channel_labels = detect_bad_channels_by_group(
                    recording_filt_full,
                    preprocessing_params["detect_bad_channels"],
                    group_property="group",
                )

                dead_channel_mask = channel_labels == "dead"
                noise_channel_mask = channel_labels == "noise"
                out_channel_mask = channel_labels == "out"
                print(f"\tBad channel detection:")
                print(
                    f"\t\t- dead channels - {np.sum(dead_channel_mask)}\n\t\t- noise channels - {np.sum(noise_channel_mask)}\n\t\t- out channels - {np.sum(out_channel_mask)}"
                )
                dead_channel_ids = recording_filt_full.channel_ids[dead_channel_mask]
                noise_channel_ids = recording_filt_full.channel_ids[noise_channel_mask]
                out_channel_ids = recording_filt_full.channel_ids[out_channel_mask]

                # ================================================================
                # NEW: Add manual bad channels to the detected ones
                # ================================================================
                manual_bad_channel_ids_resolved = np.array([], dtype=dead_channel_ids.dtype)
                if session_manual_bad_channels:
                    manual_bad_channel_ids_resolved = get_channel_ids_from_names(
                        recording_filt_full, session_manual_bad_channels
                    )
                    if len(manual_bad_channel_ids_resolved) > 0:
                        print(f"\t\t- manual channels - {len(manual_bad_channel_ids_resolved)}")
                # ================================================================

                all_bad_channel_ids = np.concatenate((dead_channel_ids, noise_channel_ids, out_channel_ids))

                skip_processing = False
                max_bad_channel_fraction = preprocessing_params["max_bad_channel_fraction"]
                if len(all_bad_channel_ids) >= int(max_bad_channel_fraction * recording.get_num_channels()):
                    print(f"\tMore than {max_bad_channel_fraction * 100}% bad channels ({len(all_bad_channel_ids)}). ")
                    preprocessing_notes += f"\n- Found {len(all_bad_channel_ids)} bad channels."
                    if preprocessing_params["remove_bad_channels"]:
                        skip_processing = True
                        print("\tSkipping further processing for this recording.")
                        preprocessing_notes += f" Skipping further processing for this recording.\n"
                    else:
                        preprocessing_notes += "\n"

                if not skip_processing:
                    if preprocessing_params["remove_out_channels"]:
                        print(f"\tRemoving {len(out_channel_ids)} out channels")
                        recording_rm_out = recording_filt_full.remove_channels(out_channel_ids)
                        preprocessing_notes += f"\n- Removed {len(out_channel_ids)} outside of the brain."
                    else:
                        recording_rm_out = recording_filt_full

                    if DESTRIPE_FLOAT32:
                        recording_rm_out = spre.astype(recording_rm_out, "float32")
                        print(f"\t[custom] EF fixes: interpolation / CMR / destripe on float32 (int16 AGC fix)")
                        preprocessing_notes += "\n- EF fixes: interpolation / CMR / destripe on float32, rounded to int16 after."

                    # Common reference denoising
                    print(f"\n\t[custom] === Common Reference (CMR) preprocessing ===")
                    try:
                        # recording_processed_cmr = spre.common_reference(
                        #     recording_rm_out, **preprocessing_params["common_reference"]
                        # )
                        recording_processed_cmr = apply_by_group(
                            recording_rm_out,
                            spre.common_reference,
                            preprocessing_params["common_reference"],
                            group_property="group",
                            func_name="common_reference (CMR)",
                        )
                        if recording_processed_cmr is None:
                            raise RuntimeError("common_reference returned None")
                        print(f"\t[custom] ✓ CMR preprocessing completed successfully")
                        print(f"\t[custom]   Output channels: {recording_processed_cmr.get_num_channels()}")
                    except Exception as e:
                        error_msg = f"Common reference preprocessing failed: {e}"
                        print(f"\t[custom] ERROR: {error_msg}")
                        raise RuntimeError(error_msg)

                    # ================================================================
                    # NEW: Combine auto-detected + manual bad channels for interpolation
                    # ================================================================
                    bad_channel_ids = np.concatenate((dead_channel_ids, noise_channel_ids))
                    
                    # Add manual bad channels (excluding out channels and duplicates)
                    if len(manual_bad_channel_ids_resolved) > 0:
                        # Filter out any that are already in bad_channel_ids or out_channel_ids
                        existing_bad = set(bad_channel_ids.tolist()) | set(out_channel_ids.tolist())
                        new_manual = [ch for ch in manual_bad_channel_ids_resolved if ch not in existing_bad]
                        if new_manual:
                            bad_channel_ids = np.concatenate((bad_channel_ids, np.array(new_manual)))
                            print(f"\t[custom] Added {len(new_manual)} manual bad channels for interpolation")
                            preprocessing_notes += f"\n- Added {len(new_manual)} manual bad channels."
                    # ================================================================

                    recording_interp = spre.interpolate_bad_channels(recording_rm_out, bad_channel_ids)
                    
                    # Destripe (highpass spatial filter) - protection against short probes
                    print(f"\n\t[custom] === Destripe (highpass spatial filter) preprocessing ===")
                    try:
                        # recording_hp_spatial = spre.highpass_spatial_filter(
                        #     recording_interp, **preprocessing_params["highpass_spatial_filter"]
                        # )
                        recording_hp_spatial = apply_by_group(
                            recording_interp,
                            spre.highpass_spatial_filter,
                            preprocessing_params["highpass_spatial_filter"],
                            group_property="group",
                            func_name="highpass_spatial_filter (destripe)",
                            kwargs_for_group=pad_to_group if DESTRIPE_PAD_TO_GROUP else None,
                        )
                        if recording_hp_spatial is None:
                            print(f"\t[custom] ⚠ Highpass spatial filter returned None")
                        else:
                            print(f"\t[custom] ✓ Destripe preprocessing completed successfully")
                            print(f"\t[custom]   Output channels: {recording_hp_spatial.get_num_channels()}")
                    except Exception as e:
                        print(f"\t[custom] ⚠ Highpass spatial filter failed: {e}")
                        recording_hp_spatial = None
                    
                    preprocessing_vizualization_data[recording_name]["timeseries"]["proc"] = dict(
                        highpass=recording_rm_out.to_dict(relative_to=data_folder, recursive=True),
                        cmr=recording_processed_cmr.to_dict(relative_to=data_folder, recursive=True),
                    )
                    if recording_hp_spatial is not None:
                        preprocessing_vizualization_data[recording_name]["timeseries"]["proc"].update(
                            dict(highpass_spatial=recording_hp_spatial.to_dict(relative_to=data_folder, recursive=True))
                        )

                    denoising_strategy = preprocessing_params["denoising_strategy"]
                    print(f"\n\t[custom] === Denoising strategy selection ===")
                    print(f"\t[custom] Requested strategy: {denoising_strategy}")
                    if denoising_strategy == "cmr":
                        recording_processed = recording_processed_cmr
                        print(f"\t[custom] ✓ Using CMR (common reference) denoising")
                    else:
                        # ================================================================
                        # FIX: Fall back to CMR if destripe (highpass_spatial) failed
                        # ================================================================
                        if recording_hp_spatial is not None:
                            recording_processed = recording_hp_spatial
                            print(f"\t[custom] ✓ Using destripe (highpass spatial filter) denoising")
                        else:
                            if DESTRIPE_STRICT:
                                raise RuntimeError("Destripe failed and --destripe-strict is set (see the error above)")
                            print(f"\t[custom] ⚠ Destripe failed, falling back to CMR")
                            recording_processed = recording_processed_cmr
                            denoising_strategy = "cmr"
                            preprocessing_notes += "\n- Destripe failed, fell back to CMR."
                            # EF fixes: record what actually ran (processing.json otherwise still says "destripe")
                            preprocessing_params["denoising_strategy"] = "cmr"
                            preprocessing_params["denoising_requested"] = "destripe"
                        # ================================================================
                    print(f"\t[custom] Final denoising strategy: {denoising_strategy}")
                    print(f"\t[custom] ===========================================\n")

                    # Safety check: ensure recording_processed is not None before removing channels
                    if recording_processed is None:
                        error_msg = (
                            f"recording_processed is None after denoising strategy '{denoising_strategy}'. "
                            f"This should not happen. recording_processed_cmr: {recording_processed_cmr is not None}, "
                            f"recording_hp_spatial: {recording_hp_spatial is not None}"
                        )
                        print(f"\t[custom] ERROR: {error_msg}")
                        raise RuntimeError(error_msg)

                    if preprocessing_params["remove_bad_channels"]:
                        print(f"\tRemoving {len(bad_channel_ids)} channels after {denoising_strategy} preprocessing")
                        recording_processed = recording_processed.remove_channels(bad_channel_ids)
                        preprocessing_notes += f"\n- Removed {len(bad_channel_ids)} bad channels after preprocessing.\n"

                    if DESTRIPE_FLOAT32 and np.dtype(recording_processed.get_dtype()) != np.dtype("int16"):
                        recording_processed = spre.astype(recording_processed, "int16", round=True)
                        print(f"\t[custom] EF fixes: rounded the processed recording back to int16")

                    # save to binary to speed up downstream processing
                    recording_bin = recording_processed.save(folder=preprocessing_output_folder)

                    # motion correction
                    recording_corrected = None
                    recording_bin_corrected = None
                    if motion_params["compute"]:
                        from spikeinterface.sortingcomponents.motion import interpolate_motion

                        preset = motion_params["preset"]
                        print(f"\tComputing motion correction with preset: {preset}")

                        detect_kwargs = motion_params.get("detect_kwargs", {})
                        select_kwargs = motion_params.get("select_kwargs", {})
                        localize_peaks_kwargs = motion_params.get("localize_peaks_kwargs", {})
                        estimate_motion_kwargs = motion_params.get("estimate_motion_kwargs", {})
                        interpolate_motion_kwargs = motion_params.get("interpolate_motion_kwargs", {})

                        motion_folder = results_folder / f"motion_{recording_name}"

                        try:
                            concat_motion = False
                            if recording_processed.get_num_segments() > 1:
                                recording_bin_c = si.concatenate_recordings([recording_bin])
                                recording_processed_c = si.concatenate_recordings([recording_processed])
                                concat_motion = True
                            else:
                                recording_bin_c = recording_bin
                                recording_processed_c = recording_processed

                            recording_bin_corrected, motion_info = spre.correct_motion(
                                recording_bin_c,
                                preset=preset,
                                folder=motion_folder,
                                output_motion_info=True,
                                detect_kwargs=detect_kwargs,
                                select_kwargs=select_kwargs,
                                localize_peaks_kwargs=localize_peaks_kwargs,
                                estimate_motion_kwargs=estimate_motion_kwargs,
                                interpolate_motion_kwargs=interpolate_motion_kwargs
                            )
                            recording_corrected = interpolate_motion(
                                recording_processed_c.astype("float32"),
                                motion=motion_info["motion"],
                                **interpolate_motion_kwargs
                            )

                            # split segments back
                            if concat_motion:
                                rec_corrected_list = []
                                rec_corrected_bin_list = []
                                for segment_index in range(recording_bin.get_num_segments()):
                                    num_samples = recording_bin.get_num_samples(segment_index)
                                    if segment_index == 0:
                                        start_frame = 0
                                    else:
                                        start_frame = recording_bin.get_num_samples(segment_index - 1)
                                    end_frame = start_frame + num_samples
                                    rec_split_corrected = recording_corrected.frame_slice(
                                        start_frame=start_frame,
                                        end_frame=end_frame
                                    )
                                    rec_corrected_list.append(rec_split_corrected)
                                    rec_split_bin = recording_bin_corrected.frame_slice(
                                        start_frame=start_frame,
                                        end_frame=end_frame
                                    )
                                    rec_corrected_bin_list.append(rec_split_bin)
                                # append all segments
                                recording_corrected = si.append_recordings(rec_corrected_list)
                                recording_bin_corrected = si.append_recordings(rec_corrected_bin_list)

                            if motion_params["apply"]:
                                print(f"\tApplying motion correction")
                                recording_processed = recording_corrected
                                recording_bin = recording_bin_corrected
                        except Exception as e:
                            print(f"\tMotion correction failed:\n\t{e}")
                            recording_corrected = None
                            recording_bin_corrected = None

                    # this is used to reload the binary traces downstream
                    dump_to_json_or_pickle(
                        recording_bin,
                        results_folder,
                        binary_output_filename,
                        relative_to=results_folder
                    )

                    # this is to reload the recordings lazily            
                    dump_to_json_or_pickle(
                        recording_processed,
                        results_folder,
                        preprocessing_output_filename,
                        relative_to=results_folder
                    )

                    # this is to reload the motion-corrected recording lazily
                    if recording_corrected is not None:     
                        dump_to_json_or_pickle(
                            recording_corrected,
                            results_folder,
                            motioncorrected_output_filename,
                            relative_to=results_folder
                        )

                    recording_drift = recording_bin
                    drift_relative_folder = results_folder

            if skip_processing:
                # in this case, processed timeseries will not be visualized
                preprocessing_vizualization_data[recording_name]["timeseries"]["proc"] = None
                recording_drift = recording_filt_full
                drift_relative_folder = data_folder
                # make a dummy file if too many bad channels to skip downstream processing
                preprocessing_output_folder.mkdir()
                error_file = preprocessing_output_folder / "error.txt"
                error_file.write_text("Too many bad channels")

            # store recording for drift visualization
            preprocessing_vizualization_data[recording_name]["drift"] = dict(
                recording=recording_drift.to_dict(relative_to=drift_relative_folder, recursive=True)
            )

            if vizualization_file_is_json_serializable:            
                with open(results_folder / f"{preprocessingviz_output_filename}.json", "w") as f:
                    json.dump(check_json(preprocessing_vizualization_data), f, indent=4)
            else:
                # then dump to pickle
                with open(results_folder / f"{preprocessingviz_output_filename}.pkl", "wb") as f:
                    pickle.dump(preprocessing_vizualization_data, f)

            t_preprocessing_end = time.perf_counter()
            elapsed_time_preprocessing = np.round(t_preprocessing_end - t_preprocessing_start, 2)

            # save params in output
            preprocessing_params["recording_name"] = recording_name
            # NEW: Add manual bad channels to saved params
            preprocessing_params["manual_bad_channel_ids"] = session_manual_bad_channels
            
            if channel_labels is not None:
                preprocessing_outputs = dict(
                    channel_labels=channel_labels.tolist(),
                )
            else:
                preprocessing_outputs = dict()
            preprocessing_process = DataProcess(
                name="Ephys preprocessing",
                software_version=VERSION,  # either release or git commit
                start_date_time=datetime_start_preproc,
                end_date_time=datetime_start_preproc + timedelta(seconds=np.floor(elapsed_time_preprocessing)),
                input_location=str(data_folder),
                output_location=str(results_folder),
                code_url=URL,
                parameters=preprocessing_params,
                outputs=preprocessing_outputs,
                notes=preprocessing_notes,
            )
            with open(preprocessing_output_process_json, "w") as f:
                f.write(preprocessing_process.model_dump_json(indent=3))

        t_preprocessing_end_all = time.perf_counter()
        elapsed_time_preprocessing_all = np.round(t_preprocessing_end_all - t_preprocessing_start_all, 2)

        print(f"PREPROCESSING time: {elapsed_time_preprocessing_all}s")
