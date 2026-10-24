#!/bin/bash

# ================================
# Run on Kempner cluster with this command (tmux recommended for running in background):
# tmux new -s spikesort
# pipeline/shun_spikesort_pipeline.sh pipeline/spike_sort.slrm
# # detach with:  Ctrl-b d (both mac and windows)
# # reattach later: tmux attach -t spikesort
# ================================

# Input arguments
#   $1        : Slurm job description file
#   $2        : Optional user profile name (from pipeline/user_profiles.conf)
#   $3 ... $N : Optional knob overrides as KEY=VALUE pairs
#               e.g.  BANDPASS_FILTER=false  BANDPASS_FREQ_MAX=6000
#
# Examples:
#   pipeline/shun_spikesort_pipeline.sh pipeline/spike_sort.slrm
#   pipeline/shun_spikesort_pipeline.sh pipeline/spike_sort.slrm emily
#   pipeline/shun_spikesort_pipeline.sh pipeline/spike_sort.slrm shunnnli BANDPASS_FILTER=false
#   pipeline/shun_spikesort_pipeline.sh pipeline/spike_sort.slrm emily BANDPASS_FREQ_MAX=10000
Slurm_file=$1
user_profile=$2
shift 2 2>/dev/null || shift $# 2>/dev/null || true

# Parse remaining positional args as KEY=VALUE knob overrides (highest priority)
declare -A cli_knob_overrides
for arg in "$@"; do
    if [[ "$arg" == *=* ]]; then
        cli_key="${arg%%=*}"
        cli_val="${arg#*=}"
        cli_knob_overrides[$cli_key]="$cli_val"
    fi
done

if [ -z "$Slurm_file" ]; then
  echo "Usage: $0 <slurm_file> [user_profile] [KEY=VALUE ...]"
  echo "  <slurm_file>   : Path to the base Slurm template (e.g., pipeline/spike_sort.slrm)"
  echo "  [user_profile] : Optional profile from pipeline/user_profiles.conf"
  echo "  [KEY=VALUE]    : Optional knob overrides (space-separated, highest priority)"
  echo "                   Available knobs: DENOISING_METHOD REMOVE_BAD_CHANNELS"
  echo "                                    BANDPASS_FILTER BANDPASS_FREQ_MIN BANDPASS_FREQ_MAX EF_FIXES"
  exit 1
fi

# ================================
# Generate job submission files
# ================================
# Resolve the Slurm file to an absolute path so we don't accidentally
# double-prefix it with the pipeline directory (e.g., pipeline/pipeline/...).
Slurm_file_path=$(realpath "$Slurm_file")

echo "Generating job submission files from the Slurm file: $Slurm_file_path"

# Extract relevant paths from the Slurm file (defaults)
top_dir=$(grep "^DATA_PATH" "$Slurm_file_path" | sed "s/DATA_PATH=//g" | tr -d '"')
work_dir=$(grep "^WORK_DIR" "$Slurm_file_path" | sed "s/WORK_DIR=//g" | tr -d '"')
out_dir=$(grep "^RESULTS_PATH" "$Slurm_file_path" | sed "s/RESULTS_PATH=//g" | tr -d '"')
backup_dir=$(grep "^BACKUP_PATH" "$Slurm_file_path" | sed "s/BACKUP_PATH=//g" | tr -d '"')

# Extract PIPELINE_PATH (expected to be "./" from your file)
pipeline_path=$(grep "^PIPELINE_PATH" "$Slurm_file_path" | sed "s/PIPELINE_PATH=//g" | tr -d '"')
# Resolve relative PIPELINE_PATH to an absolute path (based on current directory)
pipeline_code_dir=$(realpath "$pipeline_path")

# Optional user profiles config
user_profiles_config="${pipeline_code_dir}/user_profiles.conf"

# If a user profile was provided, override the default paths.
# The lookup order is:
#   1) If user_profiles.conf exists and contains "<profile>=/some/base/path",
#      use that as the base path.
#   2) Otherwise, fall back to:
#      /n/netscratch/bsabatini_lab/Lab/<profile>/spikesorting
if [ -n "$user_profile" ]; then
    echo ""
    echo "Using user profile: $user_profile"
    user_base=""

    if [ -f "$user_profiles_config" ]; then
        profile_base=$(awk -F= -v p="$user_profile" '$1==p {print $2}' "$user_profiles_config" | tail -n 1)
        if [ -n "$profile_base" ]; then
            user_base="$profile_base"
            echo "  Loaded base path from user_profiles.conf: $user_base"
        else
            echo "  Profile '$user_profile' not found in user_profiles.conf; using default pattern."
        fi
    else
        echo "  No user_profiles.conf found at: $user_profiles_config"
        echo "  Using default pattern for user base."
    fi

    if [ -z "$user_base" ]; then
        user_base="/n/netscratch/bsabatini_lab/Lab/${user_profile}/spikesorting"
    fi

    # Normalize: ensure no trailing slash
    user_base="${user_base%/}"

    top_dir="${user_base}/aind_input/todo/"
    work_dir="${user_base}/work_dir/"
    out_dir="${user_base}/aind_output_scratch/"
    backup_dir="${user_base}/aind_input/"
    echo "  DATA_PATH    -> $top_dir"
    echo "  WORK_DIR     -> $work_dir"
    echo "  RESULTS_PATH -> $out_dir"
    echo "  BACKUP_PATH  -> $backup_dir"

    # Look up per-user knob overrides.
    # Priority (highest wins):
    #   1. KEY=VALUE positional args on the command line
    #   2. <profile>.<KNOB> in user_profiles.conf
    #   3. Default value in spike_sort.slrm
    declare -A user_knob_overrides
    knob_names=(DENOISING_METHOD REMOVE_BAD_CHANNELS BANDPASS_FILTER BANDPASS_FREQ_MIN BANDPASS_FREQ_MAX EF_FIXES)
    if [ -f "$user_profiles_config" ]; then
        for knob in "${knob_names[@]}"; do
            val=$(awk -F= -v key="${user_profile}.${knob}" '$1==key {print $2}' "$user_profiles_config" | tail -n 1)
            if [ -n "$val" ]; then
                user_knob_overrides[$knob]="$val"
                echo "  $knob -> $val (from user_profiles.conf)"
            fi
        done
    fi
    # Apply command-line KEY=VALUE overrides on top (highest priority)
    for knob in "${!cli_knob_overrides[@]}"; do
        user_knob_overrides[$knob]="${cli_knob_overrides[$knob]}"
        echo "  $knob -> ${cli_knob_overrides[$knob]} (command-line override)"
    done
    echo ""
else
    # No explicit user profile: treat 'shunnnli' as the default logical profile
    # for purposes of locating job folders and download paths, using
    # user_profiles.conf as the single source of truth when available.
    DEFAULT_PROFILE="shunnnli"
    if [ -f "$user_profiles_config" ]; then
        profile_base=$(awk -F= -v p="$DEFAULT_PROFILE" '$1==p {print $2}' "$user_profiles_config" | tail -n 1)
        if [ -n "$profile_base" ]; then
            user_base="${profile_base%/}"
        fi
    fi
    # No conf overrides for the default profile, but apply any command-line KEY=VALUE overrides
    declare -A user_knob_overrides
    knob_names=(DENOISING_METHOD REMOVE_BAD_CHANNELS BANDPASS_FILTER BANDPASS_FREQ_MIN BANDPASS_FREQ_MAX EF_FIXES)
    for knob in "${!cli_knob_overrides[@]}"; do
        user_knob_overrides[$knob]="${cli_knob_overrides[$knob]}"
        echo "  $knob -> ${cli_knob_overrides[$knob]} (command-line override)"
    done
fi

# Load session-specific EXCLUDE_LAST_SEC and EXCLUDE_FIRST_SEC values from config file
exclude_config_file="${pipeline_code_dir}/exclude_seconds.conf"
declare -A exclude_map
DEFAULT_EXCLUDE_LAST_SEC=0
DEFAULT_EXCLUDE_FIRST_SEC=0

if [ -f "$exclude_config_file" ]; then
    echo "Loading exclude seconds config from: $exclude_config_file"
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        # Trim whitespace
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        if [ "$key" = "DEFAULT" ]; then
            # Parse DEFAULT value (format: first:last or just last)
            if [[ "$value" == *":"* ]]; then
                DEFAULT_EXCLUDE_FIRST_SEC="${value%%:*}"
                DEFAULT_EXCLUDE_LAST_SEC="${value#*:}"
            else
                DEFAULT_EXCLUDE_LAST_SEC="$value"
                DEFAULT_EXCLUDE_FIRST_SEC="0"
            fi
        else
            exclude_map["$key"]="$value"
        fi
    done < "$exclude_config_file"
    echo "Loaded ${#exclude_map[@]} session-specific exclude values (default: last=${DEFAULT_EXCLUDE_LAST_SEC}s, first=${DEFAULT_EXCLUDE_FIRST_SEC}s)"
else
    echo "⚠️  No exclude_seconds.conf found; using default EXCLUDE_LAST_SEC and EXCLUDE_FIRST_SEC from slurm file"
fi

# Load session-specific bad channels from config file (same pattern as exclude_seconds)
bad_channels_config_file="${pipeline_code_dir}/bad_channels.conf"
declare -A bad_channels_map

echo "DEBUG: Looking for bad_channels.conf at: $bad_channels_config_file"
if [ -f "$bad_channels_config_file" ]; then
    echo "✅ Found bad_channels.conf, loading..."
    line_num=0
    while IFS='=' read -r key value || [ -n "$key" ]; do
        line_num=$((line_num + 1))
        # Skip comments and empty lines (same as exclude_seconds)
        if [[ "$key" =~ ^#.*$ ]] || [[ -z "$key" ]]; then
            continue
        fi
        # Trim whitespace
        key_orig="$key"
        value_orig="$value"
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        # Debug output
        # echo "  DEBUG Line $line_num: key_orig='$key_orig', value_orig='$value_orig'"
        # echo "  DEBUG Line $line_num: key_trimmed='$key', value_trimmed='$value'"
        # Skip if key or value is empty after trimming
        if [[ -z "$key" ]] || [[ -z "$value" ]]; then
            echo "  DEBUG Line $line_num: SKIPPED (empty after trim)"
            continue
        fi
        bad_channels_map["$key"]="$value"
        # echo "  LOADED pattern '$key' with channels: $value"
    done < "$bad_channels_config_file"
    echo "✅ Loaded ${#bad_channels_map[@]} session-specific bad channel entries"
    if [ "${#bad_channels_map[@]}" -gt 0 ]; then
        echo "   Loaded patterns:"
        for pattern in "${!bad_channels_map[@]}"; do
            echo "     - '$pattern' = ${bad_channels_map[$pattern]}"
        done
    else
        echo "   ⚠️  WARNING: File exists but 0 entries loaded (all lines were skipped)"
    fi
else
    echo "❌ File not found: $bad_channels_config_file"
    echo "   Please create bad_channels.conf in the pipeline directory"
    echo "   Or if it exists, check the path above"
fi

# Function to look up bad channels for a given folder name
# Returns a comma-separated list of channel IDs (e.g., "20,206,3,313")
get_bad_channels_for_session() {
    local folder_name="$1"
    local matched_channels=""
    local all_channels=()
    
    echo "  Checking folder_name='$folder_name' against ${#bad_channels_map[@]} patterns" >&2
    
    for pattern in "${!bad_channels_map[@]}"; do
        # echo "    DEBUG: Testing pattern='$pattern'" >&2
        if echo "$folder_name" | grep -qi "$pattern"; then
            # echo "    DEBUG: MATCH! Pattern '$pattern' matches folder '$folder_name'" >&2
            # Extract channel list from the map value (remove spaces, split by comma)
            local channels="${bad_channels_map[$pattern]}"
            # Split by comma and add each channel to our array
            IFS=',' read -ra CHANNELS <<< "$channels"
            for ch in "${CHANNELS[@]}"; do
                ch_trimmed=$(echo "$ch" | xargs)  # trim whitespace
                if [ -n "$ch_trimmed" ]; then
                    all_channels+=("$ch_trimmed")
                fi
            done
        fi
    done
    
    # Remove duplicates while preserving order
    if [ ${#all_channels[@]} -gt 0 ]; then
        # Use awk to remove duplicates
        matched_channels=$(printf '%s\n' "${all_channels[@]}" | awk '!seen[$0]++' | tr '\n' ',' | sed 's/,$//')
    fi
    
    echo "$matched_channels"
}

# Function to look up exclude seconds value for a given folder name
# Returns the raw value string (format: first:last or just last)
get_exclude_seconds_value() {
    local folder_name="$1"
    # Try exact match first
    if [ -n "${exclude_map[$folder_name]}" ]; then
        echo "${exclude_map[$folder_name]}"
        return
    fi
    
    # Collect all partial matches (case-insensitive)
    local matching_patterns=()
    local matching_values=()
    for pattern in "${!exclude_map[@]}"; do
        if echo "$folder_name" | grep -qi "$pattern"; then
            matching_patterns+=("$pattern")
            matching_values+=("${exclude_map[$pattern]}")
        fi
    done
    
    # Check how many matches we found
    local num_matches=${#matching_patterns[@]}
    
    if [ "$num_matches" -eq 0 ]; then
        # No match found, return default
        echo "${DEFAULT_EXCLUDE_FIRST_SEC}:${DEFAULT_EXCLUDE_LAST_SEC}"
        return
    fi
    
    if [ "$num_matches" -gt 1 ]; then
        # Multiple matches - warn and use the longest pattern (most specific)
        echo "  ⚠️  WARNING: Multiple patterns match '$folder_name':" >&2
        for i in "${!matching_patterns[@]}"; do
            echo "      - ${matching_patterns[$i]}=${matching_values[$i]}" >&2
        done
        
        # Find the longest pattern (most specific match)
        local best_idx=0
        local best_len=${#matching_patterns[0]}
        for i in "${!matching_patterns[@]}"; do
            local len=${#matching_patterns[$i]}
            if [ "$len" -gt "$best_len" ]; then
                best_len=$len
                best_idx=$i
            fi
        done
        echo "      Using longest match: ${matching_patterns[$best_idx]}=${matching_values[$best_idx]}" >&2
        echo "${matching_values[$best_idx]}"
        return
    fi
    
    # Single match
    echo "${matching_values[0]}"
}

# Function to get EXCLUDE_FIRST_SEC for a given folder name
get_exclude_first_seconds() {
    local value=$(get_exclude_seconds_value "$1")
    # Parse value (format: first:last or just last)
    if [[ "$value" == *":"* ]]; then
        echo "${value%%:*}"
    else
        echo "0"
    fi
}

# Function to get EXCLUDE_LAST_SEC for a given folder name
get_exclude_last_seconds() {
    local value=$(get_exclude_seconds_value "$1")
    # Parse value (format: first:last or just last)
    if [[ "$value" == *":"* ]]; then
        echo "${value#*:}"
    else
        echo "$value"
    fi
}

echo "Top directory: $top_dir"
echo "Pipeline code path: $pipeline_code_dir"

# ================================
# Display preprocessing configuration
# ================================
# Extract USE_CUSTOM_PREPROCESSING setting from the slurm file
use_custom_preprocessing=$(grep "^USE_CUSTOM_PREPROCESSING" "$Slurm_file_path" | sed 's/USE_CUSTOM_PREPROCESSING=//g' | tr -d '"')
use_custom_preprocessing=${use_custom_preprocessing:-"true"}  # Default to true if not set

echo ""
echo "=========================================="
echo "PREPROCESSING CONFIGURATION"
echo "=========================================="

# Check custom preprocessing setting
custom_script_path="${pipeline_code_dir}/run_capsule_custom.py"
if [ "$use_custom_preprocessing" = "true" ]; then
    if [ -f "$custom_script_path" ]; then
        echo "✅ USE_CUSTOM_PREPROCESSING: true"
        echo "   Custom script: $custom_script_path"
    else
        echo "⚠️  USE_CUSTOM_PREPROCESSING: true (BUT SCRIPT NOT FOUND!)"
        echo "   Expected at: $custom_script_path"
        echo "   Will FAIL - please create the custom script or set USE_CUSTOM_PREPROCESSING=\"false\""
    fi
    
else
    echo "ℹ️  USE_CUSTOM_PREPROCESSING: false"
    echo "   Using DEFAULT preprocessing from aind-ephys-preprocessing"
    echo "   (Bad channels config will be ignored)"
fi
echo "=========================================="
echo ""

# Define where to save the pipeline job folders
# If a user profile/base is active, put job folders under that base so they are
# kept with that user's spikesorting tree. Otherwise, default to the pipeline
# code directory.
if [ -n "${user_base:-}" ]; then
    pipeline_save_path="${user_base%/}/pipeline"
else
    pipeline_save_path="${pipeline_code_dir%/}/pipeline_saved"
fi
mkdir -p "$pipeline_save_path"

# Limit concurrent spike-sort jobs to avoid Kempner GPU exhaustion (QOSMaxGRESPerUser).
# Each session uses ~2 GPUs (parent + 1 child). With kempner MaxTRESPU=16, 8 sessions fit.
MAX_CONCURRENT_JOBS=${MAX_CONCURRENT_JOBS:-8}

# Arrays to collect sessions to process (job_folder and script name)
declare -a pending_job_folders=()
declare -a pending_job_scripts=()
declare -a job_ids=()

# Get all immediate subdirectories under DATA_PATH
dir_data_array=( $(find "$top_dir" -mindepth 1 -maxdepth 1 -type d) )
dir_array_length=${#dir_data_array[@]}

echo ""
echo "Found $dir_array_length data directories"
echo "Will process in batches of ${MAX_CONCURRENT_JOBS} (set MAX_CONCURRENT_JOBS to override)"
echo ""

for element in "${dir_data_array[@]}"
do
    echo ""
    echo "Processing directory: $element"
    folder_name=$(basename "$element")

    # Check if this recording session already has spike-sorted output.
    # We assume that if the session-specific RESULTS_PATH folder has a
    # 'spikesorted' subdirectory, spike sorting has already completed.
    existing_results_folder="${out_dir%/}/${folder_name}_output"
    if [ -d "${existing_results_folder}/spikesorted" ]; then
        echo "  Detected existing spike-sorted output at ${existing_results_folder}/spikesorted"
        echo "  Skipping spike-sorting job for ${folder_name}; post-processing will still run."
        echo ""
        continue
    fi

    new_data_path="DATA_PATH=\"${element}\""
    new_results_path="RESULTS_PATH=\"${out_dir%/}/${folder_name}_output\""
    new_pipeline_path="PIPELINE_PATH=\"${pipeline_code_dir}\""

    # Look up session-specific EXCLUDE_LAST_SEC and EXCLUDE_FIRST_SEC
    session_exclude_last_sec=$(get_exclude_last_seconds "$folder_name")
    session_exclude_first_sec=$(get_exclude_first_seconds "$folder_name")
    echo "  EXCLUDE_LAST_SEC for ${folder_name}: ${session_exclude_last_sec}s"
    echo "  EXCLUDE_FIRST_SEC for ${folder_name}: ${session_exclude_first_sec}s"
    
    # Look up session-specific bad channels using pre-loaded map (only when using custom preprocessing)
    session_bad_channels=""
    if [ "$use_custom_preprocessing" = "true" ]; then
        session_bad_channels=$(get_bad_channels_for_session "$folder_name")
        if [ -n "$session_bad_channels" ]; then
            echo "  BAD_CHANNELS for ${folder_name}: $session_bad_channels"
        else
            echo "  BAD_CHANNELS for ${folder_name}: (none - no matching pattern)"
        fi
    fi

    # Create job folder inside pipeline_save_path
    job_folder="${pipeline_save_path}/pipeline_${folder_name}"
    job_slurm_script="spike_sort.slrm.${folder_name}"

    mkdir -p "$job_folder"
    cd "$job_folder"

    # Copy the original Slurm file (using its absolute path)
    cp "$Slurm_file_path" 1.tmp.slrm
    sed "s|^RESULTS_PATH=.*|$new_results_path|g" 1.tmp.slrm > 2.tmp.slrm
    sed "s|^DATA_PATH=.*|$new_data_path|g" 2.tmp.slrm > 3.tmp.slrm
    sed "s|^PIPELINE_PATH=.*|$new_pipeline_path|g" 3.tmp.slrm > 4.tmp.slrm
    sed "s|^WORK_DIR=.*|WORK_DIR=\"${work_dir}\"|g" 4.tmp.slrm > 5.tmp.slrm
    # Update EXCLUDE_LAST_SEC with session-specific value
    sed "s|^EXCLUDE_LAST_SEC=.*|EXCLUDE_LAST_SEC=\"${session_exclude_last_sec}\"|g" 5.tmp.slrm > 6.tmp.slrm
    # Update EXCLUDE_FIRST_SEC with session-specific value
    sed "s|^EXCLUDE_FIRST_SEC=.*|EXCLUDE_FIRST_SEC=\"${session_exclude_first_sec}\"|g" 6.tmp.slrm > 7.tmp.slrm
    # Update BAD_CHANNELS with session-specific value (empty string if none)
    sed "s|^BAD_CHANNELS=.*|BAD_CHANNELS=\"${session_bad_channels}\"|g" 7.tmp.slrm > 8.tmp.slrm
    # Inject per-user knob overrides (e.g. BANDPASS_FREQ_MAX, BANDPASS_FILTER, etc.)
    cp 8.tmp.slrm 9.tmp.slrm
    for knob in "${!user_knob_overrides[@]}"; do
        sed "s|^${knob}=.*|${knob}=\"${user_knob_overrides[$knob]}\"|g" 9.tmp.slrm > 10.tmp.slrm
        mv 10.tmp.slrm 9.tmp.slrm
    done
    mv 9.tmp.slrm "$job_slurm_script"
    rm -f 1.tmp.slrm 2.tmp.slrm 3.tmp.slrm 4.tmp.slrm 5.tmp.slrm 6.tmp.slrm 7.tmp.slrm 8.tmp.slrm
    
    # Extract base PREPROCESSING_ARGS from the generated script (before bad channels are appended)
    base_preprocessing_args=$(grep "^PREPROCESSING_ARGS=" "$job_slurm_script" | head -n 1 | sed 's/PREPROCESSING_ARGS=//' | tr -d '"')
    
    # Show what PREPROCESSING_ARGS will look like when executed
    if [ -n "$session_bad_channels" ] && [ "$use_custom_preprocessing" = "true" ]; then
        final_preprocessing_args="${base_preprocessing_args} --bad-channel-ids ${session_bad_channels}"
        echo "  PREPROCESSING_ARGS (final): \"${final_preprocessing_args}\""
    else
        echo "  PREPROCESSING_ARGS: \"${base_preprocessing_args}\""
    fi

    echo "  Prepared $job_slurm_script (data: ${element})"
    pending_job_folders+=("$job_folder")
    pending_job_scripts+=("$job_slurm_script")

    # Return to the pipeline code directory before processing the next folder
    cd "$pipeline_code_dir"
done

# Submit and run in batches to avoid GPU exhaustion
total_pending=${#pending_job_folders[@]}
if [ "$total_pending" -gt 0 ]; then
    echo ""
    echo "Submitting ${total_pending} spike-sorting jobs in batches of ${MAX_CONCURRENT_JOBS}..."
    batch_num=0
    offset=0
    while [ "$offset" -lt "$total_pending" ]; do
        batch_num=$((batch_num + 1))
        end=$((offset + MAX_CONCURRENT_JOBS))
        [ "$end" -gt "$total_pending" ] && end=$total_pending
        batch_size=$((end - offset))

        echo ""
        echo "=== Batch $batch_num: submitting $batch_size job(s) ==="
        job_ids=()
        for ((i=offset; i<end; i++)); do
            jf="${pending_job_folders[$i]}"
            js="${pending_job_scripts[$i]}"
            cd "$jf"
            echo "  Submitting $js"
            sbatch_output=$(sbatch "$js")
            if [ $? -ne 0 ]; then
                echo "❌ Failed to submit job: $js"
                echo "$sbatch_output"
                cd "$pipeline_code_dir"
                exit 1
            fi
            echo "    $sbatch_output"
            jid=$(echo "$sbatch_output" | awk '{print $4}')
            [[ -n "$jid" ]] && job_ids+=("$jid")
            cd "$pipeline_code_dir"
        done

        # Wait for this batch to complete
        if [ "${#job_ids[@]}" -gt 0 ]; then
            echo "  Waiting for batch $batch_num to complete..."
            while true; do
                all_done=true
                running=0
                pending=0
                completed=0
                other=0
                for jid in "${job_ids[@]}"; do
                    state=$(squeue -h -j "$jid" -o "%t" 2>/dev/null | head -n 1)
                    if [ -z "$state" ]; then
                        completed=$((completed + 1))
                        continue
                    fi
                    all_done=false
                    case "$state" in
                        PD) pending=$((pending + 1)) ;;
                        R|CG) running=$((running + 1)) ;;
                        *) other=$((other + 1)) ;;
                    esac
                done
                if $all_done; then break; fi
                echo "  Job status: total=${#job_ids[@]}, running=${running}, pending=${pending}, completed=${completed}, other=${other}"
                echo "  Waiting for 60 seconds before next status check..."
                sleep 60
            done
            echo "  Batch $batch_num complete."
        fi
        offset=$end
    done
    echo ""
    echo "All spike-sorting jobs have been submitted and completed."
else
    echo "No new spike-sorting jobs to submit (all sessions already have output or were skipped)."
fi

# Track overall success: start with true, set to false if any job fails
overall_success=true

# Jobs are waited on per-batch above; proceed to verification and post-processing.
echo ""
echo "Verifying spike-sorting outputs..."
missing_outputs=0
for element in "${dir_data_array[@]}"; do
    folder_name=$(basename "$element")
    results_folder="${out_dir%/}/${folder_name}_output"

    # Check if spikesorted directory exists (main indicator of success)
    if [ ! -d "${results_folder}/spikesorted" ]; then
        echo "⚠️  Missing spikesorted directory for ${folder_name}: ${results_folder}/spikesorted"
        missing_outputs=$((missing_outputs + 1))
        overall_success=false
    else
        echo "✅ Found spikesorted directory for ${folder_name}"
    fi

    # Also check for preprocessed directory (required for post-processing)
    if [ ! -d "${results_folder}/preprocessed" ]; then
        echo "⚠️  Missing preprocessed directory for ${folder_name}: ${results_folder}/preprocessed"
        missing_outputs=$((missing_outputs + 1))
        overall_success=false
    fi
done

if [ "$missing_outputs" -gt 0 ]; then
    echo ""
    echo "⚠️  WARNING: $missing_outputs session(s) missing expected output directories."
    echo "   The pipeline will continue but files will NOT be moved from todo folder."
fi


# ================================
# Run the AIND export / post-processing script once everything is done
# ================================
echo ""
# The postprocess scripts live at the repository root under "postprocess",
# while PIPELINE_PATH points to the "pipeline" subdirectory. So we take the
# parent of pipeline_code_dir as the repo root.
repo_root_dir=$(dirname "$pipeline_code_dir")
postprocess_script_path="${repo_root_dir%/}/postprocess/extract_aind_output.py"
if [ ! -f "$postprocess_script_path" ]; then
    echo "❌ Post-processing script not found at: $postprocess_script_path"
    echo "Please verify the location of extract_aind_output.py."
    exit 1
fi

# Ensure extract_aind_output.py sees the directory containing all recording sessions.
# This will make it process *every* recording session under this top-level directory.
export AIND_INPUT_BASE_DIR="$top_dir"
export AIND_OUTPUT_BASE_DIR="$out_dir"

echo "Starting post-processing with: $postprocess_script_path"
echo "  Using AIND_INPUT_BASE_DIR=${AIND_INPUT_BASE_DIR}"
echo "  Using AIND_OUTPUT_BASE_DIR=${AIND_OUTPUT_BASE_DIR}"

# Load cluster Python module and activate the mamba env "spikeinterface"
if command -v module >/dev/null 2>&1; then
    module load python || echo "⚠️ Failed to 'module load python'; please verify your module name."
fi

if command -v mamba >/dev/null 2>&1; then
    # Initialize mamba in this non-interactive shell and activate env
    eval "$(mamba shell hook -s bash)" 2>/dev/null || true
    mamba activate spikeinterface || echo "⚠️ Failed to activate mamba env 'spikeinterface'; assuming environment is already correct."
else
    echo "⚠️ 'mamba' command not found; assuming correct Python environment already active."
fi

# Use python3 explicitly for post-processing
if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ python3 not found in PATH; cannot run post-processing."
    exit 1
fi

python3 "$postprocess_script_path"
post_status=$?

if [ $post_status -ne 0 ]; then
    echo "❌ Post-processing script exited with status $post_status"
    overall_success=false
    # Continue to show what was processed, but don't exit yet
else
    echo "✅ Post-processing completed successfully."
fi

# Check if post-processing actually produced outputs (even if exit code was 0)
# This handles cases where the script completes but skips sessions due to missing data
echo ""
echo "Verifying post-processing outputs..."
postprocessing_verified=true
for element in "${dir_data_array[@]}"; do
    folder_name=$(basename "$element")
    results_folder="${out_dir%/}/${folder_name}_output"
    preprocessed_folder="${results_folder}/preprocessed"
    
    # Check if preprocessed folder exists and has experiment JSONs
    if [ -d "$preprocessed_folder" ]; then
        # Use ls with shell glob expansion for reliable pattern matching
        # This matches the pattern used by the Python script: block0_imec*.ap_recording1*.json
        # Shell glob expansion handles multiple wildcards better than find -name
        if ls "$preprocessed_folder"/block0_imec*.ap_recording*.json 1>/dev/null 2>&1; then
            experiment_files=$(ls "$preprocessed_folder"/block0_imec*.ap_recording*.json 2>/dev/null | wc -l)
            echo "✅ Found $experiment_files experiment JSON file(s) for ${folder_name}"
        else
            # Also check if any .json files exist at all (for debugging)
            any_json=$(find "$preprocessed_folder" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)
            if [ "$any_json" -gt 0 ]; then
                echo "⚠️  No experiment JSONs matching pattern for ${folder_name} in ${preprocessed_folder}"
                echo "   (Found $any_json other JSON file(s), but not matching expected pattern)"
                echo "   Looking for files like: block0_imec*.ap_recording*.json"
            else
                echo "⚠️  No experiment JSONs found for ${folder_name} in ${preprocessed_folder}"
            fi
            postprocessing_verified=false
            overall_success=false
        fi
    else
        echo "⚠️  Preprocessed folder missing for ${folder_name}: ${preprocessed_folder}"
        postprocessing_verified=false
        overall_success=false
    fi
done

if [ "$postprocessing_verified" = false ]; then
    echo ""
    echo "⚠️  WARNING: Post-processing did not produce expected outputs for some sessions."
    echo "   Files will NOT be moved from todo folder."
fi


# ================================
# Copy per-session output folders to download location
# ================================
echo ""

# Determine where to place the download copies.
# If a user profile/base was used, put downloads under that base.
# Otherwise, derive a sane default from RESULTS_PATH (out_dir).
if [ -n "${user_base:-}" ]; then
    copy_dest_base="${user_base}/aind_output_fordownload"
else
    out_parent_dir=$(dirname "${out_dir%/}")
    copy_dest_base="${out_parent_dir}/aind_output_fordownload"
fi

echo "Copying session output folders to: $copy_dest_base"
mkdir -p "$copy_dest_base"

for element in "${dir_data_array[@]}"; do
    folder_name=$(basename "$element")
    src_folder="${out_dir%/}/${folder_name}_output"

    if [ -d "$src_folder" ]; then
        echo "  Copying $src_folder -> $copy_dest_base/ (following symlinks)"
        # Use -L to follow symlinks so that the *contents* of linked folders
        # are copied, not the symlinks themselves. This makes the download
        # directory self-contained on your local machine.
        cp -rL "$src_folder" "$copy_dest_base/"
    else
        echo "  Skipping $folder_name: source output folder not found at $src_folder"
    fi
done

echo "✅ Finished copying all available session output folders."


# ================================
# Move recording files from todo folder to aind_input folder
# ONLY if all spike-sorting and post-processing succeeded
# ================================
echo ""
if [ "$overall_success" = true ]; then
    echo "All spike-sorting and post-processing checks passed."
    echo "Moving recording files from todo folder to the parent folder (aind_input)..."
    if [ -d "$top_dir" ] && [ "$(ls -A "$top_dir" 2>/dev/null)" ]; then
        mv "$top_dir"/* "$backup_dir" 2>/dev/null || true
        echo "✅ Finished moving recording files from todo folder to the parent folder (aind_input)."
    else
        echo "⚠️  No files to move (todo folder is empty or doesn't exist)."
    fi

    # Optionally clean up Nextflow work directory for this user/profile now that
    # all sessions have completed successfully.
    if [ -n "${work_dir:-}" ] && [ -d "$work_dir" ]; then
        echo ""
        echo "Cleaning up Nextflow work directory to free space:"
        echo "  Deleting contents of: $work_dir"
        # Remove all contents but keep the directory itself.
        rm -rf "${work_dir}/"* 2>/dev/null || true
        echo "✅ Finished cleaning work directory."
    fi
else
    echo "❌ Spike-sorting or post-processing encountered errors."
    echo "   NOT moving recording files from todo folder to parent folder."
    echo "   Files remain in: $top_dir"
    echo "   Please investigate the errors above before retrying."
    echo ""
    echo "Summary of issues:"
    echo "   - Check job exit codes and output directories above"
    echo "   - Verify that preprocessing completed successfully"
    echo "   - Ensure all expected output directories exist"
    echo ""
    echo "Where to look for detailed error logs:"
    echo "   - Slurm job logs: ${pipeline_save_path}/pipeline_<SESSION_NAME>/ephys-*.out and ephys-*.err"
    echo "   - Per-session Nextflow log: ${out_dir%/}/<SESSION_NAME>_output/nextflow.log"
    echo "   Replace <SESSION_NAME> with the folder name reported above."
    echo ""
    echo "Once issues are resolved, you can rerun the pipeline or manually move files."
    exit 1
fi
