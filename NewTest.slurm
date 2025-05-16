#!/bin/bash
#SBATCH --job-name=michel_analysis
#SBATCH --output=michel_output.txt
#SBATCH --error=michel_error.txt
#SBATCH --time=15:00:00
#SBATCH --partition=longjobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=80G

# Define base path for input files
BASE_PATH="/data13/coherent/data/d2o/processedData"

# Log job details
echo "Running on node $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Current working directory: $(pwd)"
echo "Starting job at: $(date)"

# Verify executable
if [ ! -f "./NewTest" ]; then
    echo "Error: NewTest executable not found in $(pwd)"
    exit 1
fi

# Define input files
input_files=()
# Add all files in the desired order: 17595 first, then 17955-17980
for run in 17920 {17920..17944}; do
    file="${BASE_PATH}/run${run}_processed_v5.root"
    if [ -f "$file" ]; then
        input_files+=("$file")
    else
        echo "Warning: File $file not found, skipping"
    fi
done

if [ ${#input_files[@]} -lt 2 ]; then
    echo "Error: Insufficient input files found (need at least calibration and one data file)"
    echo "Found files:"
    for file in "${input_files[@]}"; do
        echo "  $file"
    done
    exit 1
fi

# Print input files
echo "Input files to process:"
echo "  Calibration file: ${input_files[0]}"
echo "  Data files:"
for ((i=1; i<${#input_files[@]}; i++)); do
    echo "    ${input_files[$i]}"
done

# Run the analysis (first file is calibration, rest are data files)
echo "Executing: ./NewTest ${input_files[@]}"
./NewTest "${input_files[@]}"
exit_code=$?

# Check if the program ran successfully
if [ $exit_code -ne 0 ]; then
    echo "Error: NewTest failed with exit code $exit_code"
    exit $exit_code
fi

# Verify output directory (find the latest AnalysisOutput_* directory)
output_dir=$(ls -td ./AnalysisOutput_* 2>/dev/null | head -n 1)
if [ -n "$output_dir" ] && [ -d "$output_dir" ]; then
    echo "Output saved in $(pwd)/$output_dir"
    ls -l "$output_dir/"
else
    echo "Warning: No AnalysisOutput_* directory found"
fi

echo "Job finished at: $(date)"
