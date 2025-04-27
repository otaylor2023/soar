#!/bin/bash

# Script to evaluate all replay files and save results to separate files
# Usage: ./evaluate_all_replays.sh

# Path to the checkpoint
CHECKPOINT="/home/azureuser/soar/ray_results/rewardshape_flag_frenzy_ppo/PPO_FlagFrenzyEnv-v0_74231_00000_0_2025-04-22_13-51-44/checkpoint_000009"

# Create a directory for results if it doesn't exist
RESULTS_DIR="replay_evaluations"
mkdir -p "$RESULTS_DIR"

# Process each replay file
echo "Finding replay files..."
REPLAY_COUNT=0

# Use a while loop with find to properly handle filenames with spaces
find /home/azureuser/soar/Replays -name "*.json" -type f | while read -r REPLAY_FILE; do
    REPLAY_COUNT=$((REPLAY_COUNT + 1))
    
    # Extract filename without path
    FILENAME=$(basename "$REPLAY_FILE")
    
    # Create sanitized filename for output (replace spaces with underscores)
    SANITIZED_FILENAME="${FILENAME// /_}"
    OUTPUT_FILENAME="${SANITIZED_FILENAME%.json}_eval.json"
    OUTPUT_FILE="$RESULTS_DIR/$OUTPUT_FILENAME"
    
    echo "[$REPLAY_COUNT] Processing: $FILENAME"
    
    # Run the evaluation script
    python eval_replay.py --replay "$REPLAY_FILE" --checkpoint "$CHECKPOINT" --output "$OUTPUT_FILE"
    
    if [ $? -eq 0 ]; then
        echo "Results saved to: $OUTPUT_FILE"
    else
        echo "Error processing: $FILENAME"
    fi
    echo "----------------------------------------------"
done

echo "Evaluation complete. All results saved to $RESULTS_DIR/"
