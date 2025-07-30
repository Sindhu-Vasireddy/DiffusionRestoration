#!/usr/bin/env fish

cac climsim

# Ensure an argument is provided
if test (count $argv) -lt 1
    echo "Usage: sync.fish filename"
    exit 1
end

# Loop through each file in the argument list
for file in $argv
    echo ""
    echo "Processing file: $file"

    # Extract Run ID from each file
    set run_id (grep -oP 'run id \K\w+' ./out/$file)

    # Check if run ID was found
    if test -n "$run_id"
        echo "Extracted Run ID: $run_id"
        python /home/hess/projects/wandb-sync/wandb_sync.py -p discriminator-guidance -id $run_id
    else
        echo "Run ID not found in $file"
    end
end
