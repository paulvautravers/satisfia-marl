#!/bin/bash
#SBATCH --job-name=satisfia_multi_agent_param_search           # Job name
#SBATCH --output=param_search_%j_%t.out      # Standard output file (%j will be replaced with job ID)
#SBATCH --error=param_search_%j_%t.err       # Standard error file (%j will be replaced with job ID)
#SBATCH --ntasks=50                        # Number of tasks (adjust if you want multiple parallel tasks)
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
#SBATCH --mem=16GB                        # Memory per node (adjust as needed)
#SBATCH --time=08:00:00                   # Time limit (8 hours)
#SBATCH --partition=compute               # Partition to submit to (adjust to your cluster configuration)

# Load required modules or activate the environment
module load python/3.8                    # Load the Python module (adjust as needed)

# Optional: Activate a virtual environment if required
# source /path/to/virtualenv/bin/activate

# Run the Python script with arguments
srun python parameter_search.py --generations 100 --repeats 1 --simulations_per_task 100
