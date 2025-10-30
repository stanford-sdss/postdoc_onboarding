#!/bin/bash
#SBATCH --job-name=train_resnet              # create a short name for your job
#SBATCH --output=%x_%A.out      		     # file that saves any outputs
#SBATCH --error=%x_%A.err                    # file that saves any errors
#SBATCH --nodes=1                            # node count, typically 1
#SBATCH --ntasks=1                           # total number of tasks across all nodes, 1 for lightning
#SBATCH --cpus-per-task=8                    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --partition=serc                     # select the serc partition
#SBATCH --mem-per-cpu=4G                     # memory per cpu-core (4G is default, but you can request more)
#SBATCH --gres=gpu:1                         # number of gpus per requested node
#SBATCH --time=01:00:00                      # requested run time (HH:MM:SS)
#SBATCH --mail-type=begin                    # send email when job begins
#SBATCH --mail-type=end                      # send email when job ends
#SBATCH --mail-user=ellianna@stanford.edu    # fill this in with your sunet id

#Load any modules needed for your code to run
module load system 
module load python/3.12.1 
module load py-pytorch/2.4.1_py312
module load cuda/10.2.89
module load py-torchvision/0.17.1_py312

#It's a good idea to print out what GPUs are visible to your setup
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

#Run the training script
#It's also a good idea to write out the explicit path when scheduling jobs
python3 /insert/path/to/your/directory/python_scripts/resnet18_pipeline.py