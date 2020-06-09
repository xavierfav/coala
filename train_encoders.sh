#!/bin/bash
#SBATCH --job-name=rep-learning
#SBATCH --account=asignal
#SBATCH --partition=gpu
#SBATCH --time=70:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12000
#SBATCH --gres=gpu:v100:1

#SBATCH -o /scratch/asignal/favoryxa/out/%J.%u.out # STDOUT
#SBATCH -e /scratch/asignal/favoryxa/out/%J.%u.err # STDERR

printf "[----]\n"
printf "Starting execution of job $SLURM_JOB_ID from user $LOGNAME\n"
printf "Starting at `date`\n"
start=`date +%s`

# module load gcc/8.3.0
# module load pgi/19.7
# module load cuda/10.1.168
module load python-env/2019.3
module load pytorch/1.3.0

source venv/bin/activate
pip install tensorboard

srun python train_dual_ae.py 'configs/dual_e_c.json'

end=`date +%s`
printf "\n[----]\n"
printf "Job done. Ending at `date`\n"
runtime=$((end-start))
printf "It took: $runtime sec.\n"