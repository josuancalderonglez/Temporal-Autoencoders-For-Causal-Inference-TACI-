#!/bin/bash
#SBATCH --job-name=real_data_model
#SBATCH --partition=a100-8-gm320-c96-m1152,v100-8-gm128-c64-m488,a10g-4-gm96-c48-m192
#SBATCH --mem=25GB
##SBATCH --array=0
#SBATCH --time=10:00:00
#SBATCH --output=/users/jcalde2/Causation/Scripts/logs/results_%j.log
#SBATCH --error=/users/jcalde2/Causation/Scripts/logs/error_%j.log
#SBATCH --account=general
#SBATCH --gpus=1
#SBATCH --mail-user josuan.calderon@emory.edu
#SBATCH --mail-type BEGIN,END,FAIL  

conda init bash >/dev/null 2>&1
source ~/.bashrc

conda activate tcn1

echo $PATH
echo $LD_LIBRARY_PATH
nvcc --version
# nvidia-smi

# Set Python to run unbuffered
export PYTHONUNBUFFERED=1

# echo "Running Python script: voles_data_model_mating_freq.py"

##### Create Environment
# conda create --name tcn1 python=3.10
# conda activate tcn1
# conda install ipython h5py matplotlib scikit-learn pandas scipy
# pip install tensorflow[and-cuda]==2.14
# pip install keras-tcn --no-dependencies

##### Run Interactive Instance
# srun --partition=a100-8-gm320-c96-m1152,v100-8-gm128-c64-m488,a10g-4-gm96-c48-m192 -N 1 --gpus=1 --mem=20G --account=general --pty bash

##### Run Artificial Models
# python ar_processes_synthetic_data_model.py --noise 0.0 --dropout_rate_tcn 0.1 --dropout_rate_hidden 0.3 --tol_default 0.01 \
#         --length_vars 330000 --discard 30000 --encoder_type 2 --batch_size 512 --epochs 100 --scaling_method Standard \
#         --version 11

##### Run Artificial Temporal Models
python temporal_synthetic_data_model.py --noise 0.0 --dropout_rate_tcn 0.0 --dropout_rate_hidden 0.0 --tol_default 0.01  \
        --coupling_constant 0.5501 --length_vars 500000 --discard 50000 --seqs_len 10 --encoder_type 2 --batch_size 512 \
        --epochs 300 --scaling_method Standard --window_len 2000

##### Run Monkey Models
# BRAIN_AREA_KEY1=1
# BRAIN_AREA_KEY2=2

# python monkey_data_model.py --brain_area_key1 $BRAIN_AREA_KEY1 --brain_area_key2 $BRAIN_AREA_KEY2 --array_index $SLURM_ARRAY_TASK_ID \
#         --noise 0.0 --dropout_rate_tcn 0.0 --dropout_rate_hidden 0.0 --tol_default 0.05 \
#         --seqs_len 50 --encoder_type 2 --batch_size 512 --epochs 300 --scaling_method Standard --window_len 2000 --version 0

# python monkey_data_model_all.py  --noise 0.0 --dropout_rate_tcn 0.0 --dropout_rate_hidden 0.0 --tol_default 0.0001 \
#         --seqs_len 50 --encoder_type 2 --batch_size 512 --epochs 300 --scaling_method Standard --window_len 2000 --version 0

##### Run Voles Models
# for VOLE in {0..7}
# do

#         echo "Running task for Vole: $((VOLE + 1))"

#         # python -u voles_data_model.py  --array_index $SLURM_ARRAY_TASK_ID --noise 0.0 --dropout_rate_tcn 0.0 --dropout_rate_hidden 0.0 \
#         #         --tol_default 0.01 --seqs_len 50 --encoder_type 2 --batch_size 512 --epochs 300 --scaling_method Standard --version 0

#         # python -u voles_data_model_mating.py  --noise 0.0 --dropout_rate_tcn 0.0 --dropout_rate_hidden 0.0 --tol_default 0.01 \
#         #         --seqs_len 50 --encoder_type 2 --batch_size 512 --epochs 300 --scaling_method Standard --version 0

#         # python -u voles_data_model_mating_freq.py  --array_index $SLURM_ARRAY_TASK_ID --noise 0.0 --dropout_rate_tcn 0.0 --dropout_rate_hidden 0.0 \
#         #         --tol_default 0.01 --seqs_len 50 --encoder_type 2 --batch_size 512 --epochs 300 --scaling_method Standard --version 0

#         python -u voles_data_model_mating_freq_vole.py  --vole_num $VOLE --array_index $SLURM_ARRAY_TASK_ID \
#                 --noise 0.0 --dropout_rate_tcn 0.0 --dropout_rate_hidden 0.0 \
#                 --tol_default 0.001 --seqs_len 30 --encoder_type 2 --batch_size 512 --epochs 100 --scaling_method Standard \
#                 --version 0

# done