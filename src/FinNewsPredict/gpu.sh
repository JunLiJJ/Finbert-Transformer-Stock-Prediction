#!/bin/bash
#SBATCH --job-name=embedding
#SBATCH --nodes=1
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=eecs545w25_class
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joelyang@umich.edu
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-gpu=8
#SBATCH --time=00:10:00

module load anaconda3
# module load python3.10-anaconda/2023.03
module load cuda/11.8
source activate llm-env   # ✅ 激活你预先创建的环境

echo "🧠 Starting training..."
python generate_embeddings.py

echo "🧠 Training complete!"