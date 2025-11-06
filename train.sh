#!/bin/bash

#SBATCH --job-name=f1test_finetune_en18pseudo_train_endovis_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --output=Logs/train_endovis17_%j_%x.out
#SBATCH --error=Logs/train_endovis17_%j_%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=huiern214@gmail.com
#SBATCH --time=48:00:00

# module load miniconda
# conda activate surgicalsam
module load miniforge
conda activate seem

echo "Training"
# python train.py local_configs/segformer/B2/segformer.b2.512x512.en18pseudo.f0.160k.py --load-from /home/users/astar/i2r/stuhuiern/scratch/SegFormer/work_dirs/train_en17_scratch_80k_b2/segformer.b2.512x512.en17.f0.160k/iter_32000.pth
# python train.py local_configs/segformer/B2/segformer.b2.512x512.en18pseudo.f1.160k.py --load-from /home/users/astar/i2r/stuhuiern/scratch/SegFormer/work_dirs/train_en17_scratch_80k_b2/segformer.b2.512x512.en17.f1.160k/iter_76000.pth
# python train.py local_configs/segformer/B2/segformer.b2.512x512.en18pseudo.f2.160k.py --load-from /home/users/astar/i2r/stuhuiern/scratch/SegFormer/work_dirs/train_en17_scratch_80k_b2/segformer.b2.512x512.en17.f2.160k/iter_56000.pth
# python train.py local_configs/segformer/B2/segformer.b2.512x512.en18pseudo.f3.160k.py --load-from /home/users/astar/i2r/stuhuiern/scratch/SegFormer/work_dirs/train_en17_scratch_80k_b2/segformer.b2.512x512.en17.f3.160k/iter_52000.pth
# ./tools/dist_train.sh local_configs/segformer/B1/segformer.b2.512x512.en17.160k.py 4

# echo "Testing"
# python test.py local_configs/segformer/B2/segformer.b2.512x512.en17.f0.160k.py /home/users/astar/i2r/stuhuiern/scratch/SegFormer/work_dirs/segformer.b2.512x512.en17.f0.160k/iter_32000.pth 
# python test.py local_configs/segformer/B2/segformer.b2.512x512.en17.f1.160k.py /home/users/astar/i2r/stuhuiern/scratch/SegFormer/work_dirs/segformer.b2.512x512.en18pseudo.f1.160k/iter_4000.pth 
# python test.py local_configs/segformer/B2/segformer.b2.512x512.en17.f2.160k.py /home/users/astar/i2r/stuhuiern/scratch/SegFormer/work_dirs/segformer.b2.512x512.en17.f2.160k/iter_56000.pth
# python test.py local_configs/segformer/B2/segformer.b2.512x512.en17.f3.160k.py /home/users/astar/i2r/stuhuiern/scratch/SegFormer/work_dirs/segformer.b2.512x512.en17.f3.160k/iter_52000.pth
python test.py local_configs/segformer/B2/segformer.b2.512x512.en18pseudo.f1.160k.py /home/users/astar/i2r/stuhuiern/scratch/SegFormer/work_dirs/segformer.b2.512x512.en18pseudo.f1.160k/iter_4000.pth