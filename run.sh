!/Users/vatsal/Documents/EvolveGCN/run_exp.py
SBATCH -J wxh
SBATCH -p gpu

SBATCH -N 1
SBATCH --output=log.%j.out
SBATCH --error=log.%j.err
SBATCH --gres=gpu:1

#source activate torch

python run_exp.py --config_file ./experiments/parameters_ether_egcn_h_Sample.yaml
python run_exp.py --config_file ./experiments/parameters_ether_egcn_h.yaml