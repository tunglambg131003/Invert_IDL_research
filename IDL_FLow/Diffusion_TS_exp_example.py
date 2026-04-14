import os
import torch
import numpy as np

from engine.solver import Trainer
from Utils.metric_utils import visualization
from Data.build_dataloader import build_dataloader
from Utils.io_utils import load_yaml_config, instantiate_from_config
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one

from Utils.context_fid import Context_FID
from Utils.discriminative_metric import discriminative_score_metrics
from Utils.metric_utils import display_scores


class Args_Example:
    def __init__(self) -> None:
        self.config_path = './Config/sines.yaml'
        self.save_dir = './toy_exp_results'
        self.gpu = 0
        os.makedirs(self.save_dir, exist_ok=True)

args =  Args_Example()
configs = load_yaml_config(args.config_path)
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

dl_info = build_dataloader(configs, args)
model = instantiate_from_config(configs['model']).to(device)
trainer = Trainer(config=configs, args=args, model=model, dataloader=dl_info)

# trainer.train()
milestone = 10  # Replace this with the exact number of the checkpoint you want to load
trainer.load(milestone)

dataset = dl_info['dataset']
seq_length, feature_dim = dataset.window, dataset.var_num
ori_data = np.load(os.path.join(dataset.dir, f"sine_ground_truth_{seq_length}_train.npy"))
# ori_data = np.load(os.path.join(dataset.dir, f"{dataset_name}_norm_truth_{seq_length}_train.npy"))  # Uncomment the line if dataset other than Sine is used.
fake_data = trainer.sample(num=len(dataset), size_every=2001, shape=[seq_length, feature_dim])
if dataset.auto_norm:
    fake_data = unnormalize_to_zero_to_one(fake_data)
    np.save(os.path.join(args.save_dir, f'ddpm_fake_sines.npy'), fake_data)


# 2. Calculate Context-FID
# Note: Ensure the fake data being evaluated matches the shape of ori_data 
# (the example script slices fake_data up to ori_data.shape[0])
context_fid_score = []
iterations = 5
for i in range(iterations):
    context_fid = Context_FID(ori_data[:], fake_data[:ori_data.shape[0]])
    context_fid_score.append(context_fid)
    print(f'Iter {i}: ', 'context-fid =', context_fid, '\n')
      
display_scores(context_fid_score)

discriminative_score = []

for i in range(iterations):
    temp_disc, fake_acc, real_acc = discriminative_score_metrics(ori_data[:], fake_data[:ori_data.shape[0]])
    discriminative_score.append(temp_disc)
    print(f'Iter {i}: ', temp_disc, ',', fake_acc, ',', real_acc, '\n')
      
print('sine:')
display_scores(discriminative_score)
print()
