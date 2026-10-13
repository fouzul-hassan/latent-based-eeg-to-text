import os
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
import torch
import warnings
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from model.glim import GLIM
from data.datamodule import GLIMDataModule

import argparse

warnings.filterwarnings("ignore", ".*when logging on epoch level in distributed.*")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_input', action='store_true', help='Use noise input for sanity check')
    args = parser.parse_args()

    # set logger
    group_name = 'dev-dist-test'
    log_dir = './runs/' + group_name
    os.makedirs(log_dir, exist_ok=True)

    devices = [0]
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')

    # logger = None
    logger = WandbLogger(project = 'glim',
                         group = group_name,
                         save_dir = log_dir,
                        #  version = 'w25zyxe5',
                        #  offline = True,
                        )

    trainer = L.Trainer(accelerator = 'gpu',
                        devices = devices,
                        logger = logger,
                        precision = 'bf16-mixed', 
                        # limit_test_batches=2
                        )

    dm = GLIMDataModule(data_path = './data/tmp/zuco_eeg_label_8variants.df',
                        eval_noise_input = args.noise_input,
                        bsz_test = 24,
                        num_workers = 2,
                        )

    model = GLIM.load_from_checkpoint(
        "runs/dev-dist/wandb/run-20260809_114002-v1pkxohf/files-checkpoints/epoch=139-step=278320.ckpt",
        map_location = f"cuda:{devices[0]}",
        strict = False,
        weights_only = False,
        use_etes_eval = True,  # Enable ETES evaluation
        log_xai = True,  # Enable XAI logging (gate stats, histograms)
        # evaluate_prompt_embed = 'zero',
        # prompt_dropout_probs = (0.0, 0.1, 0.1),
        )
    trainer.test(model, datamodule=dm)