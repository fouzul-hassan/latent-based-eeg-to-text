import os
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
import torch
import warnings
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from model.glim import GLIM
from data.datamodule import GLIMDataModule

warnings.filterwarnings("ignore", ".*when logging on epoch level in distributed.*")
# set logger
group_name = 'dev-dist'
log_dir = './runs/' + group_name
os.makedirs(log_dir, exist_ok=True)

devices = [0]
L.seed_everything(42, workers=True)
torch.set_float32_matmul_precision('medium')

# logger = None
logger = WandbLogger(project = 'glim',
                     group = group_name,
                     save_dir = log_dir,
                    #  offline=True,
                    )
# callbacks = None
full_val_interval = 10
callbacks = [
    ModelCheckpoint(monitor = 'epoch', 
        dirpath = str(logger.experiment.dir) + '-checkpoints',
        # dirpath = log_dir,
        save_top_k = -1,
        every_n_epochs = full_val_interval,
        ),
        ]

trainer = L.Trainer(accelerator = 'gpu',
                    # strategy= 'ddp',
                    devices = devices,
                    logger = logger,
                    max_epochs = 200,
                    precision = 'bf16-mixed',
                    enable_checkpointing = True,
                    callbacks = callbacks,
                    use_distributed_sampler=False,
                    num_sanity_val_steps = 0,
                    strict_loading = False,     # T5 weights not in ckpt; allow partial restore
                    # log_every_n_steps=49,  # NOTE
                    # limit_train_batches=2,
                    # limit_val_batches=2,
                    )
# print('\U0001F60B'*10)

dm = GLIMDataModule(data_path = './data/tmp/zuco_eeg_label_8variants.df',
                    eval_noise_input = False,
                    bsz_train = 72,
                    bsz_val = 24,
                    num_workers = 4)

model = GLIM(input_eeg_len = 1280,
             hidden_eeg_len = 96,
             input_text_len = 96,
             tgt_text_len = 64,
             input_dim = 128,
             hidden_dim = 256,
             embed_dim = 1024,
             text_model_id = "google/flan-t5-large",
             prompt_nums = (3, 3, 31),
             prompt_dropout_probs = (0.0, 1.0, 1.0),
             evaluate_prompt_embed = 'src',
             n_in_blocks = 6,
             n_out_blocks = 6,
             in_temporal_modulate = True,
             out_is_causal = True,
             # Gated Attention (NeurIPS 2025 Best Paper)
             use_gated_attention = False,
             gating_type = 'elementwise',  # 'elementwise' or 'headwise'
             # Loss Configuration
             prompt_tuning_len = 0,
             dropout = 0,
             clip_loss_weight = 0.5,
             commitment_loss_weight = 0.0,
             commitment_loss_key = 'mse',
             use_y_mask = False,
             bsz_train = dm.bsz_train,
             bsz_val = dm.bsz_val,
             lr = 1e-4,
             weight_decay = 0,
             full_val_interval = full_val_interval,
             bs_retrieval = 24,
             )

# ===== LOAD PRETRAINED JEPA ENCODER =====
from pretraining.load_pretrained import load_pretrained_encoder, freeze_pretrained_encoder

PRETRAINED_CKPT_PATH = './pretraining/Results/GLIM_Pretrain5/best_model.pth'
FREEZE_ENCODER = False  # Set True to freeze pretrained in_blocks during fine-tuning

if os.path.exists(PRETRAINED_CKPT_PATH):
    print(f"🔧 Loading pretrained JEPA encoder from: {PRETRAINED_CKPT_PATH}")
    model = load_pretrained_encoder(model, PRETRAINED_CKPT_PATH)
    
    if FREEZE_ENCODER:
        model = freeze_pretrained_encoder(model, freeze_in_blocks=True)
        print("❄️ Frozen pretrained encoder (in_blocks)")
    else:
        print("🔥 Pretrained encoder will be fine-tuned")
    
    print("=" * 60)
    print("  📌 eeg_encoder.in_blocks = JEPA Pretrained Encoder")
    print("  📌 eeg_encoder.out_blocks = Q-Merger (trained from scratch)")
    print("=" * 60)
else:
    print(f"⚠️ No pretrained checkpoint found at {PRETRAINED_CKPT_PATH}")
    print("   Training from scratch (no pretraining)")
# =========================================

RESUME_CKPT = './runs/dev-dist/wandb/run-20260809_114002-v1pkxohf/files-checkpoints/epoch=139-step=278320.ckpt'
trainer.fit(model, datamodule=dm, ckpt_path=RESUME_CKPT if os.path.exists(RESUME_CKPT) else None)

