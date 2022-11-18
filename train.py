from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


from six.moves import xrange

import IPython.display as display_audio
import IPython

# import umap
import os
import soundfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import pytorch_lightning as pl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



from datasets.vas import DataModule


spec_dir_path = './data/vas/features/*/melspec_10s_22050hz'
mel_num =  80
spec_len = 860
spec_crop_len = 848
random_crop = False
batch_size = 3

data = DataModule(batch_size =batch_size, 
                  spec_dir_path = spec_dir_path, 
                  mel_num = mel_num, 
                  spec_len = spec_len, 
                  spec_crop_len =spec_crop_len, 
                  random_crop =random_crop,
                  num_workers=4)

data.setup()











from models.big_model_attn_gan import LitVQVAE

embedding_dim = 256 # 64
num_embeddings = 128 #512

commitment_cost = 0.25

learning_rate = 1e-6 * batch_size


disc_conditional= False
disc_in_channels= 1
disc_start= 2001
disc_weight= 0.8
codebook_weight= 1.0
min_adapt_weight= 1.0
max_adapt_weight= 1.0


                 

model = LitVQVAE(num_embeddings, embedding_dim, commitment_cost,
                 disc_start = disc_start, 
                 codebook_weight=codebook_weight,
                 disc_num_layers=3, 
                 disc_in_channels=disc_in_channels, 
                 disc_factor=1.0, 
                 disc_weight=disc_weight,
                 use_actnorm=False, 
                 disc_conditional=disc_conditional,
                 disc_ndf=64, 
                 min_adapt_weight=min_adapt_weight, 
                 max_adapt_weight=max_adapt_weight,
                 learning_rate = learning_rate,
                 )








from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

experiment_dir = 'lightning_logs/vqvae'

checkpoint_callback = ModelCheckpoint(save_top_k = 2,
                                       monitor="val/total_loss",
                                      mode="min",
                                      save_last= True,
                                      filename='model-{epoch:02d}'
                                      )

logger = TensorBoardLogger(save_dir = experiment_dir, name = 'big_model_attn_GAN')

trainer = pl.Trainer(default_root_dir= experiment_dir, 
                    accelerator='gpu',
                     max_epochs=120,
                     callbacks=[checkpoint_callback],
                     logger = logger,
                    #  limit_train_batches = 100,
                    #  limit_val_batches= 100,
#                      fast_dev_run = True
                     )



trainer.fit(model, 
            data, 
            #ckpt_path = "lightning_logs/vqvae/big_model_attn_GAN/version_9/checkpoints/last.ckpt"
            )


