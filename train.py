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

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models.big_model_attn_gan import LitVQVAE
from datasets.vas import DataModule

import pytorch_lightning as pl

import argparse


def init_config():
    parser = argparse.ArgumentParser(description='VQVAE for Mel spectrogram reconstruction')
 
    # model hyperparameters
    parser.add_argument('--dataset', type=str, required=True, help='dataset to use')
    parser.add_argument('--experiment', type=str, required=True, default="test", help='experiment name')

    parser.add_argument('--batch_size', type=int, default=3, help='batch size')  
    parser.add_argument('--embedding_dim', type=int, default=256, help='embeging vectro size')   
    parser.add_argument('--num_embeddings', type=int, default=128, help='codebook size')
    parser.add_argument('--learning_rate', type=float, default=1e-6 , help='learning_rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')    
    parser.add_argument('--train', type=int, default=False, help='start training process')
    parser.add_argument('--resume', type=str, default=None, help='resume_from the checkpoint')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for data',)
    parser.add_argument('--eval', type=int, default=False, help='evaluate model')
    parser.add_argument('--test', type=int, default=False, help='test model')

    parser.add_argument('--disc_conditional', type=bool, default=False, help='use conditionla discriminator')
    parser.add_argument('--disc_in_channels', type=int, default=1, help='discriminator in channels')    
    parser.add_argument('--disc_start', type=int, default=2001, help='discriminator tarining starts step')      
    parser.add_argument('--disc_weight', type=float, default=0.8, help='discriminatior loss weight')    
    parser.add_argument('--codebook_weight', type=float, default=1.0, help='codebook loss weight')   
    parser.add_argument('--min_adapt_weight', type=float, default=0.0, help='minimum of adapt loss weight between vae and discriminator')
    parser.add_argument('--max_adapt_weight', type=float, default=1.0, help='maximum of adapt loss weight between vae and discriminator')
    parser.add_argument('--noise_sigma', type=int, default=0, help='inject noise in codebook, if you use this make sure to use it from the very begining otherwise it ruines everything')
    parser.add_argument('--cd_rand_res', type=int, default=0, help='assign randomly existicng vector from output to codebook unused vector') 

    args = parser.parse_args()

    # set args.device
    args.cuda = torch.cuda.is_available()
  
    return args




def main(args):
    
    #device = torch.device("cuda" if args.cuda else "cpu")
    device = "cuda" if args.cuda else "cpu"
    args.device = device


    #### Data preparation #####
    
    spec_dir_path = './data/{}/features/*/melspec_10s_22050hz'.format(args.dataset)
    mel_num =  80
    spec_len = 860
    spec_crop_len = 848
    random_crop = False

    data = DataModule(batch_size = args.batch_size, 
                      spec_dir_path = spec_dir_path, 
                      mel_num = mel_num, 
                      spec_len = spec_len, 
                      spec_crop_len =spec_crop_len, 
                      random_crop =random_crop,
                      num_workers=args.workers)

    data.setup()




    #### model initialisation ####

    learning_rate = args.learning_rate * args.batch_size

    model = LitVQVAE(num_embeddings = args.num_embeddings, 
                     embedding_dim = args.embedding_dim, 
                     commitment_cost = 0.25,
                     disc_start = args.disc_start, 
                     codebook_weight = args.codebook_weight,
                     disc_num_layers=3, 
                     disc_in_channels=args.disc_in_channels, 
                     disc_factor = 1.0, 
                     disc_weight = args.disc_weight,
                     use_actnorm=False, 
                     disc_conditional=args.disc_conditional,
                     disc_ndf=64, 
                     min_adapt_weight = args.min_adapt_weight, 
                     max_adapt_weight = args.max_adapt_weight,
                     learning_rate = learning_rate,
                     noise_sigma = args.noise_sigma,
                     cd_rand_res = args.cd_rand_res
                     )



    ##################################### CALLBACKS and TRAINER ###############
    
    logger = TensorBoardLogger(save_dir = "lightning_logs/"+ args.experiment  + "-" + args.dataset, name = 'TensorBoardLoggs')
    checkpoint_callback = ModelCheckpoint(save_top_k = 1,
                                          monitor="val/total_loss",
                                          mode="min",
                                          save_last= True,
                                          filename= args.dataset + "-model-{epoch:02d}-{loss:.2f}",
                                          dirpath= "lightning_logs/" + args.experiment + "-" + args.dataset + "/checkpoints/version_" + str(logger.version),
                                          )    
    
    
    

    trainer = pl.Trainer(default_root_dir= "lightning_logs", 
                        accelerator=args.device, #'gpu',
                         max_epochs=args.epochs,
                         callbacks=[checkpoint_callback],
                         logger = logger,
                         num_sanity_val_steps=0,
#                          limit_train_batches = 2,
#                          limit_val_batches= 2,
#                          fast_dev_run = True
                         )

    ############################## training ##############################
    
    
    if args.train:
    
        trainer.fit(model, 
                    data, 
                    ckpt_path = args.resume,
                    )

    #################################  evaluation ###########################
    

    if args.eval == 1:

      
        trainer.validate(model, 
                         data.val_dataloader(),
                         ckpt_path = args.resume,
                        )
    

if __name__ == '__main__':
    args = init_config()
    main(args)











