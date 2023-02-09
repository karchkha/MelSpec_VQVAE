# MelSpec VQVAE

# Introduction

Welcome to the MelSpec VQVAE repository! 

This project is based on the Veqtor quantized encoder/decoder architecture presented in the paper "Taming Visually Guided Sound Generation" (https://arxiv.org/abs/2110.08791) and its corresponding repository (https://github.com/v-iashin/SpecVQGAN).

However, there are made some modifications to the original architecture in order to simplify and improve the training efficiency and speed up the process. These modifications include the implementation of codebook regularization methods, particularly, the addition of two types of noise injection. Also, the LPAPS loss has been removed in this implementation.

# Requirements

To run the code in this repository, you will need:

python 3.9.5 

torch==1.12.1+cu113
pytorch-lightning==1.8.1
SoundFile==0.10.3.post1
torchvision==0.13.1+cu113
albumentations==1.2.1

# Installation
To install MelSpec_VQVAE, follow these steps:

Clone the repository to your local machine
```bash
git clone https://github.com/karchkha/MelSpec_VQVAE
```

Navigate to the project directory and install the required dependencies

# Training

The training process of the VQVAE model is controlled by the following arguments:

* `dataset`: The dataset to be used for training the model. This argument is required.
* `experiment`: The name of the experiment. This argument is also required and will be used to name the checkpoints and tensorboard logs.
* `batch_size`: The number of samples in a batch. The default value is 3.
* `embedding_dim`: The size of the embedding vector. The default value is 256.
* `num_embeddings`: The size of the codebook. The default value is 128.
* `learning_rate`: The learning rate for the optimizer. The default value is 1e-6.
* `epochs`: The number of epochs to train the model. The default value is 100.
* `train`: A flag to start the training process. The default value is False.
* `resume`: A checkpoint file to resume training from. The default value is None.
* `workers`: The number of workers to use for data loading. The default value is 4.
* `eval`: A flag to evaluate the model. The default value is False.
* `test`: A flag to test the model. The default value is False.
* `disc_conditional`: A flag to indicate if the discriminator should be conditioned on the input. The default value is False.
* `disc_in_channels`: The number of input channels for the discriminator. The default value is 1.
* `disc_start`: The training step at which the discriminator training begins. The default value is 2001.
* `disc_weight`: The weight for the discriminator loss in the overall loss function. The default value is 0.8.
* `codebook_weight`: The weight for the codebook loss in the overall loss function. The default value is 1.0.
* `min_adapt_weight`: The minimum weight for the adversarial loss between the VAE and the discriminator. The default value is 0.0.
* `max_adapt_weight`: The maximum weight for the adversarial loss between the VAE and the discriminator. The default value is 1.0.
* `noise_sigma`: The standard deviation of the Gaussian noise added to the codebook. The default value is 0.
* `cd_rand_res`: A flag to randomly assign existing vectors from the output to the codebook's unused vectors. The default value is 0.

The training process can be started by passing the appropriate values for these arguments to the script. The model will be trained on the specified dataset for the specified number of epochs and the checkpoints will be saved in the directory specified by experiment. 

For Example the running with default velues would look like:

```bash
python train.py --experiment {experiment_name} \
                    --dataset {dataset_name} \
                    --train 1 \
                    --eval 0 \
                    --resume "lightning_logs/{experiment_name}/checkpoints/version_{version_number}/last.ckpt" \
```

The model's performance can be monitored using Tensorboard. The the loss and other metrics are logged during the training.

```bash
tensorboard --logdir lightning_logs --bind_all
```
There are few differnt models but I ended up using big_model_attn_gan.py. This one works more-less well. Takes very long to train though.

# Data

In this project, the VAS data is used [VAS](https://github.com/PeihaoChen/regnet#download-datasets).
VAS can be downloaded directly using the link provided in the [RegNet](https://github.com/PeihaoChen/regnet#download-datasets) repository.

Another way, VAS data with ready Mel_spectrograms can be downloaded by running below code:
```bash
cd ./data
# ~7GB we only download spectrograms
bash ./download_vas_features.sh
```
# Vocoder

For reconstructing Mel_spectrograms to Audio the Mel_GAN vocoder is used. Please download pretrained vocoder from: https://github.com/v-iashin/SpecVQGAN/tree/main/vocoder/logs/vggsound and put in in vocoder/logs/vggsound folder. However audio recosnstruction code is not used and is not included in this repository. It is coming in the next repository.
