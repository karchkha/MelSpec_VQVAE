# VQVAE
it's a dirty verison of VQVAE with MelSpectrograms 


main buiding idea of the encoder/decoder architecture is taken from this paper: Taming Visually Guided Sound Generation, https://arxiv.org/abs/2110.08791

VAS data can be downloaded from 

```bash
cd ./data
# ~7GB we only download spectrograms
bash ./download_vas_features.sh

```
from models please use big_model_attn_gan.py. This one works more-less well. Takes very long to train though.


Please download pretrained vocoder from: https://github.com/v-iashin/SpecVQGAN/tree/main/vocoder/logs/vggsound and put in in vocoder/logs/vggsound folder
