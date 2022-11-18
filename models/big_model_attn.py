import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl



class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) #this coresponds to "min_encoding_indices"


        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # loss1 = torch.mean((quantized.detach() - inputs) ** 2) + self._commitment_cost * torch.mean((quantized - inputs.detach()) ** 2)
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), (perplexity, encodings, encoding_indices)





class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        # return x
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h



def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)




class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
            self.pad = (0, 1, 0, 1)
        else:
            self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.with_conv:  # bp: check self.avgpool and self.pad
            x = torch.nn.functional.pad(x, self.pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = self.avg_pool(x)
        return x

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)




class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,   ######## I chabged it here and wanna see what happens
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x



class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # # print(h)
        # print("\n\n\after downsampling")
        # print(len(hs))
        # for i in range(len(hs)): 
        #   print(hs[i].shape)
        # print(temb)
        # if temb is not None:
        #   print(temb.shape)  

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # # print(h)
        # print("\n\n\after middle", h.shape)
        # print(temb)
        # if temb is not None:
        #   print(temb.shape,"\n\n\n")  

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)


        # # print(h)
        # print("\n\n\after end", h.shape)
        # print(temb)
        # if temb is not None:
        #   print(temb.shape,"\n\n\n")  

        return h








###################################







class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        # self.z_shape = (1,z_channels,curr_res,curr_res)
        # print("Working with z of shape {} = {} dimensions.".format(
        #     self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h




class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        # return x
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_






import torchvision

double_z = False
z_channels = 256
resolution = 848
in_channels =  1
out_ch = 1
ch = 128
ch_mult = [ 1,1,2,2,4]  # num_down = len(ch_mult)-1
num_res_blocks = 2
attn_resolutions = [53]
dropout = 0.0


learning_rate = 1e-3

import pytorch_lightning as pl
        
#### lighting model 
        
class LitVQVAE(pl.LightningModule):
    def __init__(self, 
                 #num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost):
        super(LitVQVAE, self).__init__()
        
        self.num_embeddings = num_embeddings

        self._encoder = Encoder( ch = ch, 
                  out_ch = out_ch, 
                  ch_mult=ch_mult, 
                  num_res_blocks = num_res_blocks,
                 attn_resolutions = attn_resolutions, 
                  dropout=0.0, 
                  resamp_with_conv=True, 
                  in_channels = in_channels,
                  resolution = resolution, 
                  z_channels = z_channels, 
                  double_z = double_z)


        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder( ch = ch, 
                  out_ch = out_ch, 
                  ch_mult=ch_mult, 
                  num_res_blocks = num_res_blocks,
                 attn_resolutions = attn_resolutions, 
                  dropout=0.0, 
                  resamp_with_conv=True, 
                  in_channels = in_channels,
                  resolution = resolution, 
                  z_channels = z_channels, 
                  double_z = double_z)
        
        self.quant_conv = torch.nn.Conv2d(z_channels, embedding_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embedding_dim, z_channels, 1)

        self.counts = [0 for _ in range(self.num_embeddings)]

    def encode(self, x):
        h = self._encoder(x) 
        z = self.quant_conv(h) 

        return z

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self._decoder(quant)

        return dec

    def decode_code(self, code_b):
        quant_b = self._vq_vae(code_b)
        dec = self.decode(quant_b)

        return dec

    def forward(self, x):

        z = self.encode(x)

        loss, quantized, info = self._vq_vae(z)
        
        x_recon = self.decode(quantized)

        # this is for controling if we use codebook well
        if not self.training:
            self.counts = [info[2].squeeze().tolist().count(i) + self.counts[i] for i in range(self.num_embeddings)]

        return loss, x_recon, info

    def get_input(self, batch):
        x = batch['image']
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch)
        
        vq_loss, data_recon, _ = self(x)

        # recon_error = F.mse_loss(data_recon, x)

        recon_error = torch.mean(torch.abs(x.contiguous() - data_recon.contiguous())) #* 15.0

        loss = recon_error  + vq_loss

        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # self.log("train_loss_alt", loss_alt, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):

        x = self.get_input(batch)
        vq_loss, data_recon, _ = self(x)
        # recon_error = F.mse_loss(data_recon, x)
        recon_error = torch.mean(torch.abs(x.contiguous() - data_recon.contiguous())) # * 15.0
        loss = recon_error + vq_loss

        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return x, data_recon
    
    def validation_epoch_end(self, val_step_output):
        if self.global_step != 0 and sum(self.counts) > 0:
          print(f'Previous Epoch counts: {self.counts}')
          zero_hit_codes = len([1 for count in self.counts if count == 0])
          used_codes = []
          for c, count in enumerate(self.counts):
              used_codes.extend([c] * count)
          self.logger.experiment.add_histogram('val/code_hits', torch.tensor(used_codes), self.global_step)
          self.logger.experiment.add_scalar('val/zero_hit_codes', zero_hit_codes, self.global_step)
          self.counts = [0 for _ in range(self.num_embeddings)]

        #log images 
        self.log_images(val_step_output)
          

    def log_images(self, step_output_pair):
        x, data_recon = step_output_pair[0]
        # grid = self.log_images(x, data_recon)

        # get to 0-1 for images
        x = (x + 1.0) / 2.0
        reconstructions = (data_recon + 1.0)/ 2.0

        # make grid
        x = torchvision.utils.make_grid(x.flip(dims=(2,)), nrow =1) #, normilize=True)
        reconstructions = torchvision.utils.make_grid(reconstructions.flip(dims=(2,)), nrow =1) #, normilize=True)
      
        self.logger.experiment.add_image("images_inputs", x, global_step=self.global_step)  
        self.logger.experiment.add_image("images_reconstructions", reconstructions, global_step=self.global_step)   


    def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(), lr=learning_rate, amsgrad=False)       
