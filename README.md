# lf_super_resolution
Super-resolution with Light Field data

# log 2018.8.11
add model SR_RGB_Bicubic_VarHisto:
1. still in RGB, 3 channels combined. The upscale layer that brings 48x48 patch to 96x96 is concatenated with the bicubic interpolated input of size 96x96x3;

2. add tf.summary.histogram() to observe the behavior(distributution) of the weights and biases of encoder and decoder. Encoder has only weights and biases for downsampling convolutions. Decoder has weights and biases for upsampling deconvolutions to construct symmetrical autoencoder structure, weights and biases for upscaling deconvolutions to achieve super resolution. The other variables for concatennate and the final 1-by-1 convolution are not included. 


# log 2018.8.14
add model SR ColorSpace Channel Skip:
1. "config_autoencoder.py": 
</br>**row 36**: added flag for whether to use bicubic interpolated input as skip connection patch in upscaling phase;
</br>**row 98-103, 127-132**: moved the layout in upscale-layers to distinguish the number of feature maps of certain channel(s);
</br>**row 144-188**: reduced number of layers such that the network could be run on single GPU machines :), should be changed back when the final structure is determined;
</br>**row 195-199**: added layout of upscale-layer if no skip connection is used for certain channel(s), mainly because of the number of feature maps;
</br>**row 246/259**: added flag in decoder-config for whether using skip connection for certain channel(s);

2. "layers.py":
</br>**row 367**: added class "decoder-variables-2D-no-SC", mainly because when no skip connection, the layout is different in number of feature maps and the layout cannot be changed directly by operations in "cnn-autoencoder.py" since it would also change the value in the "self" structure;

3. "cnn_autoencoder.py":
</br>**row 30**: bicubic interpolation flag;
</br>**row 225**: skip connection flag belonging to each decoder;
</br>**row 227-228**: generating bicubic interpolated input as a skip connection patch in upscale-phase;
</br>**row 259-272**: if no skip connection is used, then also skip generating the variables for the patches of skip connections;
</br>**row 277-284**: generating decoder-2D variables according to whether using skip connection;
</br>**row 304-319**: with/without skip connection in decoder of certain channel(s);
</br>**row 330-335**: layout changes due to whether using skip connection;
</br>**row 337-340**: layout changes due to whether using bicubic interpolated inputs;
</br>**row 360-361**: with/without bicubic interpolated inputs;

4. potential problems:
</br>**(a)**. we are using quite many flags in the model, the layouts of layers are still kind of hard-coded, right now I don't have better way to make it very clean;
</br>**(b)**. since the bottleneck now is merged, if the decoder of color channels doesn't use skip connection, the decoder would have much fewer number of feature maps. As one result of it, the first decoder layer right after the bottle neck would dramatically decrease the number of features from 576(=192 * patch-weight, row 247 in "cnn-autoencoder.py") to 160. With the skip connection it would be from 576 to 480(=160 * patch-weight). row 247 in "cnn-autoencoder.py" cannot be changed according to whether we use skip connection, or it would cause the fracture of the network at bottleneck. 
</br>**(c)**. another result of not using skip connection is that, because of fewer number of feature maps, we would have much smaller room to increase the number of layers in upscaling phase.
</br>**(d)**. I ran the model with L having skip conncetion and ab not for a little while, the loss of L channel is still satisfiying...:(
 
  
