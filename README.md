# lf_super_resolution
Super-resolution with Light Field data

# log 2018.8.11
add model SR_RGB_Bicubic_VarHisto:
1. still in RGB, 3 channels combined. The upscale layer that brings 48x48 patch to 96x96 is concatenated with the bicubic interpolated input of size 96x96x3;

2. add tf.summary.histogram() to observe the behavior(distributution) of the weights and biases of encoder and decoder. Encoder has only weights and biases for downsampling convolutions. Decoder has weights and biases for upsampling deconvolutions to construct symmetrical autoencoder structure, weights and biases for upscaling deconvolutions to achieve super resolution. The other variables for concatennate and the final 1-by-1 convolution are not included. 
