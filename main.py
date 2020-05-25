# main

# load dataset
# to dataloader
# load model
# train and output images
# generate images on random tensor

## model forward
'''
1. images through encoder -> z_mu, z_log_var
2. reparameterize z_mu, z_log_var -> z
3. draw uniform random tensor (z_p)

ENCODER UPDATE
4. generate fake images
    - z -> decoder -> x_r
    - z_p -> decoder -> x_p
6. encode fake images 
    - no grad to avoid updating decoder
    - x_r -> encoder -> z_r
    - z_p -> encoder -> z_pp
7. encoder loss using Eq. 11 and backprop

GENERATOR UPDATE
8. encode fake images with grad
    - x_r -> encoder -> z_r
    - z_p -> encoder -> z_pp
9. generator loss using Eq. 12 and backprop
'''
