# Cartoon-Style-Transfer
a cartoon transfer net based on CartoonGAN
create folder data/train/Cartoon,data/train/Cartoon_blur,data/train/Photo
create folder data/test/Photo
create folder samples/cartoon_GEN
to smooth the edge: python edgeDilate.py
to initialize: python ./train --dataroot ./data/ --cuda --initialization
train: python ./train --dataroot ./data/ --cuda --load_model ./output/initial_checkpoint.pth
test: python ./test --dataroot ./data/  --load_model ./output/checkpoint50.pth
