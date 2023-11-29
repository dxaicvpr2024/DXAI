"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

FILE=$1

if  [ $FILE == "celeba-hq-dataset" ]; then
    URL=https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip?dl=0
    ZIP_FILE=./data/celeba_hq.zip
    mkdir -p ./data
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data
    rm $ZIP_FILE

elif  [ $FILE == "afhq-dataset" ]; then
    URL=https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0
    ZIP_FILE=./data/afhq.zip
    mkdir -p ./data
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data
    rm $ZIP_FILE

elif  [ $FILE == "pretrained-network-afhq" ]; then
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=12MVga-DdXbhLk7RoKT9wfjKTnKSs7Hw5' -O afhq_resnet18_ch_3_weights.ckpt

elif  [ $FILE == "pretrained-network-celeba-hq" ]; then
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1S7yPXhkk-eK7YCvcNVjFNC6xuBPSkM2D' -O celeba_hq_resnet18_ch_3_weights.ckpt
        
else
    echo "Available arguments are pretrained-network-celeba-hq, pretrained-network-afhq, celeba-hq-dataset, and afhq-dataset."
    exit 1

fi
