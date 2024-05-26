
# VGGNet 

🛞 VGG stands for Visual Geometry Group and consists of blocks, where each block is composed of 2D Convolution and Max Pooling layers.

🛞 I [@yota](https://github.com/yotaAI) am Implementing the model from Paper 📄 : https://arxiv.org/abs/1409.1556



## 📝 Knowledge

The model owner created 3 different models for VGGNet
`📍VGG11 : 11 Weights`
`📍VGG13 : 13 Weights`
`📍VGG16 : 16 Weights`
`📍VGG19 : 19 Weights`

✏️ Previously I was implementing the training setup of VGG11 model with randomly initialized weights Currently I am initializaing model's weight with Normal Distribution with `0` mean and `10^-2` variance and bias with `0`.

✏️ As mentioned in the paper the larger of models `VGG13 VGG16 VGG19` are initialized from `Trained VGG11`.

✏️ Or we can train them directly with normal distribution of weights.
 

## 🗃️ Dataset

🗞️ For Learning Purpose I am using [Flower Dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition#).

🗞️ Keep the Flower Datset inside `./dataset`.

🗞️ Dataset is having : 
        
        🌼 daisy
        🌼 rose
        🌼 tulip
        🌼 dandelion
        🌼 sunflower

🗞️As described in paper[Page 4] The model is first trained on dataset of `Scale=256`  then the pretrained model is again trained on `Scale=384` with `Learning Rate = 10^-3`.


## 🤖 Training VGG

🏷️ Now run the `training.py` and Boom!🤯
```bash
python3 training.py -m VGG11 -mp ./vgg_11/ -l loss_vgg11.txt
```

🏷️ Model will be saved in the path you mentiond in `-mp`.

🏷️ for more details you can check `python3 training.py --help`



## 🥷🏻 Ninja Tech

⚡︎ Machine Learning ⚡︎ Deep Learning ⚡︎ CNN ⚡︎


