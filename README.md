
# VGGNet 

🛞 VGG stands for Visual Geometry Group and consists of blocks, where each block is composed of 2D Convolution and Max Pooling layers.

🛞 I [@yota](https://github.com/yotaAI) am Implementing the model from Paper 📄 : https://arxiv.org/abs/1409.1556



## 📝 Knowledge

The model owner created 3 different models for VGGNet
`📍VGG11 : 11 Weights`
`📍VGG13 : 13 Weights`
`📍VGG16 : 16 Weights`
`📍VGG19 : 19 Weights`

✏️ Previously I was implementing the training setup of VGG11 model with randomly initialized weights.

✏️ To Initialize the model's weight with Normal Distribution with `0` mean and `10^-2` variance and bias with `0` we have to `init_weights()` the `VGG Model.`

✏️ As mentioned in the paper the larger of models `VGG13 VGG16 VGG19` are initialized from `Trained VGG11`.
 

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


## 🤖 Training VGG11

🏷️ Before Starting Training  Check the `HyperParameter` section of `train_vgg11.py`.

🏷️ Now run the `train_vgg11.py` and Boom!🤯

🏷️ Training Loss will be calculated in `loss_vgg11.txt`

🏷️ Model will be saved in the path mentioned in the `HyperParamet` section of the script.

🏷️ The `first conv layer` and `last fully connected layers` of `VGG16` will be taken from trained `VGG11` model. 

```bash
python3 train_vgg11.py
```

## 🤖 Training VGG16

🏷️ Before Starting Training  Check the `HyperParameter` section of `train_vgg16.py`.

🏷️ If you are frashly training the model, you have to first Pretrained `VGG11`.

🏷️ Provide the path of the trained `VGG11` model in `vgg_11_path` of `train_vgg16.py`

🏷️ Now run the `train_vgg16.py` and Boom!🤯

🏷️ Training Loss will be calculated in `loss_vgg16.txt`

```bash
python3 train_vgg16.py
```

## 🥷🏻 Ninja Tech

⚡︎ Machine Learning ⚡︎ Deep Learning ⚡︎ CNN ⚡︎


