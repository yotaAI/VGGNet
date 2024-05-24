
# VGGNet 

🛞 VGG stands for Visual Geometry Group and consists of blocks, where each block is composed of 2D Convolution and Max Pooling layers.

🛞 I [@yota](https://github.com/yotaAI) am Implementing the model from Paper 📄 : https://arxiv.org/abs/1409.1556



## 📝 Knowledge

The model owner created 3 different models for VGGNet
`📍VGG11 : 11 Weights`
`📍VGG13 : 13 Weights`
`📍VGG16 : 16 Weights`
`📍VGG19 : 19 Weights`

✏️ Currently I am implementing the training setup of VGG11 model with randomly initialized weights.

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


## 🤖 Training

🏷️ Before Starting Training  Check the `HyperParameter` section of `train.py`.
🏷️ Now run the `train.py` and Boom!🤯
🏷️ Training Loss will be calculated in `loss.txt`
```bash
python3 train.py
```

## 🥷🏻 Ninja Tech

⚡︎ Machine Learning ⚡︎ Deep Learning ⚡︎ CNN ⚡︎


