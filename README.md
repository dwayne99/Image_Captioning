# Image Captioning

This project is based on the  "Show, Attend, and Tell" paper. It is based on using an Encoder-Decoder type of network to caption images. The encoder is a pretrained CNN (Resnet-101) while the Decoder LSTM architecture along with an Attention mechanism.

_________________

Image Captioning is the process of generating textual description of an image. It uses both Natural Language Processing and Computer Vision to generate the captions.

<p align="center">
  <img width="800" height="300" src="https://miro.medium.com/max/3548/1*6BFOIdSHlk24Z3DFEakvnQ.png">
</p>

### **High-level Overview of Image Captioning**

**Network Architecture**

<p align="center">
  <img width="800" height="300" src="https://miro.medium.com/max/1124/1*A9VldrmKKP-YKJXf9Xtzag.jpeg">
</p>

**Encoder**

The Convolutional Neural Network(CNN) can be thought of as an encoder. The input image is given to CNN to extract the features. The last hidden state of the CNN is connected to the Decoder.
Decoder

**The Decoder**

It is a Long-Short-Term-Memory Cell(LSTM) which does language modeling up to the word level. The first time step receives the encoded output from the encoder and also the <START> vector.

**Training**

The output from the last hidden state of the CNN(Encoder) is given to the first time step of the decoder. We set x1 =<START> vector and the desired label y1 = first word in the sequence. Analogously, we set xt =word vector of the first word and expect the network to predict the second word. Finally, on the last step, xt = last word, the target label yt =<END> token.
During training, the correct input is given to the decoder at every time-step, even if the decoder made a mistake before.

**Testing**

The image representation is provided to the first time step of the decoder. Set x1 =<START> vector and compute the distribution over the first word y1. We sample a word from the distribution (or pick the argmax), set its embedding vector as x2, and repeat this process until the <END> token is generated.
During Testing, the output of the decoder at time t is fed back and becomes the input of the decoder at time t+1



```
├── caption.py
├── create_input_files.py
├── data
│   └── dataset_flickr8k.json
├── data_output
├── datasets.py
├── evaluate.py
├── models.py
├── Script.ipynb
├── train.py
└── utils.py

```

## Datasets
* Common Objects in Context (COCO). A collection of more than 120 thousand images with descriptions: [Training](http://images.cocodataset.org/zips/train2014.zip), [Validation](http://images.cocodataset.org/zips/val2014.zip)
* Flickr 8K. A collection of 8 thousand described images taken from flickr.com.
* Flickr 30K. A collection of 30 thousand described images taken from flickr.com.
* [Exploring Image Captioning Datasets](http://sidgan.me/technical/2016/01/09/Exploring-Datasets), 2016

**You could also use [Andrej Karpathy's training, validation, and test splits](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip). This zip file contains the captions. You will also find splits and captions for the Flicker8k and Flicker30k datasets, so feel free to use these instead of MSCOCO if the latter is too large for your computer.**


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## Usage

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Acknowledgements

