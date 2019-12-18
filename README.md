# GraphLSTM-SLU

Code for paper "Graph LSTM with Context-Gated Mechanism for Spoken Language Understanding"

## Requirements

> - python 3.7
> - pytorch >=0.4
> - tqdm
> - allennlp

## How to run?
**For the Snips dataset:**

First, you should preprocess data:

```
python3 preprocess.py
```

To train the model:

```
python3 train.py
```

**For the ATIS dataset:**

You need to first comment the line in Constants.py:
```
TAGS = 76  # for Snips
```
and uncomment the line
```
# TAGS = 124  # for ATIS
```
Then the running process is the same as Snips. 

You can also download the preprocessed data and model checkpoints from: [Link](https://pan.baidu.com/s/1tVRhnAfeivi4k0UKaRy83g)

Extraction Code : ct8x

## Data & Thanks
1. The main data used in our paper is from [This repo](https://github.com/MiuLab/SlotGated-SLU).
2. You can also find the NE tags in [This repo](https://github.com/mesnilgr/is13).

Many thanks to them!

Since the two datasets are relatively small, you may need to tune the hyper-parameters and random seed. 




