# GraphLSTM-SLU

Code for AAAI 2020 paper "Graph LSTM with Context-Gated Mechanism for Spoken Language Understanding"

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

**Note**: Don't forget to set the parameters (e.g., -data) of train.py to the corresponding dataset. 


## data & model checkpoints
You can also download the preprocessed data and our well-trained model checkpoints from: [this link](https://pan.baidu.com/s/1tVRhnAfeivi4k0UKaRy83g)

Extraction Code : ct8x

**To use the preprocessed data and model checkpoints**:
1. Put the preprocessed data in /data directory.
2. Put the checkpoints in the root directory.
3. Set the -data parameter of train.py to be data/snips.pt or data/atis.pt (the name of your dataset).
4. Set the -restore_model of train.py parameter to be snips.chkpt or atis.chkpt (the name of your checkpoint).

## Reference
If the code is helpful to your research, please kindly cite our paper:

(To be updated)


## Thanks 
1. The main data used in our paper is from [this repo](https://github.com/MiuLab/SlotGated-SLU).
2. You can also find the NE tags in [this repo](https://github.com/mesnilgr/is13).
3. The S-LSTM is proposed in [this paper](https://arxiv.org/abs/1805.02474) and our implementation is based on [this repo](https://github.com/WildeLau/S-LSTM_pytorch).

Many thanks to them all!





