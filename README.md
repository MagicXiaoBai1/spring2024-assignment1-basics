# CS336 Spring 2024 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2024_assignment1_basics.pdf](./cs336_spring2024_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

0. Set up a conda environment and install packages:

``` sh
conda create -n cs336_basics python=3.10 --yes
conda activate cs336_basics
pip install -e .'[test]'
```

1. Run unit tests:

``` sh
pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

2. Download the TinyStories data and a subsample of OpenWebText:

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

Created： 2y ago
Runtime： 1h 24m 56s
Sweep： -
activation： silu
batch_size： 128
checkpoint_interval： 10000
checkpoint_path： checkpoints/
compile： true
context_length： 512
d_ff： 4096
d_model： 1024
d_vocab_size： -
dataset： owt
decay： 0.2
flash： -
log_interval： 100
lr： 0.0004
name： -
num_heads： 16
num_layer： 12
parallel_layers： false
post_norm： false
rotary： true
tie_embeddings： true
total_steps： 30000
use_gated_mlp： true
use_sophia： -
loss/train： 3.18867
loss/valid： 3.22499
lr： 0.00018721
perplexity/train： 24.25607
perplexity/： 25.15326