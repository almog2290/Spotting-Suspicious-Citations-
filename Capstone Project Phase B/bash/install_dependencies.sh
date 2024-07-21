#!/bin/bash

pip install lightning
pip install tensorboard
pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch_geometric
pip install cython
pip install matplotlib
