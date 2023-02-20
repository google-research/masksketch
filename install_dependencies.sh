#!/bin/bash

pip install -r requirements.txt

conda install --yes -c pytorch pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0
pip install ftfy regex tqdm

