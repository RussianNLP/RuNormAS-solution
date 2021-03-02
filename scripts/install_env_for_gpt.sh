#!/bin/bash

pip install triton==0.2.2
DS_BUILD_CPU_ADAM=1 DS_BUILD_SPARSE_ATTN=1 pip install deepspeed==0.3.7
pip install transformers==3.5.1