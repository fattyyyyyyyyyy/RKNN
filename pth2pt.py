# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 22:31:03 2022

@author: 10505
"""

import torch
import numpy as np
import yaml

from src.text import load_text_encoder
from src.asr import ASR

if __name__ == "__main__":
    pt_path = './asr.pt'
    vocab_size = 16000
    feat_dim = 120
    config = yaml.load(open('asr_example.yaml', 'r'), Loader=yaml.FullLoader)
    init_adadelta = config['hparas']['optimizer'] == 'Adadelta'
    
    model = ASR(feat_dim, vocab_size, init_adadelta, **config['model'])
    tokenizer = load_text_encoder(**config['data']['text'])
    ckpt = torch.load('best_ctc.pth', map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    
    inp = torch.Tensor(1, 1, 398, 120)
    trace_model = torch.jit.trace(model, inp) 
    trace_model.save(pt_path) 