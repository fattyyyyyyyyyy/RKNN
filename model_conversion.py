# -*- coding: utf-8 -*-
import torchvision.models as models 
import torch
import torch.jit
# from src.asr import ASR 
from rknn.api import RKNN 
 
if __name__ == '__main__': 
    # model = ASR(self.feat_dim, self.vocab_size, init_adadelta, **self.config['model']).to(self.device)
    # ckpt = torch.load(self.paras.load, map_location=self.device if self.mode == 'train' else 'cpu')
    # model.load_state_dict(ckpt['model'])

    # model path 
    pt_path = './asr.pt'
    rknn_path = './asr.rknn'
 
    # create rknn object
    rknn = RKNN() 
 
    # Set preprocessing parameters of model
    rknn.config(batch_size=1)
 
    # load pytorch model
    ret = rknn.load_pytorch(model=pt_path, input_size_list=[[1, 398, 120]])
    # ret = rknn.load_onnx(model='./ASR/asr.onnx', input_size_list=[[398, 120, 1]])
    if ret != 0: 
        print('Load Pytorch model failed!') 
        exit(ret) 

    # build rknn quantitative model
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt') 
    if ret != 0: 
        print('Build model failed!') 
        exit(ret) 
 
    # export rknn model 
    ret = rknn.export_rknn(rknn_path) 
    if ret != 0: 
        print('Export resnet18.rknn failed!') 
        exit(ret) 
 
    rknn.release()