# RKNN
## 复现步骤
这个模型是基于https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch 项目，模型定义文件位于./src/asr.py。
主要做的改动是略去了attention解码的部分，只使用了ctc解码，同时将模型的输入从(bs, 398, 120)改为了(bs, 398, 120, 1)，并在模型开头做了reshape操作。
训练得到的模型文件保存为ctc_best.pth，之后使用pth2pt.py文件将其转换成了asr.pt模型，model_conversion.py是自己编写的转rknn模型文件。

