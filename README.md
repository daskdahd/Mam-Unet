# Mam-Unet
This code is directly related to our manuscript submitted to the journal Computer Vision. We encourage readers to cite the manuscript when using this code and welcome submissions to Computer Vision.
MamUnet



![总框架图](https://github.com/user-attachments/assets/3c3d4935-f7c5-4400-9046-2f32cd2ffeca)





✨ Core Features


• A Context Anchor Attention High-Level Screening-feature Fusion Pyramid Net
works(CAHS) is proposed. By deploying horizontally-vertically cascaded convolu
tions and group-wise independent spatial modeling in the encoder-decoder bridge,
 CAHSenables precise spatial coordinate perception and adaptive weight modulation
 of multi-scale encoder features. 

 
 • A Cross-Stage Partial Feature Enhancement Layer(CSFEL) is proposed. It incor
porates progressive multi-block convolutional structures and cross-stage partial
 connection mechanisms. CSFEL conducts hierarchical and progressive information
 enhancement on encoder features across different semantic levels, thereby improv
ing feature representation capability.


 • A TransMamba Block network is proposed. Built upon the TransMamba architec
ture, it performs global sequence dependency modeling and long-range contextual
 information capture on the deepest encoder features. 

 Environment Configuration


 
torch


torchvision

tensorboard

scipy==1.15.3

numpy==1.24.4

matplotlib==3.10.3

opencv_python==  4.11.0.86

Pillow==8.2.0

h5py==2.10.0

labelme==3.16.7

mamba_ssm  == 2.2.4


Prompt：

The complete training code and data set preprocessing script will be updated after the paper is published. Please pay attention to the project update！
