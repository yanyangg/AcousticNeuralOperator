# AcousticNeuralOperator
## citation:
Yang, Yan, Angela F. Gao, Jorge C. Castellanos, Zachary E. Ross, Kamyar Azizzadenesheli, and Robert W. Clayton. "Seismic wave propagation and inversion with neural operators." The Seismic Record 1, no. 3 (2021): 126-134.

It is modified from FNO-torch.1.6 (https://github.com/zongyi-li/fourier_neural_operator/tree/master/FNO-torch.1.6). 

In the folder there are:

environment.yml: the python environment we use, mainly pytorch 1.7.1.

Salvus_acoustic_simulation.ipynb: generate training dataset using Salvus (https://mondaic.com/product/)

input_a: examples of source location and velocity model

input_u: examples of acoustic wavefield

fourier_acoustic_train.py: main code on training and evaluation

Other subfolders are just places to store outputs of fourier_acoustic_train.py.
