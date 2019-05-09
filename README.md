# cu-ssp

Implementation of [High Quality Prediction of Protein Q8 Secondary Structure by Diverse Neural Network Architectures](https://arxiv.org/abs/1811.07143),
[Iddo Drori](https://www.cs.columbia.edu/~idrori), [Isht Dwivedi](https://www.linkedin.com/in/isht7/), Pranav Shrestha, Jeffrey Wan, [Yueqi Wang](https://github.com/yueqiw), Yunchu He, Anthony Mazza, Hugh Krogh-Freeman, [Dimitri Leggas](https://www.college.columbia.edu/node/11468), Kendal Sandridge, [Linyong Nan](https://github.com/linyongnan), [Kaveri Thakoor](http://www.seismolab.caltech.edu/thakoor_k.html), [Chinmay Joshi](https://github.com/Chinmay41018), [Sonam Goenka](https://github.com/sonamg1), [Chen Keasar](https://www.cs.bgu.ac.il/~keasar), [Itsik Pe’er](http://www.cs.columbia.edu/~itsik)
NIPS Workshop on Machine Learning for Molecules and Materials, 2018.

We tackle the problem of protein secondary structure prediction using a common task framework. This lead to the introduction of multiple ideas for neural architectures based on state of the art building blocks, used in this task for the first time. We take a principled machine learning approach, which provides genuine, unbiased performance measures, correcting longstanding errors in the application domain. We focus on the Q8 resolution of secondary structure, an active area for continuously improving methods. We use an ensemble of strong predictors to achieve accuracy of 70.7% (on the CB513 test set using the CB6133filtered training set). These results are statistically indistinguishable from those of the top existing predictors. In the spirit of reproducible research we make our data, models and code available, aiming to set a gold standard for purity of training and testing sets. Such good practices lower entry barriers to this domain and facilitate reproducible, extendable research.

Q3 (left) and Q8 (right) secondary structure spheres of protein 1AKD in CB513 dataset:

<img src="https://github.com/idrori/cu-ssp/blob/master/paper/figures/1akd_q3.png" height=300><img src="https://github.com/idrori/cu-ssp/blob/master/paper/figures/1akd_q8.png" height=300>


Models
------

Model 1: Bidirectional GRU with convolution blocks
<img src="https://github.com/idrori/cu-ssp/blob/master/paper/figures/model1.png">

Model 2: U-Net with convolution blocks
<img src="https://github.com/idrori/cu-ssp/blob/master/paper/figures/model2.png">

Model 3: Temoporal convolutional network
<img src="https://github.com/idrori/cu-ssp/blob/master/paper/figures/model3.png">

Model 4: Bidirectional GRUs
<img src="https://github.com/idrori/cu-ssp/blob/master/paper/figures/model4.png">

Model 5: Bidirectional LSTM with attention

<img src="https://github.com/idrori/cu-ssp/blob/master/paper/figures/model5.png" height=400>

Model 6: Convolutions and bidirectional LSTM
<img src="https://github.com/idrori/cu-ssp/blob/master/paper/figures/model6.png">


Acknowledgments
---------------
We would like to thank the 100 CS/DSI/Stats graduate students at Columbia University of the Fall 2018 Deep Learning course for their participation in an in class protein secondary structure prediction competition. The models which achieved top performance in the competition were invited to participate in this follow-up work, which lead to the discovery of new architectures with state of the art performance. We would like to thank Tomer Sidi of BGU for thorough examination of the correct measures used for performance comparison. We would like to thank Jian Zhou and Olga Troyanskaya of Princeton for making their CB6133 dataset available and for updating their CB6133 dataset splits following our work. Chen Keasar is partially supported by grants 1122/14 from the Israel Science Foundation (ISF).

Q3 (left) and Q8 (right) secondary structure spheres of protein 1F52 in CB513 dataset:

<img src="https://github.com/idrori/cu-ssp/blob/master/paper/figures/1f52_q3_spheres.png" height=300><img src="https://github.com/idrori/cu-ssp/blob/master/paper/figures/1f52_q8_spheres.png" height=300>
