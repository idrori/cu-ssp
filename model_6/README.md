| Model \ Dataset                                    | cb513 |
| -------------------------------------------------- | ----- |
| Model 6: Convolutions and bidirectional LSTM       | 67.8% |

| Author        | Affiliation         | Email                |
| ------------- | ------------------- | -------------------- |
| Chinmay Joshi | Columbia University | caj2163@columbia.edu |
| Sonam Goenka  | Columbia University | sg3625@columbia.edu  |


Model:
1. Uses unigrams of the input protein structure and encodes each bigram into a 128 length vector.
2. Concatenates the encoded vector with their 22 profile vectors to produce input X.
3. Convolutional Layer with kernel size 11 and 64 output channels of input X to produce Z.
4. Convolutional Layer with kernel size 7 and 64 output channels of input X to produce W.
5. Concatenate X,W and Z to produce new X.
6. Convolutional Layer with kernel size 5 and 64 output channels of input X to produce Z.
7. Convolutional Layer with kernel size 3 and 64 output channels of input X to produce W.
8. Concatenate X,W and Z to produce new X.
9. Bidirectional CuDNNLSTM with input X.
5. TimeDistributed Dense Layer with softmax activation.
