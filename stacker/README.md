This directory contains the code for the "stacker". Each implementation uses a level 0 model modified to take additional features on input. These features are the predictions of the other models and can be found in the folder meta-features.

stacker_model1.ipynb assumes access to a GPU and the data loading only works in a Google Colaboratory notebook. Its best prediction accuracy (found in predictions/cb6133test_stacker_model1_6.csv) is 75.4%.

stacker_model4.ipynb makes no assumptions and can be run on a CPU. Its best prediction accuracy (found in predictions/cb6133test_stacker_model5_4.csv) is 74.5%.