# DCASE2020-TASK2-semi-VAE
The whole program is divided into two parts. First, the hidden layer vector of each audio of each epoch is trained and saved.
Secondly, the best training epoch is obtained by the standard deviation of the likelihood of the normal audio hidden vector in the training set, and the abnormal score is obtained by the likelihood difference of the hidden vector in the test set. 
There are two submission methods, one is to use small samples as training set, the other is to use the whole training set as training set.
