import numpy as np
import matplotlib.pyplot as plt

m1 = np.load('cb513_test_prob_1.npy')
m2 = np.load('cb513_test_prob_2.npy')
m3 = np.load('cb513_test_prob_3.npy')
m4 = np.load('cb513_test_prob_4.npy')
m5 = np.load('cb513_test_prob_5.npy')
m6 = np.load('cb513_test_prob_6.npy')

print (m1.shape, m2.shape, m3.shape, m4.shape, m5.shape, m6.shape)

# warped_order_1 = ['NoSeq', 'H', 'E', 'L','T', 'S', 'G', 'B',  'I']
#                    0        1    2    3   4    5    6    7     8
#                  ['L',     'B', 'E', 'G','I', 'H', 'S', 'T', 'NoSeq'] # new order
order_list = [8,5,2,0,7,6,3,1,4]
labels = ['L', 'B', 'E', 'G','I', 'H', 'S', 'T', 'NoSeq']

m1p = np.zeros_like(m4)
m2p = np.zeros_like(m4)
m3p = np.zeros_like(m4)
m4p = np.zeros_like(m4)
m5p = np.zeros_like(m4)
m6p = np.zeros_like(m4)
for count, i in enumerate(order_list):
    m1p[:,:,i] = m1[:,:700,count]
    m2p[:,:,i] = m2[:,:700,count]
    m3p[:,:,i] = m3[:,:700,count]
    m4p[:,:,i] = m4[:,:700,count]
    m5p[:,:,i] = m5[:,:700,count]
    m6p[:,:,i] = m6[:,:700,count]

def check_softmax(T):
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            try:
                assert np.abs(np.sum(T[i,j,:])-1.0)<0.0001
            except:
                print (np.sum(T[i,j,:]), 'failed')
                return 
    print ('outputs are softmaxed')

#check_softmax(m1)

summed_probs = m1p + m2p + m3p + m4p + m5# + m6p

length_list = [len(line.strip().split(',')[2]) for line in open('cb513test_solution.csv').readlines()]
print ('max protein seq length is', np.max(length_list))

ensemble_predictions = []
for protein_idx, i in enumerate(length_list):
    new_pred = ''
    for j in range(i):
        new_pred += labels[np.argmax(summed_probs[protein_idx, j ,:])]
    ensemble_predictions.append(new_pred)

with open('ensemble_predictions.csv','w') as f:
    for idx, i in enumerate(ensemble_predictions):
        f.write(str(idx)+','+i + '\n')
        
# calculating accuracy 
def get_acc(gt,pred):
    assert len(gt)== len(pred)
    correct = 0
    for i in range(len(gt)):
        if gt[i]==pred[i]:
            correct+=1
            
    return (1.0*correct)/len(gt)

gt_all = [line.strip().split(',')[3] for line in open('cb513test_solution.csv').readlines()]
acc_list = []

for gt,pred in zip(gt_all,ensemble_predictions):
	if len(gt) == len(pred):
		acc = get_acc(gt,pred)
		acc_list.append(acc)
print ('mean accuracy is', np.mean(acc_list))
