import matplotlib.pyplot as plt
import numpy as np
import datetime

def plot(data, ylabel):
        plt.figure()   
        marker = ['^','v','o','s','x']
        i = 0 
        for label in data:
            arr1 = data[label][0:len(data[label]):20]
            plt.plot(range(0, 3000, 20), arr1, label=label, linestyle=':', linewidth=1, marker = marker[i], markersize = 2)
            # plt.plot(range(len(data[label])), data[label], label=label, linestyle=':', linewidth=1)
            i = i + 1
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig('./test/test.pdf')

if __name__ == '__main__':
    labels = []
    items2 = []
    data = {}
    i=0
    with open(r'./test/test.txt', 'r') as f:
        for item in f.readlines():
            item1 = np.array(item.split())
            label = item1[0]
            labels.append(label)
            item2 = np.delete(item1, 0).astype('float32')
            items2.append(item2)
    f.close()


    for label in labels:
        data[label] = items2[i]
        i = i+1

    plot(data, 'test_acc')
    