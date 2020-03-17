import re
import sys
import matplotlib.pyplot as plt

print('reading: ',sys.argv[1])
iter = []
loss = []
for line in open(sys.argv[1]):
    it = re.search('iteration[\s\d]{,9}/[\s\d]{,8}',line)
    lm = re.search('lm loss [\d].[\d]{,9}E\+[\d]{2}',line) 
    if lm:
        iter.append(int(it.group(0)[11:18]))
        loss.append(float(lm.group(0)[8:20]))
        #print(int(it.group(0)[11:18]),float(lm.group(0)[8:20]))
plt.plot(iter,loss)
plt.xlabel('iterations')
plt.ylabel('lm loss')
plt.grid()
plt.show()
