import numpy as np
import math
x=np.array([3,1,0,4])
y=np.array([2,2,1,3])
W=[0,1]
h=[]
v=np.array([])
alpha=0.2
print("Alpha = ", alpha)
m=len(x)

def Hypo(x,W):
    h.clear()
    for i in range(0 , 4):
        a=np.array([1,x[i]])
        b=np.dot(W,a)
        h.append(b)
    return h

J=(1/(2*m)) * sum(((Hypo(x,W)-y)**2))
print("Error (J) before the gradient descent = ",J)

for i in range(0,5):
    temp0=W[0] - (alpha * ((1/m)*(sum(Hypo(x,W)-y))))
    v=Hypo(x,W)-y
    temp1=W[1] -(alpha * ((1/m)*(sum(v*x))))
    W[0]=temp0
    W[1]=temp1
    print("Theta1 = ",W[0]) 
    print("Theta2 = " , W[1])
    print(" " )
#print(Hypo(x,W))
J=(1/(2*m)) * sum(((Hypo(x,W)-y)**2))
print("Error (J) after 5 iterations of gradient descent = ",J)
