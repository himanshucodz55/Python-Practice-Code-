import numpy as np
def gradient_descent(x,y):
    m_curr=1
    b_curr=0
    itration=100000
    n=len(x)
    learning_rate=.001
    for i in range(itration):
        y_predicted=m_curr*x+b_curr
        cost=(1/n)*sum([val**2 for val in (y-y_predicted)])
        md=-(1/n)*sum(x*(y-y_predicted))
        bd=-(1/n)*sum(y-y_predicted)
        m_curr=m_curr-learning_rate*md
        b_curr=b_curr-learning_rate*bd
        print("m {},b {},cost {},itration {}".format(m_curr,b_curr,cost,i))

x=np.array([1,2,4,5])
y=np.array([1,3,2,4])

gradient_descent(x,y)