import re
import numpy as np
from matplotlib import pyplot
train='downgesture_train.list'
test='downgesture_test.list'
learning_rate=0.1
np.seterr( divide='ignore', invalid='ignore')

#1 input layer, 1 hidden layer of size 100 and 1 output node
#activation function is sigmoid function
#100 training epochs

#function to read pgm files
def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

#using sigmoid as activation function
def sigmoid(s):
    return (1.0 / (1.0 + np.exp(-s)))

#function to compute derivative of sigmoid function
def der_sigmoid(x):
    return (sigmoid(x)*(1-sigmoid(x)))

def train_function(train_data,train_label):
    epochs=1000
    input_weights=np.random.uniform(-1,1,(train_data.shape[1],100))
    hidden_weights=np.random.uniform(-1,1,(100,1))

    while(epochs):
        for i in range(train_data.shape[0]):
            X1=train_data[i]
            S1=np.dot(X1,input_weights)
            X2=sigmoid(S1)
            S2=np.dot(X2,hidden_weights)
            X3=sigmoid(S2)

            # if X3 > 0.5:
            #     prediction=1.0
            # else:
            #     prediction=0.0

            error = (X3 - train_label[i]) ** 2
            #deltaES(l)=deltaEX*deltaXS -------------->delta(l)
            deltaEX=2*(X3-train_label[i])
            deltaXS=der_sigmoid(S2)
            deltal=np.dot(deltaEX,deltaXS)
            gradient=learning_rate*np.dot(deltal,X2)
            for i in range (100):
                hidden_weights[i]=hidden_weights[i]-gradient[i]


            #deltaES(l-1)=deltaESl*deltaS(l)S(l-1)
                    #    =deltaESl*deltaS(l)X(l-1)*deltaX(l-1)S(l-1)
                    #    =delta(l)*W(input weights)*der_sigmoid(X1)


            hidden_array = np.array(hidden_weights).T
            ans = np.sum(np.array(deltal).dot(hidden_array),axis = 1)

            der_sig = np.array(der_sigmoid(S1))
            der_sig_reshape = der_sig.reshape([100,1])


            deltalminus1 = np.dot(ans.reshape([1,1]),der_sig_reshape.T)


            deltalminus1 = deltalminus1.reshape([1,100])
            X1 = X1.reshape([960,1])

            input_weights=input_weights-np.dot(X1,deltalminus1)

        epochs=epochs-1

    np.savetxt('input_weights.txt',input_weights)
    np.savetxt('hidden_weights.txt',hidden_weights)

def test_function(test_data,test_label):
    input_weights=np.loadtxt('input_weights.txt')
    hidden_weights=np.loadtxt('hidden_weights.txt')
    prediction=[]
    for i in range(test_data.shape[0]):
            X1=test_data[i]
            S1=np.dot(X1,input_weights)
            X2=sigmoid(S1)
            S2=np.dot(X2,hidden_weights)
            X3=sigmoid(S2)

            if X3 > 0.5:
                prediction.append(1.0)
            else:
                prediction.append(0.0)

    np.savetxt ('prediction.txt',prediction)
    count=0.0
    for i in range(len(prediction)):
        if prediction[i]==test_label[i]:
            count=count+1
    print (count/len(prediction))*100
#Import train data
train_data=[]
train_label=[]
for line in open(train,"r").readlines():
        train_data.append(read_pgm(line.strip()))
        if 'down' in line:
            train_label.append(1.0)
        else:
            train_label.append(0.0)
train_data=np.array(train_data,dtype="float").reshape([len(train_data),32*30])
train_label=np.array(train_label)
#print "train data"
#print np.shape(train_data)
#print train_data

#print "label"
#print train_label
#print np.shape(train_label)


#Import test data
test_data=[]
test_label=[]
for line in open(test,"r").readlines():
        test_data.append(read_pgm(line.strip()))
        if 'down' in line:
            test_label.append(1.0)
        else:
            test_label.append(0.0)
test_data=np.array(test_data,dtype="float").reshape([len(test_data),32*30])
#test_label=np.array(test_label)
#print "test data"
#print np.shape(test_data)
#print test_data

#print "label"
#print test_label
#print np.shape(test_label)
train_function(train_data,train_label)
# test_function(test_data,test_label)
