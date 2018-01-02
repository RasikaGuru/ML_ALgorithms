import re
import numpy as np
from matplotlib import pyplot
from sklearn.neural_network import MLPClassifier


train='downgesture_train.list'
test='downgesture_test.list'

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
print train_data
print train_label

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


clf = MLPClassifier(solver='lbfgs', alpha=0.1,
                    hidden_layer_sizes=(100, 1), random_state=1)

clf.fit(train_data, train_label)
print clf.predict(test_data)
