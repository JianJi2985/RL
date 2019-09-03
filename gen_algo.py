import keras
from keras.models import load_model
import numpy as np
import time

def genetic_algo(model_best1,model_best2,num=5,mutation=1):
    #model = load_model('.\\models\\actor_1-1-1-1-1.h5')
    Wbest1 = model_best1.get_weights()
    Wbest2 = model_best2.get_weights()
    #model.set_weights(W)
    child_dict = {}
    temp_dict = {'0':[],'1':[]}
    
    # Breeding
    for i in range(len(Wbest1)):
        if i%2 == 0:
            temp_dict['0'].append(Wbest1[i])
            temp_dict['1'].append(Wbest2[i])
        else:
            temp_dict['1'].append(Wbest2[i])
            temp_dict['0'].append(Wbest1[i])
    # Mutation
    for d in range(num):
        child_dict[d] = []
        val = temp_dict[str(d%2)]
        time.sleep(1)
        for i in range(len(Wbest1)):
            m = val[i].shape
            size_val = val[i].size*mutation//100 + 1
            row = np.random.randint(m[0], size=size_val)

            if len(m)==1:
                for j in range(size_val):
                    val[i][row[j]] = np.random.rand(1)[0]
            else:
                col = np.random.randint(m[1], size=size_val)
                for j in range(size_val):
                    val[i][row[j],col[j]] = np.random.rand(1)[0]
            
            child_dict[d].append(val[i])
    
    return child_dict