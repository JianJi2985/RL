import numpy as np
import random
import time
import keras
from keras import optimizers
from keras.optimizers import SGD,RMSprop,Adam,Adagrad,Adadelta,Adamax,Nadam
from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, Input, Dense, LSTM, Concatenate, Conv2D, MaxPooling2D, Conv1D, Flatten, MaxPooling1D, Dropout, Activation

from keras.utils import plot_model
#from keras.optimizers import SGD, RMSPROP


class MlModel(object):
    #count = 0
    def __init__(self):
        self.mid = None
        self.model = None
        self.design = None
        self.fn_list = {'Input':Input,\
                        'Dense':Dense,\
                        'LSTM':LSTM,\
                        'Concatenate':Concatenate,\
                        'Dropout':Dropout,\
                        'Activation':Activation,\
                        'Conv2D':Conv2D,\
                        'MaxPooling2D':MaxPooling2D,\
                        'Flatten':Flatten,\
                        'Embedding':Embedding,\
                        'Conv1D':Conv1D,\
                        'MaxPooling1D':MaxPooling1D}
        
        self.opt_list = {'SGD':SGD,\
                         'RMSprop':RMSprop,\
                         'Adam':Adam,\
                         'Adagrad':Adagrad,\
                         'Adadelta':Adadelta,\
                         'Adamax':Adamax,\
                         'Nadam':Nadam}
        self.group = None

    def create_model(self,model_param):
        if model_param['mode'] == 'sequential':
            self.create_seq_model(model_param)
        else:
            self.create_fn_model(model_param)
    def create_seq_model(self,model_param):
        self.mid = int(time.time())
        self.design = model_param
        layer_param = model_param['layers']
        compile_param = model_param['compile']
        if isinstance(model_param['compile']['optimizer'],dict):
            opt = list(model_param['compile']['optimizer'].keys())[0]
            model_param['compile']['optimizer'] = self.opt_list[opt](**(model_param['compile']['optimizer'][opt]))
        self.model = Sequential()
        for i in range(len(layer_param)):
            #print(layer)
            #print(layer_param[i])
            key = list(layer_param[i].keys())[0]
            if 'input_shape' in layer_param[i][key]:
                #print(layer_param[i][key]['input_shape'])
                layer_param[i][key]['input_shape'] = tuple(layer_param[i][key]['input_shape'])
            self.model.add(self.fn_list[key](**(layer_param[i][key][0])))
            #print(key,layer_param[i][key][0])
        self.model.compile(**(compile_param))
    
    def create_fn_model(self,model_param):
        self.mid = int(time.time())
        self.design = model_param
        layer_param = model_param['layers']
        compile_param = model_param['compile']
        self.model = Sequential()
        for i in range(len(layer_param)):
            #print(layer)
            #print(layer_param[i])
            key = list(layer_param[i].keys())[0]
            if 'input_shape' in layer_param[i][key]:
                #print(layer_param[i][key]['input_shape'])
                layer_param[i][key]['input_shape'] = tuple(layer_param[i][key]['input_shape'])
            #self.model.add(self.fn_list[key](**(layer_param[i][key][0])))
            #print(key,layer_param[i][key][0])
        self.model.compile(**(compile_param))
        
    def q_learning(self,model,env, discount = 0.9, random = False):
        
        if random:
            seq = np.random.randint(0,len(env),len(env))
        else:
            seq = np.arange(len(env))
        inp = []
        out = []
        print('##############################')
        print('Q Learning')
        for i in range(len(seq)):
            state, action, reward, new_state, game_over = env[seq[i]]
            #inp_data = state
            #print('\n')
            #print(i,action,sel_action)
            #print('\n')
            
            #print(state)
            #print(new_state)
            
            if isinstance(new_state,list):
                #print(state[0].shape,state[1].shape)
                #inp.append([np.ravel(state[0]),np.ravel(state[1])])
                action_list = np.ravel(model.predict(state))
                sel_qsa = action_list[action]
                qsa = np.max(np.ravel(model.predict(new_state)))
            else:
                
                action_list = np.ravel(model.predict(state.reshape((1,-1))))
                sel_qsa = action_list[action]
                qsa = np.max(np.ravel(model.predict(new_state.reshape((1,-1)))))
            action_list[action] = reward + discount*sel_qsa
            inp.append(state)
            out.append(action_list)
        if isinstance(new_state,list):
            #print(inp)
            input_data = []
            for i in range(len(inp[0])):
                #print(inp[i][0].shape,inp[i][1].shape)
                input_data.append(np.vstack([data[i] for data in inp]))
            model.fit(input_data,np.array(out),batch_size = 25,epochs = 1, verbose = 1)
        else:
            #print(np.array(inp).shape)
            model.fit(np.array(inp),np.array(out),batch_size = 25,epochs = 1, verbose = 1)   
        print('Q End')
        print('##############################')
        print('-',end=' ')
        return model
    
