import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker as plticker
from forex_trader_switch import ForexTraderSwitch
from forexTrader import ForexTrader
from forexprime import ForexPrime
import time
import datetime
import keras
from keras.optimizers import SGD, RMSprop
from keras.layers import Input, LSTM, Dense, Dropout
from keras.models import load_model,Model
from keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
#import matplotlib.animation as animation
#from data_config import *





if __name__ == "__main__":
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))
    
    ftlist = []
    l_model = True
    diff_acc = True
    colorlist = ['black','firebrick','sandybrown','bisque','olivedrab',\
    'slategray','steelblue','mediumspringgreen','deepskyblue',\
     'indianred','yellowgreen','navy','mediumvioletred']
    epsilon = [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
    iteration = 5000
    l_rate = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
    q_freq = 10
    funds = 1000
    mem_space = 5
    time_count = [180,720,180]
    accountID = 'account id here'
    access_token = 'access token here'
    curr_pair_list = ['EUR_JPY','EUR_USD','USD_JPY']
    #curr_pair_list = ['EUR_GBP','EUR_JPY','EUR_USD','GBP_JPY','GBP_USD','USD_JPY']
    units = 1
    fts = ForexTraderSwitch()
    fts.initial_value( accountID = accountID,access_token = access_token,\
                      curr_pair_list = curr_pair_list,units = units, time_count = time_count)
    
    order_size = ForexTraderSwitch.order.size
    signal_size = ForexTraderSwitch.signal.size
    num_curr = ForexTraderSwitch.num_curr
    currentpos_size = num_curr*3
    currentrate_size = ForexTraderSwitch.current_rate.size
    time_count = ForexTraderSwitch.time_count
    moves_allowed = ForexTraderSwitch.moves_allowed.size
    mem_size = num_curr *mem_space*3
    print(order_size,signal_size,num_curr,currentpos_size,currentrate_size,time_count,moves_allowed)
    
    #ftlist = []
    currentpath = ForexTraderSwitch.current_path
    
    accountID = 'account id here'
    modtype = '1-1-1-1-1'
    inp_size = 3*order_size + currentpos_size + 5 # for total value in usd
    out_size = moves_allowed
    model_param = {'mode':'sequential','layers':[{'Dense':[{'units':inp_size*2, 'activation':'tanh', 'input_dim':inp_size}]},\
                                                 {'Dropout':[{'rate':0.1}]},\
                                                 {'Dense':[{'units':inp_size, 'activation':'relu'}]},\
                                                 {'Dropout':[{'rate':0.1}]},\
                                                 {'Dense':[{'units':inp_size, 'activation':'relu'}]},\
                                                 {'Dropout':[{'rate':0.1}]},\
                                {'Dense':[{'units':out_size}]}],\
                      'compile':{'optimizer':{'RMSprop':{'lr':l_rate[0]}},'loss':'mse','metrics':['accuracy']}}

    if not l_model:
        dt = {}
        dt['si'] = Input(shape = (inp_size,),name = "state_input")
        dt['x'] = Dense((inp_size), activation = 'tanh')(dt['si'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['x'] = Dense((inp_size), activation = 'relu')(dt['x'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['x'] = Dense((inp_size), activation = 'relu')(dt['x'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['out1'] = Dense(out_size)(dt['x'])
        dt['out2'] = Dense(4)(dt['x'])

        modelx = Model(inputs =dt['si'],outputs = [dt['out1'],dt['out2']])
        modelx.compile(optimizer=RMSprop(lr = l_rate[7]), loss='mse', metrics=['accuracy'])
        
        inp_size += 4
        dt = {}
        dt['si'] = Input(shape = (inp_size,),name = "state_input")
        dt['x'] = Dense((inp_size), activation = 'tanh')(dt['si'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['x'] = Dense((inp_size), activation = 'relu')(dt['x'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['x'] = Dense((inp_size), activation = 'relu')(dt['x'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['out'] = Dense(2)(dt['x'])
        
        modely = Model(inputs = dt['si'],outputs = dt['out'])
        modely.compile(optimizer=RMSprop(lr = l_rate[7]), loss='mse', metrics=['accuracy'])
        
        ft = ForexTrader(accountID,access_token,modtype,modpath = [], \
                         diff_acc = diff_acc,epsilon = epsilon[0], q_freq = q_freq)
        
        ft.actor = modelx
        ft.mid_actor = int(time.time())
        ft.critic = modely
        ft.mid_critic = int(time.time())
        ft.actor.summary()
        ft.critic.summary()
        print('model %s created'%modtype)
        ft.model_save()
    else:
        modpath = modpath = [currentpath+'/models/actor_'+modtype+'.h5',currentpath+'/models/critic_'+modtype+'.h5']
        ft = ForexTrader(accountID,access_token,modtype,modpath = modpath, \
                         diff_acc = diff_acc,epsilon = epsilon[0], q_freq = q_freq)
        ft.actor.summary()
        ft.critic.summary()
        print('model %s loaded'%modtype)
    #global ftlist
    ftlist.append(ft)
    time.sleep(1)
    
    accountID = 'account id here'
    modtype = '2-1-1-1-1'
    inp_size = 3*order_size + currentpos_size + 5 # for total value in usd
    out_size = moves_allowed
    model_param = {'mode':'sequential','layers':[{'Dense':[{'units':inp_size*2, 'activation':'tanh', 'input_dim':inp_size}]},\
                                                 {'Dropout':[{'rate':0.1}]},\
                                                 {'Dense':[{'units':inp_size, 'activation':'relu'}]},\
                                                 {'Dropout':[{'rate':0.1}]},\
                                                 {'Dense':[{'units':inp_size, 'activation':'relu'}]},\
                                                 {'Dropout':[{'rate':0.1}]},\
                                {'Dense':[{'units':out_size}]}],\
                      'compile':{'optimizer':{'RMSprop':{'lr':l_rate[1]}},'loss':'mse','metrics':['accuracy']}}

    if not l_model:
        dt = {}
        dt['si'] = Input(shape = (inp_size,),name = "state_input")
        dt['x'] = Dense((inp_size), activation = 'tanh')(dt['si'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['x'] = Dense((inp_size), activation = 'relu')(dt['x'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['x'] = Dense((inp_size), activation = 'relu')(dt['x'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['out1'] = Dense(out_size)(dt['x'])
        dt['out2'] = Dense(4)(dt['x'])
        dt['out3'] = Dense(7)(dt['x'])
        dt['out4'] = Dense(5)(dt['x'])

        modelx = Model(inputs = dt['si'],outputs = [dt['out1'],dt['out2'],dt['out3'],dt['out4']])
        modelx.compile(optimizer=RMSprop(lr = l_rate[7]), loss='mse', metrics=['accuracy'])
        
        inp_size += 16
        dt = {}
        dt['si'] = Input(shape = (inp_size,),name = "state_input")
        dt['x'] = Dense((inp_size), activation = 'tanh')(dt['si'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['x'] = Dense((inp_size), activation = 'relu')(dt['x'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['x'] = Dense((inp_size), activation = 'relu')(dt['x'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['out'] = Dense(4)(dt['x'])
        
        modely = Model(inputs = dt['si'],outputs = dt['out'])
        modely.compile(optimizer=RMSprop(lr = l_rate[7]), loss='mse', metrics=['accuracy'])
        
        ft = ForexTrader(accountID,access_token,modtype,modpath = [], \
                         diff_acc = diff_acc,epsilon = epsilon[0], q_freq = q_freq)
        
        ft.actor = modelx
        ft.mid_actor = int(time.time())
        ft.critic = modely
        ft.mid_critic = int(time.time())
        ft.actor.summary()
        ft.critic.summary()
        print('model %s created'%modtype)
        ft.model_save()
    else:
        modpath = [currentpath+'/models/actor_'+modtype+'.h5',currentpath+'/models/critic_'+modtype+'.h5']
        ft = ForexTrader(accountID,access_token,modtype,modpath = modpath, \
                         diff_acc = diff_acc,epsilon = epsilon[1], q_freq = q_freq)
        ft.actor.summary()
        ft.critic.summary()
        print('model %s loaded'%modtype)
    #global ftlist
    ftlist.append(ft)
    time.sleep(1)
    
    accountID = 'account id here'
    modtype = '1-1-2-2-1'
    inp_size = 3*signal_size + currentrate_size +mem_size + 5 # for total value in usd
    out_size = moves_allowed
    model_param = {'mode':'sequential','layers':[{'Dense':[{'units':inp_size*2, 'activation':'tanh', 'input_dim':inp_size}]},\
                                                 {'Dropout':[{'rate':0.1}]},\
                                                 {'Dense':[{'units':inp_size, 'activation':'relu'}]},\
                                                 {'Dropout':[{'rate':0.1}]},\
                                                 {'Dense':[{'units':inp_size, 'activation':'relu'}]},\
                                                 {'Dropout':[{'rate':0.1}]},\
                                {'Dense':[{'units':out_size}]}],\
                      'compile':{'optimizer':{'RMSprop':{'lr':l_rate[2]}},'loss':'mse','metrics':['accuracy']}}

    if not l_model:
        dt = {}
        dt['si'] = Input(shape = (inp_size,),name = "state_input")
        dt['x'] = Dense((inp_size), activation = 'tanh')(dt['si'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['x'] = Dense((inp_size), activation = 'relu')(dt['x'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['x'] = Dense((inp_size), activation = 'relu')(dt['x'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['out1'] = Dense(out_size)(dt['x'])
        dt['out2'] = Dense(4)(dt['x'])

        modelx = Model(inputs = dt['si'],outputs = [dt['out1'],dt['out2']])
        modelx.compile(optimizer=RMSprop(lr = l_rate[7]), loss='mse', metrics=['accuracy'])
        
        inp_size += 4
        dt = {}
        dt['si'] = Input(shape = (inp_size,),name = "state_input")
        dt['x'] = Dense((inp_size), activation = 'tanh')(dt['si'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['x'] = Dense((inp_size), activation = 'relu')(dt['x'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['x'] = Dense((inp_size), activation = 'relu')(dt['x'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['out'] = Dense(2)(dt['x'])
        
        modely = Model(inputs = dt['si'],outputs = dt['out'])
        modely.compile(optimizer=RMSprop(lr = l_rate[7]), loss='mse', metrics=['accuracy'])
        
        ft = ForexTrader(accountID,access_token,modtype,modpath = [], \
                         diff_acc = diff_acc,epsilon = epsilon[0], q_freq = q_freq)
        
        ft.actor = modelx
        ft.mid_actor = int(time.time())
        ft.critic = modely
        ft.mid_critic = int(time.time())
        ft.actor.summary()
        ft.critic.summary()
        print('model %s created'%modtype)
        ft.model_save()
    else:
        modpath = [currentpath+'/models/actor_'+modtype+'.h5',currentpath+'/models/critic_'+modtype+'.h5']
        ft = ForexTrader(accountID,access_token,modtype,modpath = modpath, \
                         diff_acc = diff_acc,epsilon = epsilon[2], q_freq = q_freq)
        ft.actor.summary()
        ft.critic.summary()
        print('model %s loaded'%modtype)
    #global ftlist
    ftlist.append(ft)
    time.sleep(1)
    
    accountID = 'account id here'
    modtype = '2-1-2-2-1'
    inp_size = 3*signal_size + currentrate_size + mem_size + 5 # for total value in usd
    out_size = moves_allowed
    model_param = {'mode':'sequential','layers':[{'Dense':[{'units':inp_size*2, 'activation':'tanh', 'input_dim':inp_size}]},\
                                                 {'Dropout':[{'rate':0.1}]},\
                                                 {'Dense':[{'units':inp_size, 'activation':'relu'}]},\
                                                 {'Dropout':[{'rate':0.1}]},\
                                                 {'Dense':[{'units':inp_size, 'activation':'relu'}]},\
                                                 {'Dropout':[{'rate':0.1}]},\
                                {'Dense':[{'units':out_size}]}],\
                      'compile':{'optimizer':{'RMSprop':{'lr':l_rate[3]}},'loss':'mse','metrics':['accuracy']}}

    if not l_model:
        dt = {}
        dt['si'] = Input(shape = (inp_size,),name = "state_input")
        dt['x'] = Dense((inp_size), activation = 'tanh')(dt['si'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['x'] = Dense((inp_size), activation = 'relu')(dt['x'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['x'] = Dense((inp_size), activation = 'relu')(dt['x'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['out1'] = Dense(out_size)(dt['x'])
        dt['out2'] = Dense(4)(dt['x'])
        dt['out3'] = Dense(7)(dt['x'])
        dt['out4'] = Dense(5)(dt['x'])

        modelx = Model(inputs = dt['si'],outputs = [dt['out1'],dt['out2'],dt['out3'],dt['out4']])
        modelx.compile(optimizer=RMSprop(lr = l_rate[7]), loss='mse', metrics=['accuracy'])
        
        inp_size += 16
        dt = {}
        dt['si'] = Input(shape = (inp_size,),name = "state_input")
        dt['x'] = Dense((inp_size), activation = 'tanh')(dt['si'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['x'] = Dense((inp_size), activation = 'relu')(dt['x'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['x'] = Dense((inp_size), activation = 'relu')(dt['x'])
        dt['x'] = Dropout(rate = 0.5)(dt['x'])
        dt['out'] = Dense(4)(dt['x'])
        
        modely = Model(inputs = dt['si'],outputs = dt['out'])
        modely.compile(optimizer=RMSprop(lr = l_rate[7]), loss='mse', metrics=['accuracy'])
        
        ft = ForexTrader(accountID,access_token,modtype,modpath = [], \
                         diff_acc = diff_acc,epsilon = epsilon[0], q_freq = q_freq)
        
        ft.actor = modelx
        ft.mid_actor = int(time.time())
        ft.critic = modely
        ft.mid_critic = int(time.time())
        ft.actor.summary()
        ft.critic.summary()
        print('model %s created'%modtype)
        ft.model_save()
    else:
        modpath = [currentpath+'/models/actor_'+modtype+'.h5',currentpath+'/models/critic_'+modtype+'.h5']
        ft = ForexTrader(accountID,access_token,modtype,modpath = modpath, \
                         diff_acc = diff_acc,epsilon = epsilon[3], q_freq = q_freq)
        ft.actor.summary()
        ft.critic.summary()
        print('model %s loaded'%modtype)
    #global ftlist
    ftlist.append(ft)
    time.sleep(1)
   
    out_size = moves_allowed
    dt = {}
    dt['fid'] = Input(shape = (time_count[0],10*num_curr), name = "forex_input_daily")
    dt['lo_d'] = LSTM(time_count[0]*num_curr)(dt['fid'])
    dt['fim'] = Input(shape = (time_count[1],10*num_curr), name = "forex_input_monthly")
    dt['lo_m'] = LSTM(time_count[1]*num_curr)(dt['fim'])
    dt['fis'] = Input(shape = (time_count[2],10*num_curr), name = "forex_input_semi")
    dt['lo_s'] = LSTM(time_count[2]*num_curr)(dt['fis'])
    dt['si'] = Input(shape = (mem_size+currentrate_size+5,),name = "state_input")
    dt['x'] = keras.layers.concatenate([dt['lo_d'],dt['lo_m'],dt['lo_s'], dt['si']])
    dt['x'] = Dense((time_count[0]*num_curr//2), activation = 'tanh')(dt['x'])
    dt['x'] = Dropout(rate = 0.5)(dt['x'])
    dt['x'] = Dense((time_count[0]*num_curr//2), activation = 'relu')(dt['x'])
    dt['x'] = Dropout(rate = 0.5)(dt['x'])
    dt['x'] = Dense((time_count[0]*num_curr//2), activation = 'relu')(dt['x'])
    dt['x'] = Dropout(rate = 0.5)(dt['x'])
    dt['out1'] = Dense(out_size)(dt['x'])
    dt['out2'] = Dense(4)(dt['x'])
    
    modelx = Model(inputs =[dt['fid'],dt['fim'],dt['fis'],dt['si']],outputs = [dt['out1'],dt['out2']])
    modelx.compile(optimizer=RMSprop(lr = l_rate[6]), loss='mse', metrics=['accuracy'])
    
    dt = {}
    dt['fid'] = Input(shape = (time_count[0],10*num_curr), name = "forex_input_daily")
    dt['lo_d'] = LSTM(time_count[0]*num_curr)(dt['fid'])
    dt['fim'] = Input(shape = (time_count[1],10*num_curr), name = "forex_input_monthly")
    dt['lo_m'] = LSTM(time_count[1]*num_curr)(dt['fim'])
    dt['fis'] = Input(shape = (time_count[2],10*num_curr), name = "forex_input_semi")
    dt['lo_s'] = LSTM(time_count[2]*num_curr)(dt['fis'])
    dt['si'] = Input(shape = (mem_size+currentrate_size+9,),name = "state_input")
    dt['x'] = keras.layers.concatenate([dt['lo_d'],dt['lo_m'],dt['lo_s'], dt['si']])
    dt['x'] = Dense((time_count[0]*num_curr//2), activation = 'tanh')(dt['x'])
    dt['x'] = Dropout(rate = 0.5)(dt['x'])
    dt['x'] = Dense((time_count[0]*num_curr//2), activation = 'relu')(dt['x'])
    dt['x'] = Dropout(rate = 0.5)(dt['x'])
    dt['x'] = Dense((time_count[0]*num_curr//2), activation = 'relu')(dt['x'])
    dt['x'] = Dropout(rate = 0.5)(dt['x'])
    dt['out'] = Dense(2)(dt['x'])
    
    modely = Model(inputs =[dt['fid'],dt['fim'],dt['fis'],dt['si']],outputs = dt['out'])
    modely.compile(optimizer=RMSprop(lr = l_rate[6]), loss='mse', metrics=['accuracy'])


    #modelx.summary()
    accountID = 'account id here'
    modtype = '1-2-3-2-2'
        
    if not l_model:
        ft = ForexTrader(accountID,access_token,modtype,modpath = [], \
                         diff_acc = diff_acc,epsilon = epsilon[6], q_freq = q_freq)
        
        ft.actor = modelx
        ft.mid_actor = int(time.time())
        ft.critic = modely
        ft.mid_critic = int(time.time())
        print('model %s created'%modtype)    
        ft.model_save()
        ft.actor.summary()
        ft.critic.summary()
        time.sleep(1)
    else:
        modpath = [currentpath+'/models/actor_'+modtype+'.h5',currentpath+'/models/critic_'+modtype+'.h5']
        ft = ForexTrader(accountID,access_token,modtype,modpath = modpath, \
                         diff_acc = diff_acc,epsilon = epsilon[6], q_freq = q_freq)
        ft.actor.summary()
        ft.critic.summary()
        print('model %s loaded'%modtype)
    #global ftlist
    ftlist.append(ft)
    time.sleep(1)
    
    
    out_size = moves_allowed
    dt = {}
    dt['fid'] = Input(shape = (time_count[0],10*num_curr), name = "forex_input_daily")
    dt['lo_d'] = LSTM(time_count[0]*num_curr)(dt['fid'])
    dt['fim'] = Input(shape = (time_count[1],10*num_curr), name = "forex_input_monthly")
    dt['lo_m'] = LSTM(time_count[1]*num_curr)(dt['fim'])
    dt['fis'] = Input(shape = (time_count[2],10*num_curr), name = "forex_input_semi")
    dt['lo_s'] = LSTM(time_count[2]*num_curr)(dt['fis'])
    dt['si'] = Input(shape = (mem_size+currentrate_size+5,),name = "state_input")
    dt['x'] = keras.layers.concatenate([dt['lo_d'],dt['lo_m'],dt['lo_s'], dt['si']])
    dt['x'] = Dense((time_count[0]*num_curr//2), activation = 'tanh')(dt['x'])
    dt['x'] = Dropout(rate = 0.5)(dt['x'])
    dt['x'] = Dense((time_count[0]*num_curr//2), activation = 'relu')(dt['x'])
    dt['x'] = Dropout(rate = 0.5)(dt['x'])
    dt['x'] = Dense((time_count[0]*num_curr//2), activation = 'relu')(dt['x'])
    dt['x'] = Dropout(rate = 0.5)(dt['x'])
    dt['out1'] = Dense(out_size)(dt['x'])
    dt['out2'] = Dense(4)(dt['x'])
    dt['out3'] = Dense(7)(dt['x'])
    dt['out4'] = Dense(5)(dt['x'])
    modelx = Model(inputs =[dt['fid'],dt['fim'],dt['fis'],dt['si']],outputs = [dt['out1'],dt['out2'],dt['out3'],dt['out4']])
    modelx.compile(optimizer=RMSprop(lr = l_rate[7]), loss='mse', metrics=['accuracy'])
    
    dt = {}
    dt['fid'] = Input(shape = (time_count[0],10*num_curr), name = "forex_input_daily")
    dt['lo_d'] = LSTM(time_count[0]*num_curr)(dt['fid'])
    dt['fim'] = Input(shape = (time_count[1],10*num_curr), name = "forex_input_monthly")
    dt['lo_m'] = LSTM(time_count[1]*num_curr)(dt['fim'])
    dt['fis'] = Input(shape = (time_count[2],10*num_curr), name = "forex_input_semi")
    dt['lo_s'] = LSTM(time_count[2]*num_curr)(dt['fis'])
    dt['si'] = Input(shape = (mem_size+currentrate_size+21,),name = "state_input")
    dt['x'] = keras.layers.concatenate([dt['lo_d'],dt['lo_m'],dt['lo_s'], dt['si']])
    dt['x'] = Dense((time_count[0]*num_curr//2), activation = 'tanh')(dt['x'])
    dt['x'] = Dropout(rate = 0.5)(dt['x'])
    dt['x'] = Dense((time_count[0]*num_curr//2), activation = 'relu')(dt['x'])
    dt['x'] = Dropout(rate = 0.5)(dt['x'])
    dt['x'] = Dense((time_count[0]*num_curr//2), activation = 'relu')(dt['x'])
    dt['x'] = Dropout(rate = 0.5)(dt['x'])
    dt['out'] = Dense(4)(dt['x'])
    
    modely = Model(inputs =[dt['fid'],dt['fim'],dt['fis'],dt['si']],outputs = dt['out'])
    modely.compile(optimizer=RMSprop(lr = l_rate[6]), loss='mse', metrics=['accuracy'])

    #modelx.summary()
    accountID = 'account id here'
    modtype = '2-2-3-2-2'
    if not l_model:
        ft = ForexTrader(accountID,access_token,modtype,modpath = [], \
                         diff_acc = diff_acc,epsilon = epsilon[7], q_freq = q_freq)
        
        ft.actor = modelx
        ft.critic = modely
        ft.mid_actor = int(time.time())
        ft.mid_critic = int(time.time())
        
        print('model %s created'%modtype)    
        ft.model_save()
        time.sleep(1)
        ft.midtwin = int(time.time())
    else:
        modpath = [currentpath+'/models/actor_'+modtype+'.h5',currentpath+'/models/critic_'+modtype+'.h5']
        ft = ForexTrader(accountID,access_token,modtype,modpath = modpath, \
                         diff_acc = diff_acc,epsilon = epsilon[7], q_freq = q_freq)
        ft.actor.summary()
        ft.critic.summary()
        print('model %s loaded'%modtype)
    #global ftlist
    ftlist.append(ft)
    time.sleep(1)
    
       
    
    
    
    myFmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
    #if diff_acc:
    #    plt.title('Assets diff account')
    #else:
    #    plt.title('Assets same account')
    count = 1
    
    while count <= iteration:
        #fig.clf()
        if datetime.date.today().weekday() == 4:
            test = datetime.datetime.now().replace(hour=16,minute=55,second=0,microsecond=0)
            if datetime.datetime.now() >= test:
                count = iteration + 1
                continue
        print('\n###################################################################################')
        print('###################################################################################')
        print(int(time.time()))
        print(datetime.datetime.now())
        print('Pass %i of %i'%(count,iteration))
        print('###################################################################################')
        print('###################################################################################')
        for ftrade in ftlist:
            ftrade.runmodel()
            time.sleep(1)
        #ani = animation.FuncAnimation(fig, animate, frames=iteration)
        #if diff_acc:
        #    plt.legend(loc = 'upper left')
        #plt.show()
        if count%1 == 0:
            #ftlist[0].delmodels()
            plotlist = []
            
            for ftrade in ftlist:
                if count%10 == 0:
                    ftrade.model_save()
                    print('Model Saved')
                    ftrade.plot_data()
                    print('Report Generated')
                if diff_acc:
                    plotlist.append([ftrade.datelist,ftrade.navlist])
            fig = plt.figure(num = 2,figsize = (12,8))
            
            #legend = plt.legend(loc = 'lower left')
            #fig.clf()
            plotlist = np.array(plotlist)
            #print(plotlist)
            location = ftlist[0].current_path+'/results/'
            #legend.remove()
            plt.ylabel('Current Asset in USD')
            plt.xlabel('Time')
            plt.gca().xaxis.set_major_formatter(myFmt)
            if diff_acc:
                np.save(location+'Diff_Plot',plotlist)
                plt.title('Assets diff account')
                #plt.axhline(funds,color = colorlist[0],label = 'base')
                #x = np.linspace(0,plotlist.shape[1],plotlist.shape[1])
                start_date = ftlist[0].datelist[0]
                end_date = ftlist[-1].datelist[-1]
                plt.plot([start_date,end_date],[funds,funds],color = colorlist[0], label = 'StartPosition')
                for i in range(len(ftlist)):
                    plt.plot(ftlist[i].datelist,np.array(ftlist[i].navlist),colorlist[i+1],\
                             label = ftlist[i].modtype)
                
                plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks = 10))
                plt.gcf().autofmt_xdate()
                plt.legend(loc = 'lower left')
                fig.savefig(location+'Diff.png')
                plt.close(fig)
            else:
                plotlist = np.array([ForexPrime.same_acc_navlist,ForexPrime.same_datelist])
                np.save(location+'Same_Plot',plotlist)
                plt.title('Assets same account')
                #legend.remove()
                plt.ylabel('Current Asset in USD')
                plt.xlabel('Time')
                #plt.gca().xaxis.set_major_formatter(myFmt)
                #plt.axhline(funds,color = colorlist[0],label = 'base')
                #x = np.linspace(0,plotlist.shape[0],plotlist.shape[0])
                plt.plot(plotlist[:,0],plotlist[:,1],color = colorlist[1],label = 'forexTrader')
                plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.gcf().autofmt_xdate()
                fig.savefig(location+'Same.png')
                plt.close(fig)

        count += 1
    print('###################################################################################')
    print('###################################################################################')
    print('End of Execution')
    print('###################################################################################')
    print('###################################################################################')

    #ftlist[0].delmodels()
    for ftrade in ftlist:
        ftrade.closeAll()
        time.sleep(1)
        ftrade.model_save()
    print('###################################################################################')
    print('###################################################################################')
    print('All open order closed')
    print('###################################################################################')
    print('###################################################################################')
    
    
    '''
    plotlist = []
    location = ftlist[0].current_path+'\\results\\'
    #global diff_acc
    #if diff_acc:
    #    fig.savefig(location+'Diff.png')
    #else:
    #    fig.savefig(location+'Same.png')
    
    
    if diff_acc:
        for ftrade in ftlist:
            plotlist.append(ftrade.navlist)
        plotlist = np.array(plotlist)
        print(plotlist)
        print(plotlist.shape)
        #plt.title('Assets diff account')
        x = np.linspace(0,plotlist.shape[1],plotlist.shape[1])
        for i in range(plotlist.shape[0]):
            plt.plot(x,plotlist[i,:],colorlist[i],label = ftlist[i].modtype)
        plt.legend(loc = 'lower left')
        fig.savefig(location+'Diff.png')
        #plt.show()    
    else:
        plotlist = np.array(ForexPrime.same_acc_navlist)
        #plt.title('Assets same account')
        x = np.linspace(0,plotlist.shape[0],plotlist.shape[0])
        plt.plot(x,plotlist,colorlist[0])
        fig.savefig(location+'Same.png')
        #plt.show()
   '''     


