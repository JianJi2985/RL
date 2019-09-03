import matplotlib
matplotlib.use('Agg')
from forex_trader_switch import ForexTraderSwitch
from forexTrader import ForexTrader
from forexprime import ForexPrime
import time
import keras
from keras.layers import Input, LSTM, Dense
from keras.models import load_model,Model
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from data_config import *




def animate(i):
    global fig
    fig.clf()
    global ax
    #ax.clf()
    global legend
    global ftlist
    for ftrade in ftlist:
        ftrade.runmodel()
        time.sleep(0.1)
    global diff_acc
    global colorlist
    plt.ylabel('Current Asset in USD')
    plt.xlabel('Steps')
    if diff_acc:
        plt.title('Assets Diff account')
        for d in range(len(ftlist)):
            x = np.linspace(0,len(ftlist[d].navlist),len(ftlist[d].navlist))
            plt.plot(x,ftlist[d].navlist[:],color = colorlist[d],label = ftlist[d].modtype)
        legend.remove()
        #global legend
        legend = plt.legend(loc = 'lower left')
        
    else:
        plt.title('Assets same account')
        x = np.linspace(0,len(ForexPrime.same_acc_navlist),len(ForexPrime.same_acc_navlist))
        plt.plot(x,ForexPrime.same_acc_navlist[:])
        legend.remove()
        #global legend
        legend = plt.legend(loc = 'lower left')


if __name__ == "__main__":
    
    global ftlist
    global diff_acc
    global colorlist
    global fig
    global ax
    iteration = 300
    epsilon = 0.2
    q_freq = 1
    
    accountID = 'account id here'
    access_token = 'access token here'
    curr_pair_list = ['EUR_GBP','EUR_JPY','EUR_USD','GBP_JPY','GBP_USD','USD_JPY']
    units = 10000
    fts = ForexTraderSwitch()
    fts.initial_value( accountID = accountID,access_token = access_token,\
                      curr_pair_list = curr_pair_list,units = units)
    
    order_size = ForexTraderSwitch.order.size
    signal_size = ForexTraderSwitch.signal.size
    num_curr = ForexTraderSwitch.num_curr
    currentpos_size = num_curr*3
    currentrate_size = ForexTraderSwitch.current_rate.size
    time_count = ForexTraderSwitch.time_count
    moves_allowed = ForexTraderSwitch.moves_allowed.size
    print(order_size,signal_size,num_curr,currentpos_size,currentrate_size,time_count,moves_allowed)
    
    #ftlist = []
    
    
    accountID = 'account id here'
    modtype = '1-1-1-1-1'
    inp_size = order_size + currentpos_size + 1 # for total value in usd
    out_size = moves_allowed
    model_param = {'mode':'sequential','layers':[{'Dense':[{'units':inp_size*2, 'activation':'relu', 'input_dim':inp_size}]},\
                                {'Dense':[{'units':out_size, 'activation':'softmax'}]}],\
                      'compile':{'optimizer':'rmsprop','loss':'categorical_crossentropy','metrics':['accuracy']}}

    ft = ForexTrader(accountID,access_token,modtype,modpath = [], model_param = model_param, diff_acc = diff_acc, q_freq = q_freq)
    #global ftlist
    ftlist.append(ft)
    
    ft.model_create()
    #ft.model.summary()
    print('model %s created'%modtype)
    ft.model_save()
    time.sleep(1)
    
    accountID = 'account id here'
    modtype = '2-1-1-1-1'
    inp_size = order_size + currentpos_size + 1 # for total value in usd
    out_size = moves_allowed - num_curr
    model_param = {'mode':'sequential','layers':[{'Dense':[{'units':inp_size*2, 'activation':'relu', 'input_dim':inp_size}]},\
                                {'Dense':[{'units':out_size, 'activation':'softmax'}]}],\
                      'compile':{'optimizer':'rmsprop','loss':'categorical_crossentropy','metrics':['accuracy']}}

    ft = ForexTrader(accountID,access_token,modtype,modpath = [], model_param = model_param, diff_acc = diff_acc, q_freq = q_freq)
    #global ftlist
    ftlist.append(ft)
    
    ft.model_create()
    #ft.model.summary()
    ft.modeltwin_create()
    #ft.model_twin.summary()
    print('model %s created'%modtype)
    ft.model_save()
    time.sleep(1)
    
    accountID = 'account id here'
    modtype = '1-1-2-1-1'
    inp_size = signal_size + currentrate_size +currentpos_size + 1 # for total value in usd
    out_size = moves_allowed
    model_param = {'mode':'sequential','layers':[{'Dense':[{'units':inp_size*2, 'activation':'relu', 'input_dim':inp_size}]},\
                                {'Dense':[{'units':out_size, 'activation':'softmax'}]}],\
                      'compile':{'optimizer':'rmsprop','loss':'categorical_crossentropy','metrics':['accuracy']}}

    ft = ForexTrader(accountID,access_token,modtype,modpath = [], model_param = model_param, diff_acc = diff_acc, q_freq = q_freq)
    #global ftlist
    ftlist.append(ft)
    
    ft.model_create()
    print('model %s created'%modtype)
    ft.model_save()
    time.sleep(1)
    
    accountID = 'account id here'
    modtype = '2-1-2-1-1'
    inp_size = signal_size + currentrate_size +currentpos_size + 1 # for total value in usd
    out_size = moves_allowed - num_curr
    model_param = {'mode':'sequential','layers':[{'Dense':[{'units':inp_size*2, 'activation':'relu', 'input_dim':inp_size}]},\
                                {'Dense':[{'units':out_size, 'activation':'softmax'}]}],\
                      'compile':{'optimizer':'rmsprop','loss':'categorical_crossentropy','metrics':['accuracy']}}

    ft = ForexTrader(accountID,access_token,modtype,modpath = [], model_param = model_param, diff_acc = diff_acc, q_freq = q_freq)
    #global ftlist
    ftlist.append(ft)
    
    ft.model_create()
    ft.modeltwin_create()
    print('model %s created'%modtype)
    ft.model_save()
    time.sleep(1)
    
    accountID = 'account id here'
    modtype = '1-1-2-2-2'
    inp_size = currentrate_size +currentpos_size + 1 # for total value in usd
    out_size = moves_allowed
    model_param = {'mode':'sequential','layers':[{'Dense':[{'units':inp_size*2, 'activation':'relu', 'input_dim':inp_size}]},\
                                {'Dense':[{'units':out_size, 'activation':'softmax'}]}],\
                      'compile':{'optimizer':'rmsprop','loss':'categorical_crossentropy','metrics':['accuracy']}}

    ft = ForexTrader(accountID,access_token,modtype,modpath = [], model_param = model_param, diff_acc = diff_acc, q_freq = q_freq)
    #global ftlist
    ftlist.append(ft)
    
    ft.model_create()
    print('model %s created'%modtype)
    ft.model_save()
    time.sleep(1)
    
    accountID = 'account id here'
    modtype = '2-1-2-2-2'
    inp_size = currentrate_size +currentpos_size + 1 # for total value in usd
    out_size = moves_allowed - num_curr
    model_param = {'mode':'sequential','layers':[{'Dense':[{'units':inp_size*2, 'activation':'relu', 'input_dim':inp_size}]},\
                                {'Dense':[{'units':out_size, 'activation':'softmax'}]}],\
                      'compile':{'optimizer':'rmsprop','loss':'categorical_crossentropy','metrics':['accuracy']}}

    ft = ForexTrader(accountID,access_token,modtype,modpath = [], model_param = model_param, diff_acc = diff_acc, q_freq = q_freq)
    #global ftlist
    ftlist.append(ft)
    
    ft.model_create()
    ft.modeltwin_create()
    print('model %s created'%modtype)
    ft.model_save()
    time.sleep(1)
    
    out_size = moves_allowed
    dt = {}
    dt['fi'] = Input(shape = (time_count,num_curr), name = "forex_input")
    dt['lo'] = LSTM(time_count*num_curr)(dt['fi'])
    dt['si'] = Input(shape = (currentpos_size+1,),name = "state_input")
    dt['x'] = keras.layers.concatenate([dt['lo'], dt['si']])
    dt['x'] = Dense((time_count*num_curr//2), activation = 'relu')(dt['x'])
    dt['out'] = Dense(out_size, activation = 'softmax')(dt['x'])
    modelx = Model(inputs =[dt['fi'],dt['si']],outputs = dt['out'])
    modelx.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    #modelx.summary()
    accountID = 'account id here'
    modtype = '1-2-3-2-2'
    ft = ForexTrader(accountID,access_token,modtype, diff_acc = diff_acc, q_freq = q_freq)
    #global ftlist
    ftlist.append(ft)
    ft.model = modelx
    ft.mid = int(time.time())
    print('model %s created'%modtype)    
    ft.model_save()
    time.sleep(1)
    
    out_size = moves_allowed - num_curr
    dt = {}
    dt['fi'] = Input(shape = (time_count,num_curr), name = "forex_input")
    dt['lo'] = LSTM(time_count*num_curr)(dt['fi'])
    dt['si'] = Input(shape = (currentpos_size+1,),name = "state_input")
    dt['x'] = keras.layers.concatenate([dt['lo'], dt['si']])
    dt['x'] = Dense((time_count*num_curr//2), activation = 'relu')(dt['x'])
    dt['out'] = Dense(out_size, activation = 'softmax')(dt['x'])
    modelx = Model(inputs =[dt['fi'],dt['si']],outputs = dt['out'])
    modelx.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    #modelx.summary()
    accountID = 'account id here'
    modtype = '2-2-3-2-2'
    ft = ForexTrader(accountID,access_token,modtype, diff_acc = diff_acc, q_freq = q_freq)
    #global ftlist
    ftlist.append(ft)
    ft.model = modelx
    ft.mid = int(time.time())
    ft.model_twin = modelx
    ft.midtwin = int(time.time())
    #ft.model.summary()
    #ft.model_twin.summary()
    print('model %s created'%modtype)
    ft.model_save()
    time.sleep(1)
    
    for ftrade in ftlist:
        ftrade.closeAll()
        time.sleep(0.5)
    
    
    plt.ylabel('Current Asset in USD')
    plt.xlabel('Steps')
    
    count = 1
    try:
        while count <= iteration:
            #fig.clf()
            print('###################################################################################')
            print('###################################################################################')
            print('Pass %i of %i'%(count,iteration))
            print('###################################################################################')
            print('###################################################################################')
            ani = animation.FuncAnimation(fig, animate, frames=iteration)
            #if diff_acc:
            #    plt.legend(loc = 'upper left')
            plt.show()
            if count%10 == 0:
                #ftlist[0].delmodels()
                for ftrade in ftlist:
                    ftrade.model_save()
                    print('Model Saved')
            
            count += 1
        print('###################################################################################')
        print('###################################################################################')
        print('End of Execution')
        print('###################################################################################')
        print('###################################################################################')
        
        #ftlist[0].delmodels()
        for ftrade in ftlist:
            ftrade.closeAll()
            time.sleep(0.5)
            ftrade.model_save()
        print('###################################################################################')
        print('###################################################################################')
        print('All open order closed')
        print('###################################################################################')
        print('###################################################################################')
    except:
        pass
    
    
    #plotlist = []
    location = ftlist[0].current_path+'\\results\\'
    #global diff_acc
    if diff_acc:
        fig.savefig(location+'Diff.png')
    else:
        fig.savefig(location+'Same.png')
    '''
    if diff_acc:
        for ftrade in ftlist:
            plotlist.append(ftrade.navlist)
        plotlist = np.array(plotlist)
        plt.title('Assets diff account')
        x = np.linspace(0,plotlist.shape[1],plotlist.shape[1])
        for i in range(plotlist.shape[0]):
            plt.plot(x,plotlist[i,:],colorlist[i],label = ftlist[i].modtype)
        plt.legend(loc = 'upper left')
        fig.savefig(location+'Diff.png')
        plt.show()    
    else:
        plotlist = np.array(ForexPrime.same_acc_navlist)
        plt.title('Assets same account')
        x = np.linspace(0,plotlist.shape[0],plotlist.shape[0])
        plt.plot(x,plotlist,colorlist[0])
        fig.savefig(location+'Same.png')
        plt.show()
        
    '''

