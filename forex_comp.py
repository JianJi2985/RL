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
from gen_algo import genetic_algo
#import matplotlib.animation as animation
#from data_config import *





if __name__ == "__main__":
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))
    
    num_model = 2

    ftlist = []
    l_model = True
    diff_acc = True
    colorlist = ['black','firebrick','sandybrown','bisque','olivedrab',\
    'slategray','steelblue','mediumspringgreen','deepskyblue',\
     'indianred','yellowgreen','navy','mediumvioletred']
    epsilon = [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
    iteration = 5000
    l_rate = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
    q_freq = 1
    funds = 1000
    mem_space = 5
    time_count = [180,720,180]
    bal_dict = {}
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
    
    
    
    accountID = ['list of sub account ids']
    modtype = '1-1-2-2-1'
    inp_size = 3*signal_size + currentrate_size +mem_size + 5 +2 # for total value in usd
    out_size = moves_allowed
    model_param = {'mode':'sequential','layers':[{'Dense':[{'units':inp_size*2, 'activation':'tanh', 'input_dim':inp_size}]},\
                                                 {'Dropout':[{'rate':0.1}]},\
                                                 {'Dense':[{'units':inp_size, 'activation':'relu'}]},\
                                                 {'Dropout':[{'rate':0.1}]},\
                                                 {'Dense':[{'units':inp_size, 'activation':'relu'}]},\
                                                 {'Dropout':[{'rate':0.1}]},\
                                {'Dense':[{'units':out_size}]}],\
                      'compile':{'optimizer':{'RMSprop':{'lr':l_rate[2]}},'loss':'mse','metrics':['accuracy']}}
    def model_1_1_2_2_1(inp_size,out_size):
        dt = {}
        dt['si'] = Input(shape = (inp_size,),name = "state_input")
        dt['x'] = Dense((inp_size), activation = 'tanh')(dt['si'])
        dt['x'] = Dropout(rate = 0.1)(dt['x'])
        dt['x'] = Dense((inp_size), activation = 'relu')(dt['x'])
        dt['x'] = Dropout(rate = 0.1)(dt['x'])
        dt['x'] = Dense((inp_size), activation = 'relu')(dt['x'])
        dt['x'] = Dropout(rate = 0.1)(dt['x'])
        dt['out1'] = Dense((out_size), activation = 'softmax')(dt['x'])
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
        return modelx,modely
    if not l_model:
        modelx,modely = model_1_1_2_2_1(inp_size,out_size)
        
        for m in range(num_model):
        
            ft = ForexTrader(m,accountID[m],access_token,modtype,modpath = [], \
                             diff_acc = diff_acc,epsilon = epsilon[0], q_freq = q_freq)
            
            ft.actor = modelx
            ft.mid_actor = int(time.time())
            ft.critic = modely
            ft.mid_critic = int(time.time())
            ft.actor.summary()
            ft.critic.summary()
            print('model %s created'%modtype)
            ft.model_save()
            ftlist.append(ft)
            time.sleep(1)
            bal_dict[str(m)] = ft.trader.get_nav()['balance']
    else:
        
        for m in range(num_model):
            modpath = [currentpath+'/models/best_actor_'+modtype+'.h5',currentpath+'/models/best_critic_'+modtype+'.h5']
            ft = ForexTrader(m,accountID[m],access_token,modtype,modpath = modpath,diff_acc = diff_acc,epsilon = epsilon[2], q_freq = q_freq)
            ft.actor.summary()
            ft.critic.summary()
            print('model %s loaded'%modtype)
            ftlist.append(ft)
            time.sleep(1)
            bal_dict[str(m)] = ft.trader.get_nav()['balance']
    #global ftlist
    
    
    myFmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
    #if diff_acc:
    #    plt.title('Assets diff account')
    #else:
    #    plt.title('Assets same account')
    count = 1
    initial_time = int(time.time())
    gen_count = 0
    current_bal_dict = {}
    sum_bal_0 = 0
    sum_bal_1 = 0
    reward_bal = 0
    bal_list_0 = []
    bal_list_1 = []
    plist_0 = []
    plist_1 = []
    alist = []
    prev_time = datetime.datetime.now().minute
    while count <= iteration:
        #fig.clf()
        if datetime.date.today().weekday() == 4:
            test = datetime.datetime.now().replace(hour=16,minute=55,second=0,microsecond=0)
            if datetime.datetime.now() >= test:
                count = iteration + 1
                continue
        
        curr_time = datetime.datetime.now().minute
        
        if curr_time == prev_time:
            time.sleep(5)
            continue
        prev_time = curr_time
        
        print('\n###################################################################################')
        print('###################################################################################')
        print(int(time.time()))
        print(datetime.datetime.now())
        print('Pass %i of %i'%(count,iteration))
        print('Generation : %i'%(gen_count))
        print('###################################################################################')
        print('###################################################################################')
        for ftrade in ftlist:
            ftrade.runmodel()
            time.sleep(1)
        #ani = animation.FuncAnimation(fig, animate, frames=iteration)
        #if diff_acc:
        #    plt.legend(loc = 'upper left')
        #plt.show()
        if int(time.time()) - initial_time >= 360:
            
            score_list = []
            gen_count += 1
            
            ftlist[0].closeAll()
            ftlist[0].reward_punish = np.zeros((2))++ 0.01/funds
            ftlist[0].nav = ftlist[0].trader.get_nav()['nav']
            ftlist[0].lastnavs = [0,0,0,0,ftlist[0].funds]
            ftlist[0].fixed = ftlist[0].nav - ftlist[0].funds
            ftlist[0].nav = ftlist[0].funds

            for ftrade in ftlist:
                current_bal_dict[ftrade.id] = ftrade.trader.get_nav()['balance'] - bal_dict[ftrade.id]
                score_list.append([ftrade.id,current_bal_dict[ftrade.id]])

                ftrade.plot_data()
                bal_dict[ftrade.id] = ftrade.trader.get_nav()['balance']

            sum_bal_0 += score_list[0][1]
            bal_list_0.append(sum_bal_0)
            sum_bal_1 += score_list[1][1]
            bal_list_1.append(sum_bal_1)
            
            for ftrade in ftlist:
                ftrade.model_save()
                print('Model Saved')

                
                #ftrade.problist = []
                #ftrade.actlist = []
                
            location = currentpath +'/models/'+'timed_actor_'+modtype
            ftlist[0].actor.save(location+'.h5')
            plot_model(ftlist[0].actor, to_file=location+'.png', show_shapes = True, show_layer_names = False)
            
            location = currentpath +'/models/'+'timed_critic_'+modtype
            ftlist[0].critic.save(location+'.h5')
            plot_model(ftlist[0].critic, to_file=location+'.png', show_shapes = True, show_layer_names = False)
            
            location = currentpath +'/models/'+'open_actor_'+modtype
            ftlist[1].actor.save(location+'.h5')
            plot_model(ftlist[1].actor, to_file=location+'.png', show_shapes = True, show_layer_names = False)
            
            location = currentpath +'/models/'+'open_critic_'+modtype
            ftlist[1].critic.save(location+'.h5')
            plot_model(ftlist[1].critic, to_file=location+'.png', show_shapes = True, show_layer_names = False)
            

            fig = plt.figure(num = 2,figsize = (12,8))
            location = ftlist[0].current_path+'/results/'

            plt.ylabel('Model Balances')
            plt.xlabel('Transactions')
            np.save(location+'Comp_Plot_timed',bal_list_0)
            np.save(location+'Comp_Plot_open',bal_list_1)
            plt.title('Balance per six minutes')
            plt.plot(bal_list_0,label='timed')
            plt.plot(bal_list_1,label='open')
            plt.legend(loc='best')
            fig.savefig(location+'Comp_Plot.png')
            plt.close(fig)
            
            fig2 = plt.figure(num = 3,figsize = (12,8))
            location = ftlist[0].current_path+'/results/'

            plt.ylabel('Model Probablity')
            plt.xlabel('Transactions')
            np.save(location+'Gen_Prob_Plot_timed',ftlist[0].problist)
            np.save(location+'Gen_Prob_Plot_open',ftlist[1].problist)
            plt.title('Probability per transaction')
            plt.plot(ftlist[0].problist,label='timed')
            plt.plot(ftlist[1].problist,label='open')
            plt.legend(loc='best')
            fig2.savefig(location+'Gen_Prob_Plot.png')
            plt.close(fig2)
            
            print('###################################################################################')
            print('problist')
            print(ftlist[0].problist)
            print(ftlist[1].problist)
            print('###################################################################################')
            initial_time = int(time.time())
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


