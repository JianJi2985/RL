import keras
from keras.models import Sequential, Model, load_model
import keras.backend as K
import json
import pickle
import os.path
import numpy as np
import random
import datetime
from forexprime import *
import matplotlib.dates as mdates


class ForexTrader(ForexPrime):
    def __init__(self,idval,accountID,access_token,modtype,modpath = [], actor_param = None,critic_param = None,funds = 1000 ,\
                 diff_acc = False,epsilon = 0.2, discount = 0.9, q_freq = 5, random_sampling = True):
        super(ForexTrader,self).__init__(idval,accountID,access_token,modtype,funds)
        self.actor = None
        self.critic = None
        self.mid_actor = None
        self.mid_critic = None
        
        
        self.asset_swing_val = 0
        self.sharpe = 0
        self.over_score = 0
        self.layerlist = []
        self.complexity = ['Low','Low']
        self.complex_num = [1,1]
        self.epsilon = epsilon
        self.lrate = 0.01
        self.unit_list = [100,1000,10000,100000]
        self.epsilon_list = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
        self.lrate_list = [0.1,0.01,0.001,0.0001,0.00001]
        
        
        self.reward_punish = np.zeros((2)) + 0.01/self.funds
        self.success_fail = np.zeros((2)) + 1
        self.unit_desc = np.zeros((len(self.unit_list))) + 1
        self.random_desc = np.zeros((2)) + 1
        self.lrate_desc = np.zeros((len(self.lrate_list))) + 1
        
        self.unit_in = np.zeros((len(self.unit_list)))
        self.epsilon_in = np.zeros((len(self.epsilon_list)))
        self.lrate_in = np.zeros((len(self.lrate_list)))
        self.unit_in[0] = 1
        self.epsilon_in[0] = 1
        self.lrate_in[0] = 1
        
        self.sel_action = [0,0,self.epsilon_list.index(self.epsilon),self.lrate_list.index(self.lrate)]
        json_data = {}
        
        self.moves = np.arange(self.moves_allowed.size)
        self.move_counter = np.zeros((self.moves_allowed.size)) + 1
        
        self.modpath = modpath
        if not self.modpath:
            pass
        else:
            self.model_load()
        
        self.actor_param = actor_param
        self.critic_param = critic_param
        self.diff_acc = diff_acc
        self.q_freq = q_freq
        
        
        #self.funds = funds
        
        self.discount = discount
        self.random = random_sampling
        
        
        self.time_info = np.zeros((2))
        self.problist = []
        self.actlist = []
        #self.input = None
        #self.output = None
    
    def complexity_cal(self):
        self.complexity = ['Low','Low']
        self.complex_num = [1,1]
        mult_fac = 1
        if self.modtype.split('-')[0] == '2':
            mult_fac = 2
        
        if 'LSTM' in self.layerlist:
            self.complexity[0] = 'High'
            self.complex_num[0] = 2
        
               
        self.complex_num[1] = min(int(np.log(self.actor.count_params()*mult_fac/100)),10)
        
        if len(self.layerlist) <= 5:
            pass
        
        elif len(self.layerlist) > 5 and len(self.layerlist) < 10:
            self.complexity[1] = 'Med'
            
        elif len(self.layerlist) >= 10:
            self.complexity[1] = 'High'
        
        #print(self.complexity,self.complex_num)
    
    def asset_swing(self):
        self.asset_swing_val = int(np.var((np.array(self.navlist)/100).astype(float)))
    
    def sharpe_ratio(self):
        sumval = self.success_fail[0] + self.success_fail[1] - 2
        if int(sumval) <= 1:
            self.sharpe = 0
        else:
            daily_return = ((self.success_fail[0]-1)*2 + self.navlist[-1]/self.funds)/(sumval+1) - 1
            
            std_cal = np.zeros((int(sumval)+1))
            std_cal[0:int(self.success_fail[0])-1] = 1
            std_cal[-1] = self.navlist[-1]/self.funds - 1
            std = np.std(std_cal)
            try:
                #self.sharpe = max(round((((daily_return + 1)**365 - 1) - 0.05)/std,2),0)
                self.sharpe = max(round((daily_return - 0.05)/std,1),0)
            except:
                self.sharpe = 0
    
    def overall_score(self):
        sumval = self.success_fail[0] + self.success_fail[1] - 2
        if sumval == 0:
            success_fail = 0
        else:
            success_fail = (self.success_fail[0]-1)*100/(self.success_fail[0] + self.success_fail[1] - 2)
            
        self.over_score =min(int( 0.3*self.sharpe*25 + \
        0.2*(success_fail) + \
        0.2*(100 - self.asset_swing_val) + \
        0.2*(self.success_fail[0]*100/np.sum(self.success_fail)) + \
        0.1*(14 - np.sum(np.array(self.complex_num)))*100/14),99)
        
    def plot_data(self):
        self.asset_swing()
        self.sharpe_ratio()
        self.overall_score()
        #print(self.complexity,self.complex_num)
        self.write_trade_history()
        
                    
        fig = plt.figure(1,figsize = (16,8))
        #fig.clf()
        
        #Numlist = 7
        Alist = 4
        xl = np.arange(self.num_curr+1)
        yl = np.arange(Alist)
        #xlbl = ['None','EURGBP','EURJPY','EURUSD','GBPJPY','GBPUSD','USDJPY']
        xlbl = [item.replace('_','') for item in self.curr_pair_list]
        xlbl.insert(0,'None')
        ylbl = [' Buy ',' Sell ',' Close1 ',' CloseAll ']
        y = np.zeros((self.num_curr+1,Alist))
        
        y[0:] = (self.move_counter[0] - 1)/4
        y[1:self.num_curr+1,:] = self.move_counter[1:].reshape((self.num_curr,Alist)) - 1
        
            
        
        width = 0.4
        #fig = plt.figure(1,figsize = (20,10))
        fig.patch.set_facecolor('whitesmoke')
        plt.suptitle('Model '+self.modtype+' Perforfmance\n')
        ##############################################################
        plt.subplot(3,5,1)
        plt.title('Overall Score')
        colors = ['steelblue','mintcream']
        #print(self.over_score)
        plt.pie([self.over_score,100-self.over_score], radius=1, labels=['',''], colors=colors,\
                startangle = 0,wedgeprops = dict(linewidth = 1,width=width,edgecolor='mintcream'))
        plt.axis('equal')
        try:
            textvalue = self.over_score
        except:
            textvalue = 0
        tcolor = colors[0]
        if textvalue > 99:
            textvalue = 99
            
        elif textvalue < 50:
            tcolor = 'indianred'
        
        plt.text(x=-0.5,y=-0.3,s=str(textvalue).rjust(2,'0'),color = tcolor,weight = 600,size = 40)
        ##############################################################
        plt.subplot(3,5,2)
        plt.title('Sharpe ratio')
        colors = ['limegreen','mintcream']
        
        plt.pie([min(self.sharpe,3),3-min(self.sharpe,3)], radius=1, labels=['',''], colors=colors,\
                startangle = 0,wedgeprops = dict(linewidth = 1,width=width,edgecolor='mintcream'))
        plt.axis('equal')
        try:
            textvalue = round(self.sharpe,1)
        except:
            textvalue = 0
        tcolor = colors[0]
        if textvalue > 3:
            textvalue = '>3.0'
            
        elif textvalue < 1:
            tcolor = 'indianred'
        
        plt.text(x=-0.55,y=-0.15,s=str(textvalue).rjust(3,'0'),color = tcolor,weight = 600,size = 24)
        ##############################################################
        plt.subplot(3,5,3)
        plt.title('Design Complexity')
        colors = ['darkkhaki','mintcream']
        
        plt.pie([self.complex_num[1], 10-self.complex_num[1]], radius=1, labels=['',''], colors=colors,\
                startangle = 0,wedgeprops = dict(linewidth = 1,width=width,edgecolor='mintcream'))
        plt.axis('equal')
        try:
            textvalue = self.complexity[1]
        except:
            textvalue = 'NA'
        tcolor = colors[0]
        
        if textvalue == 'Low':
            tcolor = 'indianred'
        
        plt.text(x=-0.6,y=-0.15,s=textvalue,color = tcolor,weight = 600,size = 26)
        ##############################################################
        plt.subplot(3,5,6)
        plt.title('Success Rate')
        colors = ['yellowgreen','mintcream']
        
        plt.pie(self.success_fail, radius=1, labels=['',''], colors=colors,\
                startangle = 0,wedgeprops = dict(linewidth = 1,width=width,edgecolor='mintcream'))
        plt.axis('equal')
        try:
            textvalue = int((self.success_fail[0]/np.sum(self.success_fail))*100)
        except:
            textvalue = 0
        tcolor = colors[0]
        if textvalue > 99:
            textvalue = 99
            
        elif textvalue < 50:
            tcolor = 'indianred'
        
        plt.text(x=-0.55,y=-0.15,s=(str(textvalue)+'%').rjust(3,'0'),color = tcolor,weight = 600,size = 26)
        ##############################################################
        plt.subplot(3,5,7)
        plt.title('Asset Swing')
        colors = ['crimson','mintcream']
        
        plt.pie([self.asset_swing_val,100-self.asset_swing_val], radius=1, labels=['',''], colors=colors,\
                startangle = 0,wedgeprops = dict(linewidth = 1,width=width,edgecolor='mintcream'))
        plt.axis('equal')
        try:
            textvalue = self.asset_swing_val
        except:
            textvalue = 0
        tcolor = colors[0]
        if textvalue > 99:
            textvalue = 99
            
        elif textvalue < 50:
            tcolor = 'indianred'
        
        plt.text(x=-0.55,y=-0.15,s=(str(textvalue)+'%').rjust(3,'0'),color = tcolor,weight = 600,size = 26)
        ##############################################################
        plt.subplot(3,5,8)
        plt.title('Exploration')
        colors = ['slategray','mintcream']
        
        plt.pie(self.random_desc, radius=1, labels=['',''], colors=colors,\
                startangle = 0,wedgeprops = dict(linewidth = 1,width=width,edgecolor='mintcream'))
        plt.axis('equal')
        try:
            textvalue = int(100*self.random_desc[0]/(np.sum(self.random_desc)))
        except:
            textvalue = 0
        tcolor = colors[0]
        
        if textvalue > 99:
            textvalue = 99
            
        elif textvalue < 50:
            tcolor = 'indianred'
        
        plt.text(x=-0.55,y=-0.15,s=(str(textvalue)+'%').rjust(3,'0'),color = tcolor,weight = 600,size = 26)
        
        ##############################################################
        plt.subplot(3,5,11)
        plt.title('Reward Rate')
        colors = ['lightseagreen','mintcream']
        
        plt.pie(self.reward_punish*100/np.sum(self.reward_punish), radius=1, labels=['',''], colors=colors,\
                startangle = 0,wedgeprops = dict(linewidth = 1,width=width,edgecolor='mintcream'))
        plt.axis('equal')
        try:
            textvalue = int((self.reward_punish[0]/np.sum(self.reward_punish))*100)
        except:
            textvalue = 0
        tcolor = colors[0]
        if textvalue > 99:
            textvalue = 99
            
        elif textvalue < 50:
            tcolor = 'indianred'
        
        plt.text(x=-0.55,y=-0.15,s=(str(textvalue)+'%').rjust(3,'0'),color = tcolor,weight = 600,size = 26)
                
        ##############################################################
        plt.subplot(3,5,12)
        plt.title('Units')
        colors = ['slategray','sandybrown','lightgreen','skyblue']
        
        plt.pie(self.unit_desc, radius=1, labels=['100','1000','10000','100000'], autopct='%1.0f%%',pctdistance=0.8,colors=colors,\
                startangle = 0,wedgeprops = dict(linewidth = 1,width=width,edgecolor='mintcream'))
        plt.axis('equal')
        ##############################################################
        plt.subplot(3,5,13)
        plt.title('Learning rate')
        colors = ['slategray','sandybrown','lightgreen','skyblue','steelblue']
        
        plt.pie(self.lrate_desc, radius=1, labels=['1e-1','1e-2','1e-3','1e-4','1e-5'], autopct='%1.0f%%',pctdistance=0.8,colors=colors,\
                startangle = 0,wedgeprops = dict(linewidth = 1,width=width,edgecolor='mintcream'))
        plt.axis('equal')
        ##############################################################
        plt.subplot(3,5,(4,10),facecolor = 'whitesmoke')
        plt.bar(xl, y[:,0],color='sandybrown',edgecolor='whitesmoke' ,width = width,label = 'Buy')
        plt.bar(xl, y[:,1],color='lightgreen',edgecolor='whitesmoke' , width = width,bottom=y[:,0],label = 'Sell')
        plt.bar(xl, y[:,2],color='skyblue',edgecolor='whitesmoke' , width = width,bottom=y[:,1]+y[:,0],label = 'Close 1')
        plt.bar(xl, y[:,3],color='steelblue',edgecolor='whitesmoke' , width = width,bottom=y[:,2]+y[:,1]+y[:,0],label = 'CloseAll')

        plt.ylim(0,np.max(np.sum(y,axis = 1))*1.5) 
        plt.xticks(xl,xlbl)
        plt.legend(ncol=4,loc = 'best')#,bbox_to_anchor=(0.5,0.05 ))
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        #plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.5,wspace=0.5)
        fig.savefig(self.current_path+'/results/'+self.id+'_'+self.modtype+'_Report.png')
        #plt.show()
        plt.close(fig)
        ##############################################################
        ##############################################################
        fig1 = plt.figure(5,figsize = (16,9))
        #fig1.clf()
        #xl = [0,1,2,3,4,5]
        xl = [i for i in range(self.num_curr)]
        yl = [0,1,2,3,4,5,6,7,8,9]
        #xlbl = ['EURGBP','EURJPY','EURUSD','GBPJPY','GBPUSD','USDJPY']
        xlbl = [item.replace('_','') for item in self.curr_pair_list]
        ylbl = ['1 min','2 min','4 min','8 min','16 min','32 min','1 hr 4 min','2 hr 8 min','4 hr 16 min','8 hr 32 min']
        #plt.figure(2,figsize = (16,9))
        plt.subplot(2,5,1)
        plt.matshow(self.analytic[:,:,0],fignum = False)
        plt.xticks(xl,xlbl,rotation = 'vertical')
        plt.yticks(yl,ylbl)
        plt.title('\nTotal Long volume\n\n\n')
        plt.colorbar(fraction=0.08, pad=0.04)
        plt.subplot(2,5,2)
        plt.matshow(self.analytic[:,:,2],fignum = False)
        plt.xticks(xl,xlbl,rotation = 'vertical')
        plt.yticks(yl,ylbl)
        plt.title('\nTotal Long profits\n\n\n')
        plt.colorbar(fraction=0.08, pad=0.04)
        plt.subplot(2,5,3)
        plt.matshow(np.divide(self.analytic[:,:,0],self.analytic[:,:,4]),fignum = False)
        plt.xticks(xl,xlbl,rotation = 'vertical')
        plt.yticks(yl,ylbl)
        plt.title('\nAvg Long volume\n\n\n')
        plt.colorbar(fraction=0.08, pad=0.04)
        plt.subplot(2,5,4)
        plt.matshow(np.divide(self.analytic[:,:,2],self.analytic[:,:,4]),fignum = False)
        plt.xticks(xl,xlbl,rotation = 'vertical')
        plt.yticks(yl,ylbl)
        plt.title('\nAvg Long profits\n\n\n')
        plt.colorbar(fraction=0.08, pad=0.04)
        plt.subplot(2,5,5)
        plt.matshow(self.analytic[:,:,4],fignum = False)
        plt.xticks(xl,xlbl,rotation = 'vertical')
        plt.yticks(yl,ylbl)
        plt.title('\nTotal Long transactions\n\n\n')
        plt.colorbar(fraction=0.08, pad=0.04)
        plt.subplot(2,5,6)
        plt.matshow(self.analytic[:,:,1],fignum = False)
        plt.xticks(xl,xlbl,rotation = 'vertical')
        plt.yticks(yl,ylbl)
        plt.title('\nTotal Short volume\n\n\n')
        plt.colorbar(fraction=0.08, pad=0.04)
        plt.subplot(2,5,7)
        plt.matshow(self.analytic[:,:,3],fignum = False)
        plt.xticks(xl,xlbl,rotation = 'vertical')
        plt.yticks(yl,ylbl)
        plt.title('\nTotal Short profits\n\n\n')
        plt.colorbar(fraction=0.08, pad=0.04)
        plt.subplot(2,5,8)
        plt.matshow(np.divide(self.analytic[:,:,1],self.analytic[:,:,5]),fignum = False)
        plt.xticks(xl,xlbl,rotation = 'vertical')
        plt.yticks(yl,ylbl)
        plt.title('\nAvg Short volume\n\n\n')
        plt.colorbar(fraction=0.08, pad=0.04)
        plt.subplot(2,5,9)
        plt.matshow(np.divide(self.analytic[:,:,3],self.analytic[:,:,5]),fignum = False)
        plt.xticks(xl,xlbl,rotation = 'vertical')
        plt.yticks(yl,ylbl)
        plt.title('\nAvg Short profits\n\n\n')
        plt.colorbar(fraction=0.08, pad=0.04)
        plt.subplot(2,5,10)
        plt.matshow(self.analytic[:,:,5],fignum = False)
        plt.xticks(xl,xlbl,rotation = 'vertical')
        plt.yticks(yl,ylbl)
        plt.title('\nTotal Short transactions\n\n\n')
        plt.colorbar(fraction=0.08, pad=0.04)
        plt.suptitle('Model '+self.modtype+' Analytics')
        #plt.subplots_adjust(top=0.9, bottom=0.05, left=0.1, right=0.9, hspace=0.01,wspace=0.5)
        plt.tight_layout()
        
        fig1.savefig(self.current_path+'/results/'+self.id+'_'+self.modtype+'_Analytics.png')
        plt.close(fig1)
        ##############################################################
        ##############################################################
        #fig2 = plt.figure(6,figsize = (9,9))
        #myFmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
        #plt.ylabel('Current Level')
        #plt.xlabel('Time')
        #plt.gca().xaxis.set_major_formatter(myFmt)
        #plt.plot(self.datelist,np.array(self.levellist))
        #plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks = 10))
        #plt.gcf().autofmt_xdate()
        #plt.title('Model '+ self.modtype +' Level Details')
        #fig2.savefig(self.current_path+'/results/'+self.modtype+'_LevelInfo.png')
        #plt.close(fig2)
        ##############################################################
        ##############################################################
        #fig3 = plt.figure(5,figsize = (9,9))
        #myFmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
        #plt.ylabel('Decision Probability')
        #plt.xlabel('Time')
        #plt.gca().xaxis.set_major_formatter(myFmt)
        #plt.plot(self.datelist,np.array(self.levellist))
        #plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks = 10))
        #plt.gcf().autofmt_xdate()
        #plt.title('Model '+ self.modtype +' Level Details')
        #fig2.savefig(self.current_path+'/results/'+self.modtype+'_LevelInfo.png')
        #plt.close(fig2)
        
    def model_create(self):
        self.mlmod.create_model(self.model_param)
        self.model = self.mlmod.model
        self.mid = self.mlmod.mid
        self.layerlist = []
        for layer in self.model.layers:
            self.layerlist.append(str(layer).split(' ')[0].split('.')[-1])
        self.complexity_cal()
        
    
 
    def model_save(self):
        location = self.current_path +'/models/'+'actor_'+self.id+'_'+self.modtype
        self.actor.save(location+'.h5')
        plot_model(self.actor, to_file=location+'.png', show_shapes = True, show_layer_names = False)
        
        location = self.current_path +'/models/'+'critic_'+self.id+'_'+self.modtype
        self.critic.save(location+'.h5')
        plot_model(self.critic, to_file=location+'.png', show_shapes = True, show_layer_names = False)
        
        json_data = {'over_score':self.over_score,\
        'sharpe':self.sharpe,\
        'asset_swing_val':self.asset_swing_val,\
        'reward_punish':self.reward_punish.tolist(),\
        'success_fail':self.success_fail.tolist(),\
        'move_counter':self.move_counter.tolist(),\
        'exploration':self.random_desc.tolist(),
        'lrate':self.lrate_desc.tolist(),
        'unit':self.unit_desc.tolist()}
               
        with open(self.current_path +'/models/'+self.id+'_'+self.modtype+'_actor_envdata.txt', "wb") as fp:
            pickle.dump(self.actor_envlist, fp)
        with open(self.current_path +'/models/'+self.id+'_'+self.modtype+'_critic_envdata.txt', "wb") as fp:
            pickle.dump(self.critic_envlist, fp)
           
               
        with open(self.current_path +'/models/'+self.id+'_'+self.modtype+'_rdata.txt', 'w') as outfile:
            json.dump(json_data, outfile)  
            
    def model_load(self):
        
        self.mid_actor = int(time.time())
        self.actor = load_model(self.modpath[0])
        self.mid_critic = int(time.time())
        self.critic = load_model(self.modpath[1])
        self.lrate = K.get_value(self.actor.optimizer.lr)
        #print(self.lrate_list[1].type)
        self.sel_action[3] = self.lrate_list.index(float(str(self.lrate)))
        
        #self.sel_action[3] = 1
        self.layerlist = []
        for layer in self.actor.layers:
            self.layerlist.append(str(layer).split(' ')[0].split('.')[-1])
        self.complexity_cal()
        try:
            if os.path.isfile(self.current_path +'/models/'+self.id+'_'+self.modtype+'_rdata.txt'):
                with open(self.current_path +'/models/'+self.id+'_'+self.modtype+'_rdata.txt') as infile:
                    jdata = json.load(infile)
                self.over_score = jdata['over_score']
                self.sharpe = jdata['sharpe']
                self.asset_swing_val = jdata['asset_swing_val']
                self.reward_punish = np.array(jdata['reward_punish'])
                self.success_fail = np.array(jdata['success_fail']).astype(int)
                self.move_counter = np.array(jdata['move_counter']).astype(int)
                self.random_desc = np.array(jdata['exploration']).astype(int)
                self.lrate_desc = np.array(jdata['lrate']).astype(int)
                self.unit_desc = np.array(jdata['unit']).astype(int)
        except:
            pass
        if self.modtype.split('-')[0] == 1:
            self.lrate_desc[self.sel_action[3]] = 96
        try:
            if os.path.isfile(self.current_path +'/models/'+self.id+'_'+self.modtype+'_actor_envdata.txt'):
                with open(self.current_path +'/models/'+self.id+'_'+self.modtype+'_actor_envdata.txt', "wb") as fp:
                    self.actor_envlist = pickle.load(fp)
            if os.path.isfile(self.current_path +'/models/'+self.id+'_'+self.modtype+'_critic_envdata.txt'):
                with open(self.current_path +'/models/'+self.id+'_'+self.modtype+'_critic_envdata.txt', "wb") as fp:
                    self.critic_envlist = pickle.load(fp)
        except:
            pass
        #print(self.complexity,self.complex_num)
        #print(self.modpath[0])
        #print('model')
        
        #print(self.complexity,self.complex_num)
    def input_create(self,new_state,actor = True):
        #print('-',end=' ')
        print('###################')
        print('Create Input')
        print(datetime.datetime.now())
        curr_time = datetime.datetime.now()
        self.time_info[0] = curr_time.hour
        self.time_info[1] = curr_time.minute
        state = new_state
        
        if self.modtype == '1-1-1-1-1':
            if actor:
                new_state = np.concatenate((self.time_info.reshape((1,-1)),self.orderdaily.reshape((1,-1)),self.ordermonthly.reshape((1,-1)),self.ordersemiannually.reshape((1,-1)),\
                (self.current_pos[:,:-1]/(self.transact_memsize*max(self.unit_list))).reshape((1,-1)),\
                                             (np.array(self.lastnavs)/self.funds).reshape((1,-1))),axis = 1)
            else:
                new_state = np.concatenate((self.time_info.reshape((1,-1)),self.orderdaily.reshape((1,-1)),self.ordermonthly.reshape((1,-1)),self.ordersemiannually.reshape((1,-1)),\
                (self.current_pos[:,:-1]/(self.transact_memsize*max(self.unit_list))).reshape((1,-1)),\
                                             (np.array(self.lastnavs)/self.funds).reshape((1,-1)),\
                                             self.unit_in.reshape((1,-1))),axis = 1)
            
            #print(self.new_state)
        elif self.modtype == '2-1-1-1-1':
            if actor:
                new_state = np.concatenate((self.time_info.reshape((1,-1)),self.orderdaily.reshape((1,-1)),self.ordermonthly.reshape((1,-1)),self.ordersemiannually.reshape((1,-1)),\
                (self.current_pos[:,:-1]/(self.transact_memsize*max(self.unit_list))).reshape((1,-1)),\
                                             (np.array(self.lastnavs)/self.funds).reshape((1,-1))),axis = 1)
            else:
                new_state = np.concatenate((self.time_info.reshape((1,-1)),self.orderdaily.reshape((1,-1)),self.ordermonthly.reshape((1,-1)),self.ordersemiannually.reshape((1,-1)),\
                (self.current_pos[:,:-1]/(self.transact_memsize*max(self.unit_list))).reshape((1,-1)),\
                                             (np.array(self.lastnavs)/self.funds).reshape((1,-1)),\
                                             self.unit_in.reshape((1,-1)),self.epsilon_in.reshape((1,-1)),\
                                             self.lrate_in.reshape((1,-1))),axis = 1)
                                             
        elif self.modtype == '1-1-2-1-1':
            if actor:
                new_state = np.concatenate((self.time_info.reshape((1,-1)),self.signaldaily.reshape((1,-1)),self.signalmonthly.reshape((1,-1)),self.signalsemiannually.reshape((1,-1)),\
                self.current_rate.reshape((1,-1)),(self.current_pos[:,:-1]/(self.transact_memsize*max(self.unit_list))).reshape((1,-1)), \
                                             (np.array(self.lastnavs)/self.funds).reshape((1,-1))),axis = 1)
            else:
                new_state = np.concatenate((self.time_info.reshape((1,-1)),self.signaldaily.reshape((1,-1)),self.signalmonthly.reshape((1,-1)),self.signalsemiannually.reshape((1,-1)),\
                self.current_rate.reshape((1,-1)),(self.current_pos[:,:-1]/(self.transact_memsize*max(self.unit_list))).reshape((1,-1)), \
                                             (np.array(self.lastnavs)/self.funds).reshape((1,-1)),\
                                             self.unit_in.reshape((1,-1))),axis = 1)
        
        elif self.modtype == '2-1-2-1-1':
            if actor:
                new_state = np.concatenate((self.time_info.reshape((1,-1)),self.signaldaily.reshape((1,-1)),self.signalmonthly.reshape((1,-1)),self.signalsemiannually.reshape((1,-1)),\
                self.current_rate.reshape((1,-1)),(self.current_pos[:,:-1]/(self.transact_memsize*max(self.unit_list))).reshape((1,-1)), \
                                             (np.array(self.lastnavs)/self.funds).reshape((1,-1))),axis = 1)
            else:
                new_state = np.concatenate((self.time_info.reshape((1,-1)),self.signaldaily.reshape((1,-1)),self.signalmonthly.reshape((1,-1)),self.signalsemiannually.reshape((1,-1)),\
                self.current_rate.reshape((1,-1)),(self.current_pos[:,:-1]/(self.transact_memsize*max(self.unit_list))).reshape((1,-1)), \
                                             (np.array(self.lastnavs)/self.funds).reshape((1,-1)),\
                                             self.unit_in.reshape((1,-1)),self.epsilon_in.reshape((1,-1)),\
                                             self.lrate_in.reshape((1,-1))),axis = 1)
        
        elif self.modtype == '1-1-2-2-1':
            if actor:
                new_state = np.concatenate((self.time_info.reshape((1,-1)),self.signaldaily.reshape((1,-1)),self.signalmonthly.reshape((1,-1)),self.signalsemiannually.reshape((1,-1)),\
                self.current_rate.reshape((1,-1)),self.mem_transact.reshape((1,-1)), \
                                             (np.array(self.lastnavs)/self.funds).reshape((1,-1))),axis = 1)
            else:
                new_state = np.concatenate((self.time_info.reshape((1,-1)),self.signaldaily.reshape((1,-1)),self.signalmonthly.reshape((1,-1)),self.signalsemiannually.reshape((1,-1)),\
                self.current_rate.reshape((1,-1)),self.mem_transact.reshape((1,-1)), \
                                             (np.array(self.lastnavs)/self.funds).reshape((1,-1)),\
                                             self.unit_in.reshape((1,-1))),axis = 1)
        
        elif self.modtype == '2-1-2-2-1':
            if actor:
                new_state = np.concatenate((self.time_info.reshape((1,-1)),self.signaldaily.reshape((1,-1)),self.signalmonthly.reshape((1,-1)),self.signalsemiannually.reshape((1,-1)),\
                self.current_rate.reshape((1,-1)),self.mem_transact.reshape((1,-1)), \
                                             (np.array(self.lastnavs)/self.funds).reshape((1,-1))),axis = 1)
            else:
                new_state = np.concatenate((self.time_info.reshape((1,-1)),self.signaldaily.reshape((1,-1)),self.signalmonthly.reshape((1,-1)),self.signalsemiannually.reshape((1,-1)),\
                self.current_rate.reshape((1,-1)),self.mem_transact.reshape((1,-1)), \
                                             (np.array(self.lastnavs)/self.funds).reshape((1,-1)),\
                                             self.unit_in.reshape((1,-1)),self.epsilon_in.reshape((1,-1)),\
                                             self.lrate_in.reshape((1,-1))),axis = 1)
        elif self.modtype == '1-1-2-2-2':
            if actor:
                new_state = np.concatenate((self.time_info.reshape((1,-1)),self.current_rate.reshape((1,-1)),\
                                             (self.current_pos[:,:-1]/(self.transact_memsize*max(self.unit_list))).reshape((1,-1)), \
                                             (np.array(self.lastnavs)/self.funds).reshape((1,-1))),axis = 1)
            else:
                new_state = np.concatenate((self.time_info.reshape((1,-1)),self.current_rate.reshape((1,-1)),\
                                             (self.current_pos[:,:-1]/(self.transact_memsize*max(self.unit_list))).reshape((1,-1)), \
                                             (np.array(self.lastnavs)/self.funds).reshape((1,-1)),\
                                             self.unit_in.reshape((1,-1))),axis = 1)
                                             
        elif self.modtype == '2-1-2-2-2':
            if actor:
                new_state = np.concatenate((self.time_info.reshape((1,-1)),self.current_rate.reshape((1,-1)),\
                                             (self.current_pos[:,:-1]/(self.transact_memsize*max(self.unit_list))).reshape((1,-1)), \
                                             (np.array(self.lastnavs)/self.funds).reshape((1,-1))),axis = 1)
            else:
                new_state = np.concatenate((self.time_info.reshape((1,-1)),self.current_rate.reshape((1,-1)),\
                                             (self.current_pos[:,:-1]/(self.transact_memsize*max(self.unit_list))).reshape((1,-1)), \
                                             (np.array(self.lastnavs)/self.funds).reshape((1,-1)),\
                                             self.unit_in.reshape((1,-1)),self.epsilon_in.reshape((1,-1)),\
                                             self.lrate_in.reshape((1,-1))),axis = 1)
        
        elif self.modtype == '1-1-2-3-2':
            if actor:
                new_state = np.concatenate((self.time_info.reshape((1,-1)),self.current_rate.reshape((1,-1)),\
                                             self.mem_transact.reshape((1,-1)), \
                                             (np.array(self.lastnavs)/self.funds).reshape((1,-1))),axis = 1)
            else:
                new_state = np.concatenate((self.time_info.reshape((1,-1)),self.current_rate.reshape((1,-1)),\
                                             self.mem_transact.reshape((1,-1)), \
                                             (np.array(self.lastnavs)/self.funds).reshape((1,-1)),\
                                             self.unit_in.reshape((1,-1))),axis = 1)
                                             
        elif self.modtype == '2-1-2-3-2':
            if actor:
                new_state = np.concatenate((self.time_info.reshape((1,-1)),self.current_rate.reshape((1,-1)),\
                                             self.mem_transact.reshape((1,-1)), \
                                             (np.array(self.lastnavs)/self.funds).reshape((1,-1))),axis = 1)
            else:
                new_state = np.concatenate((self.time_info.reshape((1,-1)),self.current_rate.reshape((1,-1)),\
                                             self.mem_transact.reshape((1,-1)), \
                                             (np.array(self.lastnavs)/self.funds).reshape((1,-1)),\
                                             self.unit_in.reshape((1,-1)),self.epsilon_in.reshape((1,-1)),\
                                             self.lrate_in.reshape((1,-1))),axis = 1)    
        elif self.modtype == '1-2-3-2-2':
            for i in range(self.num_curr):
                if i == 0:
                    x = self.curr_pair_history_data_daily[i].values.reshape((self.time_count[0],10))
                    y = self.curr_pair_history_data_monthly[i].values.reshape((self.time_count[1],10))
                    z = self.curr_pair_history_data_semiannually[i].values.reshape((self.time_count[2],10))
                    #print(z.shape)
                else:
                    x = np.concatenate((x,self.curr_pair_history_data_daily[i].values.\
                                        reshape((self.time_count[0],10))),axis = 1)
                    y = np.concatenate((y,self.curr_pair_history_data_monthly[i].values.\
                                        reshape((self.time_count[1],10))),axis = 1)
                    z = np.concatenate((z,self.curr_pair_history_data_semiannually[i].values.\
                                        reshape((self.time_count[2],10))),axis = 1)
                    #print(z.shape)
                #print(y.shape)
            #x = np.divide(x[1:,:],x[:-1,:])
            if actor:
                print(x.size,y.size)
                new_state = [x.reshape((1,self.time_count[0],10*self.num_curr)),y.reshape((1,self.time_count[1],10*self.num_curr)),z.reshape((1,self.time_count[2],10*self.num_curr)),\
                              np.concatenate((self.time_info.reshape((1,-1)),self.current_rate.reshape((1,-1)),self.mem_transact.reshape((1,-1)),(np.array(self.lastnavs)/self.funds).reshape((1,-1))),axis = 1)]
            else:
                new_state = [x.reshape((1,self.time_count[0],10*self.num_curr)),y.reshape((1,self.time_count[1],10*self.num_curr)),z.reshape((1,self.time_count[2],10*self.num_curr)),\
                              np.concatenate((self.time_info.reshape((1,-1)),self.current_rate.reshape((1,-1)),self.mem_transact.reshape((1,-1)),(np.array(self.lastnavs)/self.funds).reshape((1,-1)),\
                                             self.unit_in.reshape((1,-1))),axis = 1)]
        
        elif self.modtype == '2-2-3-2-2':
            for i in range(self.num_curr):
                if i == 0:
                    x = self.curr_pair_history_data_daily[i].values.reshape((self.time_count[0],10))
                    y = self.curr_pair_history_data_monthly[i].values.reshape((self.time_count[1],10))
                    z = self.curr_pair_history_data_semiannually[i].values.reshape((self.time_count[2],10))
                else:
                    x = np.concatenate((x,self.curr_pair_history_data_daily[i].values.\
                                        reshape((self.time_count[0],10))),axis = 1)
                    y = np.concatenate((y,self.curr_pair_history_data_monthly[i].values.\
                                        reshape((self.time_count[1],10))),axis = 1)
                    z = np.concatenate((z,self.curr_pair_history_data_semiannually[i].values.\
                                        reshape((self.time_count[2],10))),axis = 1)
            #x = np.divide(x[1:,:],x[:-1,:])
            if actor:
                new_state = [x.reshape((1,self.time_count[0],10*self.num_curr)),y.reshape((1,self.time_count[1],10*self.num_curr)),z.reshape((1,self.time_count[2],10*self.num_curr)),\
                              np.concatenate((self.time_info.reshape((1,-1)),self.current_rate.reshape((1,-1)),self.mem_transact.reshape((1,-1)),(np.array(self.lastnavs)/self.funds).reshape((1,-1))),axis = 1)]
            else:
                new_state = [x.reshape((1,self.time_count[0],10*self.num_curr)),y.reshape((1,self.time_count[1],10*self.num_curr)),z.reshape((1,self.time_count[2],10*self.num_curr)),\
                              np.concatenate((self.time_info.reshape((1,-1)),self.current_rate.reshape((1,-1)),self.mem_transact.reshape((1,-1)),(np.array(self.lastnavs)/self.funds).reshape((1,-1)),\
                                             self.unit_in.reshape((1,-1)),self.epsilon_in.reshape((1,-1)),\
                                             self.lrate_in.reshape((1,-1))),axis = 1)]
        #print(new_state)
        print('Input created')
        print(datetime.datetime.now())
        print('###################')
        return state,new_state
    def output_create(self):
        #print('-',end=' ')
        #print(action)
        #act = np.argmax(action)
        #print('In output')
        #print(act)
        print('###################')
        print('Create Output')
        print(datetime.datetime.now())
        
        if self.sel_action[0] == 0:
            return
        #print('Single System')
        ticker_index = self.sel_action[0]//4
        act_index = self.sel_action[0]%4
        #print(ticker_index,act_index)
        #print('##########TMEM#############')               
        #print(self.t_memlist)    
        if act_index == 1:
            print('Buy %i %s'%(self.units,self.curr_pair_list[ticker_index]))
            print(len(self.t_memlist[ticker_index]))
            #self.variable_cost = 1/100000
            if len(self.t_memlist[ticker_index]) == self.transact_memsize:
                if self.t_memlist[ticker_index][0][2] == 0:
                    self.penalty = True
                    print('Buy length penalty')
                else:
                    print('Buy order created length mode')
                    self.trader.create_buy_order(self.curr_pair_list[ticker_index],self.units)
            else:
                print('Buy order created')
                self.trader.create_buy_order(self.curr_pair_list[ticker_index],self.units)
        elif act_index == 2:
            print(len(self.t_memlist[ticker_index]))
            print('Sell %i %s'%(self.units,self.curr_pair_list[ticker_index]))
            #self.variable_cost = 1/100000
            if len(self.t_memlist[ticker_index]) == self.transact_memsize:
                if self.t_memlist[ticker_index][0][1] == 0:
                    self.penalty = True
                    print('Sell length penalty')
                else:
                    print('Sell order created length mode')
                    self.trader.create_sell_order(self.curr_pair_list[ticker_index],self.units)
            else:
                print('Sell order created')
                self.trader.create_sell_order(self.curr_pair_list[ticker_index],self.units)
        elif act_index == 3:
            print('CloseAll',self.curr_pair_list[ticker_index])
            r = self.trader.close_positions(self.curr_pair_list[ticker_index],'ALL')
            if r == -1:
                self.penalty = True
                #self.variable_cost = 1/100000
                print('Close All penalty')
            else:
                self.variable_cost = 0.0
        else:
            ticker_index -= 1
            print('CloseFirst',self.curr_pair_list[ticker_index])
            if self.units > max(self.current_pos[ticker_index,0],self.current_pos[ticker_index,1]):
                r = self.trader.close_positions(self.curr_pair_list[ticker_index],'ALL')
            else:
                r = self.trader.close_positions(self.curr_pair_list[ticker_index],str(self.units))
            if r == -1:
                self.penalty = True
                #self.variable_cost = 1/100000
                print('Close 1 penalty')
            else:
                self.variable_cost = 0.0
        
        print('Output created')
        print(datetime.datetime.now())
        print('###################')
    def ac_learning(self):
        if self.random:
            seq = np.random.randint(0,len(self.actor_envlist[1:]),len(self.actor_envlist[1:]))
        else:
            seq = np.arange(len(self.actor_envlist[1:]))
        actor_inp = []
        actor_out = []
        critic_inp = []
        critic_out = []
        print('##############################')
        print(datetime.datetime.now())
        print('AC Learning')
        for i in range(len(seq)):
            state, action, reward, new_state, game_over = self.critic_envlist[seq[i]+1]
            actor_state,actor_action = self.actor_envlist[seq[i]+1]
            #inp_data = state
            #print('\n')
            #print(i,action,sel_action)
            #print('\n')
            
            #print(state)
            #print(new_state)
            
            if isinstance(new_state,list):
                
                reward_pred = np.ravel(self.critic.predict(state))
                next_reward_pred = np.ravel(self.critic.predict(new_state))
                
                action_list = self.actor.predict(actor_state)
                action_list = [np.ravel(item) for item in action_list]
                                
            else:
                
                reward_pred = np.ravel(self.critic.predict(state.reshape((1,-1))))
                next_reward_pred = np.ravel(self.critic.predict(new_state.reshape((1,-1))))
                #print(actor_state.shape)
                action_list = self.actor.predict(actor_state)
                action_list = [np.ravel(item) for item in action_list]
            #print(action_list)
            td_target = self.reward + self.discount*next_reward_pred
            td_error = td_target - reward_pred
            #print(td_error)
            for j in range(len(reward_pred)):
                action_list[j][actor_action[j]] += td_error[j]
                        
            critic_inp.append(state)
            critic_out.append(td_target)
            actor_inp.append(actor_state)
            actor_out.append(action_list)
            
        actor_output_data = []
        for i in range(len(actor_out[0])):
                #print(inp[i][0].shape,inp[i][1].shape)
                actor_output_data.append(np.vstack([data[i] for data in actor_out]))
                
        if isinstance(new_state,list):
            #print(inp)
            critic_input_data = []
            #critic_output_data = []
            actor_input_data = []
            
            
            for i in range(len(critic_inp[0])):
                #print(inp[i][0].shape,inp[i][1].shape)
                critic_input_data.append(np.vstack([data[i] for data in critic_inp]))
                            
            for i in range(len(actor_inp[0])):
                #print(inp[i][0].shape,inp[i][1].shape)
                actor_input_data.append(np.vstack([data[i] for data in actor_inp]))
            
                
            self.critic.fit(critic_input_data,np.array(critic_out),batch_size = 25,epochs = 1, verbose = 1)
            self.actor.fit(actor_input_data,actor_output_data,batch_size = 25,epochs = 1, verbose = 1)
        else:
            #print(np.array(inp).shape)
            #print(actor_output_data)
            self.critic.fit(np.array(critic_inp),np.array(critic_out),batch_size = 25,epochs = 1, verbose = 1)
            self.actor.fit(np.squeeze(np.array(actor_inp),axis=1),actor_output_data,batch_size = 25,epochs = 1, verbose = 1)
        print('AC End')
        print(datetime.datetime.now())
        print('##############################')
        
        
    def runmodel(self):
        #random.seed(self.mid)
        print('\n')
        print('###################################################################################')
        print(self.modtype)
        print('Id : ',self.id)
        print('Epsilon :',self.epsilon)
        print(datetime.datetime.now())
        
        #print('\n')
        #print('-',end=' ')
        #print(self.layerlist,self.complexity,self.complex_num)
        
        print('###################')
        print('Single')
        print('Learning Rate :',K.get_value(self.actor.optimizer.lr))
        print('###################')
        
        
        
        self.forexenv()
        self.modelstate()
        self.forexreward()
        if self.sel_action[0] == 0:
            self.reward -= 0.1
        self.forexstate()
        self.mem_create()
        self.model_analytics()
        print('###################')
        print('State loaded')
        print(datetime.datetime.now())
        print('###################')
        #print(self.mem_transact)
        print(self.reward,self.reward_punish)
        if self.reward > 0:
            self.reward_punish[0] += abs(self.reward)
        else:
            self.reward_punish[1] += abs(self.reward)
            
        self.actor_state,self.actor_new_state = self.input_create(self.actor_new_state)
        #print(self.new_state)
        if isinstance(self.actor_state,list):
            self.actor_envlist.append([self.actor_state,self.sel_action])
        else:
            if self.actor_state is not None:
                self.actor_envlist.append([self.actor_state,self.sel_action])
            else:
                self.actor_envlist.append([self.actor_state,self.sel_action])
        
        self.critic_state,self.critic_new_state = self.input_create(self.critic_new_state,False)
        #print(self.new_state)
        if isinstance(self.critic_state,list):
            self.critic_envlist.append([self.critic_state,self.sel_action,self.reward,self.critic_new_state,self.game_over])
        else:
            if self.critic_state is not None:
                self.critic_envlist.append([np.ravel(self.critic_state),self.sel_action,self.reward,np.ravel(self.critic_new_state),self.game_over])
            else:
                self.critic_envlist.append([self.critic_state,self.sel_action,self.reward,self.critic_new_state,self.game_over])
        
        print('Length of Env List %i'%len(self.actor_envlist))
        if isinstance(self.actor_envlist[0],list):
            
            if len(self.actor_envlist) > 2:
                
                if (len(self.actor_envlist)%self.q_freq) == 0:
                    print('Actor Critic call')
                    #print(self.actor_envlist)
                    #print(self.critic_envlist)
                    self.ac_learning()
                
                if len(self.actor_envlist)>self.buffer_length:
                    del self.actor_envlist[0]
                if len(self.critic_envlist)>self.buffer_length:
                    del self.critic_envlist[0]
        #print(self.new_state)
        self.action = self.actor.predict(self.actor_new_state)
        self.action = [np.ravel(item) for item in self.action]
        #critic_act = self.critic.predict(self.critic_new_state)
        print('Action \n',self.action)
        # Epsilon Greedy
        if random.random() >= self.epsilon:
            print('Greedy Action')
            self.sel_action[0] = np.argmax(np.ravel(self.action[0]))
            self.sel_action[1] = np.argmax(np.ravel(self.action[1]))
            self.units = self.unit_list[self.sel_action[1]]
            self.random_desc[1] += 1
            self.problist.append(np.ravel(self.action[0])[self.sel_action[0]]*100)
            self.actlist.append(self.sel_action[0])
            
            if self.modtype.split('-')[0] == '2':
                self.sel_action[2] = np.argmax(np.ravel(self.action[2]))
                self.sel_action[3] = np.argmax(np.ravel(self.action[3]))
                
                self.epsilon = self.epsilon_list[self.sel_action[2]]
                self.lrate = self.lrate_list[self.sel_action[3]]
                
        else:
            print('Non Greedy Action')
            self.random_desc[0] += 1
            #self.sel_action = random.randrange(len(self.action))
            print(self.move_counter)
            prob_m = 1 - self.move_counter/np.sum(self.move_counter)
            prob_m = prob_m/np.sum(prob_m)
            print(prob_m)
            self.sel_action[0] = np.random.choice(self.moves,1,p=prob_m)[0]
            self.problist.append(prob_m[self.sel_action[0]]*100)
            self.actlist.append(self.sel_action[0])
            print(self.unit_desc)
            prob_u = 1 - self.unit_desc/np.sum(self.unit_desc)
            prob_u = prob_u/np.sum(prob_u)
            print(prob_u)
            self.sel_action[1] = np.random.choice(np.arange(len(self.unit_list)),1,p=prob_u)[0]
            
            if self.modtype.split('-')[0] == '2':
                print(self.random_desc)
                prob_e = 1 - self.random_desc/np.sum(self.random_desc)
                prob_e = prob_e/np.sum(prob_e)
                print(prob_e)
                self.sel_action[2] = np.random.choice(np.arange(2),1,p=prob_e)[0]
                
                print(self.lrate_desc)
                prob_l = 1 - self.lrate_desc/np.sum(self.lrate_desc)
                prob_l = prob_l/np.sum(prob_l)
                print(prob_l)
                self.sel_action[3] = np.random.choice(np.arange(len(self.lrate_list)),1,p=prob_l)[0]
            
        self.units = self.unit_list[self.sel_action[1]]
        self.move_counter[self.sel_action[0]] += 1
        self.unit_desc[self.sel_action[1]] += 1
        self.unit_in.fill(0)
        self.unit_in[self.sel_action[1]] = 1
        
        if self.modtype.split('-')[0] == '2':
            self.epsilon = self.epsilon_list[self.sel_action[2]]
            self.lrate = self.lrate_list[self.sel_action[3]]
            K.set_value(self.actor.optimizer.lr, self.lrate)
            self.lrate_desc[self.sel_action[3]] += 1
                   
            self.epsilon_in.fill(0)
            self.lrate_in.fill(0)
                   
            self.epsilon_in[self.sel_action[2]] = 1
            self.lrate_in[self.sel_action[3]] = 1
        print('Action Selection : %i Units Selection : %i'%(self.sel_action[0],self.sel_action[1]))
        #print('-',end=' ')
        
        if self.sel_action[0] == 0:
            #self.variable_cost = 0.0
            print('None action chosen')
        else:
            self.output_create()
        
        print(self.nav)
        #print('-',end=' ')
        self.navlist.append(self.nav)
        self.datelist.append(datetime.datetime.utcnow())
        self.lastnavs.append(self.nav)
        
        self.levellist.append(self.level)
        
        #self.fixed += self.fixed_cost + self.units*self.variable_cost
        if len(self.navlist) > 5000:
            del self.navlist[0]
            del self.datelist[0]
            del self.levellist[0]
        if len(self.lastnavs) > 5:
            del self.lastnavs[0]
        if self.diff_acc:
            pass
            
        else:
            ForexPrime.same_acc_navlist.append(self.nav)
            ForexPrime.same_datelist.append(datetime.datetime.utcnow())
        '''
        if len(self.navlist) >= 3:
            diff1 =  self.navlist[-1] - self.navlist[-2]
            diff2 =  self.navlist[-2] - self.navlist[-3]
            try:
                ratio = np.log(abs(diff1/diff2))
            except :
                ratio = 1
            print(ratio)
            if ratio >= -0.1 and ratio <= 0.1:
                print('Between -0.1 and 0.1')
                self.epsilon += ratio/10
                K.set_value(self.model.optimizer.lr, 10 * K.get_value(self.model.optimizer.lr))
                if self.modtype.split('-')[0] == '2':
                    K.set_value(self.model_twin.optimizer.lr, 10 * K.get_value(self.model_twin.optimizer.lr))
            else:
                print('Not in Between -0.1 and 0.1')
                self.epsilon += ratio/1000
                K.set_value(self.model.optimizer.lr, 0.1 * K.get_value(self.model.optimizer.lr))
                if self.modtype.split('-')[0] == '2':
                    K.set_value(self.model_twin.optimizer.lr, 0.1 * K.get_value(self.model_twin.optimizer.lr))

        if K.get_value(self.model.optimizer.lr) > 10:
            K.set_value(self.model.optimizer.lr, 10)
        if K.get_value(self.model.optimizer.lr) < 0.001:
            K.set_value(self.model.optimizer.lr, 0.001)
        
        if self.modtype.split('-')[0] == '2':
            if K.get_value(self.model_twin.optimizer.lr) > 10:
                K.set_value(self.model_twin.optimizer.lr, 10)
            if K.get_value(self.model_twin.optimizer.lr) < 0.001:
                K.set_value(self.model_twin.optimizer.lr, 0.001)
        
        if self.epsilon >= 0.5:
            self.epsilon = 0.5
        
        if self.epsilon <= 0.2:
            self.epsilon = 0.2
        '''
        if self.game_over:
            #print(self.game_over)
            print('########################')
            print('Game Over')
            print('########################')
            self.transaction_counter = 0
            self.closeAll()
            self.nav = self.trader.get_nav()['nav']
            self.lastnavs = [0,0,0,0,self.funds]
            self.fixed = self.nav - self.funds
            self.nav = self.funds
            if self.success:
                self.success_fail[0] += 1
            else:
                self.success_fail[1] += 1
            self.game_over = False
            self.success = False
            '''
            ldiv = max(min(self.level,10),1)
            
            if self.success_fail[0] > (10//ldiv):
                self.level += 1
                self.success_fail = np.zeros((2)) + 1
            if self.success_fail[1] > (20//ldiv):
                self.level -= 1
                self.success_fail = np.zeros((2)) + 1
            if self.level < 1:
                self.level = 1
            '''
        else:
            #print(self.game_over)
            print('########################')
            print('Currently Active')
            print('########################')
            
            #print('-',end=' ')
