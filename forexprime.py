from forex_trader_switch import *
import time
import json
import talib
import os
import csv
import datetime
from mlmodel import *
from database import Database
from Oanda_Trader import *
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras
import math
from keras.layers import Input, LSTM, Dense
from keras.models import load_model,Model
from keras.utils import plot_model
class ForexPrime(ForexTraderSwitch):
    same_acc_navlist = []
    same_datelist = []
    price_history = []
    od_daily = None
    sg_daily = None
    od_monthly = None
    sg_monthly = None
    od_semi = None
    sg_semi = None
    def __init__(self,idval,accountID, access_token,modtype,funds):
        self.id = str(idval)
        
        self.db = ForexTraderSwitch.db
        self.current_path = ForexTraderSwitch.current_path
        self.curr_pair_list = ForexTraderSwitch.curr_pair_list
        self.num_curr = ForexTraderSwitch.num_curr
        self.moves_allowed = ForexTraderSwitch.moves_allowed
        self.order = ForexTraderSwitch.order
        self.signal = ForexTraderSwitch.signal
        self.orderdaily = np.zeros((self.num_curr,7,3))
        self.signaldaily = np.zeros((self.num_curr,16))
        self.ordermonthly = np.zeros((self.num_curr,7,3))
        self.signalmonthly = np.zeros((self.num_curr,16))
        self.ordersemiannually = np.zeros((self.num_curr,7,3))
        self.signalsemiannually = np.zeros((self.num_curr,16))
        self.current_rate = ForexTraderSwitch.current_rate
        self.time_count = ForexTraderSwitch.time_count
        self.curr_pair_history_data = ForexTraderSwitch.curr_pair_history_data
        self.curr_pair_history_data_daily = []
        self.curr_pair_history_data_monthly = []
        self.curr_pair_history_data_semiannually = []
        
        self.level = 1
        self.levellist = [self.level]
        self.transaction_counter = 0
        self.transaction_max = 10
        self.usd_conv = 1.0
        self.fixed_cost = 1
        self.variable_cost = 1/100000
        self.transact_memsize = 5
        self.mem_transact = np.zeros((self.num_curr,self.transact_memsize,3))
        self.units = 100
        self.trader = Oanda_Trader(accountID, access_token)
        self.closeAll()
        self.mlmod = MlModel()
        self.buffer_length = 200
        self.accountID = accountID
        self.access_token = access_token
        self.modtype = modtype
        self.funds = funds
        self.navlist = [self.funds]
        self.datelist = [datetime.datetime.utcnow()]
        ForexPrime.same_acc_navlist = [self.funds]
        ForexPrime.same_datelist = [datetime.datetime.utcnow()]
        if self.modtype == '1-1-1-1-1':
            val_to_append = ['Datetime']+[curr for curr in self.curr_pair_list]
            ForexPrime.price_history.append(val_to_append)
        
        # Net Asset value
        self.nav = self.trader.get_nav()['nav']
        
        self.lastnavs = [0,0,0,0,self.funds]
        self.fixed = self.nav - self.funds
        self.nav = self.funds
        self.old_bal = self.funds
        self.bal = self.funds
        
        # Current State, Action, Reward, New State, Game Over
        self.actor_envlist = []
        self.critic_envlist = []
        
        self.game_over = False
        self.success = False
        
        self.reward = 0
        #self.actor_reward = 0
        #self.critic_reward = 0
        self.penalty = False
        
        self.move_penalty = 0.01
        self.actor_state = None
        self.actor_new_state = None
        
        self.critic_state = None
        self.critic_new_state = None
                        
        self.past_pos = np.zeros((ForexTraderSwitch.num_curr,4))
        self.current_pos = np.zeros((ForexTraderSwitch.num_curr,4))
                       
        self.action = None
        self.eps_act = None
        self.unit_act = None
        self.lr_act = None
        
        self.t_memlist = [[],[],[],[],[],[]]
        self.t_done_memlist = [[],[],[],[],[],[]]
        self.write_done_transact = []
        self.prev_time = 0
        self.prev_len = 0
        self.analytic = np.zeros((10,len(self.curr_pair_list),6))
        self.pos_list = []
    
#    def curr_to_usd(self,curr_pair,order):
#        index_pair = self.curr_pair_list.index(curr_pair)
#        if 'USD' in curr_pair:
#            if curr_pair.split('_')[0] == 'USD':
#                if order == 'Long':
#                    self.usd_conv = np.asscalar(1/self.current_rate[index_pair])
#                else:
#                    self.usd_conv = np.asscalar(-1/self.current_rate[index_pair])
#            else:
#                if order == 'Long':
#                    self.usd_conv = 1
#                else:
#                    self.usd_conv = -1
#        else:
#            
#            if order == 'Long':
#                try:
#                    currval = curr_pair.split('_')[0]+'_USD'
#                    index_curr = self.curr_pair_list.index(currval)
#                except ValueError:
#                    currval = 'USD_'+curr_pair.split('_')[0]
#                    index_curr = self.curr_pair_list.index(currval)
#                print(currval,index_curr)
#                if currval.split('_')[0] == 'USD':
#                    self.usd_conv = np.asscalar(1/(self.current_rate[index_pair]*self.current_rate[index_curr]))
#                else:
#                    self.usd_conv = np.asscalar(self.current_rate[index_curr]/self.current_rate[index_pair])
#            else:
#                try:
#                    currval = curr_pair.split('_')[1]+'_USD'
#                    index_curr = self.curr_pair_list.index(currval)
#                except ValueError:
#                    currval = 'USD_'+curr_pair.split('_')[1]
#                    index_curr = self.curr_pair_list.index(currval)
#                print(currval,index_curr)    
#                if currval.split('_')[0] == 'USD':
#                    self.usd_conv = np.asscalar(-1/self.current_rate[index_curr])
#                else:
#                    self.usd_conv = np.asscalar(-self.current_rate[index_curr])
#        #print('Base Currency Profit factor')
#        #print(self.usd_conv)
#    def mem_write(self):
#        
#        for i in range(self.num_curr):
#            if len(self.t_memlist[i]) == 0:
#                continue
#            if isinstance(self.t_memlist[i][0],list):
#                for j,k in enumerate(self.t_memlist[i]):
#                    self.mem_transact[i,j,0:len(k)-1] = k[1:]
#            else:
#                self.mem_transact[i,0,:] = self.t_memlist[i][1:]
#    def model_analytics(self):
#        #self.analytic = np.zeros((10,len(self.curr_pair_list),6)) + 1
#        print('inside model analytics')
#        #print(self.prev_len,len(self.write_done_transact))
#        if (len(self.write_done_transact) == 0) or (self.prev_len == len(self.write_done_transact)):
#            return
#        
#        else:
#            for i in range(len(self.write_done_transact) - self.prev_len):
#                print(self.write_done_transact[-1-i])
#                self.prev_len = len(self.write_done_transact)
#                timediff = math.ceil(int(self.write_done_transact[-1-i][0])/60)
#                longunits = int(self.write_done_transact[-1-i][1])
#                shortunits = int(self.write_done_transact[-1-i][2])
#                profitloss = float(self.write_done_transact[-1-i][3])
#                currpair = self.write_done_transact[-1-i][4]
#                print(timediff,longunits,shortunits,profitloss,currpair)
#                print(timediff.bit_length() - 1)
#                if timediff > 512:
#                    timediff = 512
#                if shortunits != 0:
#                    self.analytic[timediff.bit_length() - 1,self.curr_pair_list.index(currpair),1] += shortunits
#                    self.analytic[timediff.bit_length() - 1,self.curr_pair_list.index(currpair),3] += profitloss
#                    self.analytic[timediff.bit_length() - 1,self.curr_pair_list.index(currpair),5] += 1
#                elif longunits != 0:
#                    self.analytic[timediff.bit_length() - 1,self.curr_pair_list.index(currpair),0] += longunits
#                    self.analytic[timediff.bit_length() - 1,self.curr_pair_list.index(currpair),2] += profitloss
#                    self.analytic[timediff.bit_length() - 1,self.curr_pair_list.index(currpair),4] += 1
#        #print(self.analytic)
#    def mem_create(self):
#        
#        self.mem_transact = np.zeros((self.num_curr,self.transact_memsize,3))
#        if len(self.trader.trade_history) == 0:
#            print('empty')
#            return
#        print(self.trader.trade_history[-1])
#        timestamp = self.trader.trade_history[-1][0]
#        unixtime = int(self.trader.trade_history[-1][1])
#        unit = abs(int(self.trader.trade_history[-1][2]))
#        currpair = self.trader.trade_history[-1][3]
#        price = float(self.trader.trade_history[-1][4])
#        action = self.trader.trade_history[-1][8]
#        unitcounter = unit
#        if self.prev_time == unixtime:
#            print('No new transaction')
#            return
#        self.prev_time = unixtime
#        if len(self.t_memlist[self.curr_pair_list.index(currpair)]) == 0:
#            print('First')
#            print(action)
#            if action == 'Close' or action == 'CloseAll':
#                print('Should return')
#                return
#            elif action == 'Buy':
#                list_to_app = [unixtime,unit,0,price]
#            else:
#                list_to_app = [unixtime,0,unit,price]
#            print('appending')
#            #print(list_to_app)
#            self.t_memlist[self.curr_pair_list.index(currpair)] = [list_to_app]
#            #print(self.t_memlist)
#            self.mem_write()
#            return
#        
#        if action == 'Buy' and self.t_memlist[self.curr_pair_list.index(currpair)][0][1] != 0:
#            list_to_app = [unixtime,unit,0,price]
#            self.t_memlist[self.curr_pair_list.index(currpair)].append(list_to_app)
#            print('Buy copy')
#            #print(list_to_app)
#        elif action == 'Sell' and self.t_memlist[self.curr_pair_list.index(currpair)][0][2] != 0:
#            list_to_app = [unixtime,0,unit,price]
#            self.t_memlist[self.curr_pair_list.index(currpair)].append(list_to_app)
#            print('Sell copy')
#            #print(list_to_app)
#        else:
#            transact = True
#
#            while transact:
#                if len(self.t_memlist[self.curr_pair_list.index(currpair)]) == 0:
#                    if action == 'Buy':
#                        self.t_memlist[self.curr_pair_list.index(currpair)].append([unixtime,unitcounter,0,price])
#                        print('len 0 buy')
#                        #print(unixtime,unitcounter,price)
#                    elif action == 'Sell':
#                        self.t_memlist[self.curr_pair_list.index(currpair)].append([unixtime,0,unitcounter,price])
#                        print('len 0 sell')
#                        #print(unixtime,unitcounter,price)
#                    break
#
#                if isinstance(self.t_memlist[self.curr_pair_list.index(currpair)][0],list):
#                    old_unixtime = self.t_memlist[self.curr_pair_list.index(currpair)][0][0]
#                    old_long = self.t_memlist[self.curr_pair_list.index(currpair)][0][1]
#                    old_short = self.t_memlist[self.curr_pair_list.index(currpair)][0][2]
#                    old_price = self.t_memlist[self.curr_pair_list.index(currpair)][0][3]
#                else:
#                    old_unixtime = self.t_memlist[self.curr_pair_list.index(currpair)][0]
#                    old_long = self.t_memlist[self.curr_pair_list.index(currpair)][1]
#                    old_short = self.t_memlist[self.curr_pair_list.index(currpair)][2]
#                    old_price = self.t_memlist[self.curr_pair_list.index(currpair)][3]
#                print('old entry')
#                #print(old_unixtime,old_long,old_short,old_price)
#                if old_short == 0:
#                    index = 1
#                    mult = 1
#                else:
#                    index = 2
#                    mult = -1
#                longshort = [0,0]
#                print(index)
#                if (old_long+old_short) > unitcounter:
#
#                    transact = False
#                    write_unit = old_long+old_short - unitcounter
#                    print('write unit')
#                    #print(write_unit)
#                    if isinstance(self.t_memlist[self.curr_pair_list.index(currpair)][0],list):
#                        self.t_memlist[self.curr_pair_list.index(currpair)][0][index] = write_unit
#                    else:
#                        self.t_memlist[self.curr_pair_list.index(currpair)][index] = write_unit
#                    print(unitcounter)
#                    longshort[index-1] = unitcounter
#                    unitcounter = 0
#                    print(longshort)
#                else:
#
#                    if isinstance(self.t_memlist[self.curr_pair_list.index(currpair)][0],list):
#                        del self.t_memlist[self.curr_pair_list.index(currpair)][0]
#                        print('deleted entry')
#                    else:
#                        self.t_memlist[self.curr_pair_list.index(currpair)] = []
#                        print('empty entry')
#                    unitcounter -= old_long+old_short
#                    if unitcounter == 0:
#                        transact = False
#                    #print(unitcounter)
#                    longshort[index-1] = old_long+old_short
#                    #print(longshort)
#
#                if sum(longshort) == 0:
#                    print('check completed flow')
#                    #print(unit,old_long,old_short,currpair)
#                timediff = unixtime - old_unixtime
#                
#                if mult == 1:
#                    self.curr_to_usd(currpair,'Long')
#                else:
#                    self.curr_to_usd(currpair,'Short')
#                
#                profitloss = self.usd_conv*(price - old_price)*(longshort[0]+longshort[1])
#                #print(profitloss)
#                self.t_done_memlist[self.curr_pair_list.index(currpair)]\
#                .append([timediff,longshort[0],longshort[1],round(profitloss,4)])
#                self.write_done_transact.append([timediff,longshort[0],longshort[1],round(profitloss,4),currpair])
#        self.mem_write()
#        #print(self.t_memlist[self.curr_pair_list.index(currpair)])    
#        
#        
#        
#    def closeAll(self):
#        for ticker in self.curr_pair_list:
#                self.trader.close_positions(ticker,'ALL')
#                
#    def forexenv(self):
#        if self.modtype.split('-')[-1] == '1':
#            set_sig = True
#        else:
#            set_sig = False
#        #ForexTraderSwitch.set_order_signal_db(set_sig)
#        self.get_history_data()
#    def forexreward(self):
#        
#        #self.reward = 10*(np.sum(np.subtract(self.current_pos[:,2:], self.past_pos[:,2:])) - 1)
#        #self.reward = self.nav - self.navlist[-1] - 0.1
#        #self.transaction_counter += 1
#        #self.transaction_max = 50 + (self.level - 1)*10
#        #if self.transaction_counter >= self.transaction_max:
#        #    self.game_over = True
#            
#        self.reward = (self.bal - self.old_bal)/self.funds -0.1 #+ (self.nav - self.navlist[-1])/self.funds
#        
#        if self.nav <= 0:
#            self.reward = -1
#            self.game_over = True
#        elif self.nav >= 2*self.funds:
#            self.reward = 1
#            self.game_over = True
#            self.success = True
#        
#        #else:
#            #self.reward *= 10
#        #    self.game_over = False
#        if self.penalty:
#            self.reward = -0.2
#        self.penalty = False
#    
#                
#    def forexstate(self):
#        
#        #self.order = ForexTraderSwitch.order
#        #self.signal = ForexTraderSwitch.signal
#        #self.current_rate = ForexTraderSwitch.current_rate
#        #self.curr_pair_history_data = ForexTraderSwitch.curr_pair_history_data
#        if self.id == '0':
#            self.orderdaily,self.signaldaily = self.all_algo(self.curr_pair_history_data_daily)
#            self.ordermonthly,self.signalmonthly = self.all_algo(self.curr_pair_history_data_monthly)
#            self.ordersemiannually,self.signalsemiannually = self.all_algo(self.curr_pair_history_data_semiannually)
#            ForexPrime.od_daily = self.orderdaily
#            ForexPrime.sg_daily = self.signaldaily
#            ForexPrime.od_monthly = self.ordermonthly
#            ForexPrime.sg_monthly = self.signalmonthly
#            ForexPrime.od_semi = self.ordersemiannually
#            ForexPrime.sg_semi = self.signalsemiannually
#        else:
#            self.orderdaily = ForexPrime.od_daily
#            self.signaldaily = ForexPrime.sg_daily
#            self.ordermonthly = ForexPrime.od_monthly
#            self.signalmonthly = ForexPrime.sg_monthly
#            self.ordersemiannually = ForexPrime.od_semi
#            self.signalsemiannually = ForexPrime.sg_semi
#        
#    def modelstate(self,twin = False):
#        
#        asset = self.trader.get_nav()
#        self.pastnav = self.nav
#        self.old_bal = self.bal
#        
#        self.nav = asset['nav'] - self.fixed
#        self.bal = asset['balance'] - self.fixed
#        
#        self.past_pos = self.current_pos
#
#        for ticker in self.curr_pair_list:
#            pos = self.trader.get_positions(ticker)
#            self.current_pos[self.curr_pair_list.index(ticker),0] = pos['short']
#            self.current_pos[self.curr_pair_list.index(ticker),1] = pos['long']
#            self.current_pos[self.curr_pair_list.index(ticker),2] = pos['unrealizedPL']
#            self.current_pos[self.curr_pair_list.index(ticker),3] = pos['pl']
#        pos_value = []
#        pos_value.append(datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f'))
#        for i in range(len(self.curr_pair_list)):
#            order_val = self.current_pos[i,0] + self.current_pos[i,1]
#            pos_value.append(order_val)
#            pos_temp = order_val*self.current_rate[i]
#            pos_value.append(pos_temp[0])
#        self.pos_list.append(pos_value)
#        #print(self.pos_list)
#            
#    def delmodels(self):
#        path = self.current_path+'/models/'
#        filesToRemove = [os.path.join(path,files) for files in os.listdir( path )]
#        for f in filesToRemove:
#            check = f.split('.')[-1]
#            print(f,check)
#            if check == ['h5'] or check == ['png']:
#                os.remove(f) 
#                
#    def write_trade_history(self):
#        #print(self.trader.trade_history)
#        with open(self.current_path+'/results/'+self.id+'_'+self.modtype+'_TradeHistory.csv', "w",newline='') as f:
#            writer = csv.writer(f)
#            writer.writerows(self.trader.trade_history)
#        with open(self.current_path+'/results/'+self.id+'_'+self.modtype+'_TradeCompleted.csv', "w",newline='') as g:
#            writerg = csv.writer(g)
#            writerg.writerows(self.write_done_transact)
#        with open(self.current_path+'/results/'+self.id+'_'+self.modtype+'_TradePositions.csv', "w",newline='') as p:
#            writerg = csv.writer(p)
#            writerg.writerows(self.pos_list)
#    
#    def get_current_time(self,w):
#        return w.strftime("%Y-%m-%dT%H:%M:%SZ")
#        
#    def get_history_data(self):
#        self.curr_pair_history_data_daily = []
#        for ticker in self.curr_pair_list:
#            #history = self.trader.get_history(ticker, "M1", self.time_count,\
#            #                                               self.get_current_time(datetime.datetime.utcnow()))
#            history = self.trader.get_history(ticker, "M1", self.time_count[0],\
#                                                           self.get_current_time(datetime.datetime.utcnow()))
#            self.curr_pair_history_data_daily.append(history)
#            self.current_rate[self.curr_pair_list.index(ticker)] = history.iloc[-1]['avg']
#        val_to_append = [datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')]+[item[0] for item in self.current_rate]
#        ForexPrime.price_history.append(val_to_append)
#        
#        self.curr_pair_history_data_monthly = []
#        for ticker in self.curr_pair_list:
#            #history = self.trader.get_history(ticker, "H1", self.time_count,\
#            #                                               self.get_current_time(datetime.datetime.utcnow()))
#            history = self.trader.get_history(ticker, "H1", self.time_count[1],\
#                                                           self.get_current_time(datetime.datetime.utcnow()))
#            self.curr_pair_history_data_monthly.append(history)
#            #self.current_rate[self.curr_pair_list.index(ticker)] = history.iloc[-1]['avg']
#            
#        self.curr_pair_history_data_semiannually = []
#        for ticker in self.curr_pair_list:
#            #history = self.trader.get_history(ticker, "H1", self.time_count,\
#            #                                               self.get_current_time(datetime.datetime.utcnow()))
#            history = self.trader.get_history(ticker, "D", self.time_count[2],\
#                                                           self.get_current_time(datetime.datetime.utcnow()))
#            self.curr_pair_history_data_semiannually.append(history)
#            #self.current_rate[self.curr_pair_list.index(ticker)] = history.iloc[-1]['avg']
            
    def all_algo(self,curr_pair_history_data,db=False):
        order = np.zeros((self.num_curr,7,3))
        signal = np.zeros((self.num_curr,16))
        for i in range(len(self.curr_pair_list)):
            # Processing data
            close5=(curr_pair_history_data[i]['closeAsk'].tail(5).values+\
                    curr_pair_history_data[i]['closeBid'].tail(5).values)/2
            high5=(curr_pair_history_data[i]['highAsk'].tail(5).values+\
                   curr_pair_history_data[i]['highBid'].tail(5).values)/2
            low5=(curr_pair_history_data[i]['lowAsk'].tail(5).values+\
                  curr_pair_history_data[i]['lowBid'].tail(5).values)/2
            openv5=(curr_pair_history_data[i]['openAsk'].tail(5).values+\
                    curr_pair_history_data[i]['openBid'].tail(5).values)/2

            close=(curr_pair_history_data[i]['closeAsk'].values+\
                   curr_pair_history_data[i]['closeBid'].values)/2
            high=(curr_pair_history_data[i]['highAsk'].values+\
                  curr_pair_history_data[i]['highBid'].values)/2
            low=(curr_pair_history_data[i]['lowAsk'].values+\
                 curr_pair_history_data[i]['lowBid'].values)/2
            openv=(curr_pair_history_data[i]['openAsk'].values+\
                   curr_pair_history_data[i]['openBid'].values)/2

            # Generating signals
            pattern=talib.CDL3BLACKCROWS(openv5,high5,low5,close5)
            pattern_signal=pattern[-1]

            adx=talib.ADX(high,low,close,timeperiod=14)
            rsi=talib.RSI(close,timeperiod=14)
            adx_signal=adx[-1]
            rsi_signal=rsi[-1]

            bull, bear=talib.AROON(high,low,timeperiod=14)

            sma=talib.SMA(close,timeperiod=30)
            kama=talib.KAMA(close,timeperiod=30)

            DIF,DEA,BAR=talib.MACDFIX(close,signalperiod=9)

            mfi=talib.MFI(high,low,close,curr_pair_history_data[i]['volume'].values.astype(float),timeperiod=14)
            mfi_signal=mfi[-1]

            # Storing signals
            signal[i,0] = pattern_signal
            signal[i,1] = adx_signal
            signal[i,13] = rsi_signal
            signal[i,2] = bear[-2]
            signal[i,3] = bear[-1]
            signal[i,4] = bull[-2]
            signal[i,5] = bull[-1]
            signal[i,10] = kama[-2]
            signal[i,11] = kama[-1]
            signal[i,14] = sma[-2]
            signal[i,15] = sma[-1]
            signal[i,6] = DEA[-2]
            signal[i,7] = DEA[-1]
            signal[i,8] = DIF[-2]
            signal[i,9] = DIF[-1]
            signal[i,12] = mfi_signal

            # Creating orders
            if pattern_signal>0:
                #trader.create_buy_order(ticker,units)
                order[i,0,1] = 1
            elif pattern_signal<0:
                #trader.create_sell_order(ticker,units)
                order[i,0,2] = 1
            else:
                #print('No trade made')
                order[i,0,0] = 1

            if rsi_signal>70 and adx_signal>50:
                #trader.create_buy_order(ticker,units)
                order[i,1,1] = 1
            elif rsi_signal<30 and adx_signal>50:
                #trader.create_sell_order(ticker,units)
                order[i,1,2] = 1
            else:
                #print('No trade made')
                order[i,1,0] = 1

            if rsi_signal>70:
                #trader.create_buy_order(ticker,units)
                order[i,6,1] = 1
            elif rsi_signal<30:
                #trader.create_sell_order(ticker,units)
                order[i,6,2] = 1
            else:
                #print('No trade made')
                order[i,6,0] = 1

            if (bull[-1]>70 and bear[-1]<30) or (bull[-2]<bear[-2] and bull[-1]>=bear[-1]):
                #trader.create_buy_order(ticker,units)
                order[i,2,1] = 1
            elif (bull[-1]<30 and bear[-1]>70) or (bull[-2]>=bear[-2] and bull[-1]<bear[-1]):
                #trader.create_sell_order(ticker,units)
                order[i,2,2] = 1
            else:
                #print('No trade made')
                order[i,2,0] = 1

            if kama[-1]>=sma[-1] and kama[-2]<sma[-2]:
                #trader.create_buy_order(ticker,units)
                order[i,3,1] = 1
            elif kama[-1]<=sma[-1] and kama[-2]>=sma[-2]:
                #trader.create_sell_order(ticker,units)
                order[i,3,2] = 1
            else:
                #print("No trade made")
                order[i,3,0] = 1

            if DIF[-1]>0 and DEA[-1]>0 and DIF[-2]<DEA[-2] and DIF[-1]>DEA[-1]:
                #trader.create_buy_order(ticker,units)
                order[i,4,1] = 1
            elif DIF[-1]<0 and DEA[-1]<0 and DIF[-2]>DEA[-2] and DIF[-1]<DEA[-1]:
                #trader.create_sell_order(ticker,units)
                order[i,4,2] = 1
            else:
                #print("No trade made")
                order[i,4,0] = 1

            if mfi_signal>70:
                #trader.create_buy_order(ticker,units)
                order[i,5,1] = 1
            elif mfi_signal<30:
                #trader.create_sell_order(ticker,units)
                order[i,5,2] = 1
            else:
                #print('No trade made')
                order[i,5,0] = 1

            if db:
                #write to database here
                pass
        return order,signal