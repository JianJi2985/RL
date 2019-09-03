import time
import json
import os
from datetime import datetime
import talib
from datetime import timedelta
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from mlmodel import *

from database import Database
from Oanda_Trader import *  # AI_trader

# Create actions as None, Buy, Sell, Close All(for that currency pair), Close first(for that currency pair)
# Total actions: 1 + 4*Currency Pairs

#https://stackoverflow.com/questions/10632839/python-transform-list-of-tuples-in-to-1-flat-list-or-1-matrix
#create separate process for each currency pair
#create table for order and signal, columns as algorithm and row as currency pair
# Create a class for them
class ForexTraderSwitch():
    db = None
    accountID = None
    access_token = None
    units = None
    current_path = None
    curr_pair_list = None
    num_curr = None
    moves_allowed = None
    trader = None
    order = None
    signal = None
    current_rate = None
    time_count = None
    curr_pair_history_data = []
        
    #self.model_trader = Oanda_Trader(accountID, access_token)
    def __init__(self):
        pass
    def initial_value(self, accountID, access_token, curr_pair_list = ['EUR_USD'],units = 1,  time_count = [180,720,180]):
        #ForexTraderSwitch.db = Database(connection)
        # account Id to get common data
        ForexTraderSwitch.accountID = accountID
        # access token
        ForexTraderSwitch.access_token = access_token
        ForexTraderSwitch.current_path = os.path.dirname(os.path.abspath(__file__))
        #self.model_info = []
        #self.mlmod = MlModel()
        #self.ticker = ticker
        ForexTraderSwitch.units = units
        ForexTraderSwitch.curr_pair_list = curr_pair_list
        #self.curr_list = list(set('_'.join(c_list).split('_')))
        ForexTraderSwitch.num_curr = len(ForexTraderSwitch.curr_pair_list)
        #self.nav = None
        #self.current_pos = np.zeros((ForexTraderSwitch.num_curr,3))
        ForexTraderSwitch.moves_allowed = np.zeros((4*ForexTraderSwitch.num_curr + 1))
        ForexTraderSwitch.trader = Oanda_Trader(accountID, access_token)
        #self.moves_dict = {0:(0,0),1:(0,1),2:(0,2),3:(0,3),4:(1,0),5:(1,2),6:(1,3)\
        #                   ,7:(2,0),8:(2,1),9:(2,3),10:(3,0),11:(3,1),12:(3,2)}

        # order and signal for all currency pairs
        # 3bc(0),adxrsi(1),arron(2),kamasma(3),macd3(4),mfi(5),rsi(6)
        # None,Buy,Sell
        ForexTraderSwitch.order = np.zeros((ForexTraderSwitch.num_curr,7,3))
        #pattern
        #3 blackcrows [0]
        #adx [1]
        #bear [2,3]
        #bull [4,5]
        #dea [6,7]
        #dif [8,9]
        #kama [10,11]
        #mfi [12]
        #rsi [13]
        #sma [14,15]
        ForexTraderSwitch.signal = np.zeros((ForexTraderSwitch.num_curr,16))
        # output

        # row buy, col sell
        ForexTraderSwitch.current_rate = np.zeros((ForexTraderSwitch.num_curr,1))
        #self.nav = None
        #self.history = None
        ForexTraderSwitch.time_count = time_count
        ForexTraderSwitch.curr_pair_history_data = []
        

    def get_current_time(w):
        return w.strftime("%Y-%m-%dT%H:%M:%SZ")
        
    def get_index(action):
        x = min((action-1)//(ForexTraderSwitch.num_curr-1),(action)//(ForexTraderSwitch.num_curr-1))
        #print(x)
        y = action-(x*(ForexTraderSwitch.num_curr-1))
        if y<=x:
            y-=1
        return (x,y)
    
    # Get history data for all
    def get_history_data():
        ForexTraderSwitch.curr_pair_history_data = []
        for ticker in ForexTraderSwitch.curr_pair_list:
            history = ForexTraderSwitch.trader.get_history(ticker, "M1", ForexTraderSwitch.time_count,\
                                                           ForexTraderSwitch.get_current_time(datetime.utcnow()))
            #ForexTraderSwitch.curr_pair_history_data[ticker] = history
            ForexTraderSwitch.curr_pair_history_data.append(history)
            #ticker_split = ticker.split('_')
            #index_row = self.curr_list.index(ticker_split[0])
            #index_col = self.curr_list.index(ticker_split[1])
            #ForexTraderSwitch.current_rate[index_row,index_col] = history.iloc[-1]['avg']
            #ForexTraderSwitch.current_rate[index_col,index_row] = 1/history.iloc[-1]['avg']
            ForexTraderSwitch.current_rate[ForexTraderSwitch.curr_pair_list.index(ticker)] = history.iloc[-1]['avg']
            
    # Get history data for all
    #def get_curr_pos(self):
    #    for ticker in ForexTraderSwitch.curr_pair_list:
    #        pos = self.model_trader.get_positions(ticker)
    #        self.current_pos[ForexTraderSwitch.curr_pair_list.index(ticker),0] = pos['short']
    #        self.current_pos[ForexTraderSwitch.curr_pair_list.index(ticker),1] = pos['long']
    #        self.current_pos[ForexTraderSwitch.curr_pair_list.index(ticker),2] = pos['unrealizedPL']
            
    #def get_liquid_asset(self):
    #    asset = self.model_trader.get_nav()
    #    self.nav = asset['nav']
    
    
    def three_black_crows():
        for i in range(len(ForexTraderSwitch.curr_pair_list)):
            close=(ForexTraderSwitch.curr_pair_history_data[i]['closeAsk'].tail(5).values+\
                   ForexTraderSwitch.curr_pair_history_data[i]['closeBid'].tail(5).values)/2
            high=(ForexTraderSwitch.curr_pair_history_data[i]['highAsk'].tail(5).values+\
                  ForexTraderSwitch.curr_pair_history_data[i]['highBid'].tail(5).values)/2
            low=(ForexTraderSwitch.curr_pair_history_data[i]['lowAsk'].tail(5).values+\
                 ForexTraderSwitch.curr_pair_history_data[i]['lowBid'].tail(5).values)/2
            openv=(ForexTraderSwitch.curr_pair_history_data[i]['openAsk'].tail(5).values+\
                   ForexTraderSwitch.curr_pair_history_data[i]['openBid'].tail(5).values)/2
            pattern=talib.CDL3BLACKCROWS(openv,high,low,close)
            pattern_signal=pattern[-1]
            #print(pattern_signal)
            ForexTraderSwitch.signal[i,0] = pattern_signal
            if pattern_signal>0:
                #trader.create_buy_order(ticker,units)
                ForexTraderSwitch.order[i,0,1] = 1
            elif pattern_signal<0:
                #trader.create_sell_order(ticker,units)
                ForexTraderSwitch.order[i,0,2] = 1
            else:
                #print('No trade made')
                ForexTraderSwitch.order[i,0,0] = 1


    def adx_rsi():
        for i in range(len(ForexTraderSwitch.curr_pair_list)):
            close=(ForexTraderSwitch.curr_pair_history_data[i]['closeAsk'].values+\
                   ForexTraderSwitch.curr_pair_history_data[i]['closeBid'].values)/2
            high=(ForexTraderSwitch.curr_pair_history_data[i]['highAsk'].values+\
                  ForexTraderSwitch.curr_pair_history_data[i]['highBid'].values)/2
            low=(ForexTraderSwitch.curr_pair_history_data[i]['lowAsk'].values+\
                 ForexTraderSwitch.curr_pair_history_data[i]['lowBid'].values)/2
            #open=(history['openAsk']+history['openBid'])/2
            adx=talib.ADX(high,low,close,timeperiod=14)
            rsi=talib.RSI(close,timeperiod=14)
            adx_signal=adx[-1]
            rsi_signal=rsi[-1]
            ForexTraderSwitch.signal[i,1] = adx_signal
            ForexTraderSwitch.signal[i,13] = rsi_signal
            #print("ADX: %s RSI: %s"%(adx_signal,rsi_signal))
            if rsi_signal>70 and adx_signal>50:
                #trader.create_buy_order(ticker,units)
                ForexTraderSwitch.order[i,1,1] = 1
            elif rsi_signal<30 and adx_signal>50:
                #trader.create_sell_order(ticker,units)
                ForexTraderSwitch.order[i,1,2] = 1
            else:
                #print('No trade made')
                ForexTraderSwitch.order[i,1,0] = 1
                
            if rsi_signal>70:
                #trader.create_buy_order(ticker,units)
                ForexTraderSwitch.order[i,6,1] = 1
            elif rsi_signal<30:
                #trader.create_sell_order(ticker,units)
                ForexTraderSwitch.order[i,6,2] = 1
            else:
                #print('No trade made')
                ForexTraderSwitch.order[i,6,0] = 1
   
    def aroon():
        for i in range(len(ForexTraderSwitch.curr_pair_list)):
            high=(ForexTraderSwitch.curr_pair_history_data[i]['highAsk'].values+\
                  ForexTraderSwitch.curr_pair_history_data[i]['highBid'].values)/2
            low=(ForexTraderSwitch.curr_pair_history_data[i]['lowAsk'].values+\
                 ForexTraderSwitch.curr_pair_history_data[i]['lowBid'].values)/2
            #open=(history['openAsk']+history['openBid'])/2
            bull, bear=talib.AROON(high,low,timeperiod=14)
            #print("Bull %s Bear %s" % (bull[-1], bear[-1]))
            ForexTraderSwitch.signal[i,2] = bear[-2]
            ForexTraderSwitch.signal[i,3] = bear[-1]
            ForexTraderSwitch.signal[i,4] = bull[-2]
            ForexTraderSwitch.signal[i,5] = bull[-1]
            if (bull[-1]>70 and bear[-1]<30) or (bull[-2]<bear[-2] and bull[-1]>=bear[-1]):
                #trader.create_buy_order(ticker,units)
                ForexTraderSwitch.order[i,2,1] = 1
            elif (bull[-1]<30 and bear[-1]>70) or (bull[-2]>=bear[-2] and bull[-1]<bear[-1]):
                #trader.create_sell_order(ticker,units)
                ForexTraderSwitch.order[i,2,2] = 1
            else:
                #print('No trade made')
                ForexTraderSwitch.order[i,2,0] = 1
        
    def kama_sma():
        for i in range(len(ForexTraderSwitch.curr_pair_list)):
            close=(ForexTraderSwitch.curr_pair_history_data[i]['closeAsk'].values+\
                   ForexTraderSwitch.curr_pair_history_data[i]['closeBid'].values)/2

            sma=talib.SMA(close,timeperiod=30)
            kama=talib.KAMA(close,timeperiod=30)
            #print("SMA %s KAMA %s" % (sma[-1], kama[-1]))
            ForexTraderSwitch.signal[i,10] = kama[-2]
            ForexTraderSwitch.signal[i,11] = kama[-1]
            ForexTraderSwitch.signal[i,14] = sma[-2]
            ForexTraderSwitch.signal[i,15] = sma[-1]
            if kama[-1]>=sma[-1] and kama[-2]<sma[-2]:
                #trader.create_buy_order(ticker,units)
                ForexTraderSwitch.order[i,3,1] = 1
            elif kama[-1]<=sma[-1] and kama[-2]>=sma[-2]:
                #trader.create_sell_order(ticker,units)
                ForexTraderSwitch.order[i,3,2] = 1
            else:
                #print("No trade made")
                ForexTraderSwitch.order[i,3,0] = 1
    
    def macd3():
        for i in range(len(ForexTraderSwitch.curr_pair_list)):
            close=(ForexTraderSwitch.curr_pair_history_data[i]['closeAsk'].values+\
                   ForexTraderSwitch.curr_pair_history_data[i]['closeBid'].values)/2

            DIF,DEA,BAR=talib.MACDFIX(close,signalperiod=9)
            #print("DIF %s DEA %s" % (DIF[-1], DEA[-1]))
            ForexTraderSwitch.signal[i,6] = DEA[-2]
            ForexTraderSwitch.signal[i,7] = DEA[-1]
            ForexTraderSwitch.signal[i,8] = DIF[-2]
            ForexTraderSwitch.signal[i,9] = DIF[-1]
            if DIF[-1]>0 and DEA[-1]>0 and DIF[-2]<DEA[-2] and DIF[-1]>DEA[-1]:
                #trader.create_buy_order(ticker,units)
                ForexTraderSwitch.order[i,4,1] = 1
            elif DIF[-1]<0 and DEA[-1]<0 and DIF[-2]>DEA[-2] and DIF[-1]<DEA[-1]:
                #trader.create_sell_order(ticker,units)
                ForexTraderSwitch.order[i,4,2] = 1
            else:
                #print("No trade made")
                ForexTraderSwitch.order[i,4,0] = 1
    
    def mfi():
        for i in range(len(ForexTraderSwitch.curr_pair_list)):
            close=(ForexTraderSwitch.curr_pair_history_data[i]['closeAsk'].values+\
                   ForexTraderSwitch.curr_pair_history_data[i]['closeBid'].values)/2
            high=(ForexTraderSwitch.curr_pair_history_data[i]['highAsk'].values+\
                  ForexTraderSwitch.curr_pair_history_data[i]['highBid'].values)/2
            low=(ForexTraderSwitch.curr_pair_history_data[i]['lowAsk'].values+\
                 ForexTraderSwitch.curr_pair_history_data[i]['lowBid'].values)/2
            #open=(history['openAsk']+history['openBid'])/2
            mfi=talib.MFI(high,low,close,history['volume'].values.astype(float),timeperiod=14)
            mfi_signal=mfi[-1]
            #print(mfi_signal)
            ForexTraderSwitch.signal[i,12] = mfi_signal
            if mfi_signal>70:
                #trader.create_buy_order(ticker,units)
                ForexTraderSwitch.order[i,5,1] = 1
            elif mfi_signal<30:
                #trader.create_sell_order(ticker,units)
                ForexTraderSwitch.order[i,5,2] = 1
            else:
                #print('No trade made')
                ForexTraderSwitch.order[i,5,0] = 1
         
    def set_order_signal():
        #see above class definition
        ForexTraderSwitch.get_history_data()
        ForexTraderSwitch.three_black_crows()
        ForexTraderSwitch.adx_rsi()
        ForexTraderSwitch.aroon()
        ForexTraderSwitch.kama_sma()
        ForexTraderSwitch.macd3()
        ForexTraderSwitch.mfi()
        
    def all_algo(i,db=False):
        
        # Processing data
        close5=(ForexTraderSwitch.curr_pair_history_data[i]['closeAsk'].tail(5).values+\
                ForexTraderSwitch.curr_pair_history_data[i]['closeBid'].tail(5).values)/2
        high5=(ForexTraderSwitch.curr_pair_history_data[i]['highAsk'].tail(5).values+\
               ForexTraderSwitch.curr_pair_history_data[i]['highBid'].tail(5).values)/2
        low5=(ForexTraderSwitch.curr_pair_history_data[i]['lowAsk'].tail(5).values+\
              ForexTraderSwitch.curr_pair_history_data[i]['lowBid'].tail(5).values)/2
        openv5=(ForexTraderSwitch.curr_pair_history_data[i]['openAsk'].tail(5).values+\
                ForexTraderSwitch.curr_pair_history_data[i]['openBid'].tail(5).values)/2

        close=(ForexTraderSwitch.curr_pair_history_data[i]['closeAsk'].values+\
               ForexTraderSwitch.curr_pair_history_data[i]['closeBid'].values)/2
        high=(ForexTraderSwitch.curr_pair_history_data[i]['highAsk'].values+\
              ForexTraderSwitch.curr_pair_history_data[i]['highBid'].values)/2
        low=(ForexTraderSwitch.curr_pair_history_data[i]['lowAsk'].values+\
             ForexTraderSwitch.curr_pair_history_data[i]['lowBid'].values)/2
        openv=(ForexTraderSwitch.curr_pair_history_data[i]['openAsk'].values+\
               ForexTraderSwitch.curr_pair_history_data[i]['openBid'].values)/2

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

        mfi=talib.MFI(high,low,close,ForexTraderSwitch.curr_pair_history_data[i]['volume'].values.astype(float),timeperiod=14)
        mfi_signal=mfi[-1]

        # Storing signals
        ForexTraderSwitch.signal[i,0] = pattern_signal
        ForexTraderSwitch.signal[i,1] = adx_signal
        ForexTraderSwitch.signal[i,13] = rsi_signal
        ForexTraderSwitch.signal[i,2] = bear[-2]
        ForexTraderSwitch.signal[i,3] = bear[-1]
        ForexTraderSwitch.signal[i,4] = bull[-2]
        ForexTraderSwitch.signal[i,5] = bull[-1]
        ForexTraderSwitch.signal[i,10] = kama[-2]
        ForexTraderSwitch.signal[i,11] = kama[-1]
        ForexTraderSwitch.signal[i,14] = sma[-2]
        ForexTraderSwitch.signal[i,15] = sma[-1]
        ForexTraderSwitch.signal[i,6] = DEA[-2]
        ForexTraderSwitch.signal[i,7] = DEA[-1]
        ForexTraderSwitch.signal[i,8] = DIF[-2]
        ForexTraderSwitch.signal[i,9] = DIF[-1]
        ForexTraderSwitch.signal[i,12] = mfi_signal

        # Creating orders
        if pattern_signal>0:
            #trader.create_buy_order(ticker,units)
            ForexTraderSwitch.order[i,0,1] = 1
        elif pattern_signal<0:
            #trader.create_sell_order(ticker,units)
            ForexTraderSwitch.order[i,0,2] = 1
        else:
            #print('No trade made')
            ForexTraderSwitch.order[i,0,0] = 1

        if rsi_signal>70 and adx_signal>50:
            #trader.create_buy_order(ticker,units)
            ForexTraderSwitch.order[i,1,1] = 1
        elif rsi_signal<30 and adx_signal>50:
            #trader.create_sell_order(ticker,units)
            ForexTraderSwitch.order[i,1,2] = 1
        else:
            #print('No trade made')
            ForexTraderSwitch.order[i,1,0] = 1

        if rsi_signal>70:
            #trader.create_buy_order(ticker,units)
            ForexTraderSwitch.order[i,6,1] = 1
        elif rsi_signal<30:
            #trader.create_sell_order(ticker,units)
            ForexTraderSwitch.order[i,6,2] = 1
        else:
            #print('No trade made')
            ForexTraderSwitch.order[i,6,0] = 1

        if (bull[-1]>70 and bear[-1]<30) or (bull[-2]<bear[-2] and bull[-1]>=bear[-1]):
            #trader.create_buy_order(ticker,units)
            ForexTraderSwitch.order[i,2,1] = 1
        elif (bull[-1]<30 and bear[-1]>70) or (bull[-2]>=bear[-2] and bull[-1]<bear[-1]):
            #trader.create_sell_order(ticker,units)
            ForexTraderSwitch.order[i,2,2] = 1
        else:
            #print('No trade made')
            ForexTraderSwitch.order[i,2,0] = 1

        if kama[-1]>=sma[-1] and kama[-2]<sma[-2]:
            #trader.create_buy_order(ticker,units)
            ForexTraderSwitch.order[i,3,1] = 1
        elif kama[-1]<=sma[-1] and kama[-2]>=sma[-2]:
            #trader.create_sell_order(ticker,units)
            ForexTraderSwitch.order[i,3,2] = 1
        else:
            #print("No trade made")
            ForexTraderSwitch.order[i,3,0] = 1

        if DIF[-1]>0 and DEA[-1]>0 and DIF[-2]<DEA[-2] and DIF[-1]>DEA[-1]:
            #trader.create_buy_order(ticker,units)
            ForexTraderSwitch.order[i,4,1] = 1
        elif DIF[-1]<0 and DEA[-1]<0 and DIF[-2]>DEA[-2] and DIF[-1]<DEA[-1]:
            #trader.create_sell_order(ticker,units)
            ForexTraderSwitch.order[i,4,2] = 1
        else:
            #print("No trade made")
            ForexTraderSwitch.order[i,4,0] = 1

        if mfi_signal>70:
            #trader.create_buy_order(ticker,units)
            ForexTraderSwitch.order[i,5,1] = 1
        elif mfi_signal<30:
            #trader.create_sell_order(ticker,units)
            ForexTraderSwitch.order[i,5,2] = 1
        else:
            #print('No trade made')
            ForexTraderSwitch.order[i,5,0] = 1

        if db:
            #write to database here
            pass
    # Get result of all
    def set_order_signal_db(set_sig = False):
        
        ForexTraderSwitch.get_history_data()
        if set_sig:
            for i in range(len(ForexTraderSwitch.curr_pair_list)):
                ForexTraderSwitch.all_algo(i,db=False)
    
    def ml_single_dense_blind_switch_algo(self):
        inp_size = ForexTraderSwitch.order.size + self.current_pos.size + 1 # for total value in usd
        out_size = ForexTraderSwitch.moves_allowed.size
        design = {'layers':[{'Dense':{'units':inp_size*2, 'activation':'relu', 'input_dim':inp_size}},\
                            {'Dense':{'units':out_size, 'activation':'softmax'}}],\
                  'compile':{'optimizer':'rmsprop','loss':'categorical_crossentropy','metrics':['accuracy']}}
        
        model = self.mlmod.create_seq_model(design)
        mid = self.mlmod.mid
        self.model_info.append((mid,inp_size,out_size,'Single','Dense','Blind','Switch','Algo',design,location))
        location = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.h5'
        model.save(location)
        img_loc = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.png'
        plot_model(model, to_file=img_loc, show_shapes = True, show_layer_names = False)
    def ml_twin_dense_blind_switch_algo(self):
        inp_size = ForexTraderSwitch.order.size + self.current_pos.size + 1 # for total value in usd
        out_size = (ForexTraderSwitch.moves_allowed.size + 1)/2
        design = {'layers':[{'Dense':{'units':inp_size*2, 'activation':'relu', 'input_dim':inp_size}},\
                            {'Dense':{'units':out_size, 'activation':'softmax'}}],\
                  'compile':{'optimizer':'rmsprop','loss':'categorical_crossentropy','metrics':['accuracy']}}
        
        model = self.mlmod.create_seq_model(design)
        mid = self.mlmod.mid
        self.model_info.append((mid,inp_size,out_size,'Twin','Dense','Blind','Switch','Algo',design,location))
        location = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.h5'
        model.save(location)
        img_loc = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.png'
        plot_model(model, to_file=img_loc, show_shapes = True, show_layer_names = False)
        
        model = self.mlmod.create_seq_model(design)
        mid = self.mlmod.mid
        self.model_info.append((mid,inp_size,out_size,'Twin','Dense','Blind','Switch','Algo',design,location))
        location = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.h5'
        model.save(location)
        img_loc = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.png'
        plot_model(model, to_file=img_loc, show_shapes = True, show_layer_names = False)
        
    def ml_single_dense_current_switch_algo(self):
        inp_size = ForexTraderSwitch.signal.size + ForexTraderSwitch.current_rate.size + self.current_pos.size + 1 # for total value in usd
        out_size = ForexTraderSwitch.moves_allowed.size
        design = {'layers':[{'Dense':{'units':inp_size*2, 'activation':'relu', 'input_dim':inp_size}},\
                            {'Dense':{'units':out_size, 'activation':'softmax'}}],\
                  'compile':{'optimizer':'rmsprop','loss':'categorical_crossentropy','metrics':['accuracy']}}
        
        model = self.mlmod.create_seq_model(design)
        mid = self.mlmod.mid
        self.model_info.append((mid,inp_size,out_size,'Single','Dense','Current','Switch','Algo',design,location))
        location = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.h5'
        model.save(location)
        img_loc = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.png'
        plot_model(model, to_file=img_loc, show_shapes = True, show_layer_names = False)

    def ml_twin_dense_current_switch_algo(self):
        inp_size = ForexTraderSwitch.signal.size + ForexTraderSwitch.current_rate.size + self.current_pos.size + 1 # for total value in usd
        out_size = (ForexTraderSwitch.moves_allowed.size + 1)/2
        design = {'layers':[{'Dense':{'units':inp_size*2, 'activation':'relu', 'input_dim':inp_size}},\
                            {'Dense':{'units':out_size, 'activation':'softmax'}}],\
                  'compile':{'optimizer':'rmsprop','loss':'categorical_crossentropy','metrics':['accuracy']}}
        
        model = self.mlmod.create_seq_model(design)
        mid = self.mlmod.mid
        self.model_info.append((mid,inp_size,out_size,'Single','Twin','Current','Switch','Algo',design,location))
        location = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.h5'
        model.save(location)
        img_loc = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.png'
        plot_model(model, to_file=img_loc, show_shapes = True, show_layer_names = False)
        
        model = self.mlmod.create_seq_model(design)
        mid = self.mlmod.mid
        self.model_info.append((mid,inp_size,out_size,'Single','Twin','Current','Switch','Algo',design,location))
        location = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.h5'
        model.save(location)
        img_loc = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.png'
        plot_model(model, to_file=img_loc, show_shapes = True, show_layer_names = False)
        
    def ml_single_dense_current_net_scratch(self):
        inp_size = ForexTraderSwitch.current_rate.size + self.current_pos.size + 1 # for total value in usd
        out_size = ForexTraderSwitch.moves_allowed.size
        design = {'layers':[{'Dense':{'units':inp_size*2, 'activation':'relu', 'input_dim':inp_size}},\
                            {'Dense':{'units':out_size, 'activation':'softmax'}}],\
                  'compile':{'optimizer':'rmsprop','loss':'categorical_crossentropy','metrics':['accuracy']}}
        
        model = self.mlmod.create_seq_model(design)
        mid = self.mlmod.mid
        self.model_info.append((mid,inp_size,out_size,'Single','Dense','Current','Net','Scratch',design,location))
        location = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.h5'
        model.save(location)
        img_loc = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.png'
        plot_model(model, to_file=img_loc, show_shapes = True, show_layer_names = False)
        
    def ml_twin_dense_current_net_scratch(self):
        inp_size = ForexTraderSwitch.current_rate.size + self.current_pos.size + 1 # for total value in usd
        out_size = (ForexTraderSwitch.moves_allowed.size + 1)/2
        design = {'layers':[{'Dense':{'units':inp_size*2, 'activation':'relu', 'input_dim':inp_size}},\
                            {'Dense':{'units':out_size, 'activation':'softmax'}}],\
                  'compile':{'optimizer':'rmsprop','loss':'categorical_crossentropy','metrics':['accuracy']}}
        
        model = self.mlmod.create_seq_model(design)
        mid = self.mlmod.mid
        self.model_info.append((mid,inp_size,out_size,'Twin','Dense','Current','Net','Scratch',design,location))
        location = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.h5'
        model.save(location)
        img_loc = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.png'
        plot_model(model, to_file=img_loc, show_shapes = True, show_layer_names = False)
        
        model = self.mlmod.create_seq_model(design)
        mid = self.mlmod.mid
        self.model_info.append((mid,inp_size,out_size,'Twin','Dense','Current','Net','Scratch',design,location))
        location = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.h5'
        model.save(location)
        img_loc = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.png'
        plot_model(model, to_file=img_loc, show_shapes = True, show_layer_names = False)
        
    def ml_single_lstm_last_net_scratch(self):
        out_size = ForexTraderSwitch.moves_allowed.size
        
        forex_input = Input(shape = (ForexTraderSwitch.time_count,len(ForexTraderSwitch.curr_pair_list)), name = "forex_input")
        lstm_out = LSTM(len(ForexTraderSwitch.curr_pair_list)*2)(forex_input)
        state_input = Input(shape = (self.current_pos.size + 1,),name = "state_input")
        
        x = keras.layers.concatenate([lstm_out, state_input])
        x = Dense(len(ForexTraderSwitch.curr_pair_list), activation = 'relu')(x)
        output = Dense(out_size, activation = 'softmax')(x)
        
        model = Model(inputs = [forex_input, state_input], outputs = output)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        mid = int(time.time())
        self.model_info.append((mid,'NA',out_size,'Single','LSTM','Last','Net','Scratch',design,location))
        location = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.h5'
        model.save(location)
        img_loc = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.png'
        plot_model(model, to_file=img_loc, show_shapes = True, show_layer_names = False)
        
    def ml_twin_lstm_last_net_scratch(self):
        out_size = (ForexTraderSwitch.moves_allowed.size + 1)/2
        
        forex_input = Input(shape = (ForexTraderSwitch.time_count,len(ForexTraderSwitch.curr_pair_list)), name = "forex_input")
        lstm_out = LSTM(len(ForexTraderSwitch.curr_pair_list)*2)(forex_input)
        state_input = Input(shape = (self.current_pos.size + 1,),name = "state_input")
        
        x = keras.layers.concatenate([lstm_out, state_input])
        x = Dense(len(ForexTraderSwitch.curr_pair_list), activation = 'relu')(x)
        output = Dense(out_size, activation = 'softmax')(x)
        
        model = Model(inputs = [forex_input, state_input], outputs = output)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        mid = int(time.time())
        self.model_info.append((mid,'NA',out_size,'Twin','LSTM','Last','Net','Scratch',design,location))
        location = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.h5'
        model.save(location)
        img_loc = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.png'
        plot_model(model, to_file=img_loc, show_shapes = True, show_layer_names = False)
        
        time.sleep(1)
        
        model = Model(inputs = [forex_input, state_input], outputs = output)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        mid = int(time.time())
        self.model_info.append((mid,'NA',out_size,'Twin','LSTM','Last','Net','Scratch',design,location))
        location = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.h5'
        model.save(location)
        img_loc = ForexTraderSwitch.current_path +'\\models\\'+str(mid)+'.png'
        plot_model(model, to_file=img_loc, show_shapes = True, show_layer_names = False)

# Combine results
# Use ML model
# Send ML result
