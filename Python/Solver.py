import os, sys
import time
import datetime
import pandas as pd
import numpy as np
import math
from math import radians, cos, sin, asin, sqrt 
import random
import copy

class Dynamic_programming(object):
    
    def __init__(self,Grids,State,Action,End_step,Matching_Prob,Pickup_Prob,Dest_Prob,Travel_time):
        
        '''Param'''

        self.Grids=Grids.tolist()

        self.State=State

        self.Action=Action

        self.End_step=End_step

        self.gamma=0.8

        '''Reward'''

        self.V_table={state:0.0 for state in self.State}

        self.Q_table={}

        for state in self.State:

            self.Q_table[state]={}

            for action in self.Action[state]:

                self.Q_table[state][action]=0.0

        '''Prob'''

        self.Matching_Prob=Matching_Prob
        
        self.Pickup_Prob=Pickup_Prob

        self.Dest_Prob=Dest_Prob
        
        '''Travel Time'''
        
        self.Travel_time=Travel_time

        '''Iteration'''

        self.Policy={state:state.split('-')[0] for state in self.State}

        self.Backwards()
        
    def Backwards(self):
        
        for t in range(self.End_step-2,-1,-1):
            
            print('*'*50)
            
            for grid in self.Grids:
                
                state=grid+'-'+str(int(t))
                
                print('Current step: ',t, 'Total Index: ',len(self.Grids)-1, 'Current Index: ',self.Grids.index(grid))
                
                for action in self.Action[state]:
                    
                     self.Q_table[state][action]= self.Compute_Q(state,action)
                        
                self.V_table[state]=max(self.Q_table[state].values())
                
                self.Policy[state]=max(self.Q_table[state], key=self.Q_table[state].get)
                    
    def Compute_Q(self,state,action):
        
        grid=state.split('-')[0]
        
        step=int(state.split('-')[1])
        
        action_tep=step+max(int(self.Travel_time[grid][action]),1)
        
        if action_tep<self.End_step:
        
            action_state=action+'-'+str(action_tep)

            p_m=self.Matching_Prob[action_state]

            '''Matching'''

            r_1=1

            if len(self.Pickup_Prob[action])!=0:

                for pickup_grid in self.Pickup_Prob[action]:

                    p_pickup=self.Pickup_Prob[action][pickup_grid]

                    if len(self.Dest_Prob[pickup_grid])!=0:

                        for dest_grid in self.Dest_Prob[pickup_grid]:

                            p_dest=self.Dest_Prob[pickup_grid][dest_grid]

                            dest_step=action_tep+max(int(self.Travel_time[action][pickup_grid]),1)+max(int(self.Travel_time[pickup_grid][dest_grid]),1)

                            dest_state=dest_grid+'-'+str(dest_step)

                            if dest_step<self.End_step:

                                r_1+=self.gamma**(action_tep-step)*(p_pickup*p_dest*self.V_table[dest_state])


            '''Not matching'''

            r_2=0

            r_2+=self.gamma*self.V_table[action_state]

            '''Expected reward'''

            r=r_1*p_m+(1-p_m)*r_2
            
        else:
            
            r=0.0
        
        return r

if __name__ == '__main__':

    Load_path='../Data/NYC_Network/'

    Save_path='../Data/NYC_MDP/'

    '''Load data'''

    '''Grid-related data'''

    Grids=np.load(os.path.join(Load_path,'Grids.npy'),allow_pickle=True)

    '''MDP data'''

    State=np.load(os.path.join(Save_path,'State.npy'),allow_pickle=True)

    Action=np.load(os.path.join(Save_path,'Action.npy'),allow_pickle=True).item()

    Matching_Prob=np.load(os.path.join(Save_path,'Matching_Prob.npy'),allow_pickle=True).item()

    Pickup_Prob=np.load(os.path.join(Save_path,'Pickup_Prob.npy'),allow_pickle=True).item()

    Dest_Prob=np.load(os.path.join(Save_path,'Dest_Prob.npy'),allow_pickle=True).item()

    Travel_time=np.load(os.path.join(Save_path,'Travel_time.npy'),allow_pickle=True).item()

    End_step=180

    DP=Dynamic_programming(Grids,State,Action,End_step,Matching_Prob,Pickup_Prob,Dest_Prob,Travel_time)

    np.save(os.path.join(Save_path,'Policy'),DP.Policy)

    np.save(os.path.join(Save_path,'Q_table'),DP.Q_table)

    np.save(os.path.join(Save_path,'V_table'),DP.V_table)




    