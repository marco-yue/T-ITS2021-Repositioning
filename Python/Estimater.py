import numpy as np

import pandas as pd

import copy

import os,sys

from shapely.geometry import Point, Polygon

import h3


def Get_prob(ratio):
        
    prob = 1- np.exp(-1*ratio)

    return prob


if __name__ == '__main__':


    Policy="Random_walk"

    Daily_path='../Data/NYC_Results/'

    Load_path='../Data/NYC_Network/'

    Save_path='../Data/NYC_MDP/'

    '''Load data'''

    '''Grid-related data'''

    Grid_list=np.load(os.path.join(Load_path,'Grids.npy'),allow_pickle=True)

    Grid_Point=np.load(os.path.join(Load_path,'Grid_Point.npy'),allow_pickle=True).item()

    '''Point-related data'''

    Points_list=np.load(os.path.join(Load_path,'Points_list.npy'),allow_pickle=True)

    Link_Point=np.load(os.path.join(Load_path,'Link_Point.npy'),allow_pickle=True).item()

    Point_coordinate=np.load(os.path.join(Load_path,'Point_coordinate.npy'),allow_pickle=True).item()

    Point_Grid=np.load(os.path.join(Load_path,'Point_Grid.npy'),allow_pickle=True).item()

    '''State'''

    State=np.load(os.path.join(Save_path,'State.npy'),allow_pickle=True)

    End_step=180

    max_waiting=12

    Order_DF=pd.DataFrame([],columns=['Order_id', 'Driver_id', 'Arrive_step', 'Response_step', 'Pickup_step',
       'Pickup_Latitude', 'Pickup_Longitude', 'Dropoff_Latitude',
       'Dropoff_Longitude', 'Pickup_Grid', 'Dropoff_Grid', 'Pickup_Point',
       'Dropoff_Point', 'Travel_dis', 'Travel_time'])

    Driver_DF=pd.DataFrame([],columns=['Driver_id', 'Order_id', 'Step', 'Point', 'Grid', 'Reposition_Point'])

    for day in range(1,6,1):

        print('*'*60)

        print('Day: ',day)

        Order_df=pd.read_csv(os.path.join(Daily_path,'Order_df_'+Policy+'_'+str(day)+'.csv'))

        Order_df=Order_df.drop(columns=['Unnamed: 0'])

        Driver_df=pd.read_csv(os.path.join(Daily_path,'Driver_df_'+Policy+'_'+str(day)+'.csv'))

        Driver_df=Driver_df.drop(columns=['Unnamed: 0'])
        
        Order_DF=pd.concat([Order_DF,Order_df],ignore_index=True)
        
        Driver_DF=pd.concat([Driver_DF,Driver_df],ignore_index=True)

    Order_DF=Order_DF[['Order_id','Driver_id','Arrive_step','Response_step','Pickup_Grid','Dropoff_Grid']]

    Added_item={'Order_id': list(),\
                'Driver_id':list(),\
                'Arrive_step':list(),\
                'Response_step':list(),\
                'Pickup_Grid':list(),\
                'Dropoff_Grid':list()}

    for idx,row in Order_DF.iterrows():

        print(idx, Order_DF.shape[0])

        if row['Driver_id']!='Canceled':

            if row['Response_step']>row['Arrive_step']:
            
                for step in range(row['Arrive_step']+1,row['Response_step']+1,1):

                    Added_item['Order_id'].append(row['Order_id']);
                    Added_item['Driver_id'].append(row['Driver_id']);
                    Added_item['Arrive_step'].append(step);
                    Added_item['Response_step'].append(row['Response_step']);
                    Added_item['Pickup_Grid'].append(row['Pickup_Grid']);
                    Added_item['Dropoff_Grid'].append(row['Dropoff_Grid']);


        else:

            for step in range(row['Arrive_step']+1,row['Response_step']+1+max_waiting,1):

                Added_item['Order_id'].append(row['Order_id']);
                Added_item['Driver_id'].append(row['Driver_id']);
                Added_item['Arrive_step'].append(step);
                Added_item['Response_step'].append(row['Response_step']);
                Added_item['Pickup_Grid'].append(row['Pickup_Grid']);
                Added_item['Dropoff_Grid'].append(row['Dropoff_Grid']);

                
    Added_df=pd.DataFrame(data=Added_item)

    Order_DF=pd.concat([Order_DF,Added_df],ignore_index=True)

    Order_DF.to_csv(os.path.join(Save_path,'Order_DF.csv'))

    print('Starting Matching Probability')

    '''Available Orders'''

    Order_CNT=copy.copy(Order_DF)

    Order_CNT['step']=Order_CNT.apply(lambda x:int((x['Arrive_step']-2520)/6),axis=1)

    Order_CNT=Order_CNT.drop_duplicates(subset=['step','Order_id'])

    Order_CNT=Order_CNT.groupby(['step','Pickup_Grid']).count()

    Order_CNT['Transition']=Order_CNT.index

    Order_CNT['Step']=Order_CNT.apply(lambda x:x['Transition'][0],axis=1)

    Order_CNT['Grid']=Order_CNT.apply(lambda x:x['Transition'][1],axis=1)

    Order_CNT=Order_CNT.reset_index(drop=True)

    Order_CNT=Order_CNT.rename(index=str, columns={'Order_id': 'Order_Cnt'})

    Order_CNT=Order_CNT[['Step','Grid','Order_Cnt']]

    '''Idle Drivers'''

    Driver_CNT=Driver_DF.loc[Driver_DF['Step']<=3600]

    Driver_CNT['step']=Driver_CNT.apply(lambda x:int((x['Step']-2520)/6),axis=1)

    Driver_CNT=Driver_CNT.drop_duplicates(subset=['step','Driver_id'])

    Driver_CNT=Driver_CNT.groupby(['step','Grid']).count()

    Driver_CNT['Transition']=Driver_CNT.index

    Driver_CNT['Step']=Driver_CNT.apply(lambda x:x['Transition'][0],axis=1)

    Driver_CNT['Grid']=Driver_CNT.apply(lambda x:x['Transition'][1],axis=1)

    Driver_CNT=Driver_CNT.reset_index(drop=True)

    Driver_CNT=Driver_CNT.rename(index=str, columns={'Driver_id': 'Driver_Cnt'})

    Driver_CNT=Driver_CNT[['Step','Grid','Driver_Cnt']]

    '''Overall Demand supply'''

    Demand_supply=Driver_CNT.merge(Order_CNT,on=['Step','Grid'],how='left')

    Demand_supply=Demand_supply.fillna(0)

    Demand_supply['ratio']=Demand_supply.apply(lambda x:x['Order_Cnt']/x['Driver_Cnt'],axis=1)

    Demand_supply['Prob']=Demand_supply.apply(lambda x:Get_prob(x['ratio']),axis=1)

    Demand_supply


    '''Matching Prob'''

    Matching_Prob={s:0.0 for s in State}

    for idx,row in Demand_supply.iterrows():
        
        state=row['Grid']+'-'+str(int(row['Step']))
        
        Matching_Prob[state]=row['Prob']

    np.save(os.path.join(Save_path,'Matching_Prob'),Matching_Prob)

    '''Pickup Prob'''

    print('Starting Pickup Probability')

    Order_CNT=copy.copy(Order_DF)

    Driver_CNT=copy.copy(Driver_DF.loc[(Driver_DF['Step']<=3600)& (Driver_DF['Order_id']!='Idle')])

    Driver_CNT['step']=Driver_CNT.apply(lambda x:int((x['Step']-2520)/6),axis=1)

    Pickup_df=Driver_CNT.merge(Order_CNT[['Order_id','Response_step','Pickup_Grid']],on=['Order_id'],how='left')

    Pickup_df=Pickup_df[['step','Grid','Pickup_Grid','Driver_id']]

    Pickup_CNT=Pickup_df.groupby(by=['Grid','Pickup_Grid']).count()

    Pickup_CNT['Transition']=Pickup_CNT.index

    Pickup_CNT['Grid']=Pickup_CNT.apply(lambda x:x['Transition'][0],axis=1)

    Pickup_CNT['Pickup_Grid']=Pickup_CNT.apply(lambda x:x['Transition'][1],axis=1)

    Pickup_CNT=Pickup_CNT.reset_index(drop=True)

    Pickup_CNT=Pickup_CNT.rename(index=str, columns={'Driver_id': 'Order_Num'})

    Pickup_CNT=Pickup_CNT[['Grid','Pickup_Grid','Order_Num']]

    Temp=Pickup_df.groupby(by=['Grid']).count()

    Temp['Grid']=Temp.index

    Temp=Temp.reset_index(drop=True)

    Temp=Temp.rename(index=str, columns={'Driver_id': 'Sum'})

    Temp=Temp[['Grid','Sum']]

    Pickup_CNT=Pickup_CNT.merge(Temp,on=['Grid'],how='left')

    Pickup_CNT['Prob']=Pickup_CNT.apply(lambda x:x['Order_Num']/x['Sum'],axis=1)

    '''Pickup Prob'''

    Pickup_Prob={g:{} for g in Grid_list}

    for idx,row in Pickup_CNT.iterrows():
        
        grid=row['Grid']
        
        Pickup_Prob[grid][row['Pickup_Grid']]=row['Prob']


    np.save(os.path.join(Save_path,'Pickup_Prob'),Pickup_Prob)

    '''Destination Prob'''

    print('Starting Destination Probability')

    Order_CNT=copy.copy(Order_DF)

    Dest_CNT=Order_CNT.groupby(by=['Pickup_Grid','Dropoff_Grid']).count()

    Dest_CNT['Transition']=Dest_CNT.index

    Dest_CNT['Pickup_Grid']=Dest_CNT.apply(lambda x:x['Transition'][0],axis=1)

    Dest_CNT['Dropoff_Grid']=Dest_CNT.apply(lambda x:x['Transition'][1],axis=1)

    Dest_CNT=Dest_CNT.reset_index(drop=True)

    Dest_CNT=Dest_CNT.rename(index=str, columns={'Driver_id': 'Order_Num'})

    Dest_CNT=Dest_CNT[['Pickup_Grid','Dropoff_Grid','Order_Num']]

    Temp=Order_CNT.groupby(by=['Pickup_Grid']).count()

    Temp['Pickup_Grid']=Temp.index

    Temp=Temp.reset_index(drop=True)

    Temp=Temp.rename(index=str, columns={'Driver_id': 'Sum'})

    Temp=Temp[['Pickup_Grid','Sum']]

    Dest_CNT=Dest_CNT.merge(Temp,on=['Pickup_Grid'],how='left')

    Dest_CNT['Prob']=Dest_CNT.apply(lambda x:x['Order_Num']/x['Sum'],axis=1)

    Dest_CNT

    '''Dest Prob'''

    Dest_Prob={g:{} for g in Grid_list}

    for idx,row in Dest_CNT.iterrows():
        
        grid=row['Pickup_Grid']
        
        Dest_Prob[grid][row['Dropoff_Grid']]=row['Prob']

    np.save(os.path.join(Save_path,'Dest_Prob'),Dest_Prob)

    '''Travel time'''

    print('Starting Travel time')

    Travel_time={g:{g:0 for g in Grid_list} for g in Grid_list}

    for g in Grid_list:
        for g_ in Grid_list:
            g=str(g)
            g_=str(g_)
            
            Travel_time[g][g_]=int((Point(h3.h3_to_geo(g)).distance(Point(h3.h3_to_geo(g_)))*1.3*111)/0.6)

    np.save(os.path.join(Save_path,'Travel_time'),Travel_time)









