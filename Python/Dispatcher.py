import pandas as pd

from shapely.geometry import Point, Polygon

import geopandas as gp

import numpy as np

import random

import pulp

import folium

import networkx as nx


import math

import h3

import copy

import os, sys

'''Get Neighbor ranges'''

def Get_neighbors(y):
    
    x=list()
    for y_ in y:
        for y__ in y_:
            x.append(y__)
    return x

def Compact_lists(arr):
    
    result=list()
    
    for a in arr:
        
        result=list(set(result+a))
        
    return result


def Get_travel_time(dis,speed):
    
    return int(dis/speed)

def Check_points(pnt,Points,Point_coordinate):
    
    Dic={}
    
    for point in Points:
        
        coord=Point(Point_coordinate[point])
        
        Dic[point]=pnt.distance(coord)
        
    if len(Dic)==0:
        
        return 'None'
    
    else:
        
        return min(Dic, key=Dic.get)
    
def Get_path(G,source,target,Point_coordinate,resolution,Grid_Point):

    point_path=list()
    
    shortest_dis=0

    try:

        path=nx.shortest_path(G, source=source, target=target,weight='weight')

        shortest_dis=nx.shortest_path_length(G, source=source, target=target,weight='weight')

        point_path=path

        
    except:
        
        '''Path'''
        
        start_lng,start_lat=Point_coordinate[source][1],Point_coordinate[source][0]
        
        end_lng,end_lat=Point_coordinate[target][1],Point_coordinate[target][0]
        
        '''10 parts'''
        
        for i in range(1,10,1):
            
            pnt_lng=start_lng+(end_lng-start_lng)*(i/10)
            
            pnt_lat=start_lat+(end_lat-start_lat)*(i/10)
            
            pnt=Point(pnt_lat,pnt_lng)
    
            grid=h3.geo_to_h3(pnt_lat,pnt_lng,resolution)

            if grid in Grid_Point.keys():
        
                if len(Grid_Point[grid])!=0:
                    
                    point=Check_points(pnt,Grid_Point[grid],Point_coordinate)
                    
                    if point not in [source,target] and point not in point_path:
                        
                        point_path.append(point)
                    
               
        point_path=[source]+point_path+[target]
        
        '''Distance'''
        
        for i in range(1,len(point_path),1):
            
            shortest_dis+=Point(Point_coordinate[point_path[i-1]]).distance(Point(Point_coordinate[point_path[i]]))*111000
            
        
    return point_path,shortest_dis


def MILP_Optimization(Utility):

    '''Define the problem'''

    model = pulp.LpProblem("Ride_Matching_Problems", pulp.LpMaximize)

    '''Construct our decision variable lists'''

    X = pulp.LpVariable.dicts("X",((O, D) for O in Utility.keys() for D in Utility[O].keys()),lowBound=0,upBound=1,cat='Integer')

    '''Objective Function'''

    model += (pulp.lpSum([Utility[O][D] * X[(O, D)] for O in Utility.keys() for D in Utility[O].keys()]))
    
    '''Each order can only be assigned one driver'''

    for O in Utility.keys():

        model += pulp.lpSum([X[(O, D)] for D in Utility[O].keys()]) <=1

    '''Each driver can only serve one order'''
    
    Utility_={}
    
    for O in Utility.keys():
        for D in Utility[O].keys():
            if D not in Utility_.keys():
                Utility_[D]={O:Utility[O][D]}
            else:
                Utility_[D][O]=Utility[O][D]

    for D in Utility_.keys():

         model += pulp.lpSum([X[(O, D)] for O in Utility_[D].keys()]) <=1



    model.solve()

    result={}

    for var in X:

        var_value = X[var].varValue
        
        if var_value !=0:
            
            result[var[0]]=var[1]
    

    return result

class Reposition(object):
    
    def __init__(self,Grid_list,Grid_Point,Point_Grid,Point_coordinate,resolution,speed):
        
        self.Grid_list=Grid_list
        
        self.Grid_Point=Grid_Point
        
        self.Point_Grid=Point_Grid
        
        self.Point_coordinate=Point_coordinate
        
        self.resolution=resolution
        
        self.speed=speed
        
    def Random_rep(self,point,reposition_point,grid):

        if reposition_point!=point:

            path=Get_path(G,point,reposition_point,self.Point_coordinate,self.resolution,self.Grid_Point)[0]

            dis=0

            if len(path)==2:

                point=reposition_point

            else:

                for i in range(1,len(path),1):

                    dis+=Get_path(G,path[i-1],path[i],self.Point_coordinate,self.resolution,self.Grid_Point)[1]

                    if dis>=self.speed*10:

                        point=path[i]

                        break
        else:

            r=1
            
            Neighbors=[g for g in Get_neighbors(h3.hex_range_distances(grid, 1)) if g in self.Grid_list]
            
            sample=random.choice(Neighbors)
            
            while len(Grid_Point[sample])==0:

                r+=1

                Neighbors=[g for g in Get_neighbors(h3.hex_range_distances(grid, r)) if g in self.Grid_list]
                
                sample=random.choice(Neighbors)
                
            reposition_point=random.choice(Grid_Point[sample])

        return [point,reposition_point]
    
    
    def MDP_rep(self,point,step,reposition_point,grid):

        if reposition_point!=point:

            path=Get_path(G,point,reposition_point,self.Point_coordinate,self.resolution,self.Grid_Point)[0]

            dis=0

            if len(path)==2:

                point=reposition_point

            else:

                for i in range(1,len(path),1):

                    dis+=Get_path(G,path[i-1],path[i],self.Point_coordinate,self.resolution,self.Grid_Point)[1]

                    if dis>=self.speed*10:

                        point=path[i]

                        break
        else:

            r=1
            
            s=int((step-2520)/6)
            
            state=grid+'-'+str(s)
            
            sample=max(self.Q_table[state], key=self.Q_table[state].get)
            
            while len(Grid_Point[sample])==0:

                Neighbors=[g for g in Get_neighbors(h3.hex_range_distances(grid, r)) if g in self.Grid_list]
                
                sample=random.choice(Neighbors)
                
                r+=1
                
            reposition_point=random.choice(Grid_Point[sample])

        return [point,reposition_point]
        
    def Park(self,df):
        
        df['Step']=df.apply(lambda x:x['Step']+1,axis=1)
        
        return df
    
    def Random_Walk(self,df):
        
        df['Step']=df.apply(lambda x:x['Step']+1,axis=1)
        
        df['Tuple']=df.apply(lambda x:self.Random_rep(x['Point'],x['Reposition_Point'],x['Grid']),axis=1)
        
        df['Point']=df.apply(lambda x:x['Tuple'][0],axis=1)
        
        df['Grid']=df.apply(lambda x:self.Point_Grid[x['Point']],axis=1)
        
        df['Reposition_Point']=df.apply(lambda x:x['Tuple'][1],axis=1)
        
        df=df[['Driver_id','Order_id','Step','Point','Grid','Reposition_Point']]
        
        return df
    
    def MDP_Walk(self,df):
        
        df['Step']=df.apply(lambda x:x['Step']+1,axis=1)
        
        df['Tuple']=df.apply(lambda x:self.MDP_rep(x['Point'],x['Reposition_Point'],x['Grid']),axis=1)
        
        df['Point']=df.apply(lambda x:x['Tuple'][0],axis=1)
        
        df['Grid']=df.apply(lambda x:self.Point_Grid[x['Point']],axis=1)
        
        df['Reposition_Point']=df.apply(lambda x:x['Tuple'][1],axis=1)
        
        df=df[['Driver_id','Order_id','Step','Point','Grid','Reposition_Point']]
        
        return df

if __name__ == '__main__':

    '''Param'''

    resolution = 9

    s_sec=25200

    e_sec=36000

    Start_step=2520

    End_step=3600

    Max_waiting=12

    radius=2000

    grid_radius=int(np.ceil(radius/340.0))

    speed=10 # 100 m/10seconds

    Driver_num=3000

    Policy=sys.argv[1]

    Daily_path='../Data/NYC_Feeder/'

    Load_path='../Data/NYC_Network/'

    Save_path='../Data/NYC_Results/'

    '''Load data'''

    '''Grid-related data'''

    Grid_list=np.load(os.path.join(Load_path,'Grids.npy'),allow_pickle=True)

    Grid_Point=np.load(os.path.join(Load_path,'Grid_Point.npy'),allow_pickle=True).item()


    '''Point-related data'''

    Points_list=np.load(os.path.join(Load_path,'Points_list.npy'),allow_pickle=True)

    Link_Point=np.load(os.path.join(Load_path,'Link_Point.npy'),allow_pickle=True).item()

    Point_coordinate=np.load(os.path.join(Load_path,'Point_coordinate.npy'),allow_pickle=True).item()

    Point_Grid=np.load(os.path.join(Load_path,'Point_Grid.npy'),allow_pickle=True).item()


    '''Road network Object'''

    G = nx.Graph()

    G.add_nodes_from(Points_list)

    G.add_weighted_edges_from(list(Link_Point.values()))

    '''Repositioning'''

    Rep=Reposition(Grid_list,Grid_Point,Point_Grid,Point_coordinate,resolution,speed)

    days=list(range(2,11,1))

    # for day in days:

    day=int(sys.argv[2])

    Order_df=pd.read_csv(os.path.join(Daily_path,'Order_df_'+str(day)+'.csv'))

    Order_df=Order_df.drop(columns=['Unnamed: 0'])

    Driver_df=pd.read_csv(os.path.join(Daily_path,'Driver_df.csv'))

    Driver_df=Driver_df.drop(columns=['Unnamed: 0'])

    Quatitive_results=list()

    Match_number=0

    Cancel_number=0

    for step in range(Start_step,End_step,1):

        '''(1) select unserved orders and arriving orders'''
        
        Order_batch=Order_df[(Order_df['Arrive_step']<=step)&(Order_df['Driver_id']=='Waiting')]
        
        Order_info={}
        
        for idx,row in Order_batch.iterrows():
            
            
            Order_info[row['Order_id']]={'Pickup_Point':row['Pickup_Point'],\
                                         'Dropoff_Point':row['Dropoff_Point'],\
                                         'Travel_time':int(row['Travel_time']/10),\
                                         'Match_Grids':Get_neighbors(h3.hex_range_distances(row['Pickup_Grid'], grid_radius))}
        
        Operation_Grids=Compact_lists([x['Match_Grids'] for x in Order_info.values()])
        

        '''(2) select Idle drivers'''
        
        Idle_drivers=list(Driver_df.loc[(Driver_df['Step']==step)&(Driver_df['Order_id']=='Idle'),'Driver_id'])
        
        Driver_batch=Driver_df[(Driver_df['Step']==step)&(Driver_df['Order_id']=='Idle')&(Driver_df['Grid'].isin(Operation_Grids))]
        
        Driver_info={}
        
        Grid_Drivers={}
        
        for idx,row in Driver_batch.iterrows():
            
            Driver_info[row['Driver_id']]={'Point':row['Point']}
            
            if row['Grid'] in Grid_Drivers.keys():
                
                Grid_Drivers[row['Grid']].append(row['Driver_id'])
                
            else:
                
                Grid_Drivers[row['Grid']]=[row['Driver_id']]
                
        '''(3) Construct Matching Utility'''
        
        Utility={}
        
        Pickup_time={}

        for order_id in Order_info.keys():

            Utility[order_id]={}
            
            Pickup_time[order_id]={}
            
            origin=Order_info[order_id]['Pickup_Point']
            
            Candidate_grids=Order_info[order_id]['Match_Grids']
            
            for grid in Candidate_grids:
                
                '''Existing driver?'''
                
                if grid in Grid_Drivers.keys():
                    
                    for driver_id in Grid_Drivers[grid]:
                        
                        point=Driver_info[driver_id]['Point']
                        
                        pickup_dis= Get_path(G,origin,point,Point_coordinate,resolution,Grid_Point)[1]
                        
                        Pickup_time[order_id][driver_id]=int(Get_travel_time(pickup_dis,speed)/10)
                        
                        if pickup_dis!=0.0:
                            
                            Utility[order_id][driver_id]=1/(pickup_dis)
                            
                        else:
                            
                            Utility[order_id][driver_id]=99.0
                            
             
        '''(4) Optimize matching results '''
        
        Matching_result=MILP_Optimization(Utility)
        
        Match_number+=len(Matching_result)
        
        print('Day: ',day,'Current step ',step)
        
        print('Idle Drivers: ',len(Idle_drivers),'Waiting orders: ',len(Order_info))
        
        print('Matching pairs: ',len(Matching_result),'Cumulative pairs: ',Match_number)
        
        Quatitive_results.append([step,len(Order_info),len(Idle_drivers),len(Matching_result)])
          
        '''(5) Update results of orders'''
        
        '''(5-1) Update the matched orders and drivers'''
        
        for order_id,driver_id in Matching_result.items():
            
            '''Matched order'''
            
            Order_df.loc[(Order_df['Arrive_step']<=step)&(Order_df['Order_id']==order_id),'Response_step']=step
            
            Order_df.loc[(Order_df['Arrive_step']<=step)&(Order_df['Order_id']==order_id),'Pickup_step']=step+Pickup_time[order_id][driver_id]
            
            Order_df.loc[(Order_df['Arrive_step']<=step)&(Order_df['Order_id']==order_id),'Driver_id']=driver_id
            
            '''Matched driver'''

            Driver_df.loc[(Driver_df['Step']==step)&(Driver_df['Driver_id']==driver_id)&(Driver_df['Grid'].isin(Operation_Grids)),'Order_id']=order_id
            
            Added_item={'Driver_id': driver_id,\
                        'Order_id':'Idle',\
                        'Step':step+Pickup_time[order_id][driver_id]+Order_info[order_id]['Travel_time'],\
                        'Point':Order_info[order_id]['Dropoff_Point'],\
                        'Grid': Point_Grid[Order_info[order_id]['Dropoff_Point']],\
                        'Reposition_Point':Order_info[order_id]['Dropoff_Point']}
            
            Driver_df=Driver_df.append(Added_item, ignore_index=True)
            

        '''(5-2) Update the unmatched orders'''
        
        Unmatched_orders=[O for O in Order_info.keys() if O not in Matching_result.keys()]
        
        Unit=0
        
        if len(Unmatched_orders)!=0:
            
            Unit=Order_df.loc[((step-Order_df['Arrive_step'])>Max_waiting)&(Order_df['Order_id'].isin(Unmatched_orders))].shape[0]
            
            Cancel_number+=Unit
            
            Order_df.loc[((step-Order_df['Arrive_step'])>Max_waiting)&(Order_df['Order_id'].isin(Unmatched_orders)),'Driver_id']='Canceled'
        
        print('Canceled orders ',Unit,'Cumulative Canceled orders: ',Cancel_number)
        
        print('*'*50)
        

        '''(5-3) Update the unmatehed drivers: Repositioning'''
        
        Idle_drivers=list(Driver_df.loc[(Driver_df['Step']==step)&(Driver_df['Order_id']=='Idle'),'Driver_id'])
        
        Next_Driver_df=copy.copy(Driver_df.loc[(Driver_df['Step']==step)&(Driver_df['Driver_id'].isin(Idle_drivers))])
        
        '''Just waiting at the current location'''
        
        if Policy=='Park':

            print('Policy: ',Policy)

            Next_Driver_df=Rep.Park(Next_Driver_df)
            
        elif Policy=='Random_Walk':

            print('Policy: ',Policy)
            
            Next_Driver_df=Rep.Random_Walk(Next_Driver_df)

        elif Policy=='MDP_Walk':

            print('Policy: ',Policy)
            
            Next_Driver_df=Rep.MDP_Walk(Next_Driver_df)

        Driver_df=pd.concat([Driver_df,Next_Driver_df],ignore_index=True)

    Order_df.to_csv(os.path.join(Save_path,'Order_df_'+Policy+'_'+str(day)+'.csv'))

    Driver_df.to_csv(os.path.join(Save_path,'Driver_df_'+Policy+'_'+str(day)+'.csv'))

    Quatitive_results=np.array(Quatitive_results)

    np.save(os.path.join(Save_path,'Quatitive_results_'+Policy+'_'+str(day)),Quatitive_results)
