import pandas as pd

from shapely.geometry import Point, Polygon

import geopandas as gp

import numpy as np

import random

import pulp

import folium

import networkx as nx

import math

def Check_zones(pnt,polys):
    
    key='None'
    
    for k, geom in polys.items():
        
        if pnt.within(geom):
            
            key=k
            
            break
            
    return key

def Check_links(pnt,zone,Zone_Link,Link_middle):
    
    Dic={}
    
    for link in Zone_Link[zone]:
        
        middle_pnt=Link_middle[link]
        
        Dic[link]=pnt.distance(middle_pnt)
        
    return min(Dic, key=Dic.get)

    
def Get_path(G,Points_link,Point_coordinate,source,target):
    
    link_path=list()
    
    try:
    
        path=nx.shortest_path(G, source=source, target=target,weight='weight')

        shortest_dis=nx.shortest_path_length(G, source=source, target=target,weight='weight')

        for i in range(1,len(path),1):

            pnts=path[i-1]+"&"+path[i]

            link_path.append(Points_link[pnts])
        
    except:
        
        shortest_dis=(Point_coordinate[source].distance(Point_coordinate[target])*111000)*1.3
    
    return link_path,shortest_dis

def Get_travel_time(dis,speed):
    
    return int(dis/speed)

def Get_matching_points(source,Point_coordinate,Points_list,radius):
    
    Point_candidates=list()
    
    for pnt in Points_list:
        
        dis=111000*(Point_coordinate[source].distance(Point_coordinate[pnt]))
    
        if dis<=radius:
            
            Point_candidates.append(pnt)
            
    return Point_candidates

def Compact_lists(arr):
    
    result=list()
    
    for a in arr:
        
        result=list(set(result+a))
        
    return result
            
def MILP_Optimization(Order,Driver,Utility):

    '''Define the problem'''

    model = pulp.LpProblem("Ride_Matching_Problems", pulp.LpMaximize)

    '''Construct our decision variable lists'''

    X = pulp.LpVariable.dicts("X",((i, j) for i in Order for j in Driver),lowBound=0,upBound=1,cat='Integer')

    '''Objective Function'''

    model += (pulp.lpSum([Utility[i][j] * X[(i, j)] for i in Order for j in Driver]))


    '''Each driver can only serve one order'''

    for i in Order:

        model += pulp.lpSum([X[(i, j)] for j in Driver]) <=1

    '''Each order can only be assigned one driver'''

    for j in Driver:

         model += pulp.lpSum([X[(i, j)] for i in Order]) <=1



    model.solve()

    result={}

    for var in X:

        var_value = X[var].varValue
        
        if var_value !=0:
            
            result[var[0]]=var[1]
    

    return result   
    
if __name__ == '__main__':

	'''Param'''

	s_sec=25200

	e_sec=36000

	Start_step=2520

	End_step=3600

	Max_waiting=30

	radius=1000

	speed=10 # 10 m/seconds

	Driver_num=3000

	'''Load data'''

	'''Zone-related data'''

	Taxi_Zones=np.load('../Data/NYC_Zones/Taxi_Zones.npy',allow_pickle=True).item()

	Zone_list=np.load('../Data/NYC_Zones/Zone_list.npy',allow_pickle=True)

	Zone_Center=np.load('../Data/NYC_Zones/Zone_Center.npy',allow_pickle=True).item()

	Zone_Link=np.load('../Data/NYC_Zones/Zone_Link.npy',allow_pickle=True).item()

	'''Link-related data'''

	Link_middle=np.load('../Data/NYC_Zones/Link_middle.npy',allow_pickle=True).item()

	Link_Point=np.load('../Data/NYC_Zones/Link_Point.npy',allow_pickle=True).item()

	'''Point-related data'''

	Points_list=np.load('../Data/NYC_Zones/Points_list.npy',allow_pickle=True)

	Points_link=np.load('../Data/NYC_Zones/Points_link.npy',allow_pickle=True).item()

	Point_coordinate=np.load('../Data/NYC_Zones/Point_coordinate.npy',allow_pickle=True).item()

	'''GeoSeries Object'''

	polys = gp.GeoSeries(Taxi_Zones)

	'''Road network Object'''

	G = nx.Graph()

	G.add_nodes_from(Points_list)

	G.add_weighted_edges_from(list(Link_Point.values()))

	'''Dispatching Procedure'''

	Order_df=pd.read_csv('../Data/NYC_Trips/Order_df.csv')

	Order_df=Order_df.drop(columns=['Unnamed: 0'])

	Order_df['Pickup_step']=End_step

	Order_df=Order_df[['Order_id','Driver_id','Arrive_step','Pickup_step',\
	                   'Pickup_Latitude','Pickup_Longitude',\
	                   'Dropoff_Latitude','Dropoff_Longitude',\
	                    'Pickup_Zone','Dropoff_Zone',\
	                   'Pickup_Link','Dropoff_link',\
	                   'Pickup_Point','Dropoff_Point',\
	                   'Travel_dis','Travel_time']]

	Driver_df=pd.read_csv('../Data/NYC_Trips/Driver_df.csv')

	Driver_df=Driver_df.drop(columns=['Unnamed: 0'])

	Quatitive_results=list()


	for step in range(Start_step,End_step,1):
	    
	    
	    '''(1) select unifinished orders'''
	    
	    Order_batch=Order_df[(Order_df['Arrive_step']<=step)&(Order_df['Driver_id']=='Waiting')]

	    Order_arr=list();
	    
	    Travel_time={}
	    
	    Order_info={}
	    
	    Match_Points={};
	    
	    for idx,row in Order_batch.iterrows():

	        order_id=row['Order_id']

	        Order_arr.append(order_id)
	        
	        Travel_time[order_id]=int(row['Travel_time']/10)
	        
	        Order_info[order_id]={'Pickup_Point':row['Pickup_Point'],\
	                              'Dropoff_Latitude':row['Dropoff_Latitude'],\
	                              'Dropoff_Longitude':row['Dropoff_Longitude'],\
	                              'Dropoff_Zone':row['Dropoff_Zone'],\
	                              'Dropoff_link':row['Dropoff_link'],\
	                              'Dropoff_Point':row['Dropoff_Point']}
	        
	        Match_Points[order_id]=Get_matching_points(row['Pickup_Point'],Point_coordinate,Points_list,radius)
	        
	    Operation_Points=Compact_lists(list(Match_Points.values()))
	    
	    '''(2) select Idle drivers'''
	    
	    Idle_drivers=list(Driver_df.loc[(Driver_df['Step']==step)&(Driver_df['Order_id']=='Idle'),'Driver_id'])
	    
	    Driver_batch=Driver_df[(Driver_df['Step']==step)&(Driver_df['Order_id']=='Idle')&(Driver_df['Point'].isin(Operation_Points))]
	    
	    Driver_arr=list()

	    Driver_Points={}
	    
	    Points_Drivers={}
	    
	    for idx,row in Driver_batch.iterrows():
	        
	        driver_id=row['Driver_id']

	        Driver_arr.append(driver_id)

	        Driver_Points[driver_id]=row['Point']
	        
	        if row['Point'] in Points_Drivers.keys():
	            
	            Points_Drivers[row['Point']].append(driver_id)
	            
	        else:
	            
	            Points_Drivers[row['Point']]=[driver_id]
	            
	            
	    '''(3) Construct Matching Utility'''
	    
	    Utility_matrix={}

	    for order_id in Order_arr:

	        Utility_matrix[order_id]={}
	        
	        origin=Order_info[order_id]['Pickup_Point']
	        
	        match_points=Match_Points[order_id]
	        
	        for point in match_points:
	            
	            if point in Points_Drivers.keys():
	                
	                for driver_id in Points_Drivers[point]:
	                    
	                    pickup_dis=Get_path(G,Points_link,Point_coordinate,origin,point)[1]
	                    
	                    if pickup_dis!=0.0:
	                        
	                        Utility_matrix[order_id][driver_id]=1/(pickup_dis)
	                        
	                    else:
	                        
	                        Utility_matrix[order_id][driver_id]=99.0
	                        
	        for driver_id in Driver_arr:

	            if driver_id not in Utility_matrix[order_id].keys():

	                Utility_matrix[order_id][driver_id]=0.0   
	                
	    '''(4) Optimize matching results '''
	    
	    Matching_result=MILP_Optimization(Order_arr,Driver_arr,Utility_matrix)
	    
	    print('Current step ',step,'Waiting orders: ',len(Order_arr),'Selected drivers: ',len(Driver_arr),'Matching pairs: ',len(Matching_result))
	    
	    Quatitive_results.append([step,len(Order_arr),len(Idle_drivers),len(Matching_result)])

	    print('*'*50)
	    
	    
	    '''(5) Update results of orders'''
	    
	    '''(5-1) Update the matched orders and drivers'''
	    
	    for order_id,driver_id in Matching_result.items():
	        
	        '''Matched order'''
	        
	        Order_df.loc[(Order_df['Arrive_step']<=step)&(Order_df['Order_id']==order_id),'Pickup_step']=step
	        
	        Order_df.loc[(Order_df['Arrive_step']<=step)&(Order_df['Order_id']==order_id),'Driver_id']=driver_id
	        
	        '''Matched driver'''

	        Driver_df.loc[(Driver_df['Step']==step)&(Driver_df['Driver_id']==driver_id)&(Driver_df['Point'].isin(Operation_Points)),'Order_id']=order_id
	        
	        Added_item={'Driver_id': driver_id,\
	                    'Order_id':'Idle',\
	                    'Step':step+Travel_time[order_id],\
	                    'Latitude':Order_info[order_id]['Dropoff_Latitude'],\
	                    'Longitude':Order_info[order_id]['Dropoff_Longitude'],\
	                    'Zone':Order_info[order_id]['Dropoff_Zone'],\
	                    'Link':Order_info[order_id]['Dropoff_link'],\
	                    'Point':Order_info[order_id]['Dropoff_Point']}
	        
	        Driver_df=Driver_df.append(Added_item, ignore_index=True)
	        
	    '''(5-2) Update the unmatched orders'''
	    
	    Matched_orders=list(Matching_result.keys())
	    
	    Unmatched_orders=[O for O in Order_arr if O not in Matched_orders]
	    
	    if len(Unmatched_orders)!=0:
	        
	        Order_df.loc[(step-Order_df['Arrive_step']>Max_waiting)&(Order_df['Order_id'].isin(Unmatched_orders)),'Driver_id']='Cancelled'
	        
	        
	    '''(5-3) Update the unmatehed drivers: Repositioning'''
	    
	    Idle_drivers=list(Driver_df.loc[(Driver_df['Step']==step)&(Driver_df['Order_id']=='Idle'),'Driver_id'])
	    
	    Next_Driver_df=Driver_df.loc[(Driver_df['Step']==step)&(Driver_df['Driver_id'].isin(Idle_drivers))]
	    
	    '''Just waiting at the current location'''
	    
	    Next_Driver_df['Step']=Next_Driver_df.apply(lambda x:x['Step']+1,axis=1)
	    
	    Driver_df=pd.concat([Driver_df,Next_Driver_df],ignore_index=True)
	

	Order_df.to_csv('../Data/NYC_Trips/Dispatched_Order_df.csv')

	Driver_df.to_csv('../Data/NYC_Trips/Dispatched_Driver_df.csv')

	Quatitive_results=np.array(Quatitive_results)

	np.save('../Data/NYC_Trips/Quatitive_results',Quatitive_results)
