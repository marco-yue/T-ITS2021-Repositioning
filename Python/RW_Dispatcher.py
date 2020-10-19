import pandas as pd

from shapely.geometry import Point, Polygon

import geopandas as gp

import numpy as np

import random

import pulp

import folium

import networkx as nx

import math

import copy

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

def Check_points(pnt,Zone_Points,Point_coordinate):
    
    Dic={}
    
    for point in Zone_Points:
        
        coord=Point_coordinate[point]
        
        Dic[point]=pnt.distance(coord)
        
    return min(Dic, key=Dic.get)
        

def Random_walk(zone,Connect_matrix):
    
    zone_index=int(zone.split('_')[1])
    
    candidates=np.argwhere(Connect_matrix[zone_index]==1.0)
    
    dest=random.choice(candidates)[0]
    
    return 'Zone_'+str(dest)


def Get_path(G,source,target,Point_coordinate,Zone_Point):
    
    link_path=list()
    
    try:
    
        path=nx.shortest_path(G, source=source, target=target,weight='weight')

        shortest_dis=nx.shortest_path_length(G, source=source, target=target,weight='weight')
        
        link_path=path

        
    except:
        
        '''Path'''
        
        start_lng,start_lat=list(Point_coordinate[source].coords)[0][0],list(Point_coordinate[source].coords)[0][1]
        
        end_lng,end_lat=list(Point_coordinate[target].coords)[0][0],list(Point_coordinate[target].coords)[0][1]
        
        '''10 parts'''
        
        
        for i in range(1,10,1):
            
            pnt_lng=start_lng+(end_lng-start_lng)*(i/10)
            
            pnt_lat=start_lat+(end_lat-start_lat)*(i/10)
            
            pnt=Point(pnt_lng,pnt_lat)
            
            zone=Check_zones(pnt,polys)
            
            if zone != 'None':
            
                point=Check_points(pnt,Zone_Point[zone],Point_coordinate)
                
                if point not in [source,target] and point not in link_path:

                    link_path.append(point)
               
        link_path=[source]+link_path+[target]
        
        '''Distance'''
        
        shortest_dis=Point_coordinate[source].distance(Point_coordinate[target])*111000
    
        
    return link_path,shortest_dis

def reposition(point,zone,reposition_point,speed,Connect_matrix,Point_coordinate,G,Zone_Point):
    
    if reposition_point!=point:
        
        path=Get_path(G,point,reposition_point,Point_coordinate,Zone_Point)[0]
        
        dis=0
        
        if len(path)==2:
            
            point=reposition_point
            
        else:
        
            for i in range(1,len(path),1):

                dis+=Get_path(G,path[i-1],path[i],Point_coordinate,Zone_Point)[1]
                
                if dis>=speed*10:
                    
                    point=path[i]
                    
                    break
    else:
        
        '''update reposition destination'''
        
        reposition_candidates=Zone_Point[Random_walk(zone,Connect_matrix)]
        
        while len(reposition_candidates)==0:
            
            reposition_candidates=Zone_Point[Random_walk(zone,Connect_matrix)]
            
        
        reposition_point=random.choice(reposition_candidates)
        
    return [point,reposition_point]

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

	Max_waiting=12

	radius=2000

	speed=10 # 10 m/seconds

	policy='Random_walk'

	'''Load data'''

	'''Zone-related data'''

	Taxi_Zones=np.load('../Data/NYC_Zones/Taxi_Zones.npy',allow_pickle=True).item()

	Zone_list=np.load('../Data/NYC_Zones/Zone_list.npy',allow_pickle=True)

	Zone_Center=np.load('../Data/NYC_Zones/Zone_Center.npy',allow_pickle=True).item()

	Zone_Link=np.load('../Data/NYC_Zones/Zone_Link.npy',allow_pickle=True).item()

	'''Link-related data'''

	Link_list=np.load('../Data/NYC_Zones/Link_list.npy',allow_pickle=True)

	Link_geometry=np.load('../Data/NYC_Zones/Link_geometry.npy',allow_pickle=True).item()


	'''Point-related data'''

	Points_list=np.load('../Data/NYC_Zones/Points_list.npy',allow_pickle=True)

	Link_Point=np.load('../Data/NYC_Zones/Link_Point.npy',allow_pickle=True).item()

	Point_coordinate=np.load('../Data/NYC_Zones/Point_coordinate.npy',allow_pickle=True).item()

	Zone_Point=np.load('../Data/NYC_Zones/Zone_Point.npy',allow_pickle=True).item()

	Point_zone=np.load('../Data/NYC_Zones/Point_zone.npy',allow_pickle=True).item()


	'''GeoSeries Object'''

	polys = gp.GeoSeries(Taxi_Zones)

	'''Zone matrix'''

	Connect_matrix=np.load('../Data/NYC_Zones/Connect_matrix.npy',allow_pickle=True)

	'''Road network Object'''

	G = nx.Graph()

	G.add_nodes_from(Points_list)

	G.add_weighted_edges_from(list(Link_Point.values()))

	'''Dispatching Procedure'''

	'''Dispatching Procedure'''

	Order_df=pd.read_csv('../Data/NYC_Trips/Order_df.csv')

	Order_df=Order_df.drop(columns=['Unnamed: 0'])

	Order_df['Response_step']=End_step

	Order_df['Pickup_step']=End_step

	Order_df=Order_df[['Order_id',\
	                   'Driver_id',
	                   'Arrive_step',\
	                   'Response_step',\
	                   'Pickup_step',\
	                   'Pickup_Latitude',\
	                   'Pickup_Longitude',\
	                   'Dropoff_Latitude',\
	                   'Dropoff_Longitude',\
	                   'Pickup_Zone',\
	                   'Dropoff_Zone',\
	                   'Pickup_Point',\
	                   'Dropoff_Point',\
	                   'Travel_dis',\
	                   'Travel_time']]

	Driver_df=pd.read_csv('../Data/NYC_Trips/Driver_df.csv')

	Driver_df=Driver_df.drop(columns=['Unnamed: 0'])

	Quatitive_results=list()


	for step in range(Start_step,End_step,1):
    
    
	    '''(1) select unserved orders and arriving orders'''
	    
	    Order_batch=Order_df[(Order_df['Arrive_step']<=step)&(Order_df['Driver_id']=='Waiting')]
	    
	    Order_info={}
	    
	    for idx,row in Order_batch.iterrows():
	        
	        
	        Order_info[row['Order_id']]={'Pickup_Point':row['Pickup_Point'],\
	                              'Dropoff_Point':row['Dropoff_Point'],\
	                              'Travel_time':int(row['Travel_time']/10),\
	                              'Match_Points':Get_matching_points(row['Pickup_Point'],Point_coordinate,Points_list,radius)}
	    
	    Operation_Points=Compact_lists([x['Match_Points'] for x in Order_info.values()])
	    

	    '''(2) select Idle drivers'''
	    
	    Idle_drivers=list(Driver_df.loc[(Driver_df['Step']==step)&(Driver_df['Order_id']=='Idle'),'Driver_id'])
	    
	    Driver_batch=Driver_df[(Driver_df['Step']==step)&(Driver_df['Order_id']=='Idle')&(Driver_df['Point'].isin(Operation_Points))]
	    
	    Driver_info={}
	    
	    Points_Drivers={}
	    
	    for idx,row in Driver_batch.iterrows():
	        
	        Driver_info[row['Driver_id']]={'Point':row['Point']}
	        
	        if row['Point'] in Points_Drivers.keys():
	            
	            Points_Drivers[row['Point']].append(row['Driver_id'])
	            
	        else:
	            
	            Points_Drivers[row['Point']]=[row['Driver_id']]
	            
	            
	    '''(3) Construct Matching Utility'''
	    
	    Utility={}
	    
	    Pickup_time={}

	    for order_id in Order_info.keys():

	        Utility[order_id]={}
	        
	        Pickup_time[order_id]={}
	        
	        origin=Order_info[order_id]['Pickup_Point']
	        
	        match_points=Order_info[order_id]['Match_Points']
	        
	        for point in match_points:
	            
	            '''Existing driver?'''
	            
	            if point in Points_Drivers.keys():
	                
	                for driver_id in Points_Drivers[point]:
	                    
	                    pickup_dis= Get_path(G,origin,point,Point_coordinate,Zone_Point)[1]
	                    
	                    Pickup_time[order_id][driver_id]=int(Get_travel_time(pickup_dis,speed)/10)
	                    
	                    if pickup_dis!=0.0:
	                        
	                        Utility[order_id][driver_id]=1/(pickup_dis)
	                        
	                    else:
	                        
	                        Utility[order_id][driver_id]=99.0
	                        
	        for driver_id in Driver_info.keys():

	            if driver_id not in Utility[order_id].keys():

	                Utility[order_id][driver_id]=0.0
	                
	                Pickup_time[order_id][driver_id]=0
	                
	    '''(4) Optimize matching results '''
	    
	    Matching_result=MILP_Optimization(list(Order_info.keys()),list(Driver_info.keys()),Utility)
	    
	    print('Current step ',step,'Waiting orders: ',len(Order_info),'Selected drivers: ',len(Driver_info),'Matching pairs: ',len(Matching_result))
	    
	    Quatitive_results.append([step,len(Order_info),len(Idle_drivers),len(Matching_result)])

	    print('*'*50)
	    
	    '''(5) Update results of orders'''
	    
	    '''(5-1) Update the matched orders and drivers'''
	    
	    for order_id,driver_id in Matching_result.items():
	        
	        '''Matched order'''
	        
	        Order_df.loc[(Order_df['Arrive_step']<=step)&(Order_df['Order_id']==order_id),'Response_step']=step
	        
	        Order_df.loc[(Order_df['Arrive_step']<=step)&(Order_df['Order_id']==order_id),'Pickup_step']=step+Pickup_time[order_id][driver_id]
	        
	        Order_df.loc[(Order_df['Arrive_step']<=step)&(Order_df['Order_id']==order_id),'Driver_id']=driver_id
	        
	        '''Matched driver'''

	        Driver_df.loc[(Driver_df['Step']==step)&(Driver_df['Driver_id']==driver_id)&(Driver_df['Point'].isin(Operation_Points)),'Order_id']=order_id
	        
	        Added_item={'Driver_id': driver_id,\
	                    'Order_id':'Idle',\
	                    'Step':step+Pickup_time[order_id][driver_id]+Order_info[order_id]['Travel_time'],\
	                    'Point':Order_info[order_id]['Dropoff_Point'],\
	                    'Zone': Point_zone[Order_info[order_id]['Dropoff_Point']],\
	                    'Reposition_Point':Order_info[order_id]['Dropoff_Point']}
	        
	        Driver_df=Driver_df.append(Added_item, ignore_index=True)
	        
	    '''(5-2) Update the unmatched orders'''
	    
	    
	    Unmatched_orders=[O for O in Order_info.keys() if O not in Matching_result.keys()]
	    
	    if len(Unmatched_orders)!=0:
	        
	        Order_df.loc[((step-Order_df['Arrive_step'])>Max_waiting)&(Order_df['Order_id'].isin(Unmatched_orders)),'Driver_id']='Canceled'

	        cancel_num=Order_df.loc[((step-Order_df['Arrive_step'])>Max_waiting)&(Order_df['Order_id'].isin(Unmatched_orders)),'Driver_id'].shape[0]

	        print('Cancel quantity: ',cancel_num)

	        
	    '''(5-3) Update the unmatehed drivers: Repositioning'''
	    
	    Idle_drivers=list(Driver_df.loc[(Driver_df['Step']==step)&(Driver_df['Order_id']=='Idle'),'Driver_id'])
	    
	    Next_Driver_df=Driver_df.loc[(Driver_df['Step']==step)&(Driver_df['Driver_id'].isin(Idle_drivers))]
	   
	    Next_Driver_df['Step']=Next_Driver_df.apply(lambda x:x['Step']+1,axis=1)

	    Next_Driver_df['Tuple']=Next_Driver_df.apply(lambda x:reposition(x['Point'],x['Zone'],x['Reposition_Point'],speed,Connect_matrix,Point_coordinate,G,Zone_Point),axis=1)

	    Next_Driver_df['Point']=Next_Driver_df.apply(lambda x:x['Tuple'][0],axis=1)

	    Next_Driver_df['Zone']=Next_Driver_df.apply(lambda x:Point_zone[x['Point']],axis=1)

	    Next_Driver_df['Reposition_Point']=Next_Driver_df.apply(lambda x:x['Tuple'][1],axis=1)

	    Next_Driver_df=Next_Driver_df[['Driver_id','Order_id','Step','Point','Zone','Reposition_Point']]

	    Driver_df=pd.concat([Driver_df,Next_Driver_df],ignore_index=True)
	
	Order_df.to_csv('../Data/NYC_Trips/Dispatched_Order_df_'+policy+'.csv')

	Driver_df.to_csv('../Data/NYC_Trips/Dispatched_Driver_df_'+policy+'.csv')

	Quatitive_results=np.array(Quatitive_results)

	np.save('../Data/NYC_Trips/Quatitive_results'+policy,Quatitive_results)
