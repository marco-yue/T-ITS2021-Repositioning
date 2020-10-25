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


def Get_travel_time(dis,speed):
    
    return int(dis/speed)

if __name__ == '__main__':

    '''Param'''

    resolution = 9

    s_sec=25200

    e_sec=36000

    Start_step=2520

    End_step=3600

    speed=10 # 10 m/seconds

    Driver_num=3000

    '''Load data'''

    '''Grid-related data'''

    Grid_list=np.load('../Data/NYC_Network/Grids.npy',allow_pickle=True)

    Grid_Point=np.load('../Data/NYC_Network/Grid_Point.npy',allow_pickle=True).item()


    '''Point-related data'''

    Points_list=np.load('../Data/NYC_Network/Points_list.npy',allow_pickle=True)

    Link_Point=np.load('../Data/NYC_Network/Link_Point.npy',allow_pickle=True).item()

    Point_coordinate=np.load('../Data/NYC_Network/Point_coordinate.npy',allow_pickle=True).item()

    Point_Grid=np.load('../Data/NYC_Network/Point_Grid.npy',allow_pickle=True).item()


    '''Road network Object'''

    G = nx.Graph()

    G.add_nodes_from(Points_list)

    G.add_weighted_edges_from(list(Link_Point.values()))

    '''Date'''

    days=list(range(11,21,1))

    for day in days:

        '''1. Processing the Order data'''

        '''Link based and point based'''

        Order_df=pd.read_csv('../Data/NYC_Data/pick_day_'+str(day)+'.csv',header=None,names=['Pick_Second','Trip_Duration',\
                                                             'Pickup_Latitude','Pickup_Longitude',\
                                                             'Dropoff_Latitude','Dropoff_Longitude',\
                                                             'Trip_Distance','Euclidean','Manhattan'])

        Order_df=Order_df.loc[(Order_df['Pick_Second']>=s_sec)&(Order_df['Pick_Second']<e_sec)]

        Order_df['Pickup_Grid']=Order_df.apply(lambda x:h3.geo_to_h3(x['Pickup_Latitude'],x['Pickup_Longitude'],resolution),axis=1)

        Order_df['Dropoff_Grid']=Order_df.apply(lambda x:h3.geo_to_h3(x['Dropoff_Latitude'],x['Dropoff_Longitude'],resolution),axis=1)

        Order_df=Order_df.loc[(Order_df['Pickup_Grid'].isin(Grid_list))&(Order_df['Dropoff_Grid'].isin(Grid_list))]

        Order_df=Order_df.rename(columns={'Pick_Second':'Arrive_Second'})

        Order_df['Arrive_step']=Order_df.apply(lambda x:math.ceil(x['Arrive_Second']/10),axis=1)

        Order_df=Order_df.sort_values(by=['Arrive_step'])

        Order_df=Order_df.reset_index(drop=True)

        Order_df['Order_id']=['O'+str(i) for i in Order_df.index]

        Order_df['Driver_id']='Waiting'


        Order_df=Order_df[['Order_id','Driver_id','Arrive_step',\
                           'Pickup_Latitude','Pickup_Longitude',\
                           'Dropoff_Latitude','Dropoff_Longitude',\
                            'Pickup_Grid','Dropoff_Grid']]

        Order_df['Pickup_Point']=Order_df.apply(lambda x:Check_points(Point(x['Pickup_Latitude'],x['Pickup_Longitude']),Grid_Point[x['Pickup_Grid']],Point_coordinate),axis=1)

        Order_df['Dropoff_Point']=Order_df.apply(lambda x:Check_points(Point(x['Dropoff_Latitude'],x['Dropoff_Longitude']),Grid_Point[x['Dropoff_Grid']],Point_coordinate),axis=1)

        Order_df=Order_df[(Order_df['Pickup_Point']!='None')&(Order_df['Dropoff_Point']!='None')&(Order_df['Pickup_Point']!=Order_df['Dropoff_Point'])]

        Order_df['Travel_dis']=Order_df.apply(lambda x:Get_path(G,x['Pickup_Point'],x['Dropoff_Point'],Point_coordinate,resolution,Grid_Point)[1],axis=1)

        Order_df['Travel_time']=Order_df.apply(lambda x:Get_travel_time(x['Travel_dis'],speed),axis=1)

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
                   'Pickup_Grid',\
                   'Dropoff_Grid',\
                   'Pickup_Point',\
                   'Dropoff_Point',\
                   'Travel_dis',\
                   'Travel_time']]

        Order_df.to_csv('../Data/NYC_Feeder/Order_df_'+str(day)+'.csv')

        print('*'*50)

        print('Day: ',day)

    '''2. Initialize the driver data'''

    Driver_list=['D'+str(i) for i in range(Driver_num)]

    Driver_df=pd.DataFrame(Driver_list,columns=['Driver_id'])

    Driver_df['Point']=Driver_df.apply(lambda x:random.choice(Points_list),axis=1)

    Driver_df['Grid']=Driver_df.apply(lambda x:Point_Grid[x['Point']],axis=1)

    Driver_df['Reposition_Point']=Driver_df.apply(lambda x:x['Point'],axis=1)

    Driver_df['Step']=Start_step

    Driver_df['Order_id']='Idle'

    Driver_df=Driver_df[['Driver_id','Order_id','Step','Point','Grid','Reposition_Point']]

    Driver_df.to_csv('../Data/NYC_Feeder/Driver_df.csv')