import utm
import math
import pandas as pd
import pymysql
import multiprocessing as mp
from datetime import datetime
import pytz
import glob
from haversine import haversine, Unit
import time

EARTH_RADIUS = 6378137.

SHIFT = 10000
SAMPLING_RATE = 1

#TODO: you must to change these variable
CURRENT_TIMEZONE = 'America/New_York'
CURRENT_DAY = '2019-09-23'
ZONE_NUMBER = 18
ZONE_LETTER = 'S'
df_stops_times = pd.read_csv("/home/clayson/globecom_extension/datasets/gtfs/washington_weekdays/stop_times.txt", dtype={'trip_id':'string', 'stop_headsign':'string'})
df_stops = pd.read_csv("/home/clayson/globecom_extension/datasets/gtfs/washington_weekdays/stops.txt")
df_trips = pd.read_csv("/home/clayson/globecom_extension/datasets/gtfs/washington_weekdays/trips.txt", dtype={'trip_id':'string', 'route_id':'string'})
df_shapes = pd.read_csv("/home/clayson/globecom_extension/datasets/gtfs/washington_weekdays/shapes.txt")


def euclidean_dist(origin_point, destination_point):
    euclidean_distance = math.sqrt(
        math.pow(origin_point[0] - destination_point[0], 2) + math.pow(origin_point[1] - destination_point[1], 2))
    return euclidean_distance


def equal_points(p1, p2):
    if p1[0] == p2[0] and p1[1] == p2[1]:
        return True
    else:
        return False


def linear_interpolation(p0, p1, dist):
    if equal_points(p0,p1) == True:
        return (p0[0], p1[1])
    else:
        value = p0[0] + (dist / euclidean_dist(p0, p1)) * (p1[0] - p0[0]), p0[1] + (dist / euclidean_dist(p0, p1)) * (p1[1] - p0[1])
        return value


def get_latlong_df(point):
    tup = utm.to_latlon(point[0], point[1], ZONE_NUMBER, ZONE_LETTER)  # y x
    return pd.Series(tup[:2])


def hour2ts(hour):
    date_time_obj = datetime.strptime(CURRENT_DAY+' '+hour, '%Y-%m-%d %H:%M:%S')
    tz = pytz.timezone(CURRENT_TIMEZONE)
    dt_with_tz = tz.localize(date_time_obj, is_dst=None)
    ts = (dt_with_tz - datetime(1970,1,1,tzinfo=pytz.utc)).total_seconds()
    return ts

def get_bus_stops_dataframe(trip_id):
	df_stop_times_subset = df_stops_times.loc[df_stops_times['trip_id'] == trip_id]
	df_bus_stops = pd.merge(df_stop_times_subset, df_stops, on='stop_id')
	df_bus_stops = df_bus_stops.sort_values(by=['stop_sequence'])
	df_bus_stops = df_bus_stops.reset_index(drop=True)
	df_bus_stops['stop_code'] = "NON"
	return df_bus_stops[['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence', 'stop_code', 'stop_name', 'stop_lat', 'stop_lon', 'shape_dist_traveled']]


def distance(lat1, lon1, lat2, lon2):
	return haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)


def nearest_point(df, stop_bus_point):
	df['distance'] = df.apply(lambda row : distance(stop_bus_point[0], stop_bus_point[1], row['shape_pt_lat'], row['shape_pt_lon']), axis = 1)
	df_result = df[df.distance == df.distance.min()]
	return(df_result.index[0])
	#print((df_result.iloc[0]['shape_pt_lat'], df_result.iloc[0]['shape_pt_lon'])) 


#identify bus stops in the shape
def identify_bus_stops(current_shape_id, df_bus_stops):
	route_shape = df_shapes.loc[(df_shapes["shape_id"] == current_shape_id)].copy()
	route_shape = route_shape.reset_index(drop=True)
	route_shape['id'] = list(range(1,route_shape.shape[0]+1))
	route_shape['type'] = 'non-stop'
	route_shape.at[0,'type'] = 'stop'	
	#df_bus_stops.to_csv("bus_stops_original.csv", index=False)
	df_bus_stops_aslist = df_bus_stops.values.tolist()
	for i in range(1, len(df_bus_stops_aslist)-1):
		index = nearest_point(route_shape, (df_bus_stops_aslist[i][7], df_bus_stops_aslist[i][8]))
		route_shape.at[index,'type'] = 'stop'
	route_shape.at[len(route_shape) - 1,'type'] = 'stop'

	temp_route_shape_stops = route_shape.loc[(route_shape["type"] == 'stop')]
	#temp_route_shape_stops.to_csv("bus_stops.csv", index=False)

	temp_anchor_points_aslist = temp_route_shape_stops.values.tolist()
	route_shape_stops = pd.DataFrame()
	for i in range(0, len(temp_anchor_points_aslist)):
		timestamp = hour2ts(df_bus_stops_aslist[i][1])
		route_shape_stops = route_shape_stops.append({'shape_id': temp_anchor_points_aslist[i][0], 'shape_pt_lat': temp_anchor_points_aslist[i][1] , 'shape_pt_lon': temp_anchor_points_aslist[i][2] , 'shape_pt_sequence':  temp_anchor_points_aslist[i][3], 'id': temp_anchor_points_aslist[i][4] , 
		'trip_id':df_bus_stops_aslist[i][0] , 'arrival_time': timestamp, 'departure_time': timestamp, 'stop_id': df_bus_stops_aslist[i][3], 'stop_sequence': df_bus_stops_aslist[i][4], 'stop_code': df_bus_stops_aslist[0][5], 'stop_name': df_bus_stops_aslist[i][6], 'stop_lat': df_bus_stops_aslist[i][7], 'stop_lon': df_bus_stops_aslist[i][8]}, ignore_index=True)
	#route_shape_stops.to_csv("route_shape_stops.csv", index=False)

	anchor_points = route_shape.loc[(route_shape["type"] == 'non-stop')]
	anchor_points = anchor_points.drop(columns=['type', 'distance'])
	#anchor_points.to_csv("anchor_points.csv", index=False)
	anchor_points = anchor_points[['id','shape_id','shape_pt_lat','shape_pt_lon','shape_pt_sequence']]
	route_shape_stops = route_shape_stops[['arrival_time', 'departure_time', 'id', 'shape_id', 'shape_pt_lat', 'shape_pt_lon', 'shape_pt_sequence', 'stop_lat', 'stop_lon', 'trip_id', 'stop_id', 'stop_sequence', 'stop_code', 'stop_name']]
	return anchor_points, route_shape_stops


def calculate_distance_points_acc(points_list):
    distances = []
    acc = 0
    for i in range(1,len(points_list)):
        previous_points_list = utm.from_latlon(points_list[i-1][0], points_list[i-1][1])
        current_points_list = utm.from_latlon(points_list[i][0], points_list[i][1])
        dist = euclidean_dist((previous_points_list[0], previous_points_list[1]),(current_points_list[0], current_points_list[1]))
        acc = acc + dist
        distances.append(acc)
    return distances


def identify_bus_stops_by_distance(current_shape_id, df_bus_stops):
	df_bus_stops['shape_dist_traveled'] = df_bus_stops['shape_dist_traveled'].fillna(0)
	print(df_bus_stops.head())
	df_bus_stops['point_type'] = 'stop'
	#df_bus_stops.to_csv("bus_stops.csv", index=False)
	print(len(df_bus_stops))
	df_subset_shapes = df_shapes[(df_shapes["shape_id"] == current_shape_id)]
	df_subset_shapes.drop_duplicates(subset='shape_dist_traveled', keep="first", inplace=True)
	df_subset_shapes['point_type'] = 'shape'
	#df_subset_shapes.to_csv("subset_shapes.csv", index=False)
	print(df_subset_shapes.head())
	print(len(df_subset_shapes))
	df1 = df_subset_shapes[['shape_pt_lat', 'shape_pt_lon', 'shape_dist_traveled', 'point_type']]
	df1.rename(columns={'shape_pt_lat': 'latitude', 'shape_pt_lon': 'longitude'}, inplace=True)
	df2 = df_bus_stops[['stop_lat', 'stop_lon', 'shape_dist_traveled', 'point_type']]
	df2.rename(columns={'stop_lat': 'latitude', 'stop_lon': 'longitude'}, inplace=True)
	df_shape_and_stop_points = pd.concat([df1,df2], ignore_index=True)
	df_shape_and_stop_points = df_shape_and_stop_points.sort_values(by=['shape_dist_traveled'], ascending=True)
	df_shape_and_stop_points.to_csv('shape_and_points_1.csv', index=False)

	stop_points = df_shape_and_stop_points[df_shape_and_stop_points.duplicated(['shape_dist_traveled'], keep=False)]
	stop_points.to_csv('bus_stops_1.csv', index=False)

#TODO: change the position of current_shape_id, trip_headsign, direction_id, route_id
def calibrate(trip_id):
	try:
		current_trip = df_trips[(df_trips["trip_id"] == trip_id)]
		current_trip_list = current_trip.values.tolist()
		current_shape_id = current_trip_list[0][5] #get the shape_id value for this trip_id
		o_route_id = current_trip_list[0][0] #original route id
		o_trip_headsign = current_trip_list[0][3]
		o_direction_id = current_trip_list[0][4]

		# get files stops and stop_times
		df_bus_stops = get_bus_stops_dataframe(trip_id)
		#df_bus_stops.to_csv("bus_stops.csv", index=False)

		# identify the bus stops points on shape
		anchor_points, route_shape_stops = identify_bus_stops(current_shape_id, df_bus_stops)
		#identify_bus_stops_by_distance(current_shape_id, df_bus_stops)
		#return
		#anchor_points.to_csv("anchor_points.csv", index=False)
		#route_shape_stops.to_csv("route_shape_stops.csv", index=False)

		route_shape_stops.drop_duplicates(subset=['arrival_time'], keep='last', inplace=True)
		# calibrate
		route_shape_stops_aslist = route_shape_stops.values.tolist() 
		calibrated_traj = pd.DataFrame()   
		for j in range(1, len(route_shape_stops_aslist)):
			previous_sample = route_shape_stops_aslist[j-1]
			current_sample = route_shape_stops_aslist[j]
			current_anchor_points = anchor_points[(anchor_points["shape_pt_sequence"] > previous_sample[6]) & (anchor_points["shape_pt_sequence"] < current_sample[6])]
			anchors = current_anchor_points.values.tolist()
			#goal_point = current_sample 
			goal_point = (current_sample[0], current_sample[1], current_sample[4], current_sample[5])
			anchor_index = -1      
			if len(anchors) > 0: # if there are anchor points between the previous and current points, use them
				anchor_index = 0
				goal_point = anchors[anchor_index]
			goal_point_utm = utm.from_latlon(goal_point[2], goal_point[3])
			previous_point_utm = utm.from_latlon(previous_sample[4], previous_sample[5])        
			d = euclidean_dist(previous_point_utm, goal_point_utm)
			points_list = [(previous_sample[4], previous_sample[5])]
			for anchor_point in anchors: 
				points_list.append((anchor_point[2], anchor_point[3]))
			points_list.append((current_sample[4], current_sample[5]))
			distances = calculate_distance_points_acc(points_list)                    
			total_distance = (distances[len(distances)-1])  
			temporal_distance = current_sample[0] - previous_sample[0]     
			initial_point_utm = previous_point_utm
			done = False
			speed = (total_distance / temporal_distance)
			delta_space = speed * SAMPLING_RATE
			current_time = previous_sample[0]
			elapsed_distance = 0
			calibrated_traj = calibrated_traj.append({'x':initial_point_utm[1], 'y':initial_point_utm[0], 'timestamp':current_time, 'trip_id': trip_id, 'calibrated':0, 'speed':speed, 'stop_id':previous_sample[10], 'stop_name':previous_sample[13], 'route_id':o_route_id, 'trip_headsign':o_trip_headsign, 'direction_id':o_direction_id, 'shape_id':current_shape_id}, ignore_index=True)
			if len(anchors) > 0:
				while elapsed_distance < total_distance: 
					while d > delta_space:
						y, x = linear_interpolation(initial_point_utm, goal_point_utm, delta_space)
						current_time = current_time + 1
						calibrated_traj = calibrated_traj.append({'x':x, 'y':y, 'timestamp':current_time, 'trip_id': trip_id, 'calibrated':1, 'speed':delta_space, 'stop_id': "ZZZZZ", 'stop_name':"WWWWW", 'route_id':o_route_id, 'trip_headsign':o_trip_headsign, 'direction_id':o_direction_id, 'shape_id':current_shape_id}, ignore_index=True)
						initial_point_utm = (y,x)
						elapsed_distance = elapsed_distance + delta_space
						d = euclidean_dist(initial_point_utm, goal_point_utm)
					if anchor_index == len(anchors)-1:
						goal_point_utm = utm.from_latlon(current_sample[4], current_sample[5])
						if total_distance - elapsed_distance > delta_space:
							temporal_distance = current_sample[0] - current_time
							d = euclidean_dist(initial_point_utm, goal_point_utm)
							speed = d/temporal_distance
							delta_space = speed * SAMPLING_RATE
							temporal_count = (temporal_distance / SAMPLING_RATE) - 1
							j = 1
							while temporal_count > 0:
								y, x = linear_interpolation(initial_point_utm, goal_point_utm, delta_space * j)                         
								current_time = current_time + SAMPLING_RATE
								calibrated_traj = calibrated_traj.append({'x':x, 'y':y, 'timestamp':current_time, 'trip_id': trip_id, 'calibrated':1,'speed':delta_space, 'stop_id':"ZZZZZ", 'stop_name':"WWWWW", 'route_id':o_route_id, 'trip_headsign':o_trip_headsign, 'direction_id':o_direction_id, 'shape_id':current_shape_id}, ignore_index=True)
								j = j + 1
								temporal_count = temporal_count - 1
						break
					else:
						anchor_index = anchor_index + 1
						goal_point = anchors[anchor_index]
						goal_point_utm = utm.from_latlon(goal_point[2], goal_point[3])
					d = euclidean_dist(initial_point_utm, goal_point_utm)
			else:
				temporal_count = (temporal_distance / SAMPLING_RATE) - 1
				j = 1
				while temporal_count > 0:
					y, x = linear_interpolation(previous_point_utm, goal_point_utm, delta_space * j)                         
					current_time = current_time + SAMPLING_RATE
					calibrated_traj = calibrated_traj.append({'x':x, 'y':y, 'timestamp':current_time, 'trip_id': trip_id, 'calibrated':1, 'speed':speed, 'stop_id':"ZZZZZ", 'stop_name':"WWWWW", 'route_id':o_route_id, 'trip_headsign':o_trip_headsign, 'direction_id':o_direction_id, 'shape_id':current_shape_id}, ignore_index=True)
					j = j + 1
					temporal_count = temporal_count - 1
		final_point_utm = utm.from_latlon(current_sample[4], current_sample[5])
		calibrated_traj = calibrated_traj.append({'x':final_point_utm[1], 'y':final_point_utm[0], 'timestamp':current_sample[0], 'trip_id': trip_id, 'calibrated':0, 'speed':0, 'stop_id':current_sample[10], 'stop_name':current_sample[13], 'route_id':o_route_id, 'trip_headsign':o_trip_headsign, 'direction_id':o_direction_id, 'shape_id':current_shape_id}, ignore_index=True)
		calibrated_traj_latlong = calibrated_traj[['y', 'x']].apply(get_latlong_df, axis=1)
		calibrated_traj_latlong['timestamp'] = calibrated_traj[['timestamp']]
		calibrated_traj_latlong = calibrated_traj_latlong.rename(columns={0: 'latitude', 1: 'longitude'})
		calibrated_traj_latlong['speed'] = calibrated_traj[['speed']]
		calibrated_traj_latlong['calibrated'] = calibrated_traj[['calibrated']]
		calibrated_traj_latlong['stop_id'] = calibrated_traj[['stop_id']]
		calibrated_traj_latlong['stop_name'] = calibrated_traj[['stop_name']]
		calibrated_traj_latlong['route_id'] = calibrated_traj[['route_id']]
		calibrated_traj_latlong['trip_headsign'] = calibrated_traj[['trip_headsign']]
		calibrated_traj_latlong['direction_id'] = calibrated_traj[['direction_id']]
		calibrated_traj_latlong['shape_id'] = calibrated_traj[['shape_id']]
		calibrated_traj_latlong['trip_id'] = calibrated_traj[['trip_id']]
		calibrated_traj_latlong[['trip_id','timestamp', 'latitude', 'longitude', 'speed', 'stop_id','stop_name','route_id', 'trip_headsign', 'direction_id', 'shape_id', 'calibrated']].to_csv("/home/clayson/globecom_extension/datasets/calibrated/washington_weekday/calibrated_traj_"+str(trip_id)+"_.csv", index=False)
	#return calibrated_traj_latlong[['trip_id','timestamp', 'latitude', 'longitude', 'speed', 'stop_id','stop_name','route_id', 'trip_headsign', 'direction_id', 'shape_id', 'calibrated']]
	except:
		print('Error: '+ str(trip_id))

def main_calibrated_trips(all_trip_ids):
	pool = mp.Pool(processes=20)
	pool.map(calibrate, all_trip_ids)
	pool.close()
	pool.join()

if __name__ == '__main__':
	all_trip_ids = list(df_trips['trip_id'])
	main_calibrated_trips(all_trip_ids)
	
	
	






	