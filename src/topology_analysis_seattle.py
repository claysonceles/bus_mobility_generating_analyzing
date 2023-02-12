import networkx as nx
import pandas as pd
from itertools import combinations
import multiprocessing as mp
import pymysql
import time
from math import radians, cos, sin, asin, sqrt
import utm
import csv


def execute_query(string_query):
	conn = pymysql.connect(host='localhost', port=3306, user='', passwd='', db='bustraces')
	cursor = conn.cursor()
	df = pd.read_sql(string_query, conn)
	cursor.close()
	conn.close()
	return df

# https://janakiev.com/blog/gps-points-distance-python/
def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000 # meters 


def optimized_detect_contacts(df, radius):
	edges = []
	raw_data_matrix = df.to_numpy()
	for row1, row2 in combinations(raw_data_matrix, 2):
		lat1 = row1[2]
		lon1 = row1[3]
		lat2 = row2[2]
		lon2 = row2[3]
		distance = haversine(lon1, lat1, lon2, lat2)
		if distance <= radius:
			edges.append((row1[0], row2[0]))
	return edges


def get_instantaneous_graph(df, timestamp, radius, path_filename):
	G = nx.Graph()
	raw_data_matrix = df.values.tolist()
	for row in raw_data_matrix:
		G.add_node(row[0], lat=row[2], lon=row[3], speed=row[4], route_id=row[7], trip_headsign=row[8], direction_id=row[9], shape_id=row[10])
	edges = optimized_detect_contacts(df, radius)
	G.add_edges_from(edges)
	filename = str(timestamp) + ".gpickle"
	nx.write_gpickle(G, path_filename + filename)


def create_graph(timestamp):
	radius = 500
	path_filename = "/home/clayson/globecom_extension/graphs/seattle_weekday_500m/"
	string_query = "SELECT * FROM SeattleBusWeekday20190311 WHERE timestamp="+str(timestamp)+";"
	df = execute_query(string_query)
	if not df.empty:
		get_instantaneous_graph(df, timestamp, radius, path_filename)


def write_list2csv(path_filename, output_list):
    # Open File
    result_file = open(path_filename,'w')
    # Write data to file
    for i in range(0, len(output_list)-1): 
        result_file.write(str(output_list[i]) + ",")
    result_file.write(str(output_list[len(output_list)-1])+"\n")
    result_file.close()


def write_list2csv_by_line(path_filename, output_list):
    # Open File
    result_file = open(path_filename,'w')
    # Write data to file
    for i in range(0, len(output_list)):
        result_file.write(str(output_list[i]) + "\n")
    result_file.close()


def compute_topology_metrics(graph_timestamp):
    values = []
    from_directory = "/home/clayson/globecom_extension/graphs/seattle_weekday_500m/"+str(graph_timestamp)+".gpickle"
    G = nx.read_gpickle(from_directory)
    values.append(graph_timestamp)
    values.append(G.number_of_nodes())  # Return the number of nodes in the graph.
    values.append(G.number_of_edges())  # Return the number of edges in the graph.
    values.append(nx.number_connected_components(G)) # Return the number of connected components.
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    giant = G.subgraph(Gcc[0])
    #giant = max(nx.connected_component_subgraphs(G), key=len)  # Return the largest component
    values.append(giant.number_of_nodes())  # Size of the largest component
    values.append(giant.number_of_edges())  # Number of edges of the largest connected component
    values.append(len(list(nx.isolates(G))))  # Return the number of isolate nodes in the graph.
    to_directory = "/home/clayson/globecom_extension/data_results/seattle_weekday/topology_metrics_500m/"
    write_list2csv(to_directory+str(graph_timestamp)+".csv", values)


def get_mid_point(positions):
    if(positions == []):
        return [-1,-1];
    sumx=0
    sumy=0

    positions_utm = []
    for position in positions:
        coord_utm = utm.from_latlon(position[0], position[1])
        positions_utm.append((coord_utm[0],coord_utm[1]))

    for i in positions_utm:
        sumx+=i[0]
        sumy+=i[1]
    avgx = float(sumx)/len(positions_utm)
    avgy = float(sumy)/len(positions_utm)

    lat_mid, lon_mid = utm.to_latlon(avgx, avgy, 10, 'T')
    return (lat_mid, lon_mid)


def get_components_positions(from_filename, to_filename):
	from_directory = from_filename
	G = nx.read_gpickle(from_directory)
	graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
	position_components = pd.DataFrame(columns=['lat','lon', 'intensity'])
	for i in range(0, len(graphs)):
		locations = []
		points = list(graphs[i].nodes.data())
		for j in range(0, len(points)):
			lat = points[j][1]['lat']
			lon = points[j][1]['lon']
			locations.append((lat, lon))
		location_mean = get_mid_point(locations)
		position_components = position_components.append({'lat':location_mean[0],'lon':location_mean[1], 'intensity':len(points)}, ignore_index=True)
	position_components.to_csv(to_filename, sep=',', index=False)


def compute_size_of_components(graph_timestamp):
    size_of_components = []
    from_directory = "/home/clayson/globecom_extension/graphs/seattle_weekday_500m/"+str(graph_timestamp)+".gpickle"
    G = nx.read_gpickle(from_directory)
    all_components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    for component in all_components:
        size_of_components.append(component.number_of_nodes())
    to_directory = "/home/clayson/globecom_extension/data_results/seattle_weekday/size_of_components_500m/"
    write_list2csv_by_line(to_directory+str(graph_timestamp)+".csv", size_of_components)


def get_contact_nodes(interval, filename, directory_name):
	#interval = (1600078200, 1600078299) 
	directory = directory_name
	edges = {}
	for graph_index in range(interval[0], interval[1]+1):
		try:
			G = nx.read_gpickle(directory+str(graph_index)+".gpickle")
			if G.number_of_edges() > 0:
				for edge in list(G.edges):
					key = str(edge[0])+"#"+str(edge[1])
					inverse_key  = str(edge[1])+"#"+str(edge[0])
					if key in edges or inverse_key in edges:
						edges[key].append(graph_index)
					else:
						edges[key] = [graph_index]
		except IOError:
			pass

	with open(filename, 'w') as csv_file:  
		writer = csv.writer(csv_file, escapechar=' ', quoting = csv.QUOTE_NONE)		
		for key, value in edges.items():
			timestamps = ""
			for item in value:
				timestamps = timestamps + ',' + str(item)
			writer.writerow([key, timestamps[1:]])


def get_contact_bus_lines(interval):
	#interval = (1600078200, 1600078299) 
	directory = "/home/clayson/globecom_extension/graphs/ottawa_weekdays_100m/"
	edges = {}
	for graph_index in range(interval[0], interval[1]+1):
		try:
			G = nx.read_gpickle(directory+str(graph_index)+".gpickle")
			if G.number_of_edges() > 0:
				for edge in list(G.edges):
					key = str(edge[0])+'@'+str(G.nodes[edge[0]]['route_id'])+"#"+str(edge[1])+'@'+str(G.nodes[edge[1]]['route_id'])
					inverse_key = str(edge[1])+'@'+str(G.nodes[edge[1]]['route_id'])+"#"+str(edge[0])+'@'+str(G.nodes[edge[0]]['route_id'])
					if key in edges or inverse_key in edges:
						edges[key].append(graph_index)
					else:
						edges[key] = [graph_index]
		except IOError:
			pass

	with open('/home/clayson/globecom_extension/data_results/ottawa_weekday/contacts/bus_lines/100m/contact_time_ottawa_weekday_100m.csv', 'w') as csv_file:  
		writer = csv.writer(csv_file, escapechar=' ', quoting = csv.QUOTE_NONE)		
		for key, value in edges.items():
			timestamps = ""
			for item in value:
				timestamps = timestamps + ',' + str(item)
			writer.writerow([key, timestamps[1:]])


def compute_contact_characteristics(origin_directory, destination_directory):
	input_file = open(origin_directory, 'r')
	ict_list = []
	pair_contact_list = []
	contact_duration_list = []
	for line in input_file:
		tokens = line.split(',')
		if len(tokens) > 2:
			contact_duration = 1
			last_contact_timestamp = int(tokens[1].strip())
			pair_contact = 1			
			for i in range(2, len(tokens)):
				if int(tokens[i].strip()) - int(tokens[i-1].strip()) == 1:
					contact_duration = contact_duration + 1
					last_contact_timestamp = int(tokens[i].strip())
				else:
					contact_duration_list.append(contact_duration)
					ict_list.append(int(tokens[i].strip()) - last_contact_timestamp)
					contact_duration = 1
					pair_contact = pair_contact + 1
			pair_contact_list.append(pair_contact)
			contact_duration_list.append(contact_duration)
		else:
			pair_contact_list.append(1)
			contact_duration_list.append(1)
	#write_list2csv_by_line(destination_directory+"inter_contact_time.csv", ict_list)
	#write_list2csv_by_line(destination_directory+"pairwise_contact.csv", pair_contact_list)
	write_list2csv_by_line(destination_directory, contact_duration_list)



def compute_bus_lines_contact_characteristics(origin_directory, destination_directory):
	input_file = open(origin_directory, 'r')
	ict_list = []
	pair_contact_list = []
	contact_duration_list_same_line = []
	contact_duration_list_all_lines = []
	contact_duration_list_different_lines = []
	equal_key_contact_duration = {}
	different_key_contact_duration = {}
	for line in input_file:
		tokens = line.split(',')
		if len(tokens) > 2:
			entities = tokens[0].split("#")
			line1 = entities[0].split("@")[1]
			line2 = entities[1].split("@")[1]
			if line1 == line2:
				key = line1+'#'+line2
				contact_duration = 1
				contact_duration_list = []
				for i in range(2, len(tokens)):
					if int(tokens[i].strip()) - int(tokens[i-1].strip()) == 1:
						contact_duration = contact_duration + 1
					else:
						contact_duration_list.append(contact_duration)
				if key in equal_key_contact_duration:
					equal_key_contact_duration[key].extend(contact_duration_list)
				else:
					equal_key_contact_duration[key] = [graph_index]
	write_list2csv_by_line(destination_directory+"inter_contact_time.csv", ict_list)
	write_list2csv_by_line(destination_directory+"pairwise_contact.csv", pair_contact_list)
	write_list2csv_by_line(destination_directory+"contact_duration.csv", contact_duration_list)



def get_degree_all_nodes(graph_timestamp):
	from_directory = "/home/clayson/globecom_extension/graphs/seattle_weekday_500m/"+str(graph_timestamp)+".gpickle"
	G = nx.read_gpickle(from_directory)
	degrees = [val for (node, val) in G.degree()]
	to_directory = "/home/clayson/globecom_extension/data_results/seattle_weekday/degrees_500m/"
	write_list2csv_by_line(to_directory+str(graph_timestamp)+".csv", degrees)


def main_create_graphs(interval_timestamp_weekday):
	pool = mp.Pool(processes=20)
	pool.map(create_graph, interval_timestamp_weekday)
	pool.close()
	pool.join()


def main_compute_topology_metrics(interval_timestamp_weekday):
	pool = mp.Pool(processes=20)
	pool.map(compute_topology_metrics, interval_timestamp_weekday)
	pool.close()
	pool.join()


def main_compute_size_of_components(interval_timestamp_weekday):
	pool = mp.Pool(processes=20)
	pool.map(compute_size_of_components, interval_timestamp_weekday)
	pool.close()
	pool.join()


def main_compute_all_degrees(interval_timestamp_weekday):
	pool = mp.Pool(processes=20)
	pool.map(get_degree_all_nodes, interval_timestamp_weekday)
	pool.close()
	pool.join()


if __name__ == "__main__":
	#min_max_timestamp = (1552301700, 1552373940) #Monday March 11, 2019
	min_max_timestamp = (1552303800, 1552373940) #Monday March 11, 2019
	#interval_timestamp_weekday = range(1552301700, 1552373940)
	interval_timestamp_weekday = range(1552303800, 1552373940) #  after 4:30 AM
	#main_create_graphs(interval_timestamp_weekday)
	#main_compute_topology_metrics(interval_timestamp_weekday)
	#main_compute_size_of_components(interval_timestamp_weekday)
	#main_compute_all_degrees(interval_timestamp_weekday)
	#get_components_positions("/home/clayson/globecom_extension/graphs/seattle_weekday_300m/1552316400.gpickle", "/home/clayson/globecom_extension/data_results/seattle_weekday/components_geo_positions/300m/compute_geo_comp_seattle_8_00_300m.csv")
	#get_components_positions("/home/clayson/globecom_extension/graphs/seattle_weekday_300m/1552332600.gpickle", "/home/clayson/globecom_extension/data_results/seattle_weekday/components_geo_positions/300m/compute_geo_comp_seattle_12_30_300m.csv")
	#get_components_positions("/home/clayson/globecom_extension/graphs/seattle_weekday_300m/1552350600.gpickle", "/home/clayson/globecom_extension/data_results/seattle_weekday/components_geo_positions/300m/compute_geo_comp_seattle_17_30_300m.csv")
	#get_contact_nodes(min_max_timestamp, '/home/clayson/globecom_extension/data_results/seattle_weekday/contacts/nodes/contact_time_seattle_weekday_100m.csv', "/home/clayson/globecom_extension/graphs/seattle_weekday_100m/")
	#get_contact_bus_lines(min_max_timestamp)
	#origin_directory = "/home/clayson/globecom_extension/data_results/seattle_weekday/contacts/nodes/contact_time_seattle_weekday_100m.csv"
	#destination_directory = "/home/clayson/globecom_extension/data_results/seattle_weekday/contacts/nodes/metrics/contact_time_seattle_weekday_100m.csv"
	#compute_contact_characteristics(origin_directory, destination_directory)



