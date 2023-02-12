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
	path_filename = "/home/clayson/globecom_extension/graphs/dublin_weekday_500m/"
	string_query = "SELECT * FROM DublinBusWeekday20190619 WHERE timestamp="+str(timestamp)+";"
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
    from_directory = "/home/clayson/globecom_extension/graphs/dublin_weekday_500m/"+str(graph_timestamp)+".gpickle"
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
    to_directory = "/home/clayson/globecom_extension/data_results/dublin_weekday/topology_metrics_500m/"
    write_list2csv(to_directory+str(graph_timestamp)+".csv", values)


def get_routes_from_gcc(graph_timestamp):
	from_directory = "/home/clayson/globecom_extension/graphs/dublin_weekday_500m/"+str(graph_timestamp)+".gpickle"
	G = nx.read_gpickle(from_directory)
	Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
	giant = G.subgraph(Gcc[0])
	nodes = list(giant.nodes())
	all_routes = set()
	for node in nodes:
		all_routes.add(G.nodes[node]['route_id'])
	return(all_routes)
	#to_directory = "/home/clayson/globecom_extension/data_results/dublin_weekday/gcc_100m/"
	#write_list2csv_by_line(to_directory+str(graph_timestamp)+".csv", all_routes)


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

    lat_mid, lon_mid = utm.to_latlon(avgx, avgy, 29, 'U')
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
    from_directory = "/home/clayson/globecom_extension/graphs/dublin_weekday_500m/"+str(graph_timestamp)+".gpickle"
    G = nx.read_gpickle(from_directory)
    all_components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    for component in all_components:
        size_of_components.append(component.number_of_nodes())
    to_directory = "/home/clayson/globecom_extension/data_results/dublin_weekday/size_of_components_500m/"
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
	directory = "/home/clayson/globecom_extension/graphs/dublin_weekday_300m/"
	edges = {}
	for graph_index in range(interval[0], interval[1]+1):
		try:
			G = nx.read_gpickle(directory+str(graph_index)+".gpickle")
			if G.number_of_edges() > 0:
				for edge in list(G.edges):
					key = str(edge[0])+'@'+str(G.nodes[edge[0]]['route_id'])+"#"+str(edge[1])+'@'+str(G.nodes[edge[1]]['route_id'])
					inverse_key = str(edge[1])+'@'+str(G.nodes[edge[1]]['route_id'])+"#"+str(edge[0])+'@'+str(G.nodes[edge[0]]['route_id'])
					if key in edges:
						edges[key].append(graph_index)	
					elif inverse_key in edges:
						edges[inverse_key].append(graph_index)
					else:
						edges[key] = [graph_index]
		except IOError:
			pass

	with open('/home/clayson/globecom_extension/data_results/dublin_weekday/contacts/bus_lines/contact_time_bus_lines_dublin_weekday_300m.csv', 'w') as csv_file:  
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


def compute_bus_lines_contact_characteristics(origin_directory, directory_contact_duration_same_lines, directory_contact_duration_different_lines, directory_num_contact_same_lines, directory_num_contact_different_lines, directory_ict_same_lines, directory_ict_different_lines):
	input_file = open(origin_directory, 'r')
	pair_contact_list = []
	#contact_duration_list_same_line = []
	#contact_duration_list_different_lines = []
	equal_key_contact_duration = {}
	different_key_contact_duration = {}
	equal_number_of_contacts_dict = {}
	different_number_of_contacts_dict = {}
	equal_ict_dict = {}
	different_ict_dict = {}
	for line in input_file:
		tokens = line.split(',')
		entities = tokens[0].split("#")
		line1 = entities[0].split("@")[1]
		line2 = entities[1].split("@")[1]
		key = line1+'#'+line2
		contact_duration = 1
		contact_duration_list = []
		ict_list = []
		pair_contact = 1
		if len(tokens) > 2:
			#entities = tokens[0].split("#")
			#line1 = entities[0].split("@")[1]
			#line2 = entities[1].split("@")[1]
			#key = line1+'#'+line2
			#contact_duration = 1
			#contact_duration_list = []
			#ict_list = []
			#pair_contact = 1
			last_contact_timestamp = int(tokens[1].strip())
			for i in range(2, len(tokens)):
				if int(tokens[i].strip()) - int(tokens[i-1].strip()) == 1:
					contact_duration = contact_duration + 1
					last_contact_timestamp = int(tokens[i].strip())
				else:
					contact_duration_list.append(contact_duration)
					contact_duration = 1
					pair_contact = pair_contact + 1
					ict_list.append(int(tokens[i].strip()) - last_contact_timestamp)
			contact_duration_list.append(contact_duration)
		else:
			pair_contact = 1
			contact_duration_list = [1]
			#entities = tokens[0].split("#")
			#line1 = entities[0].split("@")[1]
			#line2 = entities[1].split("@")[1]
			#key = line1+'#'+line2
		if line1 == line2:
			if key in equal_key_contact_duration:
				equal_key_contact_duration[key].extend(contact_duration_list)
			else:
				equal_key_contact_duration[key] = contact_duration_list
			if key in equal_number_of_contacts_dict:
				equal_number_of_contacts_dict[key].append(pair_contact)
			else:
				equal_number_of_contacts_dict[key] = [pair_contact]
			if key in equal_ict_dict:
				equal_ict_dict[key].extend(ict_list)
			else:
				equal_ict_dict[key] = ict_list
		else:
			inverse_key = line2+'#'+line1
			if key in different_key_contact_duration:
				different_key_contact_duration[key].extend(contact_duration_list)
			elif inverse_key in different_key_contact_duration:
				different_key_contact_duration[inverse_key].extend(contact_duration_list)
			else:
				different_key_contact_duration[key] = contact_duration_list
			if key in different_number_of_contacts_dict:
				different_number_of_contacts_dict[key].append(pair_contact)
			elif inverse_key in different_number_of_contacts_dict:
				different_number_of_contacts_dict[inverse_key].append(pair_contact)
			else:
				different_number_of_contacts_dict[key] = [pair_contact]
			if key in different_ict_dict:
				different_ict_dict[key].extend(ict_list)
			elif inverse_key in different_ict_dict:
				different_ict_dict[inverse_key].extend(ict_list)
			else:
				different_ict_dict[key] = ict_list
	file_same = open(directory_contact_duration_same_lines, "w")
	for values in equal_key_contact_duration.values():
		for value in values:
			file_same.write(str(value)+"\n")
	file_same.close()
	file_different = open(directory_contact_duration_different_lines, "w")
	for values in different_key_contact_duration.values():
		for value in values:
			file_different.write(str(value)+"\n")
	file_different.close()
	output_file = open(directory_num_contact_same_lines, "w")
	for key, values in equal_number_of_contacts_dict.items():
		output_file.write(key+","+str(sum(values))+"\n")
	output_file.close()
	#for key, values in equal_number_of_contacts_dict.items():
	#	print(key, values)
	output_file = open(directory_num_contact_different_lines, "w")
	for key, values in different_number_of_contacts_dict.items():
		output_file.write(key+","+str(sum(values))+"\n")
	output_file.close()
	output_file = open(directory_ict_same_lines, "w")
	for values in equal_ict_dict.values():
		for value in values:
			output_file.write(str(value)+"\n")
	output_file.close()
	output_file = open(directory_ict_different_lines, "w")
	for values in different_ict_dict.values():
		for value in values:
			output_file.write(str(value)+"\n")
	output_file.close()
	#write_list2csv_by_line(destination_directory+"inter_contact_time.csv", ict_list)
	#write_list2csv_by_line(destination_directory+"pairwise_contact.csv", pair_contact_list)
	#write_list2csv_by_line(destination_directory+"contact_duration.csv", contact_duration_list)


def get_degree_all_nodes(graph_timestamp):
	from_directory = "/home/clayson/globecom_extension/graphs/dublin_weekday_300m/"+str(graph_timestamp)+".gpickle"
	G = nx.read_gpickle(from_directory)
	#degrees = [(node, val) for (node, val) in G.degree()]
	#print(degrees[0:10])
	average_neighbor_degree = nx.average_neighbor_degree(G)
	degrees = pd.DataFrame(columns=['node_degree','average_neighbor_degree'])
	for key, value in average_neighbor_degree.items():
		degrees = degrees.append({'node_degree':G.degree[key],'average_neighbor_degree':value}, ignore_index=True)
	degrees.to_csv("/home/clayson/globecom_extension/data_results/dublin_weekday/assortativity_300m/"+str(graph_timestamp)+".csv", sep=',', index=False)
	#to_directory = "/home/clayson/globecom_extension/data_results/dublin_weekday/degrees_500m/"
	#write_list2csv_by_line(to_directory+str(graph_timestamp)+".csv", degrees)


def get_assortativity_degree_nodes(graph_timestamp):
	from_directory = "/home/clayson/globecom_extension/graphs/dublin_weekday_300m/"+str(graph_timestamp)+".gpickle"
	G = nx.read_gpickle(from_directory)
	degrees = pd.DataFrame(columns=['origin_node','destination_node'])
	assortativity = pd.DataFrame(columns=['timestamp','assortativity'])
	for i, j in G.edges():
		degrees = degrees.append({'origin_node':G.degree[i],'destination_node':G.degree[j]}, ignore_index=True)
		degrees = degrees.append({'origin_node':G.degree[j],'destination_node':G.degree[i]}, ignore_index=True)
	#degrees.to_csv("/home/clayson/globecom_extension/data_results/dublin_weekday/"+str(graph_timestamp)+"_assortativity_300m_temp.csv", sep=',', index=False)
	#r = nx.degree_assortativity_coefficient(G)
	#print(f"{r:3.1f}")
	#assortativity = assortativity.append({'timestamp':graph_timestamp,'assortativity':r}, ignore_index=True)
	return([degrees])
	#to_directory = "/home/clayson/globecom_extension/data_results/dublin_weekday/degrees_500m/"
	#write_list2csv_by_line(to_directory+str(graph_timestamp)+".csv", degrees)


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


def main_compute_assortativity(interval_timestamp_weekday):
	pool = mp.Pool(processes=20)
	results = pool.map(get_assortativity_degree_nodes, interval_timestamp_weekday)
	degrees = pd.DataFrame(columns=['origin_node','destination_node'])
	#assortativity = pd.DataFrame(columns=['timestamp','assortativity'])
	for result in results:
		degrees = degrees.append(result[0], ignore_index=True)
		#assortativity = assortativity.append(result[1], ignore_index=True)
	degrees.to_csv("/home/clayson/globecom_extension/data_results/dublin_weekday/degree_degree__dublin_weekday_300m.csv", index=False)
	#assortativity.to_csv("/home/clayson/globecom_extension/data_results/dublin_weekday/assortativity_dublin_weekday_300m.csv", index=False)
	pool.close()
	pool.join()

def main_compute_route_from_gcc(interval_timestamp_weekday):
	pool = mp.Pool(processes=20)
	results = pool.map(get_routes_from_gcc, interval_timestamp_weekday)
	all_routes = {}
	for result in results:
		for value in result:
			if value in all_routes:
				all_routes[value] = all_routes[value] + 1
			else:
				all_routes[value] = 1
	df = pd.DataFrame(list(zip(list(all_routes.keys()), list(all_routes.values()))), columns =['route_name', 'amount']) 
	df.to_csv("/home/clayson/globecom_extension/data_results/dublin_weekday/routes_from_gcc_dublin_500m.csv", index=False)
	pool.close()
	pool.join()


if __name__ == "__main__":
	min_max_timestamp = (1560915000, 1560985192)#1560985192) #Wednesday June 19, 2019
	#min_max_timestamp = (1560925800, 1560929400)
	interval_timestamp_weekday = range(1560915000, 1560985192)
	#main_create_graphs(interval_timestamp_weekday)
	#main_compute_topology_metrics(interval_timestamp_weekday)
	#main_compute_size_of_components(interval_timestamp_weekday)
	#main_compute_all_degrees(interval_timestamp_weekday)
	#get_components_positions("/home/clayson/globecom_extension/graphs/dublin_weekday_300m/1560928500.gpickle", "/home/clayson/globecom_extension/data_results/dublin_weekday/components_geo_positions/300m/compute_geo_comp_dublin_8_15_300m.csv")
	#get_components_positions("/home/clayson/globecom_extension/graphs/dublin_weekday_300m/1560943800.gpickle", "/home/clayson/globecom_extension/data_results/dublin_weekday/components_geo_positions/300m/compute_geo_comp_dublin_12_30_300m.csv")
	#get_components_positions("/home/clayson/globecom_extension/graphs/dublin_weekday_300m/1560963000.gpickle", "/home/clayson/globecom_extension/data_results/dublin_weekday/components_geo_positions/300m/compute_geo_comp_dublin_17_50_300m.csv")
	#get_contact_nodes(min_max_timestamp, '/home/clayson/globecom_extension/data_results/dublin_weekday/contacts/nodes/contact_time_dublin_weekday_500m.csv', "/home/clayson/globecom_extension/graphs/dublin_weekday_500m/")
	#origin_directory = "/home/clayson/globecom_extension/data_results/dublin_weekday/contacts/nodes/contact_time_dublin_weekday_500m.csv"
	#destination_directory = "/home/clayson/globecom_extension/data_results/dublin_weekday/contacts/nodes/metrics/contact_time_dublin_weekday_500m.csv"
	#compute_contact_characteristics(origin_directory, destination_directory)
	#main_compute_route_from_gcc(interval_timestamp_weekday)
	#get_contact_bus_lines(min_max_timestamp)
	#compute_bus_lines_contact_characteristics("/home/clayson/globecom_extension/data_results/dublin_weekday/contacts/bus_lines/contact_time_bus_lines_dublin_weekday_300m.csv",
	#	"/home/clayson/globecom_extension/data_results/dublin_weekday/contacts/bus_lines/metrics/contact_time_same_bus_lines_dublin_weekday_300m.csv",
	#	"/home/clayson/globecom_extension/data_results/dublin_weekday/contacts/bus_lines/metrics/contact_time_different_bus_lines_dublin_weekday_300m.csv",
	#	"/home/clayson/globecom_extension/data_results/dublin_weekday/contacts/bus_lines/metrics/number_of_contacts_same_bus_lines_dublin_weekday_300m.csv",
	#	"/home/clayson/globecom_extension/data_results/dublin_weekday/contacts/bus_lines/metrics/number_of_contacts_different_bus_lines_dublin_weekday_300m.csv",
	#	"/home/clayson/globecom_extension/data_results/dublin_weekday/contacts/bus_lines/metrics/ict_same_bus_lines_dublin_weekday_300m.csv",
	#	"/home/clayson/globecom_extension/data_results/dublin_weekday/contacts/bus_lines/metrics/ict_different_bus_lines_dublin_weekday_300m.csv")




