from IPython.display import IFrame
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import requests
# import polyline
# import folium 
# import gmplot
import random
import json
import math
import sys
from folium.plugins import HeatMap
import os 
import branca.colormap as cm
import folium
from matplotlib import cm as plt_cm
import streamlit.components.v1 as components

import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from vlm_models_for_heatmaps import *
import pandas as pd

def building_info_data_processing(building_info_path):

    # Load the CSV file
    data = pd.read_csv(building_info_path)

    # Filter the data for Answer 1 = 'no'
    filtered_data = data[data['Answer 1'] == 'no'].copy()

    # Replace specific values
    filtered_data['Answer 2'] = filtered_data['Answer 2'].replace('many', 101)
    filtered_data['Answer 5'] = filtered_data['Answer 5'].replace('10 years', 10)

    # Convert relevant columns to numeric if they are not already
    columns_to_convert = ['Answer 2', 'Answer 3', 'Answer 5']
    filtered_data[columns_to_convert] = filtered_data[columns_to_convert].apply(pd.to_numeric, errors='coerce')


    # Ensure that the ages are non-negative
    filtered_data['Answer 5'] = filtered_data['Answer 5'].apply(lambda x: max(x, 0))

    # Drop rows with NaN values in these columns
    filtered_data = filtered_data.dropna(subset=columns_to_convert)

    # Convert inf values to NaN
    filtered_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return filtered_data



def create_kde_plot(data, column, title, xlabel, ylabel):
    density = gaussian_kde(data[column].dropna(), bw_method=0.3)
    x = np.linspace(0, data[column].max(), 1000)
    y = density(x)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Density'))
    fig.add_trace(go.Scatter(x=data[column], y=np.zeros_like(data[column]) - 0.002, mode='markers', name='Datapoints', marker=dict(color='lightblue', symbol='line-ns-open', size=8)))
    fig.update_layout(
        title={'text': title},
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis=dict(showgrid=True, zeroline=False),
        yaxis=dict(showgrid=True)
    )
    return fig

def create_histogram(data, feature,label):

    fig = px.histogram(data, x=feature, color=feature,labels={feature: label},
                text_auto=True)
    fig.update_layout(
        title='Count Plot for ' + label,
        xaxis_title=label,
        yaxis_title='Count',
        yaxis=dict(range=[0, data[feature].value_counts().max() + 5],showgrid=True),  # Set y-axis range
        xaxis=dict(showgrid=True),
        # yaxis=dict(showgrid=True)
    )   

    return fig




def create_map_folium_per_prompt(start_location, kept_coord_list, kept_answer_list, index_of_interest, caption, radius=None):
    start_lat, start_lng = map(float, start_location.split(','))
    
    # Initialize the map at the start location
    fmap = folium.Map(location=[start_lat, start_lng], zoom_start=15)
    
    # Add radius circle
    if radius:
        folium.Circle([start_lat, start_lng], radius=radius, color='black', fill=True, fill_color='white').add_to(fmap)
    
    # Assign a unique score to each floor value
    values = sorted(set(kept_answer_list[index_of_interest]))
    print(values)
    scores = {floor: i for i, floor in enumerate(values, start=1)}
    
    # Create the colormap
    colormap = cm.linear.YlOrRd_09.scale(1, len(scores))
    colormap.caption = caption
    fmap.add_child(colormap)
    
    # Add heatmap layer
    heat_data = [[float(coord.split(', ')[0]), float(coord.split(', ')[1]), scores[kept_answer_list[index_of_interest][i]]] 
                 for i, coord in enumerate(kept_coord_list)]
    
    HeatMap(heat_data, 
            min_opacity=0.2,
            radius=15, 
            blur=15, 
            gradient={0.0: 'blue', 0.2: 'orange', 0.6: 'red'}
           ).add_to(fmap)
    
    fmap.save(f'{caption} heatmap.html')

    with open(f'{caption} heatmap.html', 'r') as file:
        html_content = file.read()
    
    # Use Streamlit's components to display the HTML
    components.html(html_content, height=578, width=700)

def create_map_folium_with_heatmap(start_location, kept_coord_list, scores, radius=None):
    start_lat, start_lng = map(float, start_location.split(','))
    
    # Initialize the map at the start location
    fmap = folium.Map(location=[start_lat, start_lng], zoom_start=15)
    
    # Add start location marker
    #folium.Marker([start_lat, start_lng], popup='Start Location', icon=folium.Icon(color='green')).add_to(fmap)
    
    if radius:
        folium.Circle([start_lat, start_lng], radius=radius, color='black', fill=True, fill_color='white').add_to(fmap)
    
    # Create the colormap
    coolwarm = plt_cm.get_cmap('coolwarm')
    colormap = cm.LinearColormap(colors=[coolwarm(i) for i in np.linspace(0, 1, coolwarm.N)], vmin=min(scores), vmax=max(scores))
    colormap.caption = 'Building Scores'
    fmap.add_child(colormap)
    
    # Add heatmap layer
    heat_data = [[float(coord.split(', ')[0]), float(coord.split(', ')[1]), score] for coord, score in zip(kept_coord_list, scores)]
    
    HeatMap(heat_data, 
            min_opacity=0.2,
            radius=15, 
            blur=15, 
            gradient={0.0: 'blue', 0.2: 'orange', 0.6: 'red'}
           ).add_to(fmap)
    
    fmap.save('route_map_folium.html')
    
    # display(IFrame('route_map_folium.html', width=700, height=500))
    with open('route_map_folium.html', 'r') as file:
        html_content = file.read()
    
    # Use Streamlit's components to display the HTML
    components.html(html_content, height=578, width=700)



weights = {
    'How many windows are there in the building?': {
        '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '6': 6, '8': 7, '10': 8, '12': 9, '20': 10, '30': 11, '100': 12, '101': 13
    },
    'What is the number of floor of the building?': {
        '0': 1, '1': 2, '2': 3, '3': 4, '10': 5
    },
    'Is there presence of cracks?': {
        'no': 1, 'yes': 5
    },
    'What is the estimated age of the building?': {
        '1': 1, '5': 2, '10': 3, '30': 4, '50': 5
    },
    'What is the building material?': {
        'wood': 5, 'metal': 4, 'stone': 3, 'stucco': 2, 'brick': 2, 'concrete': 1
    }
}
#Calculate the score for each building based on the weights.
def calculate_building_score(answers_lists, prompt_list, weights):
    scores = []
    for i in range(len(answers_lists[0])):
        score = 0
        for j, prompt in enumerate(prompt_list):
            answer = answers_lists[j][i]
            score += weights[prompt].get(answer, 0)
        scores.append(score)
    return scores

def get_building_status(score):
    if score > 20:
        return "High"
    elif score > 15:
        return "Medium"
    elif score > 10:
        return "Low"
    else:
        return "Safe"

def generate_information_for_heatmaps(info_route_path,images_route_path,img_list,img_name_list):

    building_info_path = os.path.join(info_route_path, 'buildng_info.csv')

    kept_images_path = f'{images_route_path}kept_images/'
    discarded_images_path = f'{images_route_path}discarded_images/'

    #------ Select your prompts
    prompt_0 = "Is this a non-building image?"
    prompt_1 = "How many windows are there in the building?"
    prompt_2 = "What is the number of floor of the building?"
    prompt_3 = "Is there presence of cracks?"
    prompt_4 = "What is the estimated age of the building?"
    prompt_5 = "What is the building material?"

    answer_to_keep = 'no'

    prompt_list = [prompt_0, prompt_1, prompt_2, prompt_3, prompt_4, prompt_5]

    if not os.path.exists(building_info_path): 
        print("New info to obtain...")
        #------ Select and download the model 
        model_name = 'ViLT'

        processor, model = get_model(model_name = model_name)  

        # Get the answers and discard the images with no buildings in them
        answer_list = get_answers_and_discard(model, processor, img_list, img_name_list, 
                        prompt_list, answer_to_keep, kept_images_path, discarded_images_path, building_info_path)
        
    else:
        print("This info is already stored. Retrieving...")
        # Read the CSV file and extract the answers
        answer_list = [[] for _ in prompt_list]
        
        with open(building_info_path, mode='r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            header = next(csv_reader)  # Skip the header
            for row in csv_reader:
                for i in range(len(prompt_list)):
                    answer_list[i].append(row[3 + i])



    kept_answer_list = [] 
    kept_coord_list = [] 

    with open(building_info_path, mode='r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            header = next(csv_reader)  # Skip the header
            for row in csv_reader:
                #the line below is commented out for keeping all the images
                # if row[3].strip().lower() == answer_to_keep:
                kept_coord_list.append(f'{row[1]}, {row[2]}')

                kept_answer_list.append(row[4:])
    kept_answer_list[0] = ['101' if x == 'many' else x for x in kept_answer_list[0]]
    kept_answer_list[3] = ['10' if x == '10 years' else x for x in kept_answer_list[3]]

    transposed_kept_answer_list = list(map(list, zip(*kept_answer_list)))
    scores = calculate_building_score(transposed_kept_answer_list, prompt_list[1:], weights)
    
    return kept_coord_list,scores,transposed_kept_answer_list

    
    # kept_images, kept_images_names = retrieve_saved_images(kept_images_path)
    # # Manually change values in answers_lists (example)
    # answer_list[1] = ['101' if x == 'many' else x for x in answer_list[1]]
    # answer_list[4] = ['10' if x == '10 years' else x for x in answer_list[1]]


####################################################################################################
#################################### Get places within a radius ####################################
# def get_places(lat, lng, radius, gmaps):
#     places_result = gmaps.places_nearby(location=(lat, lng), radius=radius) #Specify the number of places 
#     places = places_result['results']
#     return [(place['geometry']['location']['lat'], place['geometry']['location']['lng']) for place in places]

def get_places(lat, lng, radius, gmaps):
    places = []
    next_page_token = None

    while True:
        if next_page_token:
            response = gmaps.places_nearby(location=(lat, lng), radius=radius, page_token=next_page_token)
        else:
            response = gmaps.places_nearby(location=(lat, lng), radius=radius)
        
        places.extend(response['results'])
        
        next_page_token = response.get('next_page_token')
        if not next_page_token:
            break
        
        # Delay to prevent overloading the API with rapid requests
        import time
        time.sleep(2)
    
    return [(place['geometry']['location']['lat'], place['geometry']['location']['lng']) for place in places]



#################################### OR ####################################

def find_nearby_places(api_key, location, radius=500, place_type='point_of_interest'):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        'location': location,
        'radius': radius,
        'type': place_type,
        'key': api_key
    }
    response = requests.get(url, params=params)
    return response.json()
####################################################################################################

# Get the latitude and longitude of the city
def get_city_coordinates(city_name, gmaps):
    geocode_result = gmaps.geocode(city_name)
    location = geocode_result[0]['geometry']['location']
    return location['lat'], location['lng']

def chunk_waypoints(waypoints, chunk_size):
    for i in range(0, len(waypoints), chunk_size):
        yield waypoints[i:i + chunk_size]
        
def get_route(api_key, origin, destination, mode='driving', waypoints=None):
    ''' 
    Retrieves the walking route from the origin to the destination using Google Directions API.
    
    Args: 
        api_key (str): Private Google Cloud API key
        origin (str): The starting point of the route
        destination (str): The ending point of the route
        #TODO: 
        
    Returns: 
        dict: A dictionary containing the route information

    '''
    url = "https://maps.googleapis.com/maps/api/directions/json"
    if waypoints is None: 
        params = {
            'mode': mode,
            'origin': origin,
            'destination': destination,
            'key': api_key
        }
    else: 
        params = {
        'origin': origin,
        'destination': origin,
        'waypoints': 'optimize:true|'+'|'.join(waypoints),
        'mode': mode,
        'key': api_key
    }

    response = requests.get(url, params=params)
    return response.json()


def get_intermediate_steps(route, max_steps=100):
    ''' 
    Extracts intermediate points along the route from the Google Directions API response.

    Args: 
        route (dict): The route information as returned by the Google Directions API
        max_steps (int): The maximum number of steps to return
        
    Returns: 
        list: A list of tuples, where each tuple contains the latitude and longitude of an intermediate point
    '''
    steps = []
    for leg in route['routes'][0]['legs']:
        for step in leg['steps']:
            # Decode polyline to get all intermediate points in the step
            points = polyline.decode(step['polyline']['points'])
            steps.extend(points)
        # Adding the end location of the leg
        lat = leg['end_location']['lat']
        lng = leg['end_location']['lng']
        steps.append((lat, lng))
        
    # Uniformly sample the steps if there are more steps than max_steps
    if len(steps) > max_steps:
        indices = np.linspace(0, len(steps) - 1, max_steps, dtype=int)
        steps = [steps[i] for i in indices]
        
    return steps

def calculate_bearing(lat1, lon1, lat2, lon2):
    '''
    Calculates the bearing between two geographic coordinates.
    
    Args: 
        lat1 (float): Latitude of the first point
        lon1 (float): Longitude of the first point
        lat2 (float): Latitude of the second point
        lon2 (float): Longitude of the seconds points
        
    Returns: 
        float: The bearing from the first point to the second point in degrees
        
    '''
    delta_lon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    bearing = math.atan2(x, y)
    bearing = math.degrees(bearing)
    # Normalize to ensure it fails within the range of [0, 360)
    bearing = (bearing + 360) % 360
    return bearing


def reverse_geocoding(lat, lng, api_key): 
    ''' 
    Function that uses The Geocoding API to retrieve the address name of a given geographic coordinate
    
    Args: 
        lat (float): Latitude of the given point
        lng (float): Longitude of the given point
        aapi_Key (str): Private Google Cloud API key
        
    Returns: 
        dict: A dictionary containing the geocoding information
    '''
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        'latlng': f'{str(lat)}, {str(lng)}',
        'key': api_key
    }
    response = requests.get(url, params=params)
    return response.json()


def get_street_view_images(api_key, steps, adjusted_camera_angle, size="640x640", pitch=0, fov=90):
    '''
    Retrieves Street View images for each step along a route.
    
    Args: 
        api_key (str): Private Google Cloud API key.
        steps (list): A list of tuples, where each tuple contains the latitude and longitude of a point along the route.
        size (str): The size of the output image in pixels (default is "640x640").
        pitch (int): The up or down angle of the camera relative to the Street View vehicle (default is 0).
        fov (int): The field of view of the image in degrees (default is 90, maximum is 120).
        
    Returns: 
        list: A list of image data in binary format.
    '''
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    images = []
    addresses = []
    locations = []
    
    for i in range(len(steps) - 1):
        lat1, lon1 = steps[i]
        lat2, lon2 = steps[i + 1]
        bearing = calculate_bearing(lat1, lon1, lat2, lon2)
        heading = (bearing + adjusted_camera_angle) % 360  # 90 degrees to the right (This is the angle we want to take the photo)
        params = {
            'size': size,
            'location': f"{lat1},{lon1}",
            'heading': heading,
            'pitch': pitch,
            'fov': fov,
            'key': api_key
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            images.append(response.content)
            address = reverse_geocoding(lat1, lon1, api_key)
            # Get the full address name and number
            address = address['results'][0]['formatted_address'].split(',')[0]
            addresses.append(address)
            location = f"{lat1},{lon1}"
            locations.append(location)
                             
    return images, addresses, locations

def save_images(images, addresses, prefix):
    address_count = {}
    
    for idx, img in enumerate(images):
        address = addresses[idx]
        normalized_address = '_'.join(address.split())
        
        # Count occurrences of each address
        if normalized_address not in address_count:
            address_count[normalized_address] = 0
        else:
            address_count[normalized_address] += 1
        
        # Determine filename
        if address_count[normalized_address] == 0:
            filename = f"{prefix}/{normalized_address}.jpg"
        else:
            filename = f"{prefix}/{normalized_address}_{address_count[normalized_address]}.jpg"
        
        with open(filename, 'wb') as file:
            file.write(img)
            
def dict2json(dict, path): 
    with open(path, "w") as fp:
        json.dump(dict , fp)
        
def readjson(path):
    with open(path) as f_in: 
        return json.load(f_in)

def create_map_gmap(start_location, end_location=None, nearby_places=None,
                    route=None, steps=None, api_key=None, radius=None, show_radius=False):
    
    start_lat, start_lng = map(float, start_location.split(','))
    end_lat, end_lng = map(float, end_location.split(',')) if end_location else (None, None)
    
    # Initialize the map at the start location
    gmap = gmplot.GoogleMapPlotter(start_lat, start_lng, 15, apikey=api_key)
    
    # Plot the start location
    gmap.marker(start_lat, start_lng, 'green', title="Start Location")
    
    # Plot the end location
    if end_location:
        gmap.marker(end_lat, end_lng, 'red', title="End Location")
    
    # Plot nearby places
    if nearby_places:
        if type(nearby_places) == type([]): 
            for place in nearby_places: 
                lat = place[0]
                lng = place[1]
                gmap.marker(lat, lng, 'blue')
        else:
            for place in nearby_places['results']:
                lat = place['geometry']['location']['lat']
                lng = place['geometry']['location']['lng']
                name = place.get('name', 'No name')
                gmap.marker(lat, lng, 'blue', title=name)

    # Plot the route
    if route:
        route_lats, route_lngs = zip(*route)
        gmap.plot(route_lats, route_lngs, 'red', edge_width=2.5)
    
    # Plot the intermediate steps
    if steps:
        for lat, lng in steps:
            gmap.marker(lat, lng, 'purple', title="Step")
    
    if show_radius: 
        # Draw a circle to represent the radius
        gmap.circle(start_lat, start_lng, radius, color='green', ew=2.5)
    
    gmap.draw("route_map_gmap.html")
    
    # Display the map in the notebook
    display(IFrame('route_map_gmap.html', width=700, height=500))
    
def create_map_folium(start_location, end_location=None, nearby_places=None, route=None, steps = None):
    folium_map = folium.Map(location=[float(coord) for coord in start_location.split(',')], zoom_start=15)
    
    # Add the starting point
    folium.Marker(
        location=[float(coord) for coord in start_location.split(',')],
        popup='Start Location',
        icon=folium.Icon(color='green')
    ).add_to(folium_map)
    
    # Add the end point
    if end_location:
        folium.Marker(
            location=[float(coord) for coord in end_location.split(',')],
            popup='End Location',
            icon=folium.Icon(color='red')
        ).add_to(folium_map)
    
    if nearby_places:
        # Add nearby places
        for place in nearby_places['results']:
            lat = place['geometry']['location']['lat']
            lng = place['geometry']['location']['lng']
            name = place.get('name', 'No name')
            
            folium.Marker(
                location=[lat, lng],
                popup=name,
                icon=folium.Icon(color='blue')
            ).add_to(folium_map)
    
    if steps: 
        for lat, lng in steps: 
            folium.Marker(
                location=[lat,lng],
                popup = 'Step',
                icon =folium.Icon(color='purple')
            ).add_to(folium_map)
    if route: 
        folium.PolyLine(route, color="red", weight=2.5, opacity=1).add_to(folium_map)

    return folium_map

def get_points_in_circle(lat, lon, radius, num_lines, points_per_line):
    radius_in_degrees = radius / 111000  # Convert radius to degrees
    angle_increment = 360 / num_lines  # Calculate angle increment

    steps = []
    for i in range(num_lines):
        angle = math.radians(i * angle_increment)  # Convert angle to radians
        for j in range(1, points_per_line + 1):
            
            r = (radius_in_degrees * j) / points_per_line  # Calculate the radius for each point
            delta_lat = r * math.cos(angle)
            delta_lon = r * math.sin(angle) / math.cos(math.radians(lat))
            
            new_lat = lat + delta_lat
            new_lon = lon + delta_lon
            
            
            steps.append((new_lat, new_lon))
    return steps

def get_start_end_locations(route):
    start_point_lat = route['routes'][0]['legs'][0]['start_location']['lat']
    start_point_lng = route['routes'][0]['legs'][0]['start_location']['lng']
    #start_point_name = route['routes'][0]['legs'][0]['start_address'] #unused

    end_point_lat = route['routes'][0]['legs'][0]['end_location']['lat'] #unused
    end_point_lng = route['routes'][0]['legs'][0]['end_location']['lng'] #unused
    #end_point_name = route['routes'][0]['legs'][0]['end_address'] #unused

    start_location = f'{start_point_lat}, {start_point_lng}'
    end_location = f'{end_point_lat}, {end_point_lng}'
    
    return start_location, end_location


def create_images_info_folders(images_route_path, info_route_path):
    bool_new_folder = True 
    
    if not os.path.exists(images_route_path):
        os.makedirs(images_route_path)
    else:
        bool_new_folder = False
        
    if not os.path.exists(info_route_path):
        os.makedirs(info_route_path)
    
    return bool_new_folder

def retrieve_saved_images(path):
    img_list = []
    img_name_list  = []
    for image_file in os.listdir(path): 
        if image_file.endswith(".jpg"):
            img = Image.open(path+image_file)
            img_list.append(img)
            img_name_list.append(image_file)
            
    return img_list, img_name_list

def show_samples(img_list, bool_random = False, title = None):
    if bool_random: 
        random.shuffle(img_list)
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    image_list_to_plot = img_list[:10]

    for i in range(9):
        image = image_list_to_plot[i]
        axes[i // 3, i % 3].imshow(image)
        axes[i // 3, i % 3].axis('off')

    
    fig.suptitle(f"Sample Images{title}", fontsize=20, y=0.92)
    fig.subplots_adjust(wspace=0.01, hspace=0.1)  # Adjust spacing

    plt.show()