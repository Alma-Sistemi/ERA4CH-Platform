import streamlit as st
import googlemaps
from utils_for_heatpmaps import *
from building_inspection_assistant_component import render_google_maps_new



STREET_VIEW_API_KEY = "xxxxxxxxxxxxxxxxxxxxx"
# Initialize the client
gmaps = googlemaps.Client(key=STREET_VIEW_API_KEY)
# Image path 
IMAGES_PATH = 'heatmap_images/'
INFO_PATH = 'heatmaps_info/'
figure_layout={
    'width': '100%',
    'height': '100%',
    'border': '1px solid black',
    'padding': '1px',
    'zoom_level': 16,
    'margin': '0'
}

def render_heatmaps_workflow():
    st.header('Damage Assessment Heatmaps', divider='gray')
    st.markdown("<h6 style='text-align: center; color: white;'>In this workflow you can enter a specific address of interest and a desired radius for analysis.  Multiple points are generated systematically within the specified radius around the address to ensure thorough coverage of the area.For each of these generated points, images of buildings are retrieved using the Google Street View, input to a model for evaluation of their structural integrity and feature identification. The outputs from the assessment and feature extraction are used to generate heatmaps overlaid on a map of the specified area.  </h6>", unsafe_allow_html=True)
    st.markdown("#")
    col_selections,_,col_visualization = st.columns([0.3,0.03,0.67])

 
    with col_selections:
        city_name = st.text_input('Enter the city or address of your interest:',"Chania, Crete")
        radius = st.slider('Select the radius of your interest in meters', 100, 1000, value=300,step=50)

    images_route_path = f"{IMAGES_PATH}{city_name}_{radius}_random_points_more_points/"
    info_route_path = f"{INFO_PATH}{city_name}_{radius}_random_points_more_points/"
    building_info_path = os.path.join(info_route_path, 'buildng_info.csv')

    lat, lng = get_city_coordinates(city_name, gmaps)

    start_location = f'{lat}, {lng}'
    bool_new_folder = create_images_info_folders(images_route_path, info_route_path)
    # Specify the number of lines and the number of points per line
    num_lines = 20  # Number of lines
    points_per_line = 5  # Number of points per line
    marker_location = get_points_in_circle(lat, lng, radius, num_lines, points_per_line)
    if bool_new_folder:
        print('A new city/address name - radius configuration has been given.\nCollecting images...')
        
        
        # Get street view images for each step along the route
        images, addresses, locations = get_street_view_images(STREET_VIEW_API_KEY, marker_location, adjusted_camera_angle=90)
        save_images(images, locations, prefix=images_route_path)
        print(f'{len(images)} Images have been collected!')
    else:
        print('An old city/address name-radius configuration has been given.')
        
        img_list, img_name_list = retrieve_saved_images(images_route_path)
        print(f'{len(img_list)} images have been found!')
        
    img_list, img_name_list = retrieve_saved_images(images_route_path)

    # marker_location = get_points_in_circle(lat, lng, radius, num_lines, points_per_line)
    with col_selections:
        render_google_maps_new(480, 600, marker_location,lat, lng,figure_layout,circle_radius=radius)


    kept_coord_list,scores,transposed_kept_answer_list = generate_information_for_heatmaps(info_route_path,images_route_path,img_list,img_name_list)
    filtered_data = building_info_data_processing(building_info_path)
    with col_visualization:
        available_heatmaps = ["Risk assessment", 'Number of Windows' , 'Number of Floors', 'Presence of Crack', 'Age of Building', 'Building Material']
        selected_heatmap = st.radio(
            "Select the heatmap of your interest:",
            available_heatmaps,
            index=0,horizontal=True
            )

        col_heatmap,_,col_density_plot = st.columns([0.45,0.1,0.45])
        if selected_heatmap == "Risk assessment":
            with col_heatmap:
                create_map_folium_with_heatmap(start_location, kept_coord_list, scores, radius=radius)
            # Convert list to DataFrame
            data_df = pd.DataFrame(scores, columns=['score'])

            # Use the function to create and show the plot
            figure = create_kde_plot(data_df, 'score', 'Density Plot for Risk assessment scores', 'Scores', ' Nuber Density')
            with col_density_plot:
                st.plotly_chart(figure,use_container_width=True)
        else:      
            with col_heatmap:
                create_map_folium_per_prompt(start_location, kept_coord_list, transposed_kept_answer_list, available_heatmaps.index(selected_heatmap)-1, selected_heatmap, radius=radius)
           
            ################### For density plots #####################
            if selected_heatmap ==  'Number of Windows':
                # Density plot for Number of windows
                figure = create_kde_plot(filtered_data, 'Answer 2', 'Density Plot for Number of Windows', 'Number of windows', 'Number Density')
            elif selected_heatmap == "Number of Floors":
                # Density plot for Number of floors
                figure = create_kde_plot(filtered_data, 'Answer 3', 'Density Plot for Number of Floors', 'Number of floors', 'Number Density')
            elif selected_heatmap == "Presence of Crack":
                figure = create_histogram(filtered_data, 'Answer 4','Crack Presence')
            elif selected_heatmap == "Age of Building":
                # Density plot for Age of the building
                figure = create_kde_plot(filtered_data, 'Answer 5', 'Density Plot for Age of the Building', 'Age of the building (years since construction)', 'Number Density')
            elif selected_heatmap == 'Building Material':
                figure  = create_histogram(filtered_data, 'Answer 6','Material of the building')
            with col_density_plot:
                st.plotly_chart(figure,use_container_width=True)