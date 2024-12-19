import streamlit as st
import streamlit.components.v1 as components
import gmaps
import googlemaps
from streetview import search_panoramas,get_streetview
import math 
from streamlit_image_select import image_select
import copy
from ipywidgets import embed #ipywidgets version 7.7.1 is required
from utils.utils import get_street_view_images,closest_panos,calculate_distances
from models.YoloWorldEfficientSam.yoloWorld_EfficientSam import yolo_world_inference,efficientSam_YoloWorld,video_yolo_world_inference,video_efficientSam_yolo_world_inference


gmaps_key = googlemaps.Client(key="AIzaSyAv2H4eePhGGitXq-_u0bXpltDgrigqb4Y")

@st.cache_data
def render_single_address_street_view(address,default_addr):
    
    if address[0].isalpha():
        if address == "":
            g = gmaps_key.geocode(address=default_addr)
        else:
            g = gmaps_key.geocode(address=address)
            #Display map, centered to the input address point
        lat = g[0]["geometry"]["location"]["lat"]
        long = g[0]["geometry"]["location"]["lng"]
    else:
        lat, long = address.split(",")
        lat = float(lat)
        long = float(long)

    marker_locations = [(lat,long)]
    return marker_locations, lat, long

@st.cache_data
def render_go_for_a_walk_street_view(address_1,address_2,default_addr_2,default_addr):

    if address_1[0].isalpha() and address_2[0].isalpha():
        if address_1 == "":
            g_1 = gmaps_key.geocode(address=default_addr)
        else:
            g_1 = gmaps_key.geocode(address=address_1)
            #Display map, centered to the input address point
        lat = g_1[0]["geometry"]["location"]["lat"]
        long = g_1[0]["geometry"]["location"]["lng"]

        if address_2 == "":
            g_2 = gmaps_key.geocode(address=default_addr_2)
        else:
            g_2 = gmaps_key.geocode(address=address_2)
            #Display map, centered to the input address point
        lat_2 = g_2[0]["geometry"]["location"]["lat"]
        long_2 = g_2[0]["geometry"]["location"]["lng"]
    else:
        lat, long = address_1.split(",")
        lat = float(lat)
        long = float(long)

        lat_2, long_2 = address_2.split(",")
        lat_2 = float(lat_2)
        long_2 = float(long_2)

    marker_locations = [(lat,long),(lat_2,long_2)] 
    lat_diff = lat_2 - lat
    long_diff = long_2 - long

    threshold_diff = 0.0004
    lat_list = []
    long_list = []

    #include the first point in lists
    lat_list.append(lat)
    long_list.append(long)

    new_lat = copy.copy(lat)
    new_long = copy.copy(long)
    # Find the intermediate points(lat, long) between the starting and the ending address
    while abs(lat_diff) > threshold_diff:
        if lat_diff > 0:
            new_lat = new_lat + threshold_diff
            lat_list.append(new_lat)
            lat_diff = lat_diff - threshold_diff
        else:
            new_lat = new_lat - threshold_diff
            lat_list.append(new_lat)
            lat_diff = lat_diff + threshold_diff    

    while abs(long_diff) > threshold_diff:
        if long_diff > 0:
            new_long = new_long + threshold_diff
            long_list.append(new_long)
            long_diff = long_diff - threshold_diff
        else:
            new_long = new_long - threshold_diff
            long_list.append(new_long)
            long_diff = long_diff + threshold_diff           
    # the intermedate points for lat and long may not be the same, so make the length of the list same, by repeating the last point of the small list
    while len(long_list) != len(lat_list):
        if len(long_list) > len(lat_list):
            lat_list.append(lat_2)
        else:
            long_list.append(long_2)
    #include the last point in lists
    lat_list.append(lat_2)
    long_list.append(long_2)

    return marker_locations, lat_list, long_list, lat, long

@st.cache_data
def render_google_maps(height, width, marker_locations,lat, long,figure_layout):
    
    map = gmaps.figure(layout = figure_layout,center = (lat,long),zoom_level=17)
    markers = gmaps.marker_layer(marker_locations)
    map.add_layer(markers)
    snippet = embed.embed_snippet(views=map)
    html = embed.html_template.format(title="", snippet=snippet)
    components.html(html, height=height,width=width)
    print("rendering mapsss")

@st.cache_data
def render_google_maps_new(height, width, marker_locations, lat, long, figure_layout,circle_radius=None):

    api_key = "AIzaSyAv2H4eePhGGitXq-_u0bXpltDgrigqb4Y"
    # Function to generate markers script
    def generate_markers_script(marker_locations):
        script = ""
        for marker in marker_locations:
            script += f"""
            new google.maps.Marker({{
              position: {{lat: {marker[0]}, lng: {marker[1]}}},
              map: map
            }});
            """
        return script

    # Generate the markers script
    markers_script = generate_markers_script(marker_locations)

    # Define the HTML for the Google Maps iframe
    html_code = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <title>Google Map</title>
        <st src="https://maps.googleapis.com/maps/api/js?key={api_key}&callback=initMap" async defer></script>
        <style>
          #map {{
            height: 100%;
          }}
          html, body {{
            height: 100%;
            margin: {figure_layout.get('margin', '0')};
            padding: {figure_layout.get('padding', '0')};
          }}
        </style>
        <script>
          function initMap() {{
            var center = {{lat: {lat}, lng: {long}}};
            var map = new google.maps.Map(document.getElementById('map'), {{
              zoom: {figure_layout.get('zoom_level', 13)},
              center: center
            }});
            {markers_script}



            // Conditionally add a circle if radius is provided
            if ({circle_radius} !== null) {{
              var circle = new google.maps.Circle({{
                strokeColor: '#FF0000',
                strokeOpacity: 0.8,
                strokeWeight: 2,
                fillColor: '#FF0000',
                fillOpacity: 0.35,
                map: map,
                center: center,
                radius: {circle_radius}  // radius in meters
              }});
            }}



          }}
        </script>
      </head>
      <body>
        <div id="map" style="width:{figure_layout.get('width', '100%')}; height:{figure_layout.get('height', '100%')}; border:{figure_layout.get('border', 'none')}; padding:{figure_layout.get('padding', '1px')};"></div>
      </body>
    </html>
    """

    # Embed the HTML into the Streamlit app
    components.html(html_code, height=height, width=width)





def render_single_address_right_column(lat,long):

    # Search panoramas from street view based on the input address point (lat,long)
    panos = search_panoramas(lat=lat, lon=long)
    print("PANORAMAS ARE ", panos)

    distances_from_each_pano = calculate_distances(panos,lat,long)

    # You can check the dates of the images, so you will get the most recent ones
    closest_panos_list = closest_panos(distances_from_each_pano, panos, 5)

    angle_fixed = 180
    images = []
    for cl_pan in closest_panos_list:
        images.append(get_street_view_images(cl_pan.pano_id,angle_fixed))

    selected_image = image_select(
        label="Select the best image - Default Camera Angle: 180",
        images=images)

    angle = st.slider('Select the angle of camera that you prefer', 0, 270, value=180,step=45)
    _,col22, _ = st.columns([0.3,0.4,0.3])
    with col22:
        for i in range(len(images)):
            if(images[i] == selected_image and angle != angle_fixed):
                selected_image = get_street_view_images(closest_panos_list[i].pano_id,angle)
                
        markdown_text = f"<p style='text-align: center ;font-size: 18px; color: white;'>Camera Angle: {angle}</p>"
        # Display the Markdown text using st.markdown()
        st.markdown(markdown_text, unsafe_allow_html=True)
        st.image(selected_image,use_column_width=True)
        selected_image.save("download_image.png")

        with open("download_image.png","rb") as image_file:
            download_button_2 = st.download_button(
                label="Download image",
                data=image_file,
                file_name="image_in_"+str(angle) +".png",
                mime="image/png",
                use_container_width=True,
                key="button_2"
            )
        return selected_image

def render_images_retrieval_and_selection(lat_list,long_list,lat,long,num_of_point):
    panos = search_panoramas(lat=lat_list[num_of_point], lon=long_list[num_of_point])
    distances_from_each_pano = calculate_distances(panos,lat,long)
    # You can check the dates of the images, so you will get the most recent ones
    closest_panos_list = closest_panos(distances_from_each_pano, panos, 5)

    angle_fixed = 180
    images = []
    for cl_pan in closest_panos_list:
        images.append(get_street_view_images(cl_pan.pano_id,angle_fixed))

    selected_image = image_select(
        label="Select the best image out of five closer images - Default Camera Angle:180",
        images=images)
    return images,angle_fixed,closest_panos_list,selected_image




def render_changed_angle_image_and_map(lat_list,long_list,num_of_point,angle_fixed,angle,_closest_panos_list,_images,_selected_image,figure_layout):

    col21, _ ,col22 = st.columns([0.45,0.1,0.45])
    
    with col22:
        marker_location = [(lat_list[num_of_point], long_list[num_of_point])] 
        markdown_text = f"<p style='text-align: center ;font-size: 18px; color: white;'>Google Maps Pin for intermediate point: {num_of_point}</p>"
        # Display the Markdown text using st.markdown()
        st.markdown(markdown_text, unsafe_allow_html=True)
        # render_google_maps(480, 480, marker_location,lat_list[num_of_point], long_list[num_of_point],figure_layout)
        render_google_maps_new(480, 480, marker_location,lat_list[num_of_point], long_list[num_of_point],figure_layout)
    selected_image = _selected_image
    with col21:
        for i in range(len(_images)):
            if(_images[i] == selected_image and angle != angle_fixed):
                    selected_image= get_street_view_images(_closest_panos_list[i].pano_id,angle)
                
        # Use f-string to inject the `angle` variable value into the Markdown text
        markdown_text = f"<p style='text-align: center ;font-size: 18px; color: white;'>Intermediate point: {num_of_point}, Camera Angle: {angle}</p>"
        # Display the Markdown text using st.markdown()
        st.markdown(markdown_text, unsafe_allow_html=True)
        st.image(selected_image,use_column_width=True)

        return selected_image



def render_gor_for_a_walk_right_column(lat_list,long_list,lat,long,num_of_point,figure_layout):

    images,angle_fixed,closest_panos_list,selected_image = render_images_retrieval_and_selection(lat_list,long_list,lat,long,num_of_point)
    angle = st.slider('Select the angle of camera that you prefer', 0, 270, value=180,step=45)
    selected_image = render_changed_angle_image_and_map(lat_list,long_list,num_of_point,angle_fixed,angle,closest_panos_list,images,selected_image,figure_layout)
    return selected_image,angle

################################################################################ building_inspection_component #################################
def building_inspection_component():

    st.header('Building Inspection Assistant', divider='gray')

    st.markdown("<h2 style='text-align: left; color: white;'>1. Image Retrieval</h2>", unsafe_allow_html=True)

    st.markdown("<h6 style='text-align: center; color: white;'>Select between 'Single address' or 'Go for a walk' mode. For the 'Single address' mode, enter the address or the coordinates of the building that you want to inspect. In case of 'Go for a walk',  enter a starting and destination point. Images from multiple intermediate points will be captured like walking from the starting point to the destination. The closer images to the point in the map, will be visualised at the right handside. Use the slider to change the capture angle of each image.</h6>", unsafe_allow_html=True)
    default_addr = "Nearchou,3,Chania"


    # figure_layout = {
    #     'width': '800px',
    #     'height': '770px',
    #     'border': '1px solid black',
    #     'padding': '1px'
    # }
    figure_layout={
        'width': '100%',
        'height': '100%',
        'border': '1px solid black',
        'padding': '1px',
        'zoom_level': 17,
        'margin': '0'
    }
    gmaps.configure(api_key="AIzaSyAv2H4eePhGGitXq-_u0bXpltDgrigqb4Y") # Your Google API key
    # gmaps_key = googlemaps.Client(key="AIzaSyAv2H4eePhGGitXq-_u0bXpltDgrigqb4Y")

    st.markdown("#")

    col1,_,col2 = st.columns([0.4,0.03,0.57])
    with col1:
        single_or_two_adds = st.radio(
        "Select between 'Single address' or 'Go for a walk' mode:",
        ["Single address", "Go for a walk" ],
        index=0,horizontal=True
        )
        if single_or_two_adds == "Single address":
            address = st.text_input('Enter the address of your interest or coordinates in latitude,longtitude format',default_addr)
            marker_locations, lat, long = render_single_address_street_view(address,default_addr)

        elif single_or_two_adds == "Go for a walk":
            default_addr_2 = "Nearchou,23,Chania"

            address_1 = st.text_input('Enter the starting address of your interest or coordinates in latitude,longtitude format',default_addr)
            address_2 = st.text_input('Enter the ending address of your interest in the same format as the starting one ',default_addr_2)
            marker_locations, lat_list, long_list, lat, long = render_go_for_a_walk_street_view(address_1,address_2,default_addr_2,default_addr)


        # render_google_maps(700, 700, marker_locations ,lat, long,figure_layout)
        render_google_maps_new(700, 700, marker_locations ,lat, long,figure_layout)

    with col2:
        if single_or_two_adds == "Go for a walk":
            num_of_point = st.slider('Select the intermediate point of your interest', 0, len(lat_list)-1, value=0,step=1)
            selected_image,angle = render_gor_for_a_walk_right_column(lat_list,long_list,lat,long,num_of_point,figure_layout)
            selected_image.save("download_image.png")
            with open("download_image.png","rb") as image_file:
                download_button = st.download_button(
                    label="Download image",
                    data=image_file,
                    file_name="image_in_"+str(angle) +".png",
                    mime="image/png",
                    use_container_width=True
                )
            

        elif single_or_two_adds == "Single address":
            selected_image = render_single_address_right_column(lat,long)




    st.markdown("#")
    st.markdown("<h2 style='text-align: left; color: white;'>2. Facade Element Detection</h2>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: white;'>When the best image was selected, you can perform object detection or segmentation using YoloWOrld model and EfficientSAM respectively, in order to detect the facade elements of your interest from the available ones .</h6>", unsafe_allow_html=True)

    st.markdown("#")
    _,yolocol1,yolocol2,_ = st.columns([0.35,0.20,0.2,0.25])

    with yolocol1:

        element_for_detection_2 = st.radio(
            "Choose the element that you want to detect:",
            ["Window", "Door", "Balcony"],
            index=0
            )

        # st.write("You selected:", element_for_detection_2)

    with yolocol2:
        detection_technique = st.radio(
            "Choose the detection method:",
            ["Object detection", "Instance segmentation"],
            index=0
            )

    st.markdown("#")
    _,yolocol11,_ = st.columns([0.45,0.1,0.45])
    with yolocol11:
        yolo_world_button = st.button(
            'Run ' + detection_technique)
    _,yolocol12,_ = st.columns([0.3,0.4,0.3])
    with yolocol12:
        if yolo_world_button and detection_technique == "Object detection":
            annotated_image,detections = yolo_world_inference(image = selected_image,element_for_detection=element_for_detection_2)
            markdown_text = f"<p style='text-align: center ;font-size: 18px; color: white;'>{detection_technique} - Selected facade element: {element_for_detection_2}</p>"
            # Display the Markdown text using st.markdown()
            print("Num of detections is ", len(detections))
            st.markdown(markdown_text, unsafe_allow_html=True)
            st.image(annotated_image,use_column_width=True)
        elif yolo_world_button and detection_technique == "Instance segmentation":

            annotated_image = efficientSam_YoloWorld(selected_image,element_for_detection_2)
            markdown_text = f"<p style='text-align: center ;font-size: 18px; color: white;'>{detection_technique} - Selected facade element: {element_for_detection_2}</p>"
            # Display the Markdown text using st.markdown()
            st.markdown(markdown_text, unsafe_allow_html=True)
            st.image(annotated_image,use_column_width=True)