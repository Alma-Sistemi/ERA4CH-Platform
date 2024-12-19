from transformers import ViltProcessor, ViltForQuestionAnswering
import csv
import os     
from models.YoloWorldEfficientSam.yoloWorld_EfficientSam import yolo_world_inference
from models.Floor_detection.Floor_detection import inference_floor_detection

def get_model(model_name):
    if model_name == 'ViLT': 
        return get_vilt() 
    
def get_vilt():    
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    return processor, model

def get_answers_and_discard(model, processor, img_list, img_name_list, 
                prompt_list, answer_to_keep, kept_images_path, discarded_images_path, csv_path):
    
    os.makedirs(kept_images_path, exist_ok = True)
    os.makedirs(discarded_images_path, exist_ok = True)
    
    answers_lists = [[] for _ in prompt_list]
    
    unique_locations = set()
    kept_images = set() 

    with open(csv_path, mode='w', newline='') as csv_file: 
        csv_writer = csv.writer(csv_file)
        header = ['Image Path', 'Lat', 'Lng'] + [f'Answer {i+1}' for i in range(len(prompt_list))]
        csv_writer.writerow(header)
        
        for image_file, image in zip(img_name_list, img_list):
            # Extract the lat and long from the image filename
        
            #lat_lng = image_file.split('.')[0]
            lat_lng = os.path.splitext(image_file)[0]
            if '_' in lat_lng: 
                lat_lng = lat_lng.split('_')[0]
           
            lat, lng = lat_lng.split(',')
            
            # Check if the location is already processed
            if lat_lng in unique_locations:
                continue
            unique_locations.add(lat_lng)
            
            row = [os.path.join(kept_images_path, image_file), lat, lng]
            

# ******************************************HERE ARE  THE LINES THAT SHOULD BE EDITTED FOR THE MODELS INTEGRATION*************************************************************
            for i, prompt in enumerate(prompt_list):

                #  ****************************** the code below is useful **********************
                # encoding = processor(image, prompt, return_tensors="pt")

                # outputs = model(**encoding)
                # logits = outputs.logits
                # idx = logits.argmax(-1).item()

                # answer = model.config.id2label[idx]
                # answers_lists[i].append(answer)
                # **********************************************************************************


                # HERE THE ANSWER OF THE MODEL FOR EVERY PROMPT IS WRITTEN IN THE FILE
                if "windows" in prompt:
                    element_for_detection = "Window"
                    annotated_image,detections = yolo_world_inference(image = image,element_for_detection=element_for_detection)
                    answer = len(detections)
                    
                elif "floor" in prompt:
                     floor_annotated_image,num_of_floors = inference_floor_detection(image)
                     answer = num_of_floors
                     

                elif "non-building" in prompt:
                    annotated_image,detections = yolo_world_inference(image = image,element_for_detection="building")
                    
                    # JpegImageFile' object has no attribute 'shape' so the below line cannot be used
                    import numpy as np
                    image_np = np.array(image)
                    image_area = image_np.shape[0] * image_np.shape[1] 
                     
                    

                    building_exists = False
                    answer="yes"

                    for i in range(len(detections)):

                        bb_area = (detections[i].xyxy[0][2] - detections[i].xyxy[0][0]) * detections[i].xyxy[0][3] - detections[i].xyxy[0][1]
                        
                        coverage_percentage = (bb_area/image_area)*100
                        
                        # Change the threshold for the coverage_percentage, to keep images that bulding covers more area in the image
                        if coverage_percentage > 30:
                            building_exists = True
                            answer="no"
                            
                    
                    if answer.lower() == answer_to_keep: 
                        
                        image.save(os.path.join(kept_images_path, image_file))
                        kept_images.add(lat_lng)
                    else: 
                        
                        image.save(os.path.join(discarded_images_path, image_file))

                    
                    # if building_exists:
                    #     answer="no"

                else:
                    encoding = processor(image, prompt, return_tensors="pt")

                    outputs = model(**encoding)
                    logits = outputs.logits
                    idx = logits.argmax(-1).item()

                    answer = model.config.id2label[idx]
                    answers_lists[i].append(answer)                    

                row.append(answer)
                
                # if i==0:
                #     #Only the first prompt is related to discarding or not the images
                #     if answer.lower() == answer_to_keep: 
                #         print('SAVED IN KEPT')
                #         image.save(os.path.join(kept_images_path, image_file))
                #         kept_images.add(lat_lng)
                #     else: 
                #         print('SAVED IN DISCARDED')
                #         image.save(os.path.join(discarded_images_path, image_file))

                  
            csv_writer.writerow(row)
            
    return answers_lists


def get_answers(model, processor, img_list, img_name_list, prompt_list):
    
    answers_lists = [[]] * len(prompt_list)
    for image_file, image in zip(img_name_list, img_list):
        for i, prompt in enumerate(prompt_list):
            encoding = processor(image, prompt, return_tensors="pt")
            
            outputs = model(**encoding)
            logits = outputs.logits
            idx = logits.argmax(-1).item()

            answer = model.config.id2label[idx]
            answers_lists[i].append(answer)

    return answers_lists