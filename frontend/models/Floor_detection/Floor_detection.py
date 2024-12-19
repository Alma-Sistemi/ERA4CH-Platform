import cv2
import numpy as np
import os

# Loading weights and config file
#.cfg file is modified cfg file with modification for classes and filter size

def inference_floor_detection(image):
    
    if "UploadedFile" not in  str(type(image)):
        file_name = image.filename
        file_name = file_name.split("/")
        file_name = file_name[-1]
        
    else:    
        file_name = image.name
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    file_path = os.path.join("uploads", file_name)  # Define your desired directory

    if "UploadedFile" not in  str(type(image)):
        print("the type is ", str(type(image)))
        image.save(file_path, format='JPEG')
    else:
        with open(file_path, "wb") as f:
            f.write(image.getvalue())

    SOURCE_IMAGE_PATH = "./uploads/" + file_name


    net = cv2.dnn.readNet("yolov3.cfg","yolov3_last.weights")
    layer_names = net.getLayerNames()
    # output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    output_layers = [layer_names[idx - 1] for idx in net.getUnconnectedOutLayers()]
    #target class to detect
    classes = ["floor"]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Loading image
    img = cv2.imread(SOURCE_IMAGE_PATH)
    scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    height, width, channels = img.shape
    # finding object using open cv DNN
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (2, 2, 2), True, crop=False)
    net.setInput(blob)
    output_of_layer = net.forward(output_layers)
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    num_detection=0
    for output in output_of_layer:
        for detection in output: #all detection in output but we need threshold to select which detection to take
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3: # how better is detection, more the value more sure the model is taking for detection
                num_detection=num_detection+1
                centroid_x = int(detection[0] * width)
                centeroid_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(centroid_x - w / 2)
                y = int(centeroid_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    # print("Total no. of floors in building=",num_detection)
    #Non maxima supression other NMS like plain NS can be used as well
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # cv2.putText(img, label, (x, y + 30), font, 3, color, 2)

    print("Total no. of floors in building=",len(indexes))
    return img,len(indexes)
