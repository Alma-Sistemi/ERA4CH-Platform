import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import io
import cv2
import supervision as sv

import os

from tqdm import tqdm
from inference.models.yolo_world.yolo_world import YOLOWorld
import numpy as np
import copy
import shutil



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load model in cuda
def load(device: torch.device) -> torch.jit.ScriptModule:
    if device.type == "cuda":
        # model = torch.jit.load(HOME + "/.autodistill/" + GPU_EFFICIENT_SAM_CHECKPOINT)
        model = torch.jit.load("./efficient_sam_vitt_torchscript.pt")
        model = model.to(device)
    else:
        # model = torch.jit.load(HOME + "/.autodistill/" + CPU_EFFICIENT_SAM_CHECKPOINT)
        model = torch.jit.load("./efficient_sam_vitt_torchscript.pt")
        model = model.to(device)
        print("No available CUDA, the model was loaded in CPU")

    model.eval()
    return model

class EfficientYOLOWorld_custom():

    def __init__(self):
        self.detection_model = YOLOWorld(model_id="yolo_world/l")
        self.segmentation_model = load(device=DEVICE)


    def box_segmentation_inference(self,image, pts_sampled, pts_labels,device ):

        image_np = image
        img_tensor = ToTensor()(image_np)
        pts_sampled = torch.reshape(torch.tensor(pts_sampled), [1, 1, -1, 2])
        pts_labels = torch.reshape(torch.tensor(pts_labels), [1, 1, -1])
        predicted_logits, predicted_iou = self.segmentation_model(
            img_tensor[None, ...].to(device),
            pts_sampled.to(device),
            pts_labels.to(device),
        )
        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
        predicted_logits = torch.take_along_dim(
            predicted_logits, sorted_ids[..., None, None], dim=2
        )

        return torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()

    def predict_with_yoloworld(self,classes, input: str=None,frame=None, confidence: int = 0.5) -> sv.Detections:

        if input != None:
          image = cv2.imread(input)
        else:
          image = frame
        #predict using non max suppresion (NMS) to Eliminate Double Detection
        model = self.detection_model
        model.set_classes(classes)
        results = model.infer(image,confidence=confidence)
        result = sv.Detections.from_inference(results).with_nms(threshold=0.1)

        return result

    def predict_with_efficientsam(self, detections_from_yolo,input: str=None,frame=None) -> sv.Detections:

        if input != None:
          image = cv2.imread(input)
        else:
          image = frame

        detections_from_yolo.mask = np.array([None] * len(detections_from_yolo.xyxy))

        for i, [x_min, y_min, x_max, y_max] in enumerate(detections_from_yolo.xyxy):
            y_min, y_max = int(y_min), int(y_max)
            x_min, x_max = int(x_min), int(x_max)
            input_image = image[y_min:y_max, x_min:x_max]
            input_point = np.array([[x_min, y_min], [x_max, y_max]])
            input_label = np.array([2,3])
            image_for_inference = image
            mask_efficient_sam_vits = self.box_segmentation_inference(image_for_inference, input_point, input_label,DEVICE)

            detections_from_yolo.mask[i] = mask_efficient_sam_vits

        return detections_from_yolo



def yolo_world_inference(element_for_detection,image=None,img_file_buffer=None):
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        image= np.array(image)
    elif image is not None:
        image = np.array(image)

    
    classes = [element_for_detection]
    base_model = EfficientYOLOWorld_custom()
    detections = base_model.predict_with_yoloworld(classes, frame=image, confidence=0.0005)

    BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
    LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=1, text_scale=0.3, text_color=sv.Color.BLACK,text_padding = 1)

    annotated_image = image.copy()
    annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
    annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)
    return annotated_image,detections

def efficientSam_YoloWorld(image,element_for_detection):
    image = np.array(image)
    base_model = EfficientYOLOWorld_custom()

    annotated_image, detections = yolo_world_inference(image = image,element_for_detection = element_for_detection)

    segmented_result = base_model.predict_with_efficientsam(copy.deepcopy(detections),frame=image)
    mask_annotator = sv.MaskAnnotator()
    annotated_image = mask_annotator.annotate(
        scene=annotated_image,
        detections=segmented_result,
    )

    return annotated_image


def video_yolo_world_inference(video,element_for_detection):
    file_name = video.name
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    file_path = os.path.join("uploads", file_name)  # Define your desired directory
    with open(file_path, "wb") as f:
        f.write(video.getvalue())

    TARGET_VIDEO_PATH = f"./uploads/result_object_detection_video.mp4"
    SOURCE_VIDEO_PATH = "./uploads/" + file_name
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

    width, height = video_info.resolution_wh
    frame_area = width * height

    
    BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=3)
    LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=1, text_scale=0.7, text_color=sv.Color.BLACK,text_padding = 1)

    classes = [element_for_detection]
    base_model = EfficientYOLOWorld_custom()

    import streamlit as st
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    frames_counter = 0
    with sv.VideoSink(target_path=TARGET_VIDEO_PATH, video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            percentage = int(((frames_counter/video_info.total_frames)*100))
            my_bar.progress(percentage, text=progress_text)
            frames_counter = frames_counter +1
            frame = np.array(frame)
            # predict the bounding boxes via YOLOWorld
            detections = base_model.predict_with_yoloworld(classes, frame=frame, confidence=0.0005)

            #remove the bbs that cover more than 10% of the whole frame
            detections = detections[(detections.area / frame_area) < 0.10]

            # #segment the remaining bbs via efficientSAM
            # detections = base_model.predict_with_efficientsam(detections,frame=frame)

            #Annotate the bbs along with the segmentations mask and save
            annotated_frame = frame.copy()
            annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(annotated_frame, detections)
            annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, detections)
            # annotated_frame = mask_annotator.annotate(scene=annotated_frame,detections=detections)
            sink.write_frame(annotated_frame)
    my_bar.empty()


def video_efficientSam_yolo_world_inference(video,element_for_detection):
    file_name = video.name
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    file_path = os.path.join("uploads", file_name)  # Define your desired directory
    with open(file_path, "wb") as f:
        f.write(video.getvalue())

    TARGET_VIDEO_PATH = f"./uploads/result_segmented_video.mp4"
    SOURCE_VIDEO_PATH = "./uploads/" + file_name
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

    width, height = video_info.resolution_wh
    frame_area = width * height

    
    BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=3)
    LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=1, text_scale=0.7, text_color=sv.Color.BLACK,text_padding = 1)

    classes = [element_for_detection]
    base_model = EfficientYOLOWorld_custom()
    mask_annotator = sv.MaskAnnotator()

    import streamlit as st
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    frames_counter = 0
    with sv.VideoSink(target_path=TARGET_VIDEO_PATH, video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            percentage = int(((frames_counter/video_info.total_frames)*100))
            my_bar.progress(percentage, text=progress_text)
            frames_counter = frames_counter +1
            frame = np.array(frame)
            # predict the bounding boxes via YOLOWorld
            detections = base_model.predict_with_yoloworld(classes, frame=frame, confidence=0.0005)

            #remove the bbs that cover more than 10% of the whole frame
            detections = detections[(detections.area / frame_area) < 0.10]

            # #segment the remaining bbs via efficientSAM
            detections = base_model.predict_with_efficientsam(detections,frame=frame)

            #Annotate the bbs along with the segmentations mask and save
            annotated_frame = frame.copy()
            annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(annotated_frame, detections)
            annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, detections)
            annotated_frame = mask_annotator.annotate(scene=annotated_frame,detections=detections)
            sink.write_frame(annotated_frame)
    my_bar.empty()