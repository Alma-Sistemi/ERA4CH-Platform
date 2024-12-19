from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
from PIL import Image
import io
import json
import torch
import numpy as np
import math
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from streetview import get_streetview


def apiPostRequest(image, server_url: str, element_for_detection):
    print("In post request")
    m = MultipartEncoder(
        fields={
            'file': ('filename', image, 'image/jpeg'),
            'text': element_for_detection
        }
    )
    print(m.content_type)
    r = requests.post(server_url,
                      data=m,
                      headers={'Content-Type': m.content_type},
                      timeout=8000)

    return r


def apiPostRequestForCracks(image, server_url: str):

    print("In cracks post request")
    image_bytes = image.read()
    r = requests.post(server_url, files={"file": image_bytes})
    path_to_overlay_image = r.content.decode()
    print("In cracks post request: ", path_to_overlay_image)
    print("type is ", type(path_to_overlay_image))
    return path_to_overlay_image


def show_anns(
    image,
    predictions,
    boxes,
    detected_element,
    figsize=(8, 6),
    axis="off",
    cmap="viridis",
    alpha=0.4,
    add_boxes=True,
    box_color="r",
    box_linewidth=1,
    title=None,
    output=None,
    blend=True,
    **kwargs,
):
    """Show the annotations (objects with random color) on the input image.

    Args:
        figsize (tuple, optional): The figure size. Defaults to (12, 10).
        axis (str, optional): Whether to show the axis. Defaults to "off".
        cmap (str, optional): The colormap for the annotations. Defaults to "viridis".
        alpha (float, optional): The alpha value for the annotations. Defaults to 0.4.
        add_boxes (bool, optional): Whether to show the bounding boxes. Defaults to True.
        box_color (str, optional): The color for the bounding boxes. Defaults to "r".
        box_linewidth (int, optional): The line width for the bounding boxes. Defaults to 1.
        title (str, optional): The title for the image. Defaults to None.
        output (str, optional): The path to the output image. Defaults to None.
        blend (bool, optional): Whether to show the input image. Defaults to True.
        kwargs (dict, optional): Additional arguments for matplotlib.pyplot.savefig().
    """

    warnings.filterwarnings("ignore")

    anns = predictions

    if anns is None:
        print("Please run predict() first.")
        return
    elif len(anns) == 0:
        print("No objects found in the image.")
        return

    # plt.figure(figsize=figsize)

    plt.imshow(image, aspect="auto")

    if add_boxes:
        for i,box in enumerate(boxes):
            # Draw bounding box
            box = box.cpu().numpy()  # Convert the tensor to a numpy array
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=box_linewidth,
                edgecolor=box_color,
                facecolor="none",
            )
            plt.gca().add_patch(rect)
            ax =  plt.gca()
            rx= box[0]
            ry = box[1]
            cx = rx + (box[2] - box[0])/10.0
            cy = ry - (box[3] - box[1])/8.3
            label = detected_element + ", id:" + str(i + 1)
            l = ax.text(
                cx, cy,
                label,
                
                fontsize=5,
            
                color="red",
                ha='left',
                va='top'
            )

    if "dpi" not in kwargs:
        kwargs["dpi"] = 100

    if "bbox_inches" not in kwargs:
        kwargs["bbox_inches"] = "tight"

    plt.imshow(anns, cmap=cmap, alpha=alpha)

    if title is not None:
        plt.title(title)
    plt.axis(axis)

    if output is not None:
        if blend:
            plt.savefig(output, **kwargs)
        else:
            array_to_image(predictions, output, image)

    # the below code is used to return segmented image as a np.array
    # buffer = io.BytesIO()

    # plt.savefig(buffer, format="png")
    # plt.savefig("segmented.png")

    # buffer.seek(0)

    # Open the in-memory image buffer as a NumPy array
    # overlayed_image = Image.open(buffer)
    # overlayed_image_array = np.array(overlayed_image)

    # return buffer, so I can use it directly in the fastApi response.
    # return buffer
    # return overlayed_image_array


def create_mask_overlay(image_pil, selected_boxes, masks):

    mask_multiplier = 255,
    dtype = np.uint8
    image_np = np.array(image_pil)

    if selected_boxes.nelement() == 0:  # No "object" instances found
        print("No objects found in the image. - in create_mask_overlay")
        return
    else:
        # Create an empty image to store the mask overlays
        mask_overlay = np.zeros_like(
            image_np[..., 0], dtype=dtype
        )  # Adjusted for single channel

        for i, (box, mask) in enumerate(zip(selected_boxes, masks)):
            # Convert tensor to numpy array if necessary and ensure it contains integers
            if isinstance(mask, torch.Tensor):
                mask = (
                    mask.cpu().numpy().astype(dtype)
                )  # If mask is on GPU, use .cpu() before .numpy()
            mask_overlay += ((mask > 0) * (i + 1)).astype(
                dtype
            )  # Assign a unique value for each mask

        # Normalize mask_overlay to be in [0, 255]
        mask_overlay = (
            mask_overlay > 0
        ) * mask_multiplier  # Binary mask in [0, 255]

    return mask_overlay


def select_masked_above_threshold(boxes, masks, logits, threshold):

    keep_masks = []
    keep_boxes = []
    keep_logits = []

    #print("boxes before: ", boxes)
    #print("masks before: ", masks)
    #print("logits before: ", logits)

    for i, log in enumerate(logits):
        if log > threshold:
            keep_masks.append(masks[i])
            keep_boxes.append(boxes[i])
            keep_logits.append(log)

    keep_masks_tensor = torch.tensor(
        keep_masks, dtype=torch.int)
    keep_boxes_tensor = torch.tensor(
        keep_boxes, dtype=torch.int)

    return keep_boxes_tensor, keep_masks_tensor, keep_logits


def create_json(logits_list, boxes_tensor, masks_tensor, detected_element):
    # Create a list to store multiple annotations
    annotations = []

    # Sample bbox tensor
    # Iterate through the bbox tensor and create annotations

    for i, bbox in enumerate(boxes_tensor):
        bbox_list = bbox.tolist()
        bbox_area = (bbox_list[2] - bbox_list[0]) * \
            (bbox_list[3] - bbox_list[1])
        width = (bbox_list[2] - bbox_list[0])
        height = (bbox_list[3] - bbox_list[1])
        annotation = {
            "id": i + 1,  # You can use a unique identifier for each annotation
            "bounding box": bbox_list,
            "bounding box area": bbox_area,
            "width": width,
            "height": height,
            "area of segmented mask": torch.count_nonzero(masks_tensor[i]).item(),
            # Replace with actual stability_score data
            "confidence score": logits_list[i]
        }
        annotations.append(annotation)

    # Create a dictionary to hold the list of annotations
    data = {detected_element: annotations}
    # Convert the dictionary to a JSON string
    json_string = json.dumps(data, indent=2)

    # Print the JSON string
    # print(json_string)
    return json_string


def calculate_distances(panoramas, lat, long):
    distances = []
    for i in range(len(panoramas)):


        p = [lat, long] 
        q = [panoramas[i].lat, panoramas[i].lon] 

        # Calculate Euclidean distance
        dist = math.dist(p, q)
        
        distances.append(dist)
    return distances

def closest_panos(distances, panos,num_of_panos):
    distances_for_sort = distances.copy()
    distances_for_sort.sort()

    closest_panos_list = []
    for i in range(num_of_panos):
        index = distances.index(distances_for_sort[i])
        closest_panos_list.append(panos[index])
    
    return closest_panos_list

def get_street_view_images (pano_id, angle):
    image = get_streetview(
    pano_id=pano_id,
    api_key='AIzaSyAv2H4eePhGGitXq-_u0bXpltDgrigqb4Y',
    heading=angle
    )
    return image
