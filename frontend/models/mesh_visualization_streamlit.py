import open3d as o3d
import pyvista as pv
import streamlit as st
import numpy as np

from stpyvista import stpyvista

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import os


def mesh_generation(image):

    feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

    # load and resize the input image

    file_name = image.name
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    file_path = os.path.join("uploads", file_name)  # Define your desired directory
    with open(file_path, "wb") as f:
        f.write(image.getvalue())
    SOURCE_IMAGE_PATH = "./uploads/" + file_name

    # image = Image.open("../../facade-identification/3D mesh from an image/DJI_0488.JPG")
    image = Image.open(SOURCE_IMAGE_PATH)
    new_height = 480 if image.height > 480 else image.height
    new_height -= (new_height % 32)
    new_width = int(new_height * image.width / image.height)
    diff = new_width % 32
    new_width = new_width - diff if diff < 16 else new_width + 32 - diff
    new_size = (new_width, new_height)
    image = image.resize(new_size)

    # prepare image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")

    # get the prediction from the model
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # remove borders
    pad = 16
    output = predicted_depth.squeeze().cpu().numpy() * 1000.0
    output = output[pad:-pad, pad:-pad]
    image = image.crop((pad, pad, image.width - pad, image.height - pad))


    width, height = image.size

    depth_image = (output * 255 / np.max(output)).astype('uint8')
    image_np = np.array(image)

    # create rgbd image
    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(image_np)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)

    # camera settings
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(width, height, 500, 500, width/2, height/2)

    # create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

    # outliers removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.0)
    pcd = pcd.select_by_index(ind)

    # estimate normals
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()

    # surface reconstruction
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, n_threads=1)[0]

    # rotate the mesh
    rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(rotation, center=(0, 0, 0))

    # save the mesh
    o3d.io.write_triangle_mesh(f'./mesh_new.obj', mesh)

    # Convert Open3D mesh to PyVista for Streamlit visualization
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    faces = np.hstack([[3] + list(face) for face in faces]).reshape(-1, 4)

    # Create a PyVista mesh
    pv_mesh = pv.PolyData(vertices, faces)

    # Set up Streamlit
    st.title("3D Mesh Visualization")

    # Create a PyVista plotter object
    plotter = pv.Plotter(notebook=False)
    ## Add some scalar field associated to the mesh
    pv_mesh['myscalar'] = pv_mesh.points[:, 2] * pv_mesh.points[:, 0]
    # plotter.add_mesh(pv_mesh, color="lightblue")
    ## Add mesh to the plotter
    plotter.add_mesh(
        pv_mesh,
        scalars="myscalar",
        # cmap="prism",
        # show_edges=True,
        # edge_color="#001100",
        ambient=0.2,
    )

    ## Some final touches
    plotter.background_color = "lightblue"
    plotter.view_isometric()

    ## Pass a plotter to stpyvista
    stpyvista(plotter)