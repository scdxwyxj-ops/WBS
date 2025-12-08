import os
import numpy as np
from PIL import Image

try:
    import networkx as nx  # type: ignore
except ImportError:
    nx = None  # type: ignore

from skimage.segmentation import slic
import cv2

import torch
import torch.nn.functional as F
from torchvision import transforms
from einops import rearrange

from image_processings.simple_graph import SimpleGraph



def change_image_type(image, target_type):
    origin_image_type = type(image)
    
    if origin_image_type == torch.Tensor:
        if target_type == 'np.array':
            image = rearrange(image, 'c h w -> h w c')
            return image.numpy()
        elif target_type == 'PIL.Image':
            image = rearrange(image, 'c h w -> h w c')
            return Image.fromarray(image.numpy())
        elif target_type == 'tensor':
            return image
        else:
            raise ValueError("Unknown target type")
    
    elif origin_image_type == np.ndarray:
        if target_type == 'tensor':
            image = rearrange(image, 'h w c -> c h w')
            return torch.from_numpy(image)
        elif target_type == 'PIL.Image':
            return Image.fromarray(image)
        elif target_type == 'np.array':
            return image
        else:
            raise ValueError("Unknown target type")
    
    elif origin_image_type == Image.Image:
        if target_type == 'tensor':
            image = rearrange(np.array(image), 'h w c -> c h w')
            return torch.from_numpy(image)
        elif target_type == 'np.array':
            return np.array(image)
        elif target_type == 'PIL.Image':
            return image
        else:
            raise ValueError("Unknown target type")
    
    else:
        raise ValueError("Unsupported image type")
    
def get_resize_shape(new_size, image_tensor):
    image_tensor = change_image_type(image_tensor, 'tensor')
    dim = len(image_tensor.shape)
    if dim == 3:
        _, h, w = image_tensor.shape
    if dim == 4:
        _, _, h, w = image_tensor.shape
    
    if w >= h:
        ratio = new_size/w
        new_h = int(h*ratio)
        if new_h%2 != 0:
            new_h = new_h - 1
        resized_shape = (new_h, new_size)
    else:
        ratio = new_size/h
        new_w = int(w*ratio)
        if new_w%2 != 0:
            new_w = new_w - 1
        resized_shape = (new_size, new_w)
    return dim, resized_shape

def get_padding_tensor(new_size, image_tensor):
    shape_of_origin_image = image_tensor.shape
    hp = int((new_size - shape_of_origin_image[1]) / 2)
    vp = int((new_size - shape_of_origin_image[2]) / 2)
    return (vp, vp, hp, hp)
    

def resize_and_pad(new_size, image_tensor):
    dim, resized_shape = get_resize_shape(new_size=new_size, image_tensor=image_tensor)##get_resize_shape
    resize_function = transforms.Resize(resized_shape, interpolation=transforms.InterpolationMode.NEAREST,antialias=True)
    if dim == 3:
        image_tensor = resize_function(image_tensor.unsqueeze(0)) ##resize
        padding = get_padding_tensor(new_size, image_tensor)
        return F.pad(input=image_tensor, pad=padding, mode='constant', value=0).squeeze(0) ##padding
    if dim == 4:
        image_tensor = resize_function(image_tensor) ##resize
        padding = get_padding_tensor(new_size, image_tensor) 
        return F.pad(input=image_tensor, pad=padding, mode='constant', value=0)##padding
        


def to_grey(image):
    image_tensor = change_image_type(image= image, target_type= 'tensor')
    to_gray = transforms.Grayscale(num_output_channels=1)
    return to_gray(image_tensor)


def resize_slic_pad(image, new_size, max_num_segments, compactness = 15, sigma = 1, min_size_factor = 0.5, max_size_factor = 1.2):
    ## get tensor and size
    image_tensor = change_image_type(image= image, target_type= 'tensor') ## turn to tensor

    ## resize
    _ ,resized_shape = get_resize_shape(new_size=new_size, image_tensor=image_tensor)
    resize_function = transforms.Resize(resized_shape, interpolation=transforms.InterpolationMode.NEAREST, antialias=True)
    image_resized = resize_function(image_tensor.unsqueeze(0)).squeeze(0)

    ##slic
    image_array = change_image_type(image_resized, 'np.array')
    segments = slic(
        image_array,
        n_segments=max_num_segments,
        compactness=compactness,
        sigma=sigma,
        min_size_factor=min_size_factor,
        max_size_factor=max_size_factor,
        start_label=1,
    )
    num_of_different_segments = np.max(segments)
    segments = torch.tensor(segments)

    ##padding
    padding = get_padding_tensor(new_size, image_resized)
    image_resized_padding = F.pad(input=image_resized, pad=padding, mode='constant', value=0)
    segments_padding = F.pad(input=segments, pad=padding, mode='constant', value=0)

    return image_resized_padding, segments_padding, num_of_different_segments, segments, image_resized

def segment2graph(segment, num_of_nodes_in_graph, num_of_different_segments = None):
    if type(segment) != np.ndarray:
        segment = np.array(segment)
    if num_of_different_segments == None:
        num_of_different_segments = np.max(segment)
    G = nx.Graph() if nx is not None else SimpleGraph()
    # Add regular nodes with a segment attribute
    for i in range(num_of_different_segments):
        G.add_node(i)
    # Add the global node
    G.add_node(num_of_nodes_in_graph-1)
    # Connect each regular node to the global node
    for i in range(num_of_different_segments):
        G.add_edge(i, num_of_nodes_in_graph-1)
    # Add additional isolated nodes to reach self.num_nodes
    for i in range(num_of_different_segments, num_of_nodes_in_graph-1):
        G.add_node(i)
    # Add edge for adjacent segment
    height, width = segment.shape
    for y in range(height):
        for x in range(width):
            segment1 = segment[y, x]
            if x + 1 < width:
                segment2 = segment[y, x + 1]
                if segment1 != segment2:
                    G.add_edge(segment1 - 1, segment2 - 1)
            if y + 1 < height:
                segment3 = segment[y + 1, x]
                if segment1 != segment3:
                    G.add_edge(segment1 - 1, segment3 - 1)
    ## Add self_loop for each node
    for node in G.nodes():
        G.add_edge(node, node)
    return G



class image_i_segment:
    def __init__(self, 
                 name = None, 
                 label = None, 
                 image = None, 
                 new_size_of_image = 260,
                 num_node_for_graph = 48, ##max of different segments in SLIC is equal to num_node_for_graph
                 compactness_in_SLIC = 15,
                 sigma_in_SLIC = 1,
                 min_size_factor_in_SLIC = 0.5, 
                 max_size_factor_in_SLIC = 1.2):
        maindir = os.path.join("/app/data_quyuxuan_filteres_by_aichenjin1/", str(label))
        if image is None:
            self.image = cv2.imread(os.path.join(maindir, name))
        else:
            self.image = change_image_type(image, target_type='PIL.Image')
        self.new_size_of_image = new_size_of_image
        self.segment_without_padding = None
        self.segments_padding = None
        self.image_resized = None
        self.image_resized_padding = None
        self.graph = None
        self.num_of_different_segments_in_SLIC = None
        self.num_node_for_graph = num_node_for_graph
        self.compactness_in_SLIC = compactness_in_SLIC
        self.sigma_in_SLIC = sigma_in_SLIC
        self.min_size_factor_in_SLIC = min_size_factor_in_SLIC
        self.max_size_factor_in_SLIC = max_size_factor_in_SLIC

        self.resize_slic_pad_for_self()
        self.get_graph_for_self_with_segment()

    def resize_slic_pad_for_self(self):
        self.image_resized_padding, self.segments_padding, self.num_of_different_segments_in_SLIC, self.segment_without_padding, self.image_resized = resize_slic_pad(self.image, self.new_size_of_image, max_num_segments = self.num_node_for_graph, compactness = self.compactness_in_SLIC, sigma=self.sigma_in_SLIC, min_size_factor=self.min_size_factor_in_SLIC, max_size_factor=self.max_size_factor_in_SLIC)

    def get_graph_for_self_with_segment(self):
        if self.segments_padding == None:
            self.resize_slic_pad_for_self(self)
        else:
            self.graph = segment2graph(self.segment_without_padding, self.num_node_for_graph, num_of_different_segments = self.num_of_different_segments_in_SLIC)
