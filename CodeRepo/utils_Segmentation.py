from stardist.models.base import StarDistPadAndCropResizer
from csbdeep.utils import axes_check_and_normalize
import tensorflow as tf
import numpy as np
import skimage as ski
from numpy.lib import stride_tricks as st
import warnings
from PIL import Image
from numba import jit
from tqdm import trange
from pathlib import Path

class Resizer(StarDistPadAndCropResizer):
    def __init__(self, grid_dict):
        super().__init__(grid=grid_dict)
        
    def _axes_div_by(self, query_axes,unet_depth):
        query_axes = axes_check_and_normalize(query_axes)
        div_by = dict(zip(
            query_axes.replace('C',''),
            tuple(p**unet_depth * g for p,g in zip([2,2],(1,1)))
        ))
        self.axes_net_div_by = tuple(div_by.get(a,1) for a in query_axes)
    
def get_UNet_tf(model_dir):
    axes_net='YXC'
    axes=axes_net.replace('C','')
    grid_dict = dict(zip(axes,(1,1)))
    resizer=Resizer(grid_dict)
    resizer._axes_div_by(axes_net,3)
    physical_devices=tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device,True)
    return tf.keras.models.load_model(model_dir,compile=False),resizer,axes

def predict_UNet(X,modelUNet,resizer,axes):
    testimg=resizer.before(X,axes,resizer.axes_net_div_by)
    testimg=np.expand_dims(testimg,-1)
    testimg=np.expand_dims(testimg,0)
    res=modelUNet(testimg,training=False)
    res=resizer.after(res[0,..., 0],axes)
    return np.where(res>0.5,1,0)

def load_img(path):
    x = np.array(Image.open(path))  
    return x

def get_mean(Dir,end='max',ending='jpg',skip=10):
    files=sorted(Path(Dir).glob('*'+ending))
    counter=0
    if end=='max':
        end=len(files)
    for i in trange(0,end,skip):
        x=load_img(files[i]) 
        if i==0:
            av=x.astype(np.float32)
        else:
            av=np.add(av,x)    
        counter+=1
    mean=av/counter
    return mean

def fillSmallHoles(img, size, Value,connectivity):
    labels = ski.measure.label(img, connectivity=connectivity)
    props = ski.measure.regionprops_table(
        labels, img, properties=['label', 'area'])
    for count, a in zip(props.get('label'), props.get('area')):
        if (a < size):
            img = np.where(labels == count, Value, img)
    return img

def checkLabelsforMask(labelsSD,imgMask):
    for i in range(1,np.max(labelsSD)+1):
        points=np.argwhere(labelsSD==i)
        if np.count_nonzero(imgMask[points[:,0],points[:,1]])==0:
            labelsSD[points[:,0],points[:,1]]=0


@jit(nopython=True)
def controlled_dilation(labels, LabelsNeighbors, imgMask, imgIntersec):
    labels_copy = np.copy(labels)
    dilated = False
    for i in range(len(labels)):
        for j in range(len(labels[0])):
            if (labels[i, j] > 0) and (np.count_nonzero(LabelsNeighbors[i, j]) < 9):
                for e in range(3):
                    for ee in range(3):
                        if (i+e-1 >= 0) and (j+ee-1 >= 0) and (i+e-1 < len(labels)) and (j+ee-1 < len(labels[0])):
                            if (LabelsNeighbors[i, j, e, ee] == 0) and (imgMask[i+e-1, j+ee-1] > 0) and (imgIntersec[i+e-1, j+ee-1] == 0):
                                labels_copy[i+e-1, j+ee-1] = labels[i, j]
                                dilated = True
    return labels_copy, dilated

def dilateToMask(labels,imgMask):
    LabelsNeighbors = st.sliding_window_view(np.pad(labels, 1), (3, 3))
    labels_copy, dilated = controlled_dilation(
        labels, LabelsNeighbors, imgMask)
    while(dilated):
        LabelsNeighbors = st.sliding_window_view(
            np.pad(labels_copy, 1), (3, 3))
        labels_copy, dilated = controlled_dilation(
            labels_copy, LabelsNeighbors, imgMask)
    return labels_copy  

@jit(nopython=True)
def controlled_dilation_points(points,labels, LabelsNeighbors, imgMask):
    next_points=[]
    labels_copy = np.copy(labels)
    for k in range(len(points)):
        i=points[k][0]
        j=points[k][1]
        if (labels[i, j] > 0) and (np.count_nonzero(LabelsNeighbors[i, j]) < 9):
            for e in range(3):
                for ee in range(3):
                    if (i+e-1 >= 0) and (j+ee-1 >= 0) and (i+e-1 < len(labels)) and (j+ee-1 < len(labels[0])):
                        if (LabelsNeighbors[i, j, e, ee] == 0) and (imgMask[i+e-1, j+ee-1] > 0)and( labels_copy[i+e-1, j+ee-1] != labels[i, j]):
                            labels_copy[i+e-1, j+ee-1] = labels[i, j]
                            next_points.append((i+e-1, j+ee-1))
    return labels_copy, next_points

def dilateToMask_points(labels,imgMask):
    LabelsNeighbors = st.sliding_window_view(np.pad(labels, 1), (3, 3))
    points=np.argwhere(labels > 0)
    labels_copy, next_points=controlled_dilation_points(points,labels, LabelsNeighbors, imgMask)
    while len(next_points)>0:
        LabelsNeighbors = st.sliding_window_view(np.pad(labels_copy, 1), (3, 3))
        labels_copy, next_points=controlled_dilation_points(np.array(next_points),labels_copy, LabelsNeighbors, imgMask)
    return labels_copy  

def combinedPrediction(X,modelSD,imgMask,size=0,usePoints=True):
    labelsSD=modelSD.predict_instances(X)[0]
    if size>0:
        labelsSD=fillSmallHoles(labelsSD,size,0,1)
    checkLabelsforMask(labelsSD,imgMask)
    if usePoints:
        labels=dilateToMask_points(labelsSD,imgMask)
    else:
        labels=dilateToMask(labelsSD,imgMask)
    return labels,labelsSD 