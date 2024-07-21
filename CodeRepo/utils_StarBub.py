import numpy as np
from numpy.lib import stride_tricks as st
from skimage.draw import polygon_perimeter, polygon
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import patches
import math
import numpy.linalg as lag
from numba import jit
import csv
from PIL import Image
from matplotlib.patches import Ellipse
from skimage.measure import EllipseModel
import json
from scipy.ndimage.filters import uniform_filter1d


class RDObj():
    """ RadialDistanceObject class

    Parameters
    ----------
    id : int
        Object ID.
    num_rays: int
        Number of radial rays defining the object.
    center: Tuple
        Tuple (y,x) for the center coordinates of the object.
    dists: float  
        Array containing the length of each radial ray from the center to the object boundary.
    points:
        Numpy array (y,x,bool) containing the position of the radial end points and a bool whether the point touches
        another segmentation instance or not.
    """

    def __init__(self, id, num_rays, center=None, dists=None, points=None):
        self.id = id
        self.num_rays = num_rays
        self.center = center
        self.dists = dists
        self.points = points

    def getCenter(self, img):
        points = np.argwhere(img == self.id)
        if len(points) > 0:
            self.center = (np.mean(points[:, 0]), np.mean(points[:, 1]))

    def touchImgBorder(self, img, i, j):
        if (i >= len(img)) or (i <= 0) or (j >= len(img[0])) or (j <= 0):
            return True
        return False

    def get_dists(self):
        if self.points is not None:
            self.dists = np.sqrt(np.square(
                self.center[0]-self.points[:, 0])+np.square(self.center[1]-self.points[:, 1]))
        else:
            raise("Points None for ID "+self.id)

    def get_points(self):
        if self.dists is not None:
            phis = np.linspace(0, 2*np.pi, self.num_rays, endpoint=False)
            distx = np.ones(phis.shape)*np.cos(phis)
            disty = np.ones(phis.shape)*np.sin(phis)
            self.points[:,0] =self.center[0]+disty*self.dists
            self.points[:,1] =self.center[1]+distx*self.dists
            return self.points
        else:
            raise("Dists None for ID "+self.id)

    def generateRD_manual(self, img):
        phis = np.linspace(0, 2*np.pi, self.num_rays, endpoint=False)
        if (self.center is None):
            self.getCenter(img)
        if (self.center is None):
            return
        distx = np.ones(phis.shape)*np.cos(phis)
        disty = np.ones(phis.shape)*np.sin(phis)
        stretch = 5
        points = np.zeros((len(phis), 3))
        StretchBool = True
        mask = img != self.id
        mask[:, 0] = 1
        mask[:, len(mask[0])-1] = 1
        mask[0, :] = 1
        mask[len(mask)-1, :] = 1
        while StretchBool:
            points_strechted = [(self.center[0]+disty*stretch),
                                (self.center[1]+distx*stretch)]
            points_strechted[0] = np.where(
                points_strechted[0] < 0, 0, points_strechted[0])
            points_strechted[1] = np.where(
                points_strechted[1] < 0, 0, points_strechted[1])
            points_strechted[0] = np.where(points_strechted[0] >= len(
                img)-1, len(img)-1, points_strechted[0])
            points_strechted[1] = np.where(points_strechted[1] >= len(
                img[0])-1, len(img[0])-1, points_strechted[1])
            touch_mask = mask[points_strechted[0].astype(
                int), points_strechted[1].astype(int)]
            compare = ((points[:, 2] != 1) & (touch_mask == 1))
            points[:, 0] += (self.center[0]+disty*(stretch-1))*compare
            points[:, 1] += (self.center[1]+distx*(stretch-1))*compare
            points[:, 2] = np.where(compare == 1, 1, points[:, 2])
            stretch += 1
            if (np.count_nonzero(points[:, 2] == 0) == 0):
                StretchBool = False
        points[:, 2] = 0
        self.points = points.astype(int)
        self.get_dists()
        self.getTouchingCandidates(img)

    def getTouchingCandidates(self, img):
        LabelNeighbors = st.sliding_window_view(np.pad(img, 1), (3, 3))
        for point in self.points:
            if (point[0] < len(img)) and (point[1] < len(img[0])) and (point[0] >= 0) and (point[1] >= 0):
                if (np.count_nonzero(LabelNeighbors[int(point[0]), int(point[1])] == 0) == 0) or (self.touchImgBorder(img, int(point[0]), int(point[1])) == True):
                    point[2] = 1
            else:
                point[2] = 1

    def transformRDToArray(self, metric):
        RDArray = self.dists*metric
        return RDArray

    def stretchPoints(self, stretch):
        phis = np.linspace(0, 2*np.pi, self.num_rays, endpoint=False)
        distx = np.ones(phis.shape)*np.cos(phis)
        disty = np.ones(phis.shape)*np.sin(phis)
        self.points[:, 0] = (self.center[0]+disty*stretch).astype(int)
        self.points[:, 1] = (self.center[1]+distx*stretch).astype(int)

    def drawRD(self, ax, color='g', linewidths=0.6):
        dist_lines = np.empty((self.num_rays, 2, 2))
        if len(self.points) == 0:
            print("No Endpoints evaluated so far. Try generateRD_manual() function")
        else:
            dist_lines[:, 0, 0] = self.points[:, 1]
            dist_lines[:, 0, 1] = self.points[:, 0]
            dist_lines[:, 1, 0] = self.center[1]
            dist_lines[:, 1, 1] = self.center[0]
            ax.add_collection(LineCollection(
                dist_lines, colors=color, linewidths=linewidths))

    def drawTouchingCandidates(self, ax, color='r', linewidths=0.6):
        num_candidates = np.count_nonzero(self.points[:, 2] == 1)
        if num_candidates > 0:
            dist_lines = np.empty((num_candidates, 2, 2))
            counter = 0
            for point in self.points:
                if point[2] == 1:
                    dist_lines[counter, 0, 0] = point[1]
                    dist_lines[counter, 0, 1] = point[0]
                    counter += 1
            dist_lines[:, 1, 0] = self.center[1]
            dist_lines[:, 1, 1] = self.center[0]
            ax.add_collection(LineCollection(
                dist_lines, colors=color, linewidths=linewidths))

    def plot_polygon(self, ax, color='g', lineType='--', linewidth=1.2):
        a, b = list(self.points[:, 1]), list(self.points[:, 0])
        a += a[:1]
        b += b[:1]
        ax.plot(a, b, lineType, alpha=1,  zorder=1,
                color=color, linewidth=linewidth)

    def polygon_on_img(self, shape):
        img = np.zeros(shape).astype(int)
        r = np.array(self.points[:, 0])
        c = np.array(self.points[:, 1])
        rr, cc = polygon(r, c)
        img[rr, cc] = 1
        return img

    def get_grad(self,gradimg):
        gradimg_neigh = st.sliding_window_view(np.pad(gradimg, 1), (3, 3))
        return get_grad(self.points,gradimg_neigh)


class Bubble():
    """ Bubble class

    Parameters
    ----------
    
    metric: float
        Real pixel size in [m] needed to calculate physical sizes.
    Diameter: float
        Sphere-volume equivalent diameter. 
    Position: Tuple
        Tuple (y,x) for the center coordinates of the bubble.
    Major: float
        Majoraxis length, determined as the longest distance of all boundary pixels.
    Minor: float
        Majoraxis length, determined as the longest distance of all boundary pixels perpendicular to the Majoraxis.
    Volume: float
        Spheroidal volume of the bubble determined with Major and Minor.
    Timestep: float
        Timestep when the bubble was detected (e.g. image number).
    Velocity: float (tracking not implemented yet!)
        Bubble velocity.
    """

    def __init__(self, points, metric,
                 Diameter=None, Position=None,
                 Major=None, Minor=None,
                 Volume=None, Timestep=0.0,
                 Velocity=None, ID=1,
                 tracking=False, calc_area=False,
                 bbox=None,grad=0,occu_rate=0):
        self.occu_rate=occu_rate
        self.grad =grad
        if isinstance(points,RDObj):
            Volume, Dia = self.calcDiameterFromRDC(points,metric)           
            self.Dia=Dia
            points=points.points
        if Diameter is None:
            Major, Minor, Volume, Diameter, Position, area = self.getBubbleProps(
                points, metric, calc_area=calc_area)
        else:
            area = None

        if bbox is None:
            self.getBbox(points)
        else:
            self.Bbox=bbox
        self.Diameter = Diameter
        self.Position = Position
        self.Major = Major
        self.Minor = Minor
        self.Volume = Volume
        self.Timestep = Timestep
        self.Velocity = Velocity
        self.ID = ID
        self.area = area
        
        if tracking:
            self.points = points

    def calcDiameterFromRDC(self,RDCO,metric):
        Volume=np.sum(4/3*math.pi*(RDCO.dists*metric)**3/RDCO.num_rays)
        d_Sphere = (6*Volume/math.pi)**(1/3)
        return Volume,d_Sphere

    def getBbox(self, points):
        self.Bbox = (np.min(points[:, 1]), np.min(
            points[:, 0]), np.max(points[:, 1]), np.max(points[:, 0]))
    
    def plotBbox(self,ax,color='r'):
        if self.Bbox is None:
            self.getBbox(self.points)
        rect=plt.Rectangle((self.Bbox[0],self.Bbox[1]),self.Bbox[2]-self.Bbox[0],self.Bbox[3]-self.Bbox[1],edgecolor=color,fill=False)
        ax.add_patch(rect)

    def getBubbleProps(self, points, metric, calc_area=False):
        MajorP, MinorP, center = self.getMajorMinor(points)
        if (MinorP[0] is None):
            return None, None, None, None, None, None
        Major = math.sqrt((MajorP[0][0]-MajorP[1][0])
                          ** 2+(MajorP[0][1]-MajorP[1][1])**2)/2*metric
        Minor = math.sqrt((MinorP[0][0]-MinorP[1][0])
                          ** 2+(MinorP[0][1]-MinorP[1][1])**2)/2*metric
        V_Ellipsoid = math.pi*4/3*Major**2*Minor
        d_Sphere = (6*V_Ellipsoid/math.pi)**(1/3)
        self.max_x, self.max_y, self.min_x, self.min_y = np.max(points[:, 1]), np.max(
            points[:, 0]), np.min(points[:, 1]), np.min(points[:, 0])
        if calc_area:
            area = self.calc_corrected_area(points)
            return Major, Minor, V_Ellipsoid, d_Sphere, center, area
        return Major, Minor, V_Ellipsoid, d_Sphere, center, None

    def getMajorMinor(self, points):

        # Getting the center of all points belonging to the object
        center = np.mean(points[:, 0]), np.mean(points[:, 1])

        # Major axis as largest distance of all border points
        MajorP1, MajorP2 = getMaxDistAxis(points)

        # Minor axis as largest distance of all points perpendicular to Major axis
        VecMajor = [MajorP2[0]-MajorP1[0], MajorP2[1]-MajorP1[1]]
        Perp_points = []

        # Use more boundary points to fullfill perpendicular criterion
        if len(Perp_points) < 2:
            allpoints = polygon_peri(points)
            Perp_points = []
            for p1 in allpoints:
                VecTemp = [center[0]-p1[0], center[1]-p1[1]]
                scalarProd = np.dot(VecMajor, VecTemp)
                if abs(scalarProd)/(lag.norm(VecMajor)*lag.norm(VecTemp)) < (1/lag.norm(VecTemp)):
                    Perp_points.append(p1)
        MinorP1, MinorP2 = getMaxDistAxis(np.array(Perp_points))

        return ([MajorP1, MajorP2], [MinorP1, MinorP2], center)

    def ValuesToString(self):
        if self.Velocity == None:
            return [str(self.Position[1]), str(self.Position[0]), str(self.Diameter), str(self.Major), str(self.Minor), str(self.Timestep), str(self.ID)]
        else:
            return [str(self.Position[1]), str(self.Position[0]), str(self.Diameter), str(self.Major), str(self.Minor), str(self.Velocity), str(self.Timestep), str(self.ID)]

    def ConvertToRDC(self)->RDObj:
        if self.points is not None:
            rdc= RDObj(self.ID,len(self.points))
            rdc.points=self.points
            rdc.center=self.Position
            return rdc

def HiddenReco(labels, metric, timestep=0,n_rays = 64, useRDC=False, model=None, boolPlot=False, ax=plt.gca(), OnlyPoints=False, tracking=False,gradimg=None,img=None,Outdir=None):
    if model == None:
        useRDC = False
    if useRDC == False:
        ell = EllipseModel()
    Bubbles = []
    if useRDC:
        RDArrays = []
        idx = []
        RDCs = []
        counter = 0
        for i in range(1, np.max(labels)+1):
            Rdc = RDObj(i, n_rays)
            Rdc.generateRD_manual(labels)
            if (Rdc.center != None):
                RDCs.append(Rdc)               
                if (np.count_nonzero(Rdc.points[:, 2] == 1) > 1):
                    RDArray = Rdc.transformRDToArray(metric)
                    RDArrays.append(RDArray)
                    idx.append(counter)
                counter += 1
        if len(RDArrays)>0:
            yhat = model.predict(np.asarray(RDArrays))
            for i, Id in enumerate(idx):
                stretch = yhat[i]/metric
                stretch = np.where(
                    stretch*RDCs[Id].points[:, 2] > RDCs[Id].dists, stretch, RDCs[Id].dists)
                stretch = uniform_filter1d(stretch, size=4)
                RDCs[Id].stretchPoints(stretch)
        for i, Rdc in enumerate(RDCs):
            # Cutting and writing out single bubble images for other models
            boolcut=False
            if img is not None:
                points=np.argwhere(labels==Rdc.id)
                
                if np.min(points[:,0])>0 and np.max(points[:,0])<len(img)-1 and np.min(points[:,1])>0 and np.max(points[:,1])<len(img[0])-1:
                    cut=img[np.min(points[:,0]):np.max(points[:,0]),np.min(points[:,1]):np.max(points[:,1])] 
                    cut=np.where(cut==0,2,cut)
                    cut=np.where(cut==-1,0,cut)
                    np.save(Outdir+str(i),cut.astype('uint8'))
                else:
                    boolcut=True
            if boolPlot:
                random_color = tuple(
                    (np.random.choice(range(255), size=3))/255)
                Rdc.plot_polygon(ax=ax, color=random_color,
                                 lineType='-', linewidth=1.5)
            if OnlyPoints == False:
                mean_grad=0
                if gradimg is not None:
                    mean_grad=Rdc.get_grad(gradimg)
                occu_rate=np.count_nonzero(Rdc.points[:,2])/Rdc.num_rays
                if occu_rate==0.0 and boolcut:
                    occu_rate=0.01
                Bub = Bubble(Rdc, metric, Timestep=timestep,
                             ID=i, tracking=tracking,grad=mean_grad,occu_rate=occu_rate)
                if Bub.Diameter is not None:
                    Bubbles.append(Bub)
            else:
                points = Rdc.points[:, :2]
                Bubbles.append((i, points))
    else:
        for i in range(1, np.max(labels)+1):
            Rdc = RDObj(i, n_rays)
            Rdc.generateRD_manual(labels)
            if (Rdc.center != None):
                pointsEllipse = []
                pointsbackup = []
                for point in Rdc.points:
                    if point[2] == 0:
                        pointsEllipse.append([point[0], point[1]])
                    pointsbackup.append([point[0], point[1]])
                areaEllipse = 0
                if len(pointsEllipse) > 0:
                    pointsEllipse = np.array(pointsEllipse)
                    ell.estimate(pointsEllipse)
                    try:
                        x0, y0, a, b, phi1 = ell.params
                        phi = 0.5*np.pi-phi1
                        areaEllipse = math.pi*a*b
                    except:
                        areaEllipse = 0
                        print('Error in ellipse fit, retry with backup')
                if (areaEllipse < np.count_nonzero(labels == i)) or (areaEllipse > 20*np.count_nonzero(labels == i)):
                    pointsEllipse = np.array(pointsbackup)
                    ell.estimate(pointsEllipse)
                    try:
                        x0, y0, a, b, phi1 = ell.params
                        phi = 0.5*np.pi-phi1
                        areaEllipse = math.pi*a*b
                    except:
                        areaEllipse = 0
                        print(f'Unable to fit ellipse for label {i}')
                if boolPlot:
                    random_color = tuple(
                        (np.random.choice(range(255), size=3))/255)
                    ellipse = Ellipse(
                        (y0, x0), 2*a, 2*b, angle=math.degrees(phi), alpha=0.25, color=random_color)
                    ax.add_artist(ellipse)
                if math.isnan(areaEllipse) == False:
                    major_el = a if a > b else b
                    minor_el = b if b < a else a
                    major_el = major_el*metric
                    minor_el = minor_el*metric
                    V_Ellipsoid = math.pi*4/3*major_el**2*minor_el
                    d_Sphere = (6*V_Ellipsoid/math.pi)**(1/3)
                    Bubbles.append(Bubble(None, None, Diameter=d_Sphere, Position=[
                                   y0, x0], Major=major_el, Minor=minor_el, Volume=V_Ellipsoid, Timestep=timestep))
    return Bubbles


def SaveCSV_List(Bubbles, directory, name, header=None):
    f = open(directory+name+'.csv', "w")
    wr = csv.writer(f)
    if header:
        f.write(str(header+'\n'))
    for bub in Bubbles:
        wr.writerow(bub.ValuesToString())
    f.close()


def scale(X, x_min, x_max):
    nom = (X-X.min())*(x_max-x_min)
    denom = X.max()-X.min()
    denom = denom+(denom == 0)
    return x_min + nom/denom


def Save_Labels(labels, directory, name, scaleGrey=False):
    if scaleGrey == True:
        labels = scale(labels, 0, 255)
    img = Image.fromarray(labels.astype('uint8'))
    img.save(directory+name+".png")


def polygon_peri(points):
    r = np.array(points[:, 0])
    c = np.array(points[:, 1])
    rr, cc = polygon_perimeter(r, c)
    ret_points = np.zeros((len(rr), 2))
    ret_points[:, 0] = rr
    ret_points[:, 1] = cc
    return ret_points


@jit(nopython=True)
def getMaxDistAxis(points):
    Max_dist = 0
    MaxdistP1 = None
    MaxdistP2 = None
    for p1 in points:
        for p2 in points:
            dist = math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
            if dist > Max_dist:
                Max_dist = dist
                MaxdistP1 = p1
                MaxdistP2 = p2
    return MaxdistP1, MaxdistP2


def writeOutJSONPoints(Bubbles, directory, name):
    Bubbles_dict = []
    for bub in Bubbles:
        Bubble_dict = {}
        Bubble_dict['ID'] = bub[0]
        XPoints = []
        YPoints = []
        for b in bub[1]:
            YPoints.append(int(b[0]))
            XPoints.append(int(b[1]))
        Bubble_dict['YPoints'] = YPoints
        Bubble_dict['XPoints'] = XPoints
        Bubbles_dict.append(Bubble_dict)
    with open(directory+name+'.json', "w") as out:
        json.dump(Bubbles_dict, out, indent=1)

@jit(nopython=True)
def get_grad(points,gradimg_neigh):
    Mean_grad=0
    counter=0
    for p in points:
        if p[2]==0:
            Mean_grad+=np.max(gradimg_neigh[p[0],p[1]])
            counter+=1
    if counter>0:
        return Mean_grad/counter
    else:
        return 0