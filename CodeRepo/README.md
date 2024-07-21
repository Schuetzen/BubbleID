# Star Convex Bubble Finder

The present repository contains the code to identify and reconstruct overlapping bubbles in 2D images as described in

- H. Hessenkemper, S. Starke, Y. Atassi, T. Ziegenhein, D. Lucas 
Bubble identification from images with machine learning methods. *International Journal of Multiphase Flows,Volume 155, 104169, https://doi.org/10.1016/j.ijmultiphaseflow.2022.104169*

It is based on the [*StarDist*](https://github.com/stardist/stardist) approach to predict star-convex polygons as object representation in crowded images. Further optional improvements can be achieved with a object mask provided by a *UNet* and a correction of the occluded bubble parts with a so-called *Radial Distance Correction*. For applying these three models, please download the pretrained [*Models*](http://doi.org/10.14278/rodare.1471).

## Installation

The code is compatible with python version 3.7. Please install the required packages using [conda](https://docs.anaconda.com/free/anaconda/install/index.html) with the following procedure. At first, the packages listed in the requirements.txt file can be installed using `pip`

> pip install -r requirements.txt

Then the correct numpy version needs to be reinstalled with

> pip install numpy==1.20.0

Ignore all incoming warnings.

## Usage

After installation the demonstration Prediction_demo.ipynb notebook can be used to identify and reconstruct overlapping bubbles in images. For this place the Models folder in the same directory where the notebook is located.

![](RDC.png)

If you have any questions, please contact h.hessenkemper@hzdr.de
