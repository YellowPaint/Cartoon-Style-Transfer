import cv2 as cv
import math
import numpy as np
import os

def edge_detect(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_RGB2GRAY)

    edge_output = cv.Canny(gray, 50, 100)
    return edge_output

def edge_dilate(image,edge_output):
    image_dilate = image;
    
    for col in range(2,255):
    	for raw in range(2,255):
            if edge_output[col,raw] == 255:
                if(edge_output[col-1,raw-1] != 255):
                    image_dilate[col-1,raw-1,:] = image[col,raw,:]
                if(edge_output[col+1,raw-1] != 255):
                    image_dilate[col+1,raw-1,:] = image[col,raw,:]
                if(edge_output[col-1,raw+1] != 255):
                    image_dilate[col-1,raw+1,:] = image[col,raw,:]
                if(edge_output[col+1,raw+1] != 255):
                    image_dilate[col+1,raw+1,:] = image[col,raw,:]
    return image_dilate

class MyGaussianBlur():
    #initization
    def __init__(self, radius=1, sigema=1.5):
        self.radius=radius
        self.sigema=sigema    
    #the calculation of Gauss
    def calc(self,x,y):
        res1=1/(2*math.pi*self.sigema*self.sigema)
        res2=math.exp(-(x*x+y*y)/(2*self.sigema*self.sigema))
        return res1*res2
    #the Gauss module
    def template(self):
        sideLength=self.radius*2+1
        result = np.zeros((sideLength, sideLength))
        for i in range(sideLength):
            for j in range(sideLength):
                result[i,j]=self.calc(i-self.radius, j-self.radius)
        all=result.sum()
        return result/all    
    #the filter function
    def filter(self, image, edge_output, template): 
        height = image.shape[0]
        width = image.shape[1]
        imageEdgeSmooth = image
        for i in range(self.radius, height-self.radius):
            for j in range(self.radius, width-self.radius):
                if(edge_output[i,j] == 255 or edge_output[i-1,j-1] == 255 or edge_output[i-1,j+1] == 255 or
                edge_output[i+1,j-1] == 255 or edge_output[i+1,j+1] == 255):
                    for k in range(0,3):
                        t = image[i-self.radius:i+self.radius+1, j-self.radius:j+self.radius+1,k]
                        a = np.multiply(t, template)
                        imageEdgeSmooth[i,j,k] = a.sum()
        return imageEdgeSmooth


GBlur=MyGaussianBlur(radius=2, sigema=1.5)
temp=GBlur.template()

file_dir = './data/train/Cartoon/'
if not os.path.exists('./data/train/Cartoon_blur/'):
    os.makedirs('./data/train/Cartoon_blur/')
for _, _,files in os.walk(file_dir):
    #read images
    for i in range(0,len(files)):
        path = file_dir + files[i]
        src = cv.imread(path)
        edge_output = edge_detect(src)
        image_dilate = edge_dilate(src,edge_output)
        image_edgeBlur=GBlur.filter(image_dilate, edge_output, temp)
        print("\r",i+1,"/",len(files),end = "")

        savePath = './data/train/Cartoon_blur/' + str(i).zfill(4) + '.jpg'
        cv.imwrite(savePath, image_edgeBlur)