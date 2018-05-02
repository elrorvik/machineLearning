from skimage.io import imread_collection, imshow, concatenate_images
from skimage import io
import numpy as np
import skimage
import matplotlib.pyplot as plt
from skimage import color, feature
from PIL import Image
from scipy import misc
import imageio
import cv2
import os


#Retruns 20,20 images of the input using sliding expecting to contain characters.
#Used sliding window detection with input grid size
#Also makes use of canny edge detection and erosion techniques
#Saves to output folder with tag "name" if save is True
#Requires cv2, scipy, pillow, matplotlib, numpy, skimage

def sliding_window(image,grid=(22,22),name="output",save=False):
    print("Sliding window grid: ",grid)
    image = color.rgb2gray(image)
    image = image/image.max()
    lx,ly = len(image[0]),len(image)
    canny = feature.canny(image)

    #Sliding window on canny edge image
    coord = np.zeros([ly,lx])
    for y in range(ly - grid[0]):
        for x in range(lx - grid[1]):
            w = canny[y:y+grid[0],x:x+grid[1]]
            m = np.mean(w[1:,1:])
            sx = np.mean(w[0,:])
            sy = np.mean(w[:,0])
            if(m>0.1 and sx<0.05 and sy<0.05):
                coord[y+1,x+1] = m

    peaks = coord.copy()
    dx,dy = 2,2
    #Erode less intense neighbours, dinstance dx,dy
    for y in range(dy,ly - grid[1]-dy):
        for x in range(dx,lx - grid[0]-dx):
            p = coord[y,x]
            if(p<coord[y-dy:y+dy,x-dx:x+dx].max()):
                peaks[y,x] = 0


    #Erode neighbours with less neighbours, distance dx,dy
    n = np.zeros([ly,lx])
    for y in range(dy,ly - grid[1]-dy):
        for x in range(dx,lx - grid[0]-dy):
            if(peaks[y,x] > 0):
                n[y,x] =np.count_nonzero(peaks[y-dy:y+dy,x-dx:x+dx])
      
    for y in range(dy,ly - grid[1]-dy):
        for x in range(dx,lx - grid[0]-dx):
            if(n[y,x]<n[y-dy:y+dy,x-dx:x+dx].max()):
                n[y,x] = 0

    #Find results from neighbourhood matrice with remaining most popular pixels
    result = n>0
    segments = []
    segment_cord = []
    for y in range(ly):
        for x in range(lx):
            if(result[y,x]):
                r = cv2.resize(image[y:y+grid[0],x:x+grid[1]],(20,20))
                segments.append(r)
                segment_cord.append((x,y))

    #If save, write to output folder
    output_folder = "output/segments"
    if(save):
        if not os.path.exists(output_folder): #create the folder if it does not exist
            os.makedirs(output_folder)
        result_plot = result*3 + canny*0.2
        Image.fromarray(result_plot*255.0).convert('RGB').save(output_folder+name+"Plot.jpeg")

        imageio.mimsave(output_folder+name+".gif",np.asarray(segments))
        for i in range(len(segments)):
            Image.fromarray(segments[i]*255.0).convert('RGB').save(output_folder
                    +name+"_"+str(i)+".jpeg")

    return np.asarray(segments),segment_cord

def extract_windows(image,name,save,a=20,b=25):
    array_seg, array_cord = [],[] 
    for i in range(a,b):
        temp_seg, temp_cord = sliding_window(image,(i,i),name,save)
        array_seg.append(temp_seg)
        array_cord.append(temp_cord)
    segments = array_seg[0]
    coordinates = array_cord[1]
    for j in range(1,len(array_seg)):
        segments = np.concatenate((segments,array_seg[j]))
        coordinates = np.concatenate((coordinates,array_cord[j]))
    return segments,coordinates

    
        



def get_image_classify(folder_path):
    return imread_collection(folder_path)

def main():
    clas_dir = 'detection-images/*.jpg'
    print("===Classify: ",clas_dir )
    clas_images = get_image_classify(clas_dir)
    extract_window(clas_images[0],(20,20),name="image2",save=True)



if __name__ == "__main__":
    main()
