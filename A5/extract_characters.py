from skimage.io import imread_collection, imshow, concatenate_images
from skimage import io
import numpy as np
import skimage
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage import data, color, exposure, feature
from sklearn import preprocessing
from sklearn.svm import LinearSVC
import skimage.filter
import sklearn.svm as ssv
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
from scipy import misc
import imageio

def extract_window(image,grid=(20,20),name="output",save=False):
    print("Extracting window segments: ",grid)
    image = color.rgb2gray(image)
    image = image/image.max()
    lx,ly = len(image[0]),len(image)
    canny = feature.canny(image)
    #plt.imshow(canny, interpolation='nearest')
    #plt.show()

    coord = np.zeros([ly,lx])
    for y in range(ly - grid[0]):
        for x in range(lx - grid[1]):
            w = canny[y:y+grid[0],x:x+grid[1]]
            m = np.mean(w)
            sx = np.mean(w[0,:])
            sy = np.mean(w[:,0])
            if(m>0.1 and sx<0.05 and sy<0.05):
                coord[y,x] = m

    #plt.imshow(coord, interpolation='bilinear')
    #plt.show()

    peaks = coord.copy()
    dx,dy = 2,2
    for y in range(dy,ly - grid[1]-dy):
        for x in range(dx,lx - grid[0]-dx):
            p = coord[y,x]
            if(p<coord[y-dy:y+dy,x-dx:x+dx].max()):
                peaks[y,x] = 0

    #plt.imshow(peaks, interpolation='bilinear')
    #plt.show()

    dx,dy = 2,2
    pruned = peaks.copy()
    n = np.zeros([ly,lx])
    for y in range(dy,ly - grid[1]-dy):
        for x in range(dx,lx - grid[0]-dy):
            if(peaks[y,x] > 0):
                n[y,x] =np.count_nonzero(peaks[y-dy:y+dy,x-dx:x+dx])
      
    m = n.copy()
    for y in range(dy,ly - grid[1]-dy):
        for x in range(dx,lx - grid[0]-dx):
            if(n[y,x]<n[y-dy:y+dy,x-dx:x+dx].max()):
                n[y,x] = 0

    result = n>0

    segments = []
    for y in range(ly):
        for x in range(lx):
            if(result[y,x]):
                segments.append(image[y:y+grid[0],x:x+grid[1]])
    if(save):
        result_plot = result*3 + canny*0.2
        Image.fromarray(result_plot*255.0).convert('RGB').save("output/"+name+"Plot.jpeg")

        imageio.mimsave("output/"+name+".gif",np.asarray(segments))
        for i in range(len(segments)):
            Image.fromarray(segments[i]*255.0).convert('RGB').save("output/"
                    +name+"_"+str(i)+".jpeg")
    return segments


def get_image_classify(folder_path):
    return imread_collection(folder_path)

def main():
    clas_dir = 'detection-images/*.jpg'
    print("===Classify: ",clas_dir )
    clas_images = get_image_classify(clas_dir)
    extract_window(clas_images[0],(22,22),name="image1",save=True)



if __name__ == "__main__":
    main()
