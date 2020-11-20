import cv2
import matplotlib
import numpy as np
import scipy
import scipy.io as sio
import imutils
import os
import mahotas as mt
from imutils import contours
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
from sklearn.neural_network import MLPClassifier
import csv
p=1;
# Loading image
os.chdir('C:/Users/sarka/Desktop/Major Project/Latest/Input/Code/CROP CODE - PY/test');   ## change here the test image path
img = cv2.imread('2.jpg');
os.chdir('C:/Users/sarka/Desktop/Major Project/Latest/Input/Code/CROP CODE - PY');## change here to original code path
cv2.imshow('Input Image',img);

##Gaussian Blurring
kernel = np.ones((7,7),np.float32)/25
img1 = cv2.filter2D(img,-1,kernel)
cv2.imshow('Gaussian Image',img1)

## Bilateral Filter for Edge Enhancement
img3 = cv2.bilateralFilter(img1,9,75,75)
cv2.imshow('Bilateral Filtered Image',img3)


## RGB to Gray conversion
GRAY_Img = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
cv2.imshow('GRAY Image',GRAY_Img)

Data2Ext=GRAY_Img;
cv2.imwrite('ImageRedist.jpg',Data2Ext);
print ("Testing")

roi1=GRAY_Img;
r,c=roi1.shape;
if p==1:
    roi = roi1.reshape((roi1.shape[0] * roi1.shape[1], 1))
print(roi)
## KMEANS clustering
imgkmeans = KMeans(n_clusters=2, random_state=0);
imgkmeans.fit(roi);
label_values=imgkmeans.labels_;
Label_reshped = np.reshape(label_values,(roi1.shape[0] ,roi1.shape[1]));

segmentregions=roi1;
blobregions=roi1;

rows,cols = roi1.shape;
# Thresholding for segmentation
for i in range(0,rows):
    for j in range(0,cols):
        pixl=Label_reshped[i,j];
        if pixl==0:
            segmentregions[i,j]=255;
            
        else:
            segmentregions[i,j]=0;

cv2.imshow('Segemented Image',segmentregions)


# Thresholding for segmentation
NewImage = cv2.imread('ImageRedist.jpg');
NewImage= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
extractedregions=NewImage;

for k in range(0,rows):
    for l in range(0,cols):
        pixl1=segmentregions[k,l];
        if pixl1==0:
##            print 'ok'
            extractedregions[k,l]=NewImage[k,l];
##            print extractedregions[k,l]
        else:
##            print 'no'
            extractedregions[k,l]=0;

cv2.imshow('Extracted Regions Image',extractedregions)


## GLCM Features Extractor
def extract_features(image):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(image)

        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean


GLCMfeatures = extract_features(extractedregions);
(means, stds) = cv2.meanStdDev(extractedregions)
## Normal Mean Standard Deviation
print (GLCMfeatures)
print (means, stds)

al=np.size(GLCMfeatures);

Id=np.zeros((al+2,), dtype=np.float);
for i in range(0,al+2):
    if i<13:
        Id[i]=GLCMfeatures[i];
    elif i==13:
        Id[i]=means;
    else:
        Id[i]=stds;

valuu=np.mean(Id);

print (np.mean(Id))

p=1;
W=np.zeros((15), dtype=np.float);
DBVal=np.zeros((20,), dtype=np.float);
for im in range(1,20):
    # Loading image
    os.chdir('C:/Users/sarka/Desktop/Major Project/Latest/Input/Code/CROP CODE - PY/dataset');
    img = cv2.imread(str(im)+'.jpg');
    os.chdir('C:/Users/sarka/Desktop/Major Project/Latest/Input/Code/CROP CODE - PY');
##    cv2.imshow('Input Image',img);

    ##Gaussian Blurring
    kernel = np.ones((7,7),np.float32)/25
    img1 = cv2.filter2D(img,-1,kernel)
##    cv2.imshow('Gaussian Image',img1)

    ## Bilateral Filter for Edge Enhancement
    img3 = cv2.bilateralFilter(img1,9,75,75)
##    cv2.imshow('Bilateral Filtered Image',img3)


    ## RGB to Gray conversion
    GRAY_Img = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
##    cv2.imshow('GRAY Image',GRAY_Img)

    Data2Ext=GRAY_Img;
    cv2.imwrite('ImageRedist.jpg',Data2Ext);
    print ("Training")

    roi1=GRAY_Img;
    r,c=roi1.shape;
    if p==1:
        roi = roi1.reshape((roi1.shape[0] * roi1.shape[1], 1))

    ## KMEANS clustering
    imgkmeans = KMeans(n_clusters=2, random_state=0);
    imgkmeans.fit(roi);
    label_values=imgkmeans.labels_;
    Label_reshped = np.reshape(label_values,(roi1.shape[0] ,roi1.shape[1]));

    segmentregions=roi1;
    blobregions=roi1;

    rows,cols = roi1.shape;
    # Thresholding for segmentation
    for i in range(0,rows):
        for j in range(0,cols):
            pixl=Label_reshped[i,j];
            if pixl==0:
                segmentregions[i,j]=255;
                
            else:
                segmentregions[i,j]=0;

##    cv2.imshow('Segemented Image',segmentregions)

##    _, contours, hierarchy = cv2.findContours(segmentregions, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
##    contour_list = []
##    for contour in contours:
##        area = cv2.contourArea(contour)
##        if area > 100 :
##            contour_list.append(contour)
##
##    cnt = contours[1:];
##    cv2.drawContours(segmentregions, contour_list,  -1, (255,0,0), 2)
##    cv2.imshow('Regions Detected',segmentregions)
    ##print contour_list

    # Thresholding for segmentation
    NewImage = cv2.imread('ImageRedist.jpg');
    NewImage= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    extractedregions=NewImage;

    for k in range(0,rows):
        for l in range(0,cols):
            pixl1=segmentregions[k,l];
            if pixl1==0:
                extractedregions[k,l]=NewImage[k,l];
            else:
                extractedregions[k,l]=0;
##    cv2.imshow('Extracted Regions Image',extractedregions)


    ## GLCM Features Extractor
    def extract_features(image):
            # calculate haralick texture features for 4 types of adjacency
            textures = mt.features.haralick(image)

            # take the mean of it and return it
            ht_mean = textures.mean(axis=0)
            return ht_mean


    GLCMfeatures = extract_features(extractedregions);
    (means, stds) = cv2.meanStdDev(extractedregions)
    ## Normal Mean Standard Deviation
    print (GLCMfeatures)
    print (means, stds)

    al=np.size(GLCMfeatures);

    Id=np.zeros((al+2,), dtype=np.float);
    for i in range(0,al+2):
        if i<13:
            Id[i]=GLCMfeatures[i];
        elif i==13:
            Id[i]=means;
        else:
            Id[i]=stds;
    DBVal[im-1]=np.mean(Id);
    print (np.mean(Id))


################################
################################
################################
## Testing
for kk1 in range(0,20):
    if valuu==DBVal[kk1]:
        det=kk1+1;

        if det==1:
            print ('Zea mays)')
            print('Maize')
        elif det==2:
            print ('Oryza sativa, Oryza glaberrima')
            print('Rice')
        elif det==3:
            print ('Triticum aestivum')
            print('Wheat')
        elif det==4:
            print ('Sorghum vulgare')
            print('Jowar')
        elif det==5:
            print ('Dolichos biffoeus')
            print('Horse Gram')
        elif det==6:
            print ('Cojonus cgjan')
            print('Red Gram')
        elif det==7:
            print ('Phaseolies auicus')
            print('Green Gram')
        elif det==8:
            print ('Plasoes mungo')
            print('Black Gram') 
        else:
            print ('Paddy')



cv2.waitKey(0)
cv2.destroyAllWindows()


            

