#from grabCutCrop import cropMask 
import cv2
import numpy as np

import sys

def cropMaskNo(imgH,imgW):
    mask = np.ones((imgH,imgW),np.uint8)
    return  mask
def cropMaskPortion(imgH,imgW,xp1,yp1,xp2,yp2,xp3,yp3,xp4,yp4):
    mask = np.zeros((imgH,imgW),np.uint8)
    y1 = yp1*imgH
    x1 = xp1*imgW
    y2 = yp2*imgH
    x2 = xp2*imgW
    y3 = yp3*imgH
    x3 = xp3*imgW
    y4 = yp4*imgH
    x4 = xp4*imgW
    cv2.fillConvexPoly(mask, np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], np.int32), 1, 8)
    return  mask

if len(sys.argv)<3:
    print 'Computes no-crop and mean-crop baselines, generates two files {gtFile}_fullno.res and {gtFile}_fullmean.res'
    print 'usage: '+sys.argv[0]+' gtFile.csv imageDir [mean_x1 mean_y1 mean_x2 mean_y2 mean_x3 mean_y3 mean_x4 mean_y4]'
    print '                                           (optional mean box)'
    exit()

gtFile=sys.argv[1]
imageDir=sys.argv[2]
if imageDir[-1]!='/':
    imageDir+='/'
reverse=False

sumIOU_no=0
sumIOU_mean=0
countIOU=0

scale=1

print 'eval on '+gtFile

outFile_no = gtFile+'_fullno.res'
outFile_mean = gtFile+'_fullmean.res'
numLines=0
try:
    with open(outFile_no,'r') as f:
        numLines = len(f.readlines())
except IOError:
    numLines=0
try:
    out_no = open(outFile_no,'a')
except IOError:
    out_no = open(outFile_no,'w')

try:
    out_mean = open(outFile_mean,'a')
except IOError:
    out_mean = open(outFile_mean,'w')

with open(gtFile) as f:
    cc=0
    lines = f.readlines()
    xp1=0
    yp1=0
    xp2=0
    yp2=0
    xp3=0
    yp3=0
    xp4=0
    yp4=0
    imgHW={}
    for line in lines:
        p = line.split(',')
        imagePath = p[0]
        x1 = float(p[1])*scale
        y1 = float(p[2])*scale
        x2 = float(p[3])*scale
        y2 = float(p[4])*scale
        x3 = float(p[5])*scale
        y3 = float(p[6])*scale
        x4 = float(p[7])*scale
        y4 = float(p[8])*scale
        if reverse:
            tmpX=x4
            tmpY=y4
            x4=x3
            y4=y3
            x3=tmpX
            y4=tmpY
        #type = p[9]
        if x1<0:
            continue
        cc+=1
        image = cv2.imread(imageDir+imagePath)
        imgHW[imagePath]=(int(image.shape[0]*scale),int(image.shape[1]*scale))
        xp1 += x1/(image.shape[1]*scale)
        yp1 += y1/(image.shape[0]*scale)
        xp2 += x2/(image.shape[1]*scale)
        yp2 += y2/(image.shape[0]*scale)
        xp3 += x3/(image.shape[1]*scale)
        yp3 += y3/(image.shape[0]*scale)
        xp4 += x4/(image.shape[1]*scale)
        yp4 += y4/(image.shape[0]*scale)
    xp1/=cc
    yp1/=cc
    xp2/=cc
    yp2/=cc
    xp3/=cc
    yp3/=cc
    xp4/=cc
    yp4/=cc
    if len(sys.argv)>9:
        xp1=float(sys.argv[3])
        yp1=float(sys.argv[4])
        xp2=float(sys.argv[5])
        yp2=float(sys.argv[6])
        xp3=float(sys.argv[7])
        yp3=float(sys.argv[8])
        xp4=float(sys.argv[9])
        yp4=float(sys.argv[10])
    else:
        print str(xp1)+' '+str(yp1)+' '+str(xp2)+' '+str(yp2)+' '+str(xp3)+' '+str(yp3)+' '+str(xp4)+' '+str(yp4)

    cc=0
    for line in lines:
        p = line.split(',')
        imagePath = p[0]
        x1 = float(p[1])*scale
        y1 = float(p[2])*scale
        x2 = float(p[3])*scale
        y2 = float(p[4])*scale
        x3 = float(p[5])*scale
        y3 = float(p[6])*scale
        x4 = float(p[7])*scale
        y4 = float(p[8])*scale
        if reverse:
            tmpX=x4
            tmpY=y4
            x4=x3
            y4=y3
            x3=tmpX
            y4=tmpY
        #type = p[9]
        if x1<0:
            continue

        cc+=1
        if cc<=numLines:
            continue

        #image = cv2.imread(imageDir+imagePath)
        hw = imgHW[imagePath]
        #print hw
        mask_no = cropMaskNo(hw[0],hw[1])
        mask_mean = cropMaskPortion(hw[0],hw[1],xp1,yp1,xp2,yp2,xp3,yp3,xp4,yp4)
        gtMask = np.zeros(mask_no.shape,np.uint8)
        cv2.fillConvexPoly(gtMask, np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], np.int32), 1, 8)

        intersection_no = np.sum(mask_no&gtMask)
        union_no = np.sum(mask_no|gtMask)
        sumIOU_no += float(intersection_no)/union_no
        out_no.write(imagePath+' '+str(float(intersection_no)/union_no)+'\n')
        out_no.flush()

        intersection_mean = np.sum(mask_mean&gtMask)
        union_mean = np.sum(mask_mean|gtMask)
        sumIOU_mean += float(intersection_mean)/union_mean
        out_mean.write(imagePath+' '+str(float(intersection_mean)/union_mean)+'\n')
        out_mean.flush()

        countIOU += 1

out_no.write('mean IOU for '+gtFile+': '+str(sumIOU_no/countIOU)+'\n')
out_no.close()

out_mean.write('mean IOU for '+gtFile+': '+str(sumIOU_mean/countIOU)+'\n')
out_mean.close()

