#from grabCutCrop import cropMask 
from multiprocessing.dummy import Pool as ThreadPool 
import cv2
import numpy as np

import sys

def cropMask(imageDir,saveDir,imagePath, border):
    interName = saveDir+'grabCut_'+imagePath.replace('/','_')+'.png'
    mask=cv2.imread(interName,0)
    if mask is None or mask.shape[0]==0:
        img = cv2.imread(imageDir+imagePath)
        if img is None:
            print "Could not read: "+imageDir+imagePath
        mask = np.zeros(img.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (border,border,img.shape[1]-2*border,img.shape[0]-2*border) #(x,y,w,h)
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        cv2.imwrite(interName,mask)


    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

    revMask = 1-mask2
    #fix the mask so that there is only one CC
    num, labels, stats, centiods = cv2.connectedComponentsWithStats(revMask, 8, cv2.CV_32S)
    maxLabel=None
    maxSize=0
    for l in range(1,num):
        if stats[l,cv2.CC_STAT_AREA] > maxSize:
            maxSize = stats[l,cv2.CC_STAT_AREA]
            maxLabel = l


    return  (labels!=maxLabel).astype('uint8')

if len(sys.argv)<5:
    print 'This evalutes using grab cut'
    print 'usage: '+sys.argv[0]+' gtFile.csv imageDir saveIntermDir numThreads' # [reverse]'
    exit(0)

gtFile=sys.argv[1]
imageDir=sys.argv[2]
if imageDir[-1]!='/':
    imageDir+='/'
saveDir=sys.argv[3]
if saveDir[-1]!='/':
    saveDir+='/'
numThreads = int(sys.argv[4])
reverse=False
#if len(sys.argv)>5 and sys.argv[5][0]=='r':
#    reverse=True

sumIOU=0
countIOU=0

scale=1

print 'eval on '+gtFile

outFile = gtFile+'_fullgrab.res'
#numLines=0
#try:
#    with open(outFile,'r') as f:
#        numLines = len(f.readlines())
#except IOError:
#    numLines=0
try:
    out = open(outFile,'a')
except IOError:
    out = open(outFile,'w')



def worker(line):
    global imageDir, saveDir, reverse
    try:
        p = line.split(',')
        imagePath = p[0]
        x1 = float(p[1])
        y1 = float(p[2])
        x2 = float(p[3])
        y2 = float(p[4])
        x3 = float(p[5])
        y3 = float(p[6])
        x4 = float(p[7])
        y4 = float(p[8])
        if reverse:
            tmpX=x4
            tmpY=y4
            x4=x3
            y4=y3
            x3=tmpX
            y4=tmpY
        #type = p[9]
        if x1<0:
            return None,None

        #cc+=1
        #if cc<=numLines:
        #    return None, None

        #image = cv2.imread(imageDir+imagePath)
        #if image.shape[0]==0:
        #    print 'failed to open '+imageDir+imagePath
        #image = cv2.resize(image,(0,0),None,scale,scale)
        mask = cropMask(imageDir,saveDir,imagePath,5)
        gtMask = np.zeros(mask.shape,np.uint8)
        cv2.fillConvexPoly(gtMask, np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], np.int32), 1, 8)

        intersection = np.sum(mask&gtMask)
        union = np.sum(mask|gtMask)
        if float(intersection)/union < 0.6:
            cv2.imwrite(saveDir+"ERR_"+imagePath.replace('/','_')+'.png',mask*255);
        return imagePath, float(intersection)/union
    except:
        print 'Error: '
        print sys.exc_info()

with open(gtFile) as f:
    pool = ThreadPool(numThreads)
    results = pool.map(worker, f.readlines())
    for (imagePath, iou) in results:
        if imagePath is not None:
            sumIOU += iou
            out.write(imagePath+' '+str(iou)+'\n')
            countIOU += 1

out.write('mean IOU for '+gtFile+': '+str(sumIOU/countIOU)+'\n')
out.close()
print 'mean IOU for '+gtFile+': '+str(sumIOU/countIOU)


