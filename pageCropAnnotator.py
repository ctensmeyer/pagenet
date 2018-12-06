#!/usr/bin/python

#This takes a list of image files and acts as a tool to mark the crop region of the page

import re
import xml.etree.ElementTree as ET
import os
import sys
from StringIO import StringIO
import cv2



def showControls():
    print(' -----------------------------------------------')
    print('| CONTROLS:                                     |')
    print('| * set new corner (base on loc):   left-click  |')
    print('| * set new seam corner (two-page): middle-click|')
    print('| * confirm corners:                enter       |')
    print('| * mark page as abnormal:          a           |')
    print('| * undo:                           backspace   |')
    print('| * start previous page over:       backspace(+)|')
    #print('| * start current page over:        delete      |')
    print('| * exit:                           esc         |')
    print('|                                               |')
    print(' -----------------------------------------------')



lastDidList=[]
tl=(-1,-1)
tr=(-1,-1)
bl=(-1,-1)
br=(-1,-1)
tm=(-1,-1)
bm=(-1,-1)

image=None
orig=None

abnorm=False

def draw():
    global image,tl,tr,bl,br,tm,bm,abnorm
    if tm[0]>=0 and bm[0]>=0:
        cv2.line(image, tm, bm, (0,255,0), 2)
    if tm[0]>=0 and tl[0]>=0:
        cv2.line(image, tm, tl, (0,255,0), 2)
    if tm[0]>=0 and tr[0]>=0:
        cv2.line(image, tm, tr, (0,255,0), 2)
    if bm[0]>=0 and bl[0]>=0:
        cv2.line(image, bm, bl, (0,255,0), 2)
    if bm[0]>=0 and br[0]>=0:
        cv2.line(image, bm, br, (0,255,0), 2)
    if tl[0]>=0 and tr[0]>=0:
        cv2.line(image, tl, tr, (255,0,0), 2)
    if br[0]>=0 and tr[0]>=0:
        cv2.line(image, br, tr, (255,0,0), 2)
    if br[0]>=0 and bl[0]>=0:
        cv2.line(image, br, bl, (255,0,0), 2)
    if tl[0]>=0 and bl[0]>=0:
        cv2.line(image, tl, bl, (255,0,0), 2)


    if tl[0]>=0:
        image[tl[1],tl[0]]=(0,0,255)
        cv2.circle(image, tl, 2, (0,0,200), 1)
        cv2.circle(image, tl, 5, (0,0,200), 2)
    if tr[0]>=0:
        image[tr[1],tr[0]]=(0,0,255)
        cv2.circle(image, tr, 2, (0,0,200), 1)
        cv2.circle(image, tr, 5, (0,0,200), 2)
    if bl[0]>=0:
        image[bl[1],bl[0]]=(0,0,255)
        cv2.circle(image, bl, 2, (0,0,200), 1)
        cv2.circle(image, bl, 5, (0,0,200), 2)
    if br[0]>=0:
        image[br[1],br[0]]=(0,0,255)
        cv2.circle(image, br, 2, (0,0,200), 1)
        cv2.circle(image, br, 5, (0,0,200), 2)
    if tm[0]>=0:
        image[tm[1],tm[0]]=(0,0,255)
        cv2.circle(image, tm, 2, (0,100,200), 1)
        cv2.circle(image, tm, 5, (0,100,200), 2)
    if bm[0]>=0:
        image[bm[1],bm[0]]=(0,0,255)
        cv2.circle(image, bm, 2, (0,100,200), 1)
        cv2.circle(image, bm, 5, (0,100,200), 2)

    if abnorm:
        cv2.putText(image, 'ABNORMAL', (image.shape[1]/2,image.shape[0]/2), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255))


    cv2.imshow("image", image)

bimage=None
def clicker(event, x, y, flags, param):
        # grab references to the global variables
        global image,tl,tr,bl,br,tm,bm,lastDidList,orig

        """if event == cv2.EVENT_LBUTTONDOWN:
            if len(segPts)>0:
                #change last boundary
                image=bimage.copy()
                segPts[-1]=x
                ll=max(0,segPts[-1]-1)
                rr=min(image.shape[1], segPts[-1]+1)
                image[:,ll:rr,0] = color[(colorIdx+len(color)-1)%len(color)][0] * image[:,ll:rr,0]
                image[:,ll:rr,1] = color[(colorIdx+len(color)-1)%len(color)][1] * image[:,ll:rr,1]
                image[:,ll:rr,2] = color[(colorIdx+len(color)-1)%len(color)][2] * image[:,ll:rr,2]
                cv2.imshow("image", image)
"""
        if event == cv2.EVENT_LBUTTONDOWN:
                # a new boundary
                if x<image.shape[1]/2 and y<image.shape[0]/2:
                    tl=(x,y)
                    if 0 in lastDidList:
                        image=orig.copy()
                        lastDidList.remove(0)
                    lastDidList.append(0)
                if x>image.shape[1]/2 and y<image.shape[0]/2:
                    tr=(x,y)
                    if 1 in lastDidList:
                        image=orig.copy()
                        lastDidList.remove(1)
                    lastDidList.append(1)
                if x<image.shape[1]/2 and y>image.shape[0]/2:
                    bl=(x,y)
                    if 2 in lastDidList:
                        image=orig.copy()
                        lastDidList.remove(2)
                    lastDidList.append(2)
                if x>image.shape[1]/2 and y>image.shape[0]/2:
                    br=(x,y)
                    if 3 in lastDidList:
                        image=orig.copy()
                        lastDidList.remove(3)
                    lastDidList.append(3)
                draw()

        elif event == cv2.EVENT_MBUTTONDOWN:
                # a new boundary
                if y<image.shape[0]/2:
                    tm=(x,y)
                    if 1 in lastDidList:
                        image=orig.copy()
                        lastDidList.remove(1)
                    lastDidList.append(4)
                if y>image.shape[0]/2:
                    bm=(x,y)
                    if 3 in lastDidList:
                        image=orig.copy()
                        lastDidList.remove(3)
                    lastDidList.append(5)
                draw()

def segmenter(imDir,imagePath,dispHeight):
    global image,tl,tr,bl,br,tm,bm,lastDidList,orig,abnorm
    print 'opening '+imDir+imagePath
    orig = cv2.imread(imDir+imagePath)
    scale = orig.shape[0]/dispHeight
    orig = cv2.resize(orig,(0,0),None,1.0/scale,1.0/scale)
    #print 'opened'
    assert orig is not None
    redo=True
    while redo: #undo loop
        abnorm=False
        lastDidList=[]
        tl=(-1,-1)
        tr=(-1,-1)
        bl=(-1,-1)
        br=(-1,-1)
        tm=(-1,-1)
        bm=(-1,-1)
        redo=False
        image = orig.copy()
        draw()
        while True:
            # display the imageWork and wait for a keypress
            key = cv2.waitKey(33) & 0xFF #so it is robust on all systems
            #print key
            if key == 13 and tl[0]>=0 and tr[0]>=0 and bl[0]>=0 and br[0]>=0: #enter
                toWrite = imagePath+','+str(int(scale*tl[0]))+','+str(int(scale*tl[1]))+','+str(int(scale*tr[0]))+','+str(int(scale*tr[1]))+','+str(int(scale*br[0]))+','+str(int(scale*br[1]))+','+str(int(scale*bl[0]))+','+str(int(scale*bl[1]))
                if abnorm:
                    if tm[0]>=0 and bm[0]>=0:
                        toWrite += ',ABNORMAL,'+str(int(scale*tm[0]))+','+str(int(scale*tm[1]))+','+str(int(scale*bm[0]))+','+str(int(scale*bm[1]))
                    else:
                        toWrite += ',ABNORMAL'
                else:
                    if tm[0]>=0 and bm[0]>=0:
                        toWrite += ',DOUBLE,'+str(int(scale*tm[0]))+','+str(int(scale*tm[1]))+','+str(int(scale*bm[0]))+','+str(int(scale*bm[1]))
                    else:
                        toWrite += ',SINGLE'
                toWrite+='\n';
                return toWrite, False, False
            elif key == 8: #backspace
                if len(lastDidList)>0:
                    imageWork = orig.copy()
                    lastDid=lastDidList.pop()
                    if lastDid==0:
                        tl=(-1,-1)
                    elif lastDid==1:
                        tr=(-1,-1)
                    elif lastDid==2:
                        bl=(-1,-1)
                    elif lastDid==3:
                        br=(-1,-1)
                    elif lastDid==4:
                        tm=(-1,-1)
                    elif lastDid==5:
                        bm=(-1,-1)
                    image=orig.copy()
                    draw()
                else:
                    return '', True, False
            elif key == 127: #del
                #if len(lastDidList)>0:
                    print('[CLEAR]')
                    redo=True
                    break
                #else:
                #    return '', True, False
            elif key == 27: #esc
                print('esc')
                return '', False, True
                #exit(0)
                #break
            elif key == 97: #'a'
                #return imagePath+',-1,-1,-1,-1,-1,-1,-1,-1,ABNORMAL\n', False, False
                abnorm = not abnorm
                image=orig.copy()
                draw()

    #return newWords, newWordBoxes

if len(sys.argv)<4:
    print 'usage: '+sys.argv[0]+' imgDir imgList outAnn.csv [displayHeight]'
    print 'output format: imageFile, tlx, tly, trx, try, brx, bry, blx, bly, type (,tmx, tmy, bmx, bmy)'
    exit(0)

inFile = sys.argv[2]
imDir = sys.argv[1]
if imDir[-1]!='/':
    imDir+='/'
outFile = sys.argv[3]
dispHeight=500.0
if len(sys.argv)>4:
    dispHeight=float(sys.argv[4])

cv2.namedWindow("image")
cv2.setMouseCallback("image", clicker)

didCount=0
did=[]
try:
    check = open(outFile,'r')
    did = check.read().splitlines()
    didCount=len(did)
    check.close()
    print 'found '+outFile+', appending. Note: this is sychronizing based on count alone, if '+inFile+' hash changed, but sure to align '+outFile
except IOError:
    print ('making new out:'+outFile)

out = open(outFile,'w')

print ' =============================================== '
print ' !!! INSTRUCTIONS !!!'
print ' If the page does not contain a single page, or  '
print ' an open book, mark as abnormal with INSERT (e.g.'
print ' two seperate pages).'
print ' Click on the four corners to include all the    '
print ' full pages in the image (including two pages if '
print ' fully present).'
print ' If two pages a full present also mark page seam '
print ' (middle-click).'
print ' On placing points, prioritize the following to  '
print ' be included/discluded from the polygons in the  '
print ' following order:'
print '   1. Including the present page(s) content.'
print '   2. Discluding other pages and background.'
print '   3. Discluding the present page(s) boudary.'
print '   4. Including the present page(s) white area.'
#print ' book). If a corner is torn, click where it ought'
#print ' to be, based on page edges. The page seem on an '
#print ' open book is the page edge.'
print ' Use ESC to exit or the latest page you finished '
print ' will be lost.'

#i=didCount
i=0
#pageCount=-1
prevSeg=''
seg=''
showControls()
inF = open(inFile,'r')
images = inF.read().splitlines()
end=False
doneOne=False
while i<len(images) and not end:
    if i%10==9:
        showControls()
    print(str(i+1)+' of '+str(len(images)))

    if len(did)>i:
        line = did[i].strip().split(',')
        typ = line[8]
        #print typ
        if typ != '-1':
            out.write(did[i].strip()+'\n')
            i+=1
            continue
        seg, undo, end = segmenter(imDir, images[i],dispHeight)
        out.write(seg)
        seg=''
        i+=1
    else:


        seg, undo, end = segmenter(imDir, images[i],dispHeight)
        if len(seg)>0:
            doneOne=True
        
        if undo and i>0 and doneOne:
            prevSeg=''
            print(str(i)+' of '+str(len(images)))
            prevSeg, undo, end = segmenter(imDir, images[i-1],dispHeight)
        else:
            out.write(prevSeg)
            prevSeg=seg
            seg=''
            i+=1
out.write(prevSeg)
out.write(seg)
