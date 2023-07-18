#%%

'''
See the Readme.md to set the conda environment.

This script is meant to be run inside of VS Code, similar to a Jupyter notebook.
'''

import open3d as o3d
from pathlib import Path
import numpy as np
import json,math

import cv2                      # (note, needed to use - conda install -c defaults opencv)
import matplotlib.pyplot as plt

from getDataBits import *     # our helper file

#%%

'''
Build a point cloud from a data set
- the 'fldr' specifies the data set in the simData folder
'''
def getPointCloud(fldr):
    # XYZ point cloud
    npXYZ = getPointCloudData(fldr)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(npXYZ)

    # and RGB
    npImg = getPointCloudRGB(fldr)
    npImg = npImg[:,0:3]
    pcd.colors = o3d.utility.Vector3dVector(npImg)
    return pcd

if False:   # testing
    pcd = getPointCloud(testFldr)
    o3d.io.write_point_cloud("sync.ply", pcd)
    o3d.visualization.draw_geometries([pcd])

#%%

'''
Create the needed zones:
- pallet
- allowed pallet
- safety zone
- in front of pallet
and, for the rotations
- allowed pallet - payload only
'''

def getRects(testFldr):
  twoDbboxs,threeDbboxs = getClassNumbers(testFldr)
  pallet3DLocation = get3DPalletAtOrigin(threeDbboxs) # pal3Dbbox,lDepth,palWidth,palHeight,threeDXForm
  pal3Dbbox = pallet3DLocation[0] # x_min,y_min,z_min,x_max,y_max,z_max
  # btw, X and Y are interchanged in the Isaac Sim data
  # other 'funk' is that Y grows downward.   
  yMin = pal3Dbbox[0] / 100.0   # Isaac Sim operates in cm 
  xMin = pal3Dbbox[1] / 100.0
  zMin = pal3Dbbox[2] / 100.0
  yMax = pal3Dbbox[3] / 100.0
  xMax = pal3Dbbox[4] / 100.0
  zMax = pal3Dbbox[5] / 100.0

  # pallet, as specified by Isaac Sim
  baseMin = np.array([xMin, yMin, zMin])
  baseMax = np.array([xMax, yMax, zMax])

  # constants
  floorExclude = 0.01
  overhang = 0.10     # extend bounds to fully see pallet
  payloadMaxHeight = 2.0
  safetyBounds = 0.2
  littleExtra = 0.025   # sim model had pallets w/ a tad too much overhang in front...

  # Allowed pallet   Want to at least see the pallet (front)
  palletMin = np.array([xMin-overhang, yMin-overhang-littleExtra, floorExclude])
  palletMax = np.array([xMax+overhang, yMax+overhang, zMax + payloadMaxHeight])

  # Safety zone
  palSafetyMin = np.array([xMin-safetyBounds, 
                            yMin-safetyBounds, 
                            floorExclude])
  palSafetyMax = np.array([xMax+safetyBounds,
                            yMax+safetyBounds, 
                            zMax + payloadMaxHeight])

  # In front of pallet
  inFrontMin = np.array([xMin-overhang, 
                              -3.0,              # camera position, more or less              
                              floorExclude])
  inFrontMax = np.array([xMax+overhang,
                              yMin-safetyBounds, 
                              zMax + payloadMaxHeight])
  
    # Allowed pallet   Want to at least see the pallet (front)
  payloadMin = np.array([xMin-overhang, yMin-overhang-littleExtra, zMax])
  payloadMax = np.array([xMax+overhang, yMax+overhang, zMax + payloadMaxHeight])


  # Return the bounds
  return [baseMin,baseMax],[palletMin,palletMax],[palSafetyMin,palSafetyMax],[inFrontMin,inFrontMax],[payloadMin,payloadMax]  

## Testing
if False:
  testFldr = tstFldrs[2]  # 5 leans

  base,pallet,safetyBounds,inFrontBounds,payloadBounds = getRects(testFldr)

  pcd = getPointCloud(testFldr)
  pcBaseCropd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(base[0],base[1]))
  pcPalletCropd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(pallet[0],pallet[1]))
  pcSafetyCropd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(safetyBounds[0],safetyBounds[1]))
  pcInFrontCropd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(inFrontBounds[0],inFrontBounds[1]))
  pcPayloadCropd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(payloadBounds[0],payloadBounds[1]))

  print(len(pcSafetyCropd.points),len(pcPalletCropd.points),len(pcInFrontCropd.points))

  ## ways to look at the various point clouds

  # # use test case #0
  # o3d.visualization.draw_geometries_with_editing([pcBaseCropd])

  # # use test case #0
  # o3d.visualization.draw_geometries_with_editing([pcPalletCropd])

  # # use test case #3
  # o3d.visualization.draw_geometries_with_editing([pcSafetyCropd])

  # # use test case #8
  # o3d.visualization.draw_geometries_with_editing([pcInFrontCropd])

  # use test case #1
  o3d.visualization.draw_geometries_with_editing([pcPayloadCropd])

#%%

'''
Test the zone geometry tests:
    1. The payload extends beyond where it should.
    2. The payload is too close to another object.
    3. There is something between the forklift and the front of the pallet.

For 1 and 2 - this is 
      len(pcSafetyCropd.points) - len(pcPalletCropd.points) < a small number
      We use 'a small number' in case of noise in the point cloud
For 3 - this is 
      len(pcInFrontCropd.points) > a small number

'''

aSmallNumber = 100    # 10 - 100 ?   Could also be a percentage of the payload size

def test123(testN):
  testFldr = tstFldrs[testN]
  pc = getPointCloud(testFldr)

  base,pallet,safetyBounds,inFrontBounds,payloadBounds = getRects(testFldr)

  pcPalletCropd = pc.crop(o3d.geometry.AxisAlignedBoundingBox(pallet[0],pallet[1]))
  pcSafetyCropd = pc.crop(o3d.geometry.AxisAlignedBoundingBox(safetyBounds[0],safetyBounds[1]))
  pcInFrontCropd = pc.crop(o3d.geometry.AxisAlignedBoundingBox(inFrontBounds[0],inFrontBounds[1]))
  
  points12 = len(pcSafetyCropd.points) - len(pcPalletCropd.points)
  points3 = len(pcInFrontCropd.points)

  b12 = points12 > aSmallNumber
  b3 = points3 > aSmallNumber

  return b12,b3

'''
tstFldrs = ['_out_sdrec_perfect',
            '_out_sdrec_slightRight',
            '_out_sdrec_veryRight',
            '_out_sdrec_adjacent',
            '_out_sdrec_adjacentFine',
            '_out_sdrec_adjacentLean',
            '_out_sdrec_forward',
            '_out_sdrec_fallRight',
            '_out_sdrec_inFront',
            ]
'''
correctAnswers = [[False,False],
                  [False,False],
                  [True,False],       # leans too far
                  [True,False],       # adjacent pallet
                  [False,False],      
                  [True,False],       # forklift too close
                  [True,True],        # forward leaning also moves out front
                  [True,False],   
                  [False,True],]     # safety cone in front

## This is the full testing for the point cloud cases
if True:
  # do tests and validate
  for i in range(9):
    b12,b3 = test123(i)
    c12,c3 = correctAnswers[i]
    if b12 != c12 or b3 != c3:
      print("FAILURE: Test ",i," - results ",b12,b3," - should be ",c12,c3)
    print("Test ",i," - correct results ",b12,b3)


#%%

'''
Now to test for the leaning pallets...

First, we need a view of the pallet from the front, followed by the side (noting
that we'll probably only see the front of the pallet, but that's ok).
'''

imgSize = 256   # size of image - small enough that we tend to fill all needed pixels.

def castPCto2D(pts):
  npImg = np.zeros((imgSize,imgSize),dtype=np.uint8) 
  minXVal = np.min(pts[:,0])
  maxXVal = np.max(pts[:,0])
  rangeXVal = maxXVal - minXVal

  minYVal = np.min(pts[:,1])
  maxYVal = np.max(pts[:,1])
  rangeYVal = maxYVal - minYVal

  for pt in pts:
    x = int( ( ( pt[0] - minXVal ) / rangeXVal ) * (imgSize-1))
    y = (imgSize-1) - int( ( ( pt[1] - minYVal ) / rangeYVal ) * (imgSize-1))
    npImg[y,x] = 255

  brdSz = 20
  npImgBorder = cv2.copyMakeBorder(npImg, brdSz, brdSz, brdSz, brdSz, cv2.BORDER_CONSTANT, value=0)
  return npImgBorder

def getTwoPerspectives(testN):
  testFldr = tstFldrs[testN]
  pc = getPointCloud(testFldr)
  base,pallet,safetyBounds,inFrontBounds,payloadBounds = getRects(testFldr)
  pcPayload = pc.crop(o3d.geometry.AxisAlignedBoundingBox(payloadBounds[0],payloadBounds[1]))

  pts = np.asarray(pcPayload.points)

  # Two perspectives
  xzpts = np.delete(pts,1,1)
  yzpts = np.delete(pts,0,1)

  xzImg = castPCto2D(xzpts)
  yzImg = castPCto2D(yzpts)

  return xzImg,yzImg,xzpts,yzpts  


if False:
  testN = 1
  xzImg,yzImg,xzpts,yzpts = getTwoPerspectives(testN)

  if True:
    plt.imshow(xzImg)
  else:
    plt.imshow(yzImg)

  plt.show()

#%%

def doAngleThing(testN,bFront):
  xzImg,yzImg,xzpts,yzpts = getTwoPerspectives(testN)


  if bFront:
    img = xzImg
    pts = xzpts
  else:
    img = yzImg
    pts = yzpts

  # # surprisingly different answer - so not the way to go, eh?
  # vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
  # print("from float points",vx, vy, x0, y0)
  # print(math.atan2(vy,vx),math.atan2(vy,vx)*180.0/math.pi)

  contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  largest_contour = max(contours, key=cv2.contourArea)
  vx, vy, x0, y0 = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
  print(vx, vy, x0, y0)

  angRad = math.atan2(vy,vx)
  angDeg = angRad*180.0/math.pi

  line_length = 100
  line_start = (int(x0 - vx * line_length), int(y0 - vy * line_length))
  line_end = (int(x0 + vx * line_length), int(y0 + vy * line_length))

  # yes, feels kludgey, but it works
  if bFront:
    line_start = (int(x0 - (-vy) * line_length), int(y0 - vx * line_length))
    line_end = (int(x0 + (-vy) * line_length), int(y0 + vx * line_length))
  else:
    line_start = (int(x0 - vx * line_length), int(y0 - vy * line_length))
    line_end = (int(x0 + vx * line_length), int(y0 + vy * line_length))
    angDeg =  angDeg - 90

  imgRGB = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
  cv2.line(imgRGB, line_start, line_end, [0,128,255], thickness=2)
  # cv2.circle(imgRGB, (ix0,iy0), 10, [255,0,0], 2)
  exa = "Angle: {:.1f} deg".format(angDeg)
  plt.text(150, 148, exa, fontsize=12, color='red')

  return angDeg,imgRGB

if True:
  testN = 1

  if True:
    ang,img = doAngleThing(testN,True)
  else:
    ang,img = doAngleThing(testN,False)
  plt.imshow(img)
  plt.show()

  # in case we want without the x/y axis.
  # though need to put the text on the image, not the plot
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # cv2.imwrite("img.png",img)


#%%

