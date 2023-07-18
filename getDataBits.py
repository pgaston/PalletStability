#%%

'''
getDataBits.py
    NVidia Isaac Sim is used to create sample data for testing.
    It outputs a number of files, e.g., bounding_box_2d_tight_0000.npy,
    this script is used to read those files and extract the data needed.
    - 2D bounding box
    - 3D bounding box, plus pose transform and size of object

    Note that, for the 3D bounding box, we're setting the pallet of interest at 0,0,0.

    btw, I'm using the Synthetic Data Recorder function in a static mode to get this data.
'''

from pathlib import Path
import numpy as np
import json,math

import cv2
import open3d as o3d


clsName = "pallet"      # defined in Isaac Sim
dataFP = Path('simData') 
assert(dataFP.exists())

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

testFldr = tstFldrs[1]

#%%

def getFilePath(fldr,fileName):
    baseFP = dataFP / Path(fldr)
    assert(baseFP.exists())
    fPath = baseFP / Path(fileName)
    assert(fPath.exists())
    return fPath

def getNpyDepth(fldr):
    npyFP = getFilePath(fldr,'distance_to_camera_0000.npy')
    npyData = np.load(npyFP)
    assert(npyData.shape == (720, 1280))
    npyData = npyData.reshape((720, 1280))
    return npyData

def getPointCloudData(fldr):
    npyFP = getFilePath(fldr,'pointcloud_0000.npy')
    npyData = np.load(npyFP)
    assert(npyData.shape == (720*1280, 3))    
    return npyData

def getPointCloudRGB(fldr):
    npyFP = getFilePath(fldr,'pointcloud_rgb_0000.npy')
    npyData = np.load(npyFP)
    assert(npyData.shape == (720*1280, 4))  
    npyData = npyData[:,0:3]    # RGB only
    npyData = npyData.astype(np.float64)
    npyData = npyData / 255.0
    return npyData

def getCameraStuff(fldr):
    camFP = getFilePath(fldr,'camera_params_0000.json')
    with open(camFP) as f:
        jData = json.load(f)
        return jData

def getCameraIntrinsics(fldr):
    jData = getCameraStuff(fldr)
    res = jData['renderProductResolution']
    width = res[0]
    height = res[1]
    fl = jData['cameraFocalLength']
    camIntrin = o3d.camera.PinholeCameraIntrinsic(width,height,
                                                     fl,fl,
                                                     width/2,height/2)
    return camIntrin
 
def getImage(fldr):
    imgFP = getFilePath(fldr,'rgb_0000.png')
    img = o3d.io.read_image(str(imgFP))
    return img

def getNpyImage(fldr):
    imgFP = getFilePath(fldr,'rgb_0000.png')
    img = cv2.imread(str(imgFP))
    np_img = np.asarray(img)
    return np_img

def getDepthImage(fldr):
    npyDepth = getNpyDepth(fldr)
    depthImg = o3d.geometry.Image(npyDepth)
    return depthImg

## Testing
if False:
    import matplotlib.pyplot as plt
    npyData = getNpyDepth(testFldr)
    plt.imshow(npyData, cmap='gray')
    plt.show()

if False:
    npyData = getPointCloud(testFldr)
    print(npyData.shape)

if False:
    camData = getCameraStuff(testFldr)
    print(camData['cameraProjection'])

if False:
    camIntrin = getCameraIntrinsics(testFldr)
    print(camIntrin.intrinsic_matrix)

if False:
    img = getImage(testFldr)
    print(img)

if False:
    img = getDepthImage(testFldr)
    print(img)

if False:
    nptData = getPointCloudRGB(testFldr)
    print(nptData.shape)
    print(nptData[0:10,:])


#%%

# must do a linear search...
def getClassFromJson(jData):
    for key, value in jData.items():
        if value['class'] == clsName:
            return int(key)
    return None

def collectEm(jData,clsNum):
    ret = []
    for entry in jData:
        if entry[0] == clsNum:
            ret.append(entry)
    return ret

def getClassNumbers(fldr):
    # 2D bounding box
    class2DFP = getFilePath(fldr,'bounding_box_2d_tight_labels_0000.json')
    # print("class2DFP",class2DFP)
    with open(class2DFP) as f:
        jData = json.load(f)
        twoDClassNum = getClassFromJson(jData)

    npy2DataFP = getFilePath(fldr,'bounding_box_2d_tight_0000.npy')
    npyData = np.load(npy2DataFP)
    twoDbboxs = collectEm(npyData, twoDClassNum)

    # 3D bounding box
    class3DFP = getFilePath(fldr,'bounding_box_3d_labels_0000.json')
    with open(class3DFP) as f:
        jData = json.load(f)
        threeDClassNum = getClassFromJson(jData)

    npy3DataFP = getFilePath(fldr,'bounding_box_3d_0000.npy')
    npyData = np.load(npy3DataFP)
    threeDbboxs = collectEm(npyData, threeDClassNum)

    return twoDbboxs,threeDbboxs
  
if False:
    twoDbboxs,threeDbboxs = getClassNumbers(testFldr)
    print("twoDbboxs", len(twoDbboxs))
    print("threeDbboxs", len(threeDbboxs))


#%%

def get2DPalletOrigin(npPallets):
    for npPallet in npPallets:
        semId = npPallet[0]
        x1 = npPallet[1]
        y1 = npPallet[2]
        x2 = npPallet[3]
        y2 = npPallet[4]
        frac = npPallet[5]

        mid = (x1 + x2) / 2.0
        wid = x2 - x1
        hgt = y2 - y1
        area = wid * hgt

        if math.isclose(mid,1280/2,abs_tol=0.001):
            return (x1,y1,x2,y2,mid,wid,hgt,frac,area)
        
    return None

if False:
    twoDbboxs,threeDbboxs = getClassNumbers(testFldr)
    pallet2DLocation = get2DPalletOrigin(twoDbboxs)
    print("pallet2DLocation",pallet2DLocation)

'''
pallet2DLocation (411, 450, 869, 590, 640.0, 458, 140, 0.3311, 64120)
'''

#%%

# Of all the pallets in the scene - one should be at 'origin'
def get3DPalletAtOrigin(npPallets):
    def show4DMat(mat):
        for i in range(0,mat.shape[0]):
            print("%.2f, %.2f, %.2f, %.2f" % (mat[i,0],mat[i,1],mat[i,2],mat[i,3]))
    def palletAtOrigin(mat):
        return math.isclose(mat[3,0],0.0,abs_tol=0.001) and math.isclose(mat[3,1],0.0,abs_tol=0.001) and math.isclose(mat[3,2],0.0,abs_tol=0.001)

    for npPallet in npPallets:
        # 3D information - npPallet was found earlier
        semId = npPallet[0]     # class number
        x_min = npPallet[1]     # extent of pallet
        y_min = npPallet[2]
        z_min = npPallet[3]
        x_max = npPallet[4]
        y_max = npPallet[5]
        z_max = npPallet[6]
        lDepth = x_max - x_min
        palWidth = y_max - y_min
        palHeight = z_max - z_min

        pal3Dbbox = [x_min,y_min,z_min,x_max,y_max,z_max]
        threeDXForm = npPallet[7]   # 4x4 transformation/rotation matrix

        if palletAtOrigin(threeDXForm):
            # mat = threeDXForm
            # i = 3
            # print("%.2f, %.2f, %.8f, %.2f" % (mat[i,0],mat[i,1],mat[i,2],mat[i,3]))
            # print("3D information:")
            # print(f"pallet w/d/h ({palWidth:0.3f} {palDepth:0.3f} {palHeight:0.3f})")
            # print(f"pallet min ({x_min:0.3f} {y_min:0.3f} {z_min:0.3f})")
            # print(f"pallet max ({x_max:0.3f} {y_max:0.3f} {z_max:0.3f})")
            # print("3D pose transform:")
            # show4DMat(threeDXForm)
            return (pal3Dbbox,lDepth,palWidth,palHeight,threeDXForm)

    return None

if False:
    twoDbboxs,threeDbboxs = getClassNumbers(testFldr)
    pallet3DLocation = get3DPalletAtOrigin(threeDbboxs)
    print("pallet3DLocation",pallet3DLocation)

'''
pallet3DLocation ([-60.507656, -50.0, 2.2888184e-05, 60.815845, 50.28731, 21.111975], 121.3235, 100.28731, 21.111952, array([[ 2.220446e-18, -1.000000e-02,  0.000000e+00,  0.000000e+00],
       [ 1.000000e-02,  2.220446e-18,  0.000000e+00,  0.000000e+00],
       [ 0.000000e+00,  0.000000e+00,  1.000000e-02,  0.000000e+00],
       [ 0.000000e+00,  0.000000e+00,  3.337860e-08,  1.000000e+00]],
      dtype=float32))
'''

#%%
