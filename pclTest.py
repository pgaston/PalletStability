#%%

from pathlib import Path
import numpy as np
# import cv2

import pclpy
# from pclpy import pcl
import matplotlib.pyplot as plt


#%%

# look at a numpy file
# then input to pcl

baseFP = Path('simData') 

baseFP = baseFP / Path('_out_sdrec_slightRight')

# npyFP = baseFP / Path('distance_to_camera_0000.npy')
npyFP = baseFP / Path('pointcloud_0000.npy')

npyFP.exists()

#%%

npyData = np.load(npyFP)

npyData.shape,npyData.dtype,npyData.size,npyData[0]

#%%
cloud = pclpy.pcl.PointCloud.PointXYZ()
p3 = cloud.from_array(npyData)
print(p3.size())


# Get a list of all the methods available for the object
methods = dir(p3)

# Print the list of methods
print(methods)

#%%
a = p3.xyz



#%%



#%%




#%%
xyz = p3.xyz

# Plot the point cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz[:,0], xyz[:,1], -xyz[:,2], s=0.1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%

a = npyData[:, 0].reshape((-1, 1))
b = npyData[:, 1].reshape((-1, 1))
c = npyData[:, 2].reshape((-1, 1))

pNpy = np.concatenate((a, b, c), axis=1)
a.shape,b.shape,c.shape,pNpy.shape
pNpy[0]




#%%

# Load a PCD file
cloud = pcl.PointCloud.PointXYZ()
p3 = cloud.from_array(pNpy)
print(p3.size())
p3.width,p3.height

#%%
xyz = p3.to_array()
xyz



xyz.shape
#%%


# Plot the point cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], s=0.1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

#%%


#%%

# Load the point cloud data
npyFP = Path('simData/_out_sdrec_slightRight/distance_to_camera_0000.npy')
npyData = np.load(npyFP)
cloud = pcl.PointCloud.PointXYZ()
cloud.from_array(npyData)

# Convert the point cloud to a numpy array
xyz = cloud.to_array()

#%%

import pclpy
from pclpy import pcl
cloud = pcl.PointCloud.PointXYZ()

# xyz = cloud.to_array()
cloud.width

#%%
