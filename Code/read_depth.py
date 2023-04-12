import struct
import numpy as np
import open3d as o3d
def loadDepthImageCompressed(fname):

    # now read the depth image
    pFile = open(fname, "rb")
    if not pFile:
        print("could not open file ", fname)
        return None

    im_width = 0
    im_height = 0
    success = True

    # read width of depthmap
    im_width = struct.unpack('i', pFile.read(4))[0]
    # read height of depthmap
    im_height = struct.unpack('i', pFile.read(4))[0]
    # depth_img = [0] * (im_width * im_height)
    depth_img = np.zeros((im_height * im_width,),np.short)

    p = 0

    while p < im_width * im_height:
        numempty = struct.unpack('i', pFile.read(4))[0]

        for i in range(numempty):
            depth_img[p + i] = 0
        numfull = struct.unpack('i', pFile.read(4))[0]
        temp = struct.unpack('h' * numfull, pFile.read(2 * numfull))
        depth_img[p + numempty : p + numempty + numfull] = temp
        p += numempty + numfull

    pFile.close()

    if success:
        return np.resize(depth_img, (im_height, im_width))
    else:
        return None
    