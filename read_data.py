"""
read data
"""

import os
import glob as glob

import numpy as np

import cv2
import pymesh

#from save_data import save_mesh

def depth_img_to_mesh(rgb_img, depth_img, mask, K, scale = 1.0, downsample = 1):

    H, W, C = rgb_img.shape

    inds = -1 * np.ones((H, W)).astype(np.int)

    downsample_mask = np.zeros((H, W)).astype(np.bool)
    downsample_mask[0::downsample, 0::downsample] = True

    mask = np.logical_and(mask, downsample_mask)

    numPnts = np.sum(mask)
    inds.reshape(-1)[ mask.reshape(-1) ] = range(numPnts)

    """ faster version """
    grid_v, grid_u = np.mgrid[0:H, 0:W]
    grid_v_sample = grid_v.reshape((1,-1))
    grid_v_sample = grid_v_sample[mask.reshape((1,-1))]

    grid_u_sample = grid_u.reshape((1,-1))
    grid_u_sample = grid_u_sample[mask.reshape((1,-1))]

    vertices = np.dot(
        np.linalg.inv(K),
        scale * depth_img[mask].reshape((1,-1)) *
        np.vstack((grid_u_sample,
                   grid_v_sample,
                   np.ones((1, numPnts))))
    ).astype(np.float32).T

    colors = rgb_img[grid_v_sample, grid_u_sample, :]

    mask_ = mask.reshape((-1,1))
    inds_ = inds.reshape((-1,1))

    face_mask =  np.logical_and( np.logical_and( mask_[0:-W, :], mask_[W:, :] ),
                                 mask_[1:-W+1, :] ).reshape(-1)

    faces = np.hstack(( inds_[0:-W, :],
                        inds_[W:, :],
                        inds_[1:-W+1, :] ))

    faces = faces[face_mask]

    face_mask2 = np.logical_and( np.logical_and( mask_[0:-W, :], mask_[W-1:-1,:] ),
                                mask_[W:, :] ).reshape(-1)

    faces2 = np.hstack(( inds_[0:-W, :],
                         inds_[W-1:-1, :],
                         inds_[W:, :] ))
    faces2 = faces2[face_mask2]

    faces = np.vstack((faces, faces2))

    edges = np.hstack((np.vstack((faces[:,0],faces[:,1])),
                       np.vstack((faces[:,0],faces[:,2])),
                       np.vstack((faces[:,1],faces[:,2]))))

    E = edges
    F = faces
    V = np.hstack((vertices, colors))

    return (V, F, E)

def read_data(image_path, model_file, intrinsics_file, start_frame, end_frame,
              format = 'rgb*.png', use_mesh = 1, depth_min=1000, depth_max = 6000, scale=0.001):

    images = sorted( glob.glob(os.path.join(image_path, format)) )[0: end_frame - start_frame + 1]
    intrinsics = np.loadtxt( intrinsics_file )

    if(use_mesh):

        mesh = pymesh.load_mesh(model_file)

        V = mesh.vertices
        F = mesh.faces

        r = mesh.get_attribute('vertex_red') / 255.0
        g = mesh.get_attribute('vertex_green') / 255.0
        b = mesh.get_attribute('vertex_blue') / 255.0

        V = np.hstack((V, r.reshape(-1,1), g.reshape(-1,1), b.reshape(-1,1)))

        E = np.hstack((np.vstack((F[:,0],F[:,1])), np.vstack((F[:,0],F[:,2])), np.vstack((F[:,1],F[:,2]))))

        mesh = (V, F, E)

    else:
        """ use depth image for tracking """

        rgb_img = cv2.imread(images[0], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        """ change from BGR to RGB """
        rgb_img = rgb_img[:, :, ::-1] / 255.0

        depth_img = cv2.imread(model_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        mask = np.logical_and( depth_img > depth_min, depth_img < depth_max )

        mesh = depth_img_to_mesh(rgb_img, depth_img, mask, intrinsics, scale=scale)

        # """ save the constructed mesh """
        # save_mesh('./test.ply', mesh[0], mesh[1], np.zeros((3,1)), np.zeros((3,1)))

    return {'images':images, 'mesh': mesh, 'K': intrinsics}