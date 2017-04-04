"""
rigid photometric tracking
input: data(images, mesh, K)

"""

import cv2

import numpy as np

from scipy.optimize import least_squares
from operators_py import rigid_tracking_residual_py

from save_data import save_mesh

class RigidTracking(object):

    def __init__(self, data, params = {}):

        """
        initialize our rigid tracking object, copy over images, mesh, K
        """

        self.images = data['images']
        self.mesh = data['mesh']
        self.K = data['K']

        self.initialized = True

        """
        camera pose, first three for axis-angle rotation, last three for translation
        """
        self.cur_frame = 0
        self.cur_rot = np.zeros(3)
        self.cur_trans = np.zeros(3)

        """
        parameters
        """
        self.params = params

        self.do_rigid_tracking = self.params.get('do_rigid_tracking', True)
        self.save_folder = self.params.get('save_folder', './results')
        self.frame_offset = self.params.get('frame_offset', 0)
        self.iters = self.params.get('iters', 20)

        self.image_downsampling_list = self.params.get('image_downsampling_list', [1])
        self.blur_sizes_list = self.params.get('blur_sizes_list', [0])

        self.data_term = self.params.get('data_term', 'gray')

        """ prepare the reference mesh if we do non-rigid tracking """
        if(not self.do_rigid_tracking):
            self.ref_V = self.mesh[0].copy()
            self.ref_E = self.mesh[2].copy()

    def track_sequence(self):

        while(self.cur_frame < len(self.images)):
            self.track_one_frame()
            self.cur_frame += 1

        print "tracking done"

    def track_one_frame(self):

        """
        load new image
        do tracking
        save mesh
        """

        cur_img = cv2.imread(self.images[ self.cur_frame ],
                           cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        """ change from BGR to RGB """
        cur_img = cur_img[:, :, ::-1]
        cur_img = cur_img / 255.0

        self.track_frame_pyramid(cur_img, self.cur_rot, self.cur_trans)

        file_name = self.save_folder + 'mesh{:04d}.ply'.format( self.cur_frame + self.frame_offset + 1 )
        save_mesh(file_name, self.mesh[0], self.mesh[1], self.cur_rot, self.cur_trans)

    """ tracking with a pyramid """
    def track_frame_pyramid(self, img, rot, trans):

        vertices = self.mesh[0][:, 0:3].transpose().reshape((3, -1))
        colors = self.mesh[0][:, 3::].reshape((-1, 3))

        grays = colors[:, 0] * 0.299 + colors[:, 1] * 0.587 + colors[:, 2] * 0.114
        img_gray = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114

        grays = grays.reshape((-1,1))
        img_gray = img_gray.reshape(( img_gray.shape[0], img_gray.shape[1], 1 ))

        num_levels = len(self.image_downsampling_list)

        for i in range(num_levels):

            down_factor = self.image_downsampling_list[i]
            K = np.vstack((self.K[0] / down_factor, self.K[1] / down_factor, self.K[2]))

            ksize = self.blur_sizes_list[i]

            if (ksize > 0):
                img_i = cv2.GaussianBlur(img, (ksize, ksize))
                img_gray_i = cv2.GaussianBlur(img_gray, (ksize, ksize))
            else:
                img_i = img
                img_gray_i = img_gray

            img_i = cv2.resize(img_i, (0,0), fx=1.0/down_factor, fy=1.0/down_factor)
            img_gray_i = cv2.resize(img_gray_i, (0,0), fx=1.0/down_factor, fy=1.0/down_factor)

            if(self.data_term == 'gray'):
                self.rigid_tracking(rot, trans, vertices, grays, K, img_gray_i)
            elif(self.data_term == 'color'):
                self.rigid_tracking(rot, trans, vertices, colors, K, img_i)
            else:
                raise Exception("data term " +  self.data_term + " is not supported")

    def rigid_tracking(self, rot, trans, vertices, colors, K, img):

        func, jac = rigid_tracking_residual_py( rot, trans, vertices, colors, K,
                                                image=img, return_value=False )

        x0 = np.concatenate((rot.reshape(-1), trans.reshape(-1)))

        solution = least_squares(func, x0, jac=jac, method='lm')

        self.cur_rot = solution.x[0:3]
        self.cur_trans = solution.x[3:6]

        print self.cur_rot, self.cur_trans
