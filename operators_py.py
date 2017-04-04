"""
useful operators implemented in python
"""

import numpy as np

"""
convert from axis-angle representation to rotation matrix
http://www.ethaneade.com/latex2html/lie_groups/node37.html

w in so3 to rotation matrix:

R = exp(w_x) = I + (sin(\theta) / \theta)w_x + ((1-cos(\theta))\theta^2) w_x^2

derivative dR_dwi:

      --- w_i [w]x + [w x (I - R)e_i]x
	  ----------------------------------- R
      --- 	 ||w||^{2}


"""

def axis_angle_to_rotation_matrix_forward(w):

    theta = np.linalg.norm(w)

    if(theta > 0):

        wx = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        R = np.eye(3) + np.sin(theta) / theta * wx + (( 1 - np.cos(theta) ) / theta ** 2 ) * wx.dot(wx)

    else:

        R = np.array([[1, -w[2], w[1]], [w[2], 1, -w[0]], [-w[1], w[0], 1]])

    return R

def axis_angle_to_rotation_matrix_backward(w, R = []):
    """
    :param w:
    :return: dR_dw(9x3)
    """

    theta = np.linalg.norm(w)

    if(R == []):
        R = axis_angle_to_rotation_matrix_forward(w)

    dR_dw = np.zeros((9,3))

    if (theta > 0):

        wx = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

        for i in range(3):

            ei = np.zeros((3,1))
            ei[i] = 1

            temp = wx.dot(np.eye(3) - R).dot(ei)

            dR_dwi = (w[i] * wx + np.array([[0, -temp[2][0], temp[1][0]],
                                            [temp[2][0], 0, -temp[0][0]],
                                            [-temp[1][0], temp[0][0], 0]])).dot(R) / theta ** 2

            dR_dw[:, i] = dR_dwi.reshape(-1)

    else:

        dR_dw = np.array([[0, 0, 0],
                          [0, 0,-1],
                          [0, 1, 0],
                          [0, 0, 1],
                          [0, 0, 0],
                          [-1,0, 0],
                          [0, -1,0],
                          [1, 0, 0],
                          [0, 0, 0]])

    return dR_dw

def perspective_projection_backward(X, K):
    """
    :param X: 3 X P
    :param K:  3 x 3
    :return: P x 2 x 3
    """

    """
    dp_dkx = [1/z, 0, -x/z^2;
             0, 1/z, -y/z^2]
    """

    P = X.shape[1]
    KX = K.dot(X)

    dp_dkx = np.zeros((P, 2, 3))

    dp_dkx[:, 0, 0] = 1 / KX[2]
    dp_dkx[:, 0, 2] = - KX[0] / KX[2] ** 2
    dp_dkx[:, 1, 1] = 1 / KX[2]
    dp_dkx[:, 1, 2] = - KX[1] / KX[2] ** 2

    """
    dkx_dx = K
    """
    dkx_dx = K

    dp_dx = dp_dkx.dot(dkx_dx)

    return dp_dx

def rigid_transformation_backward(vertices, rot):
    """
    :param vertices: 3 X P
    :param rot:  3 x 1
    :param trans:  3 x 1
    :return: dx_dw P x 3 x 3
    :return: dx_dt P x 3 x 3
    """

    """
    dx_dR = [x,y,z,0,0,0,0,0,0;
             0,0,0,x,y,z,0,0,0;
             0,0,0,0,0,0,x,y,z]
    """

    P = vertices.shape[1]
    dx_dR = np.zeros((P, 3, 9))

    dx_dR[:, 0, 0:3] = vertices.transpose()
    dx_dR[:, 1, 3:6] = vertices.transpose()
    dx_dR[:, 2, 6:9] = vertices.transpose()

    """
    dR_dw(9x3)
    """
    dR_dw = axis_angle_to_rotation_matrix_backward(rot)
    dx_dw = dx_dR.dot(dR_dw)

    """
    dx_dt is just identity
    """
    dx_dt = np.repeat( np.eye(3).reshape((1,3,3)), P, axis=0 )

    return dx_dw, dx_dt

"""
sampling values from H * W * C images based on 2d projections

img: W * H * C
proj: 2 * P
values: P * C
dv_dp: P * C * 2

img is provided as parameters, proj as input and sampled values as output

"""

def sample_from_images_forward(proj, img):

    H, W, C = img.shape
    P = proj.shape[1]

    values = np.zeros((P, C))
    numVis = 0

    for i in range(P):

        if(proj[0][i] > 0 and proj[0][i] < W - 1 and
           proj[1][i] > 0 and proj[1][i] < H - 1):

            numVis += 1

            ul = np.int(proj[0][i])
            vl = np.int(proj[1][i])

            delta_u = proj[0][i] - ul
            delta_v = proj[1][i] - vl

            uu = ul + 1
            vu = vl + 1

            values[i] = (1-delta_u)*(1-delta_v)*img[vl, ul] + \
                        (1-delta_u)*delta_v*img[vu, ul] + \
                        delta_u*(1-delta_v)*img[vl, uu] + \
                        delta_u*delta_v*img[vu, uu]

    # print "number of visible points {:g}".format(numVis)

    return values

def sample_from_images_backward(proj, img):

    H, W, C = img.shape
    P = proj.shape[1]

    dv_dp = np.zeros((P, C, 2))

    for i in range(P):

        if(proj[0][i] > 0 and proj[0][i] < W - 1 and
           proj[1][i] > 0 and proj[1][i] < H - 1):

            ul = np.int(proj[0][i])
            vl = np.int(proj[1][i])

            delta_u = proj[0][i] - ul
            delta_v = proj[1][i] - vl

            uu = ul + 1
            vu = vl + 1

            dv_dp[i][:, 0] = -(1 - delta_v)*img[vl, ul] - delta_v*img[vu, ul] + \
                             (1- delta_v)*img[vl, uu] + delta_v*img[vu, uu]

            dv_dp[i][:, 1] = -(1 - delta_u)*img[vl, ul] + (1-delta_u)*img[vu, ul] - \
                             delta_u*img[vl, uu] + delta_u*img[vu, uu]

    return dv_dp

def sample_from_images_faster_forward(proj, img):

    H, W, C  = img.shape
    P = proj.shape[1]

    values = np.zeros((P, C))

    mask = np.logical_and( proj[0] > 0, proj[1] > 0)
    mask2 = np.logical_and( proj[0] < W - 1, proj[1] < H - 1)
    mask = np.logical_and(mask, mask2)

    ul = proj[0][mask].astype(np.int)
    vl = proj[1][mask].astype(np.int)

    delta_u = (proj[0][mask] - ul).reshape((-1,1))
    delta_v = (proj[1][mask] - vl).reshape((-1,1))

    uu = ul + 1
    vu = vl + 1

    values[mask] = (1-delta_u)*(1-delta_v)*img[vl, ul] + \
                   (1-delta_u)*delta_v*img[vu, ul] + \
                   delta_u*(1-delta_v)*img[vl, uu] + \
                   delta_u*delta_v*img[vu, uu]

    # print "number of visible points {:g}".format( np.sum(mask) )

    return values

def sample_from_images_faster_backward(proj, img):

    H, W, C = img.shape
    P = proj.shape[1]

    dv_dp = np.zeros((P, C, 2))

    mask = np.logical_and( proj[0] > 0, proj[1] > 0)
    mask2 = np.logical_and( proj[0] < W - 1, proj[1] < H - 1)
    mask = np.logical_and(mask, mask2)

    ul = proj[0][mask].astype(np.int)
    vl = proj[1][mask].astype(np.int)

    delta_u = (proj[0][mask] - ul).reshape((-1,1))
    delta_v = (proj[1][mask] - vl).reshape((-1,1))

    uu = ul + 1
    vu = vl + 1

    dv_dp[:,:,0][mask] = -(1 - delta_v)*img[vl, ul] - delta_v*img[vu, ul] + \
                         (1- delta_v)*img[vl, uu] + delta_v*img[vu, uu]

    dv_dp[:,:,1][mask] = -(1 - delta_u)*img[vl, ul] + (1-delta_u)*img[vu, ul] - \
                         delta_u*img[vl, uu] + delta_u*img[vu, uu]

    return dv_dp

def rigid_tracking_residual_py(rot, trans, vertices, colors, K, **kwargs):

    return_value = kwargs.get('return_value', True)
    img = kwargs.get('image')

    def rigid_tracking_func(x):

        rot = x[0:3]
        trans = x[3:6]

        R = axis_angle_to_rotation_matrix_forward(rot)
        X = R.dot(vertices) + trans.reshape((3,1))
        KX = K.dot(X)

        proj = KX[0:2] / KX[2]
        # values = sample_from_images_forward(proj, img)
        values = sample_from_images_faster_forward(proj, img)

        residuals = values - colors

        return residuals.reshape(-1)

    def rigid_tracking_jac(x):

        rot = x[0:3]
        trans = x[3:6]

        R = axis_angle_to_rotation_matrix_forward(rot)
        X = R.dot(vertices) + trans.reshape((3,1))
        KX = K.dot(X)

        proj = KX[0:2] / KX[2]

        # dv_dp = sample_from_images_backward(proj, img)
        dv_dp = sample_from_images_faster_backward(proj, img)

        dp_dx = perspective_projection_backward(X, K)
        dx_dw, dx_dt = rigid_transformation_backward(vertices, rot)

        P, C = colors.shape
        dr_dw = np.zeros((P, C, 3))
        dr_dt = np.zeros((P, C, 3))

        for p in range(P):
            dr_dw[p,:,:] = dv_dp[p,:,:].dot(dp_dx[p,:,:]).dot(dx_dw[p,:,:])
            dr_dt[p,:,:] = dv_dp[p,:,:].dot(dp_dx[p,:,:]).dot(dx_dt[p,:,:])

        jac = np.hstack( (dr_dw.reshape((-1,3)), dr_dt.reshape((-1,3))) )

        return jac

    if(return_value):

        R = axis_angle_to_rotation_matrix_forward(rot)
        X = R.dot(vertices) + trans.reshape((3,1))
        KX = K.dot(X)

        proj = KX[0:2] / KX[2]
        values = sample_from_images_faster_forward(proj, img)
        residuals = values - colors
        residuals = residuals.reshape((-1,1))

        dv_dp = sample_from_images_faster_backward(proj, img)
        dp_dx = perspective_projection_backward(X, K)
        dx_dw, dx_dt = rigid_transformation_backward(vertices, rot)

        P, C = colors.shape
        dr_dw = np.zeros((P, C, 3))
        dr_dt = np.zeros((P, C, 3))

        for p in range(P):
            dr_dw[p,:,:] = dv_dp[p,:,:].dot(dp_dx[p,:,:]).dot(dx_dw[p,:,:])
            dr_dt[p,:,:] = dv_dp[p,:,:].dot(dp_dx[p,:,:]).dot(dx_dt[p,:,:])

        jac = np.hstack( (dr_dw.reshape((-1,3)), dr_dt.reshape((-1,3))) )

        return residuals, jac

    else:

        return rigid_tracking_func, rigid_tracking_jac
