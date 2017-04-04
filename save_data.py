import os

import numpy as np

import pymesh

from operators_py import axis_angle_to_rotation_matrix_forward

"""
save mesh according to rotation and translation
"""
def save_mesh(file_name, V, F, rot, trans):

    R = axis_angle_to_rotation_matrix_forward(rot)
    vertices = np.dot(R, V[:, 0:3].transpose()) + trans.reshape(3, 1)
    vertices = vertices.transpose()

    mesh = pymesh.form_mesh(vertices, F)

    mesh.add_attribute("vertex_red")
    mesh.set_attribute("vertex_red", V[:,3] * 255 )

    mesh.add_attribute("vertex_green")
    mesh.set_attribute("vertex_green", V[:, 4] * 255 )

    mesh.add_attribute("vertex_blue")
    mesh.set_attribute("vertex_blue", V[:, 5] * 255 )

    """check if dir exists"""

    dir =  os.path.dirname( file_name )
    if(not os.path.exists( dir )):
        os.mkdir(dir)

    pymesh.save_mesh(file_name, mesh, *mesh.get_attribute_names(), use_float=True)