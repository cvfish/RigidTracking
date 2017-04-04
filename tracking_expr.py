"""
do rigid photometric tracking use depth image or template mesh
"""

import argparse

from read_data import *
from rigid_tracking import RigidTracking

parser = argparse.ArgumentParser(description='6DoF rigid photometric tracking in python')

parser.add_argument("--start_frame", type=int, default=1, metavar='N', help='start frame for tracking (default: 1)')
parser.add_argument("--end_frame", type=int, default=10, metavar='N', help='end frame for tracking (default: 10)')

parser.add_argument("--image_path", default='./rgb/', help='rgb image folder (default: "./rgb")')
parser.add_argument("--depth_file", default='./depth/depth.png', help='depth image (default: "./depth/depth.png" ) ')
parser.add_argument("--mesh_file", default='./mesh/mesh.ply', help='mesh template (default: "./mesh/mesh.ply") ')

parser.add_argument("--save_folder", default='./results/', help='folder for saving results (default: "./results/")')
parser.add_argument("--frame_offset", type=int, default=0, metavar='N', help='frame offset value (default: 0)')

parser.add_argument("--use_depth_image", default=True, type=bool, help='use depth image by default')

parser.add_argument("--intrinsics_file", default='./intrinsics.txt', help='txt file with intrinsic camera parameters')

args = parser.parse_args()

start_frame = args.start_frame
end_frame = args.end_frame

image_path = args.image_path
depth_file = args.depth_file
mesh_file = args.mesh_file

save_folder = args.save_folder
frame_offset = args.frame_offset

use_depth_image = args.use_depth_image
intrinsics_file = args.intrinsics_file

if(use_depth_image):
    data = read_data(image_path, depth_file, intrinsics_file, start_frame, end_frame, format='*.png', use_mesh = 0)
else:
    data = read_data(image_path, mesh_file, intrinsics_file, start_frame, end_frame, format='frame*.png', use_mesh = 1)

rt = RigidTracking(data, {'frame_offset': frame_offset,
                          'save_folder': save_folder,
                          'iters': 20,
                          'do_rigid_tracking': True,
                          'image_downsampling_list': [4],
                          'blur_sizes_list': [0],
                          'data_term': 'color'
                          })

rt.track_sequence()