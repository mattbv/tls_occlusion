# -*- coding: utf-8 -*-
"""
Module to manage the command line execution of the package.

@author: Matheus Boni Vicari (2017)
"""

from main import run_matching, leaf_matmap, process_obj, process_map
import argparse
import glob
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--function", dest="function",
                        help="function to run")
    parser.add_argument("-b", dest="b", help="run batch of match processing")
    parser.add_argument("-o", "--obj", dest="obj_file", help="object file",
                        metavar="FILE")
    parser.add_argument("-p", "--path", dest="pc_path", help="point cloud\
 path", metavar="FILE")
    parser.add_argument("-p", "--point_cloud", dest="pc_file",
                        help="point cloud file", metavar="FILE")
    parser.add_argument("-m", "--matmap", dest="matmap_file",
                        help="material mapping file", metavar="FILE")
    parser.add_argument("-i", "--image", dest="img_file", help="object file",
                        metavar="FILE")
    parser.add_argument("--grid_size", dest="grid_size",
                        help="grid size for the material image downsampling")
    parser.add_argument("--pc_correction", dest="pc_correction",
                        help="distance correction to match the point cloud with\
 the object")

    p = parser.parse_args()

    function = p.function

    if function == 'process_map':
        process_map(p.img_file, p.grid_size)

    elif function == 'process_obj':
        process_obj(p.obj_file)

    elif function == 'leaf_matmap':
        leaf_matmap(p.obj_file, p.matmap_file)

    elif function == 'run_matching':

        if p.b:
            files = glob.glob(p.pc_path + os.sep + '*.csv')

            for f in files:
                run_matching(p.obj_file, f, p.pc_correction)

        else:
            run_matching(p.obj_file, p.pc_file, p.pc_correction)
