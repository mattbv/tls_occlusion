# -*- coding: utf-8 -*-
"""
Module to manage the command line execution of the package.

@author: Matheus Boni Vicari (2017)
"""

import matplotlib.pyplot as plt
import argparse
import glob
import os
import numpy as np
from parser.hips import read_hips
from parser.material_map import generate_map
from parser.obj_parser import parse_obj_matmap as p_matmap
from parser.obj_parser import parse_obj_clone as p_clone
from point_in_facets import pfmatch
from utils.point_cloud import correct


def matmap2facets():

    filename = os.path.basename(p.img_file).split('.')[:-1]

    distmap, tri = generate_map(p.img_file, datum=p.datum,
                                grid_size=p.grid_size)

    np.save(('%s%s_distmap_%s_.npy' % (result_folder, filename, p.grid_size)),
            distmap)
    np.save(('%s%s_triangulation_%s_.npy' % (result_folder, filename,
                                             p.grid_size)), tri)


def convert_hips():

    """
    fname = hips file name

    """

    filename = os.path.basename(p.hips_file).split('.')[:-1]

    img, res_x, res_y, fmt = read_hips(p.hips_file)
    plt.imshow(img)
    plt.imsave('%s%s.png' % (result_folder, filename))
    plt.close()


def parse_obj_matmap():

    filename = os.path.basename(p.obj_file).split('.')[:-1]

    obj_dataframe, obj_tri = p_matmap(p.obj_file, p.matmap_file)
    obj_dataframe.to_csv(('%s%s_obj_facets_.csv' % (result_folder, filename)))
    np.save('%s%s_triangulation.npy' % (result_folder, filename), obj_tri)


def parse_obj_clone():

    filename = os.path.basename(p.obj_file).split('.')[:-1].split('_wood')[0]

    obj_dataframe, obj_tri = p_clone(p.wood_obj_file, p.foliage_obj_file,
                                     p.leaf_obj_file)
    obj_dataframe.to_csv(('%s%s_obj_facets_.csv' % (result_folder, filename)))
    np.save('%s%s_triangulation.npy' % (result_folder, filename), obj_tri)


def pc_correction():

    filename = os.path.basename(p.point_cloud).split('.')[:-1]

    corrected_pc = correct(p.point_cloud, axis_order=p.corr_axis,
                           dist_from_center=p.corr_dist)

    np.savetxt('%s%s.txt' % (result_folder, filename), corrected_pc,
               fmt='%1.3f')


def batch_pc_correction():

    pc_folder = os.path.join(p.point_cloud_folder, '')
    files = glob.glob(pc_folder + '*.txt')

    for f in files:
        arr = np.loadtxt(f)
        filename = os.path.basename(p.obj_file).split('.')[:-1]
        corrected_pc = correct(arr, axis_order=p.corr_axis,
                               dist_from_center=p.corr_dist)
        np.savetxt('%s%s.txt' % (result_folder, filename), corrected_pc,
                   fmt='%1.3f')


def match_pf():

    fname_obj = os.path.basename(p.facets).split('.')[:-1].split('_wood')[0]
    fname_pc = os.path.basename(p.point_cloud).split('.')[:-1]

    facets, pc_df = pfmatch(p.facets, p.point_cloud)

    facets.to_csv(('%s%s-%s_macthed_facets_.csv' % (result_folder,
                                                    fname_obj, fname_pc)))
    pc_df.to_csv(('%s%s-%s_macthed_points_.csv' % (result_folder,
                                                   fname_pc, fname_obj)))


def batch_match_pf():

    pc_folder = os.path.join(p.point_cloud_folder, '')
    files = glob.glob(pc_folder + '*.txt')

    fname_obj = os.path.basename(p.facets).split('.')[:-1]

    for f in files:
        fname_pc = os.path.basename(f).split('.')[:-1]
        arr = np.loadtxt(f)
        facets, pc_df = pfmatch(p.facets, arr)

        facets.to_csv(('%s%s-%s_macthed_facets_.csv' % (result_folder,
                                                        fname_obj, fname_pc)))
        pc_df.to_csv(('%s%s-%s_macthed_points_.csv' % (result_folder,
                                                       fname_pc, fname_obj)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--obj", dest="obj_file", help="single object\
 file.", metavar="FILE")
    parser.add_argument("--obj_folder", dest="obj_folder",
                        help="multiple obj  clouds folder path.",
                        metavar="FILE")
    parser.add_argument("-p", "--point_cloud", dest="point_cloud",
                        help="single point cloud file path.", metavar="FILE")
    parser.add_argument("--pc_folder", dest="point_cloud_folder",
                        help="multiple point clouds folder path.",
                        metavar="FILE")
    parser.add_argument("-m", "--matmap", dest="matmap_file",
                        help="material mapping file.", metavar="FILE")
    parser.add_argument("--hips", dest="hips_file", help="hips image\
 file.", metavar="FILE")
    parser.add_argument("-i", "--image", dest="img_file", help="converted\
 image file.", metavar="FILE")
    parser.add_argument("--grid_size", dest="grid_size", type=int, default=10,
                        help="grid size for the material image downsampling")
    parser.add_argument("--correction_dist", dest="corr_dist", type=float,
                        help="distance correction to match the point cloud with\
 the object.")
    parser.add_argument("--correction_axis", dest="corr_axis",
                        help="axis order correction to match the point cloud\
 with the object.")
    parser.add_argument("--datum", dest="datum", type=int, nargs='+',
                        default=[512, 0],
                        help="datum of material map image.")
    parser.add_argument("-r", "--result_folder", dest="result_folder",
                        default='',
                        help="folder path where to save processing results.",
                        metavar="FILE")
    parser.add_argument("--wood_obj", dest="wood_obj_file", help="single wood\
 object file.", metavar="FILE")
    parser.add_argument("--foliage_obj", dest="foliage_obj_file",
                        help="single foliage object file.", metavar="FILE")
    parser.add_argument("--leaf_obj", dest="leaf_obj_file", help="single leaf\
 object file.", metavar="FILE")
    parser.add_argument("--facets_file", dest="facets", help="processed obj\
 file converted to facets (.csv).", metavar="FILE")

    p = parser.parse_args()

    result_folder = os.path.join(p.result_folder, '')
