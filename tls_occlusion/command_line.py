# -*- coding: utf-8 -*-
"""
Module to manage the command line execution of the package.

@author: Matheus Boni Vicari (2017)
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
from parser.hips import read_hips
from parser.material_map import generate_map
from parser.obj_parser import parse_obj_matmap as p_matmap
from parser.obj_parser import parse_obj_clone as p_clone
from point_in_facets import pfmatch
from utils.point_cloud import correct


def matmap2facets():
    
    p = parse_args()
    result_folder = os.path.join(p.result_folder, '')
    filename = os.path.basename(p.hips_file).split('.')[:-1][0]
 
    hips, res_x, res_y, fmt = read_hips(p.hips_file)
 
    img_u = np.triu(hips)
    img_l = np.tril(hips)
 
    d1 = [0, hips.shape[1]]
    d2 = [hips.shape[0], 0]
 
    map_l, tri_l = generate_map(img_l, datum=d1, grid_size=p.grid_size)
    map_u, tri_u = generate_map(img_u, datum=d2, grid_size=p.grid_size)
 
    area_ratio = (((np.sum(img_l) + np.sum(img_u))/255).astype(float) /
                  (img_l.shape[0] * img_l.shape[1]))
 
    matmap = {'distmap': {'lower': map_l, 'upper': map_u},
              'triangles': {'lower': tri_l, 'upper': tri_u},
              'area_ratio': area_ratio}
 
    np.save(('%s%s_matmap_%s_.npy' % (result_folder, filename, p.grid_size)),
            matmap)
    
    
#def convert_hips():
#
#    """
#    fname = hips file name
#
#    """
#    
#    p = parse_args()
#    result_folder = os.path.join(p.result_folder, '')
#
#    filename = os.path.basename(p.hips_file).split('.')[:-1][0]
#
#    img, res_x, res_y, fmt = read_hips(p.hips_file)
#    plt.imshow(img)
#    plt.imsave('%s%s.png' % (result_folder, filename), img)
#    plt.close()


def parse_obj_matmap():
    
    p = parse_args()
    result_folder = os.path.join(p.result_folder, '')

    filename = os.path.basename(p.obj_file).split('.')[:-1][0]

    obj_dataframe, obj_tri = p_matmap(p.obj_file, p.matmap_file)
    obj_dataframe.to_csv(('%s%s_obj_facets_.csv' % (result_folder, filename)))
    np.save('%s%s_triangulation.npy' % (result_folder, filename), obj_tri)


def parse_obj_clone():
    
    p = parse_args()
    result_folder = os.path.join(p.result_folder, '')

    filename = os.path.basename(p.obj_file).split('.')[:-1].split('_wood')[0]

    obj_dataframe, obj_tri = p_clone(p.wood_obj_file, p.foliage_obj_file,
                                     p.leaf_obj_file)
    obj_dataframe.to_csv(('%s%s_obj_facets_.csv' % (result_folder, filename)))
    np.save('%s%s_triangulation.npy' % (result_folder, filename), obj_tri)


def pc_correction():
    
    p = parse_args()
    result_folder = os.path.join(p.result_folder, '')
    
    arr = np.loadtxt(p.point_cloud, delimiter=",")
    
    filename = os.path.basename(p.point_cloud).split('.')[:-1][0]

    corrected_pc = correct(arr, axis_order=p.corr_axis,
                           dist_from_center=p.corr_dist)

    np.savetxt('%s%s.txt' % (result_folder, filename), corrected_pc,
               fmt='%1.3f', delimiter=',')


def batch_pc_correction():
    
    p = parse_args()
    result_folder = os.path.join(p.result_folder, '')

    pc_folder = os.path.join(p.point_cloud_folder, '')
    files = glob.glob(pc_folder + '*.txt')

    for f in files:
        arr = np.loadtxt(f, delimiter=",")
        filename = os.path.basename(f).split('.')[:-1][0]
        corrected_pc = correct(arr, axis_order=p.corr_axis,
                               dist_from_center=p.corr_dist)
        np.savetxt('%s%s.txt' % (result_folder, filename), corrected_pc,
                   fmt='%1.3f', delimiter=',')


def match_pf():
    
    p = parse_args()
    result_folder = os.path.join(p.result_folder, '')

    fname_pc = os.path.basename(p.point_cloud).split('.')[:-1][0]

    facets_df = pd.read_csv(p.facets)

    facets, pc_df = pfmatch(facets_df, p.point_cloud)
    pc_df.index.name = 'point_id'

    facets.to_csv(('%s%s_macthed_facets_.csv' % (result_folder,
                                                 fname_pc)))
    pc_df.to_csv(('%s%s_macthed_points_.csv' % (result_folder,
                                                fname_pc)))


def batch_match_pf():

    p = parse_args()
    result_folder = os.path.join(p.result_folder, '')

    pc_folder = os.path.join(p.point_cloud_folder, '')
    files = glob.glob(pc_folder + '*.txt')
    
    facets_df = pd.read_csv(p.facets)

    for f in files:
        fname_pc = os.path.basename(f).split('.')[:-1][0]
        arr = np.loadtxt(f, delimiter=',')
        facets, pc_df = pfmatch(facets_df, arr)
        pc_df.index.name = 'point_id'

        facets.to_csv(('%s%s_macthed_facets_.csv' % (result_folder,
                                                        fname_pc)))
        pc_df.to_csv(('%s%s_macthed_points_.csv' % (result_folder,
                                                    fname_pc)))


def parse_args():
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
    parser.add_argument("--correction_dist", dest="corr_dist", type=int,
                        help="distance correction to match the point cloud with\
 the object.", default=0)
    parser.add_argument("--correction_axis", dest="corr_axis", type=int,
                        nargs='+',
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
    
    return p

    
if __name__ == '__main__':

    p = parse_args()
    result_folder = os.path.join(p.result_folder, '')
