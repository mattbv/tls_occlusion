# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:31:15 2017

@author: mathe
"""

from hipsmap import read_hips
from distmap import generate_map
from map2facet import replace_leaf
from objparser import filter_obj, parse_facets
from point_in_facets import pfmatch
import pandas as pd
import numpy as np
import os


def run_matching(obj_file, pc_file, pc_correction):

    point_cloud = np.loadtxt(pc_file, delimiter=',')
    facets = pd.read_csv(obj_file)

    basename1 = obj_file.split('.')[:-1][0]
    basename2 = os.path.basename(pc_file).split('.')[0]

    facets_df, pc_df = pfmatch(facets, point_cloud, pc_correction)

    facets_df.to_csv(basename1 + basename2 + '_facet_data.csv')
    pc_df.to_csv(basename1 + basename2 + '_pc_data.csv')


def leaf_matmap(obj_file, matmap_file):

    facets_df = pd.read_csv(obj_file)
    filename = obj_file.split('.')[:-1][0]

    matmap = np.load(matmap_file).item()

    leaf_mask = facets_df['material'].str.contains("eaf")
    leaf_facets = facets_df.ix[leaf_mask]
    wood_facets = facets_df.ix[~leaf_mask]

    new_leaf, leaf_tri = replace_leaf(leaf_facets, matmap)

    wood_tri = np.zeros([wood_facets.shape[0], 3])
    for i in range(0, wood_facets.shape[0], 3):
        base_id = i
        wood_tri[base_id, :] = np.array([base_id, base_id + 1, base_id + 2])
        wood_tri[base_id + 1, :] = np.array([base_id + 1, base_id + 2,
                                             base_id])
        wood_tri[base_id + 2, :] = np.array([base_id + 2, base_id,
                                             base_id + 1])

    newobj = pd.concat([wood_facets, new_leaf], join='inner')

    leaf_tri = leaf_tri + wood_tri.shape[0]
    triangles = np.vstack((wood_tri, leaf_tri))

    newobj.to_csv(filename + '_obj_final.csv')
    np.save(filename + '_triangulation_final.npy', triangles)


def process_obj(obj_file):

    filename = obj_file.split('.')[:-1][0]

    filter_obj(obj_file, obj_file + '_temp.txt')

    df, tri = parse_facets(obj_file + '_temp.txt')
    df.to_csv(filename + '_facets.csv')
    np.save(filename + '_triangulation.npy', tri)

    os.remove(obj_file + '_temp.txt')


def process_map(img_file, grid_size):

    hips, res_x, res_y, fmt = read_hips(img_file)
    out_name = img_file.split('.')[:-1][0]

    img_u = np.triu(hips)
    img_l = np.tril(hips)

    d1 = [0, hips.shape[1]]
    d2 = [hips.shape[0], 0]

    map_l, tri_l = generate_map(img_l, datum=d1, grid_size=grid_size)
    map_u, tri_u = generate_map(img_u, datum=d2, grid_size=grid_size)

    area_ratio = (((np.sum(img_l) + np.sum(img_u))/255).astype(float) /
                  (img_l.shape[0] * img_l.shape[1]))

    matmap = {'distmap': {'lower': map_l, 'upper': map_u},
              'triangles': {'lower': tri_l, 'upper': tri_u},
              'area_ratio': area_ratio}

    np.save(out_name + '_grid_' + str(grid_size) + '_matmap.npy', matmap)
