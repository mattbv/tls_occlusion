# -*- coding: utf-8 -*-
"""
Module to replace an existing set of square(ish) facets, defined by pairs of
triangles, by a new set of facets created from a material mapping processing.

@author: Matheus Boni Vicari (2017).
"""

import numpy as np
from pandas import DataFrame
from itertools import ifilter
from itertools import izip
from parse_geometry import cylinder
from parse_geometry import sphere
import pandas as pd
from ..utils.string import containsAny
from ..utils.geometry import project
from ..utils.geometry import poly_area


def matmap2facets(facets, matmap):

    """
    Function to perform the leaf replacement. This function will take as input
    the pandas.DataFrame containing the original leaf facets. These facets
    are originally defined by pairs of triangles, here referenced as lower
    and upper triangles.

    Parameters
    ----------
    facets: pandas.DataFrame
        DataFrame containing all pairs of leaf facets to replace. The facets
        must be ordered together, so each pair is in sequential rows and
        facet indices.
    matmap: numpy.data
        Material map to replace te leaf.

    Returns
    -------
    newleaf: pandas.DataFrame
        New set of facets that composes the leaf material. This new DataFrame
        has the same columns as the input 'facets' variable.
    tri: numpy.ndarray
        Nx3 vertices order of the triangulation of the facets in newleaf.


    """

    # Selecting the xyz vertices and the facet ids in facet.
    vertices = facets.loc[:, ['facet', 'vx', 'vy', 'vz']]

    # Unpacking the matmat into the map distances and triangulations for
    # upper and lower triangles.
    map_l = matmap['distmap']['lower']
    map_u = matmap['distmap']['upper']
    tri_l = matmap['triangles']['lower']
    tri_u = matmap['triangles']['upper']
    area_ratio = matmap['area_ratio']

    # Generating a list with 'new_leaf' and leght equals to all vertices in
    # matmap (both upper and lower).
    mat = ['new_leaf'] * (map_l.shape[0] + map_u.shape[0])

    # Obtaining unique facets ids in vertices.
    v_ids = np.unique(vertices.facet).astype(int)

    # Initializing new pandas.DataFrame.
    newleaf = DataFrame(columns=['facet', 'vx', 'vy', 'vz', 'area',
                                 'material'])

    # Initializing empy triangulation list.
    tri = []

    # Looping over each unique even facet id.
    for fi in range(0, v_ids.shape[0], 2):

        # Obtaining the vertices for the current facet.
        i = v_ids[fi]

        # Selecting upper and lower facets based on the facet id.
        # Calculating the upper and lower facet areas.
        facet_u = np.array(vertices.loc[vertices['facet'] ==
                                        i].ix[:, 1:]).astype(float)
        area_u = np.mean(facets.loc[vertices['facet'] ==
                                    i].ix[:, 'area'])

        facet_l = np.array(vertices.loc[vertices['facet'] ==
                                        i + 1].ix[:, 1:]).astype(float)
        area_l = np.mean(facets.loc[vertices['facet'] ==
                                    i + 1].ix[:, 'area'])

        # Projecting upper and lower sections of the new leaf facet and then
        # stacking both into a single facet.
        proj_u = project(facet_u[1, :], facet_u[2, :], facet_u[0, :], map_u)
        proj_l = project(facet_l[2, :], facet_l[0, :], facet_l[1, :], map_l)
        proj = np.vstack((proj_l, proj_u))

        # Creating a list of ids (repetition) and calculated area for the new
        # facet.
        f = [i] * proj.shape[0]
        area_new = [area_ratio * (area_l + area_u)] * proj.shape[0]

        # Creating pandas.DataFrame using the new facet id, vertices, area and
        # material
        d = {'facet': f, 'vx': proj[:, 0], 'vy': proj[:, 1],
             'vz': proj[:, 2], 'area': area_new, 'material': mat}
        df = DataFrame(data=d, columns=['facet', 'vx', 'vy', 'vz',
                                        'area', 'material'])
        # Appending temporary facet DataFrame 'df' to final DataFrame 'newleaf'
        newleaf = newleaf.append(df, ignore_index=True)

        # Stacking upper and lower triangulations and increasing the indices
        # values based on the last stack triangulation indices.
        tritemp = np.vstack((tri_l, (np.max(tri_l) + 1) + tri_u))
        try:
            trimax
        except:
            tri.append(tritemp)
        else:
            tri.append((trimax + 1) + tritemp)

        # Calculating trimax based on the max id in the stacked triangulation.
        trimax = np.max(tri[-1])

    # Flattening triangulation into a single Nx3 array.
    tri = np.concatenate(tri, axis=0)

    return newleaf, tri


def parse_foliage(foliage_obj, leaf_obj):

    with open(foliage_obj) as f:
        obj_str = list(ifilter(lambda line: 'clone' in line, f))

    clone_str = []
    transform_str = []
    for i in obj_str:
        t = i.split(' Transform ')
        transform_str.append(t[1].split(' ')[:-1])
        clone_str.append(t[0].split('clone ')[1].split(' '))

    clone_params = np.asarray(clone_str).astype(float)
    transform_params = np.asarray(transform_str).astype(float)

    leaf_coord, leaf_tri = parse_leaf(leaf_obj)
    n_rows = leaf_coord.shape[0]

    foliage_vertices = []
    foliage_tri = []
    area = []
    facet = []
    count = 0

    for c, t in izip(clone_params, transform_params):

#        foliage_vertices.append(c + np.dot(leaf_coord, t.reshape([3, 3])))
        foliage_vertices.append(c + np.dot(leaf_coord, t.reshape([3, 3]).T))
        area.append([poly_area(foliage_vertices[-1])] * n_rows)
        facet.append([count] * n_rows)
        count += 1

        try:
            trimax
        except:
            foliage_tri.append(leaf_tri)
        else:
            foliage_tri.append(trimax + leaf_tri)

        trimax = len(foliage_vertices) * leaf_coord.shape[0]

    foliage_tri = np.concatenate(foliage_tri, axis=0)

    foliage_vertices = np.concatenate(foliage_vertices, axis=0)
    foliage_vertices = np.array(foliage_vertices, ndmin=2)
    facet = np.concatenate(facet, axis=0)
    facet = np.array(facet, ndmin=2).T
    area = np.concatenate(area, axis=0)
    area = np.array(area, ndmin=2).T
#
#    foliage_vertices = np.zeros([clone_params.shape[0] * n_rows, 5])
#    foliage_tri = []
##    area = []
##    facet = []
#    count = 0
#
#    for c, t in izip(clone_params, transform_params):
#
##        foliage_vertices.append(c + np.dot(leaf_coord, t.reshape([3, 3])))
#        begin = count * n_rows
#        end = (count * n_rows) + n_rows
#        foliage_vertices[begin:end, 1:4] = np.asarray(c + np.dot(leaf_coord, t.reshape([3, 3]).T))
#        foliage_vertices[begin:end, 0] = np.asarray([count] * n_rows)
#        foliage_vertices[begin:end, 4] = np.asarray([poly_area(foliage_vertices[-1])] * n_rows)
#
#        count += 1
#
#        try:
#            trimax
#        except:
#            foliage_tri.append(leaf_tri)
#        else:
#            foliage_tri.append(trimax + leaf_tri)
#
#        trimax = len(foliage_vertices) * leaf_coord.shape[0]
#
##    foliage_vertices = np.concatenate(foliage_vertices, axis=0)
#    foliage_tri = np.concatenate(foliage_tri, axis=0)
##    facet = np.concatenate(facet, axis=0)
##    area = np.concatenate(area, axis=0)

#    facet = pd.Series(facet, name='facets')
#    area = pd.Series(area, name='area')
    foliage = np.hstack((facet, foliage_vertices, area))
    df = pd.DataFrame(foliage, columns=['facet', 'vx', 'vy', 'vz', 'area'])
#    df = pd.concat([df, material], axis=1)
#    df =

#    return foliage_vertices, foliage_tri, area, facet
    return df, foliage_tri


def parse_leaf(leaf_obj):

    # Reading the object file into obj as a single string.
    with open(leaf_obj) as f:
        obj = f.read()

    # Spliting the initial string into clusters by the keyword 'usemtl' which
    # means that every facet will be separated into a different string.
    # After splitting the string, filter all generated strings to keep only
    # those containing vertices information 'v'.
    facet_str = obj.split('usemtl ')
    facet_str = [i for i in facet_str if containsAny(i, 'v') is True]
    sph_str = list(ifilter(lambda line: 'sph' in line, facet_str))
    cyl_str = list(ifilter(lambda line: 'cyl' in line, facet_str))
    facet_str = list(ifilter(lambda line: 'f' in line, facet_str))

    # Initializing variables and allocating space for later user.
    coords = np.zeros([len(facet_str) * 3, 3])
    triangles = np.zeros([len(facet_str) * 3, 3])

    # Looping over every facet string data.
    for fid in xrange(len(facet_str)):
        # Setting the base_id that represents the initial DataFrame row to
        # assign the current facet's information.
        base_id = fid * 3
        # Spliting the current facet info by lines.
        temp = facet_str[fid].split('\n')
        # Filtering all substring (lines) and obtaining the vertices 'v' data.
        c = [i for i in temp[1:] if containsAny(i, 'v')]

        # Creating vertices coordinates and assigning them to the coords array,
        # based on the base_id.
        v1 = tuple(map(float, c[0].split('v ')[1].split(' ')))
        v2 = tuple(map(float, c[1].split('v ')[1].split(' ')))
        v3 = tuple(map(float, c[2].split('v ')[1].split(' ')))
        coords[base_id, :] = v1
        coords[base_id + 1, :] = v2
        coords[base_id + 2, :] = v3

        # Assigning the triangulation order (arbitrarily created) for the
        # facet. This step is just a repetition of the same order of vertices
        # ids,
        # 0, 1, 2
        # 1, 2, 0
        # 2, 1, 0
        # but increasing the ids acording to the base_id.
        triangles[base_id, :] = np.array([base_id, base_id + 1, base_id + 2])
        triangles[base_id + 1, :] = np.array([base_id + 1, base_id + 2,
                                             base_id])
        triangles[base_id + 2, :] = np.array([base_id + 2, base_id,
                                             base_id + 1])

    for c in cyl_str:
        vt, st = cylinder(c)
        coords = np.vstack((coords, vt))
        triangles = np.vstack((triangles, st + np.max(triangles) + 1))

    for s in sph_str:
        vt, st = sphere(s)
        coords = np.vstack((coords, vt))
        triangles = np.vstack((triangles, st + np.max(triangles) + 1))

    return coords, triangles
