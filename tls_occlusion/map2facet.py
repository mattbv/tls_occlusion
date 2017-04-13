# -*- coding: utf-8 -*-
"""
Module to replace an existing set of square(ish) facets, defined by pairs of
triangles, by a new set of facets created from a material mapping processing.

@author: Matheus Boni Vicari (2017)
"""

import numpy as np
from pandas import DataFrame


def replace_leaf(facets, matmap):

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


def project(base, vert1, vert2, distmap):

    """
    Function to project a set of vertices based on a distance mapping.

    Parameters
    ----------
    base: numpy.ndarray
        1x3 point coordinates of the triangle base vertex.
    vert1: numpy.ndarray
        1x3 point coordinates of the triangle vertex 1.
    vert2: numpy.ndarray
        1x3 point coordinates of the triangle vertex 2.
    distmap: numpy.ndarray
        Distance map of the points to project.

    Returns
    -------
    proj_points: numpy.ndarray
        Nx3 projected points coordinates.

    """

    # Projecting along axis base->vert1 and base->vert2.
    # Calculating the projection of points in axis 1.
    proj0 = (base + np.array(distmap[:, 0], ndmin=2).T *
             (vert1 - base))
    # Calculating the projection of points in axis 2.
    proj1 = (base + np.array(distmap[:, 1], ndmin=2).T *
             (vert2 - base))
    # Sum of the vectors to project both axis over the triangle.
    proj_points = (proj0 + proj1) - base

    return proj_points
