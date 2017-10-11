# -*- coding: utf-8 -*-
"""
Module to test if a point is placed inside the bounds of a facet.

@author: Matheus Boni Vicari
"""

import numpy as np
from pandas import DataFrame
from collections import defaultdict
from utils import geometry as geo
from utils import bbox
import pandas as pd


def pfmatch(facets, point_cloud):

    """
    Function to test the matching a set of points to a set of facets.

    Parameters
    ----------
    facets: pandas.DataFrame
        3D model facets information.
    point_cloud: numpy.ndarray
        Nx3 point cloud coordinates.

    Returns
    -------
    facets_df: pandas.DataFrame
        3D model facets with initial information plus the new count of points
        intersected on each facet.
    pc_df: pandas.DataFrame
        Matching information for the input point cloud.

    """

    pc_df = assign_facets(point_cloud[:, :3], facets)

    # Grouping the info in the point cloud DataFrame by facet number, to have
    # a final count of points per facet.
    gfacet = pc_df.groupby('facet')

    # Looping over the facet informatio (items) and assigning the count of
    # points into the designed facet id in facets_df.
    for items in gfacet.groups.items():
        facets.loc[facets.facet == items[0], 'n_points'] = len(items[1])

    # Calculating point density in each facet.
    facets['density'] = facets.n_points / facets.area

    return facets, pc_df


def assign_facets(points, facets):

    """
    Function to assign points to facets.

    Parameters
    ----------
    points: numpy.ndarray
        Nx3 point cloud coordinates.
    facets: pandas.DataFrame
        3D facets information.

    Returns
    -------
    pc_df: pandas.DataFrame
        Matching information for the input point cloud.

    """

    # Initializing a defaultdict and pandas.DataFrame to store the matching
    # information.
    pc_dict = defaultdict(list)
    pc_df = DataFrame(columns=['facet', 'dist'])
    pc_arr = np.zeros([points.shape[0], 2])

    coords = np.array(facets.ix[:, 'vx':'vz']).astype(np.float)
    facs = {}
    for f in facets.groupby('facet'):
        # Getting the facet vertices coordinates.
        facs[f[0]] = coords[f[1].ix[:, 0]]

    # Looping over each facet.
    for f in facets.groupby('facet'):

        # Getting the facet vertices coordinates.
        fcoord = coords[f[1].ix[:, 0]]

        # Generating a bounding box around the facet vertices.
        min_x = np.min(fcoord[:, 0])
        max_x = np.max(fcoord[:, 0])
        min_y = np.min(fcoord[:, 1])
        max_y = np.max(fcoord[:, 1])
        min_z = np.min(fcoord[:, 2])
        max_z = np.max(fcoord[:, 2])

        bbox_mask = bbox.get_points(points, min_x, max_x, min_y, max_y, min_z,
                                    max_z)
        inidx = np.where(bbox_mask)[0]

        # Looping over every point inside the Bbox and appending the currenct
        # facet id into the point id in pc_dict.
        for id_ in inidx:
            pc_dict[id_].append(f[0].astype(int))

    # Looping all over point cloud dictionary keys.
    for p, f in pc_dict.iteritems():

        # Initializing mindist and f_id.
        mindist = np.inf
        f_id = np.nan

        unique_facets = np.unique(f)

        for u in unique_facets:

            fcoord = facs[u]

            for j in range(0, fcoord.shape[0], 3):

                ft = fcoord[j:j+3, :]

                if ft.shape[0] == 3:
                    # Calcu lating the distance of facet 'f' to point 'p'.
                    fdist = geo.dist_to_plane(points[p], ft)

                    # Testing if current distance is smaller the minimum
                    # distance.
                    # This steps is designed to select only the closest facet
                    # to current point 'p'.
                    if abs(fdist) <= abs(mindist):
                        # Set facet id as current facet 'f' and 'mindist' to
                        # current calculated dist of 'fdist'.
                        f_id = u
                        mindist = abs(fdist)

        pc_arr[p, 0] = f_id
        pc_arr[p, 1] = mindist

        # Assigning values for column 'facet' and 'dist'.
        pc_df = pd.DataFrame(pc_arr, columns=['facet', 'dist'])

    return pc_df
