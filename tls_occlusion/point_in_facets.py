# -*- coding: utf-8 -*-
"""
Module to test if a point is placed inside the bounds of a facet.

@author: Matheus Boni Vicari
"""

import numpy as np
from pandas import DataFrame
from collections import defaultdict
import tpf


def pfmatch(facets_df, point_cloud, pc_correction=0):

    """
    Function to test the matching a set of points to a set of facets.

    Parameters
    ----------
    facets_df: pandas.DataFrame
        3D model facets information.
    point_cloud: numpy.ndarray
        Nx3 point cloud coordinates.
    pc_correction: int
        Correction distance to align the point cloud to the 3D model.

    Returns
    -------
    facets_df: pandas.DataFrame
        3D model facets with initial information plus the new count of points
        intersected on each facet.
    pc_df: pandas.DataFrame
        Matching information for the input point cloud.

    """

    # pc_correction = -45

    # Saving material information from the point cloud before correcting
    # coordinates.
    point_material = point_cloud[:, 3].copy()
    # Correcting point cloud xy coordinates.
    point_cloud = correct_pc(point_cloud, pc_correction)

    # Setting matcode as True (1) all rows with leaf facets.
    facets_df['matcode'] = np.asarray(facets_df['material'].str.contains("eaf")
                                      ).astype(int)

    # Creating wood mask (matcode == 0).
    w_mask = np.asarray(facets_df.ix[:, 'matcode'] == 0).astype(bool)

    # Splitting the input facets into wood (w_material) and leaf facets
    # (l_material).
    w_material = facets_df.ix[w_mask]
    l_material = facets_df.ix[~w_mask]

    # Splitting input point cloud into wood (w_pc) and leaf (l_pc) points.
    w_pc = point_cloud[point_material == 0]
    l_pc = point_cloud[point_material == 1]

    # Executing the function assign_facets to facet-points pair of wood and
    # leaf material separatey.
    w_df = assign_facets(w_pc, w_material)
    l_df = assign_facets(l_pc, l_material)

    # Joining the wood and leaf point DataFrame.
    pc_df = w_df.append(l_df, ignore_index=True)

    # Grouping the info in the point cloud DataFrame by facet number, to have
    # a final count of points per facet.
    gfacet = pc_df.groupby('facet')

    # Looping over the facet informatio (items) and assigning the count of
    # points into the designed facet id in facets_df.
    for items in gfacet.groups.items():
        facets_df.loc[facets_df.facet == items[0], 'n_points'] = len(items[1])

    # Calculating point density in each facet.
    facets_df['density'] = facets_df.n_points / facets_df.area

    return facets_df, pc_df


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

    # Looping over each facet.
    for f in facets.groupby('facet'):

        # Getting the facet vertices coordinates.
        fcoord = np.array(f[1].ix[:, 'vx':'vz']).astype(np.float)

        # Generating a bounding box around the facet vertices.
        bbox = Bbox(fcoord)

        # Selecting all points in the point cloud that are inside the bounding
        # box.
        inidx = np.where(np.all(np.logical_and(bbox.min <= points,
                                               points <= bbox.max), axis=1))[0]

        # Looping over every point inside the Bbox and appending the currenct
        # facet id into the point id in pc_dict.
        for id_ in inidx:
            pc_dict[id_].append(f[0])

    # Looping all over point cloud dictionary keys.
    for p in pc_dict.keys():

        # Initializing mindist and f_id.
        mindist = np.inf
        f_id = np.nan

        # Looping over all facets for the current key in pc_dict.
        for f in pc_dict[p]:
            # Setting the current facet as the temporary facet for the
            # current point id (key).
            tfacet = facets[facets['facet'] == f]

            # Selecting current facet vertices coordinates.
            fcoord = np.array(tfacet.ix[:, 'vx':'vz']).astype(np.float)

            # Testing if point is inside the current facet.
            # This option is currently not used as the complexity of the
            # leaf material makes it fail to function properly.
#            test = tpf.point_in_facet(points[p], fcoord)
            test = 1

            # Checking if test is True.
            if test:
                # Calculating the distance of facet 'f' to point 'p'.
                fdist = dist_to_plane(points[p], fcoord)

                # Testing if current distance is smaller the minimum distance.
                # This steps is designed to select only the closest facet to
                # current point 'p'.
                if abs(fdist) <= abs(mindist):
                    # Set facet id as current facet 'f' and 'mindist' to
                    # current calculated dist of 'fdist'.
                    f_id = f
                    mindist = fdist

        # Assigning values for column 'facet' and 'dist'.
        pc_df.set_value(p, 'facet', f_id)
        pc_df.set_value(p, 'dist', mindist)

    return pc_df


class Bbox:

    """
    Bounding Box class.

    """

    def __init__(self, points, tol=0.03):

        """
        Function to initiate and generate a Bounding Box around a
        set of coordinates.

        Parameters
        ----------
        points: numpy.ndarray
            Nx3 coordinates of the points.
        tol: float
            Buffer distance to add to the bounding box vertices.

        """

        self.min = np.min(points, axis=0) - tol
        self.max = np.max(points, axis=0) + tol


def correct_pc(point_cloud, dist):

    """
    Function to correct the point cloud x and y coordinates based on a fixed
    distance.

    Parameters
    ----------
    point_cloud: numpy.ndarray
        Nx3 point cloud coordinates.
    dist: float
        Fixed distance to add to x and y coordiantes of point_cloud.

    Returns
    -------
    point_cloud:  numpy.ndarray
        Nx3 corrected point cloud coordinates.

    """

    # Adding the fixed distance to x and y.
    x = point_cloud[:, 1] + dist
    y = point_cloud[:, 0] + dist
    z = point_cloud[:, 2]

    return np.vstack((x, y, z)).T


def dist_to_plane(p, facet):

    """
    Function to calculate the distance of a point to a facet (plane)

    Parameters
    ----------
    p: numpy.ndarray
        1x3 point coordinates.
    facet: numpy.ndarray
        Nx3 facet vertices coordinates.

    Returns
    -------
    dist: float
        Distance of the point 'p' to the plane 'facet'.

    """

    u = facet[1] - facet[0]
    v = facet[2] - facet[0]

    n = np.cross(u, v)
    n /= np.linalg.norm(n)

    p_ = p - facet[0]
    return np.dot(p_, n)


def containsAny(str, set):
    """Check whether 'str' contains ANY of the chars in 'set'"""
    return 1 in [c in str for c in set]
