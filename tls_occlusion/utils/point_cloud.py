# -*- coding: utf-8 -*-
"""
@author: Matheus Boni Vicari
"""
import numpy as np


def correct(point_cloud, axis_order=[0, 1, 2], dist_from_center=0):

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
    x = point_cloud[:, axis_order[0]] + dist_from_center
    y = point_cloud[:, axis_order[1]] + dist_from_center
    z = point_cloud[:, axis_order[2]]

    return np.vstack((x, y, z)).T
