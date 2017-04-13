# -*- coding: utf-8 -*-
"""
@author: Matheus Boni Vicari (2017).
"""

from numba import autojit
import numpy as np


@autojit
def point_in_facet(P, facet):

    """
    Function to test if a point lies inside a facet defined by 3 vertices.
    The algorithm was based on a solution presented by the user blackpawn.

    Parameters
    ----------
    P: array
        1 x n array containing the coordinates of the point P in an
        n-dimensional space.
    facet: array
        3 x n array containing the coordinates of the 3 vertices that composes
        the facet in an n-dimensional space.

    Returns
    -------
    point_in_facet: bool
        Result of the test to check if point P is inside facet. Returns 'True'
        if P is inside facet and 'False' otherwise.

    References
    ----------
    .. [1] Point in triangle test. http://blackpawn.com/texts/pointinpoly/

    """

    v0 = facet[2, :] - facet[0, :]
    v1 = facet[1, :] - facet[0, :]
    v2 = P - facet[0, :]

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    return (u >= 0) & (v >= 0) & (u + v < 1)
