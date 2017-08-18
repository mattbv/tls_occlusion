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


@autojit
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


@autojit
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


@autojit
# Unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):

    """
    Based on the post from Jamie Bull at http://stackoverflow.com/questions/
    12642256/python-find-area-of-polygon-from-xyz-coordinates

    """
    x = np.linalg.det([[1, a[1], a[2]],
                       [1, b[1], b[2]],
                       [1, c[1], c[2]]])
    y = np.linalg.det([[a[0], 1, a[2]],
                       [b[0], 1, b[2]],
                       [c[0], 1, c[2]]])
    z = np.linalg.det([[a[0], a[1], 1],
                       [b[0], b[1], 1],
                       [c[0], c[1], 1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)


@autojit
def poly_area(poly):

    """
    Based on the post from Jamie Bull at http://stackoverflow.com/questions/
    12642256/python-find-area-of-polygon-from-xyz-coordinates

    """

    if len(poly) < 3:  # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i+1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)
