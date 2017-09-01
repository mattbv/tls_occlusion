# -*- coding: utf-8 -*-
"""
Module to import and convert a .hips material image into a material map.

@author: Matheus Boni Vicari (2017).
"""

import numpy as np
from scipy.spatial import Delaunay
from skimage import feature


def generate_map(img, datum=[512, 0], grid_size=10):

    """
    Function to generate a distance map and a set of triangles indices to
    compose the new facets.

    This function will map the edges of the material in the input image
    simplify them into a simpler polygon, based on the downsampling grid,

    Parameters
    ----------
    img: numpy.ndarray
        2D array representing the material image to map.
    datum: list/array
        Indices of the "origin" to calculate the distance from.
    grid_size: int
        Downsampling distance in which to group the original edge points
        of the material image.

    Returns
    -------
    distmap: numpy.ndarray
        Distance map of the downsampled edged from the material image.

    s: numpy.ndarray
        Triangulation indices for the new facets created from distmap.

    """

    # Mapping the edges of the material in img.
    contour_coords = map_edges(img, datum)
    # Simplifying the material edges.
    poly = simplify_poly(contour_coords, grid_size)
    # Generating the distmap by calculating the distance of each point on
    # the simplified edges to the origin (datum).
    distmap = calculate_distance(poly, img)
    # Triangulating the polygon edges to generate facets.
    # This step could use some improvement on the constrained triangulation.
    s, vd = constrained_triangulation(distmap)

    return distmap, s


def map_edges(img, datum):

    """
    Function to map the edges of a material image and return them as
    uv (xy) coordinates.

    Parameters
    ----------
    img: numpy.ndarray
        2D array representing the material image to map.
    datum: list/array
        Indices of the "origin" to calculate the distance from.

    Returns
    -------
    contour_coords: numpy.ndarray
        2D coordinates of the contours (edge) points from the material
        image.

    """

    # Detecting edge pixels using Canny edge detection.
    edge = feature.canny(img)

    # Initiating empty list of contour points.
    contour_points = []

    # Looping over every pixel inside the edge range to obtain their uv
    # coordinates.
    for y in range(0, edge.shape[0]):
        for x in range(0, edge.shape[1]):
            # Checking if current pixel is edge or not. Pixel with zero value
            # is not an edge pixel.
            if edge[x, y] != 0:
                # Appending pixel coordinate to contour points.
                contour_points = contour_points + [[y, x]]

    # Calculating distance of contour points to user-defined origin (datum).
    contour_coords = np.abs(np.array(contour_points) - datum)

    return contour_coords


def simplify_poly(poly, grid_size):

    """
    Function to simplify a set of polygon points based on a downsampling
    distance. The simplification is calculated by the median of the
    coordinates inside each cell of a grid defined by the grid spacing
    'grid_size'.

    Parameters
    ----------
    poly: numpy.ndarray
        Set of coordinates for the polygon vertices.
    grid_size: int
        Downsampling distance in which to group the original edge points
        of the material image.

    Returns
    -------
    median: numpy.ndarray
        Set of coordinates of the simplified polygon vertices.

    """

    # Generating blocks/cells ids for every point in poly.
    ids = poly / grid_size

    # Initiating the median list.
    median = []

    # Looping over all x and y cell ids.
    for i in np.unique(ids[:, 0]):
        for j in np.unique(ids[:, 1]):
            # Selecting all points inside cell defined by the ids i,j.
            pts = poly[(ids[:, 0] == i) & (ids[:, 1] == j)]

            # Checking for the number of points inside current cell.
            if pts.shape[0] > 1:
                # If more than 1 point, calculate the median of all points
                # and append to median list.
                median.append(np.median(pts, axis=0))

            else:
                # If only one point, assign point to median. If no point
                # was found in the current cell, append an empty list.
                median.append(pts)

    # Flattening the median list into a median array of the simplified
    # poly vertices coordinates.
    median = np.asarray([i for i in median if len(i) > 1])

    return median


def calculate_distance(coords, img):

    """
    Function to calculate the relative distance in coords based on the original
    image (img) size.

    Parameters
    ----------
    coords: numpy.ndarray
        Set of 2D coordinates.
    img: numpy.ndarray
        2D array representing the material image to map.

    Returns
    -------
    distance: numpy.ndarray
        Relative distance, as floats, of every absolute distance in coord.

    """

    return coords.astype(float) / img.shape


def constrained_triangulation(points):

    """
    Function to perform the constrained triangulation of a set of points.
    As of now, this function is incomplete as it performs a simple Delaunay
    triangulation with no constraints. In the future, a better (constrained)
    triangulation will replace this approach.

    Parameters
    ----------
    points: numpy.ndarray
        Set of point coordinates to triangulate.

    Returns
    -------
    sim: numpy.ndarray
        Set of simplices in the triangulation. These simplices are the
        set of indices for every triangle vertices.
    vert: numpy.ndarray
        Set of vertices coordinates used in the triangulation.

    """
    
    # Triangulating input points.
    tri = Delaunay(points)

    # Obtaining vertices of triangulation.
    sim = tri.simplices
    vert = points[sim]

    return sim, vert
