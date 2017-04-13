# -*- coding: utf-8 -*-
"""
Module to parse and convert an Wavefront object file (.obj) used for
radiative transfer simulations (ray tracing) into a comma-separated table.
The purpose of this module is to allow the loading of an .obj file into
Python for further data analysis.

@author: Matheus Boni Vicari (2017)
"""

import pandas as pd
from itertools import ifilter
import numpy as np
from area import poly_area


def filter_obj(obj_path, out_file):

    """
    Function to filter an .obj file, keeping only information containing
    material bounding boxes that contain vertices.
    """

    with open(obj_path, 'rb') as f, open(out_file, 'wb') as g:
        g.writelines(ifilter(lambda line: ('v' in line) | ('usemtl' in line),
                     f))


def parse_facets(obj_path):

    """
    Function to read an object file (.obj) after an initial filtering and
    parse the object into a pandas.DataFrame to create a table of facets
    information and the triangulation vertices to allow the 3D display of
    the facets.

    The following lines are an example of what a facet looks like in the
    object file (after filtering):
    usemtl ACPL_leaf
    v 0. 0.0193 0.0685
    v 0. 0.0393 0.0548
    v 0. 0.0393 0.0644

    The first line states the material used in the represented facet. The
    second to fourth lines represent the facet vertices in 3D space.

    Parameters
    ----------
    obj_path: str
        Path of the wavefron object file to import. If the folder where this
        object is stored is not on system PATH, use the full path as input.

    """

    # Reading the object file into obj as a single string.
    with open(obj_path) as f:
        obj = f.read()

    # Spliting the initial string into clusters by the keyword 'usemtl' which
    # means that every facet will be separated into a different string.
    # After splitting the string, filter all generated strings to keep only
    # those containing vertices information 'v'.
    facet_str = obj.split('usemtl ')
    facet_str = [i for i in facet_str if containsAny(i, 'v') is True]

    # Initializing variables and allocating space for later user.
    coords = np.zeros([obj.count('v'), 5])
    triangles = np.zeros([obj.count('v'), 3])
    mat = [0] * (obj.count('v'))

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
        coords[base_id, 1:4] = v1
        coords[base_id + 1, 1:4] = v2
        coords[base_id + 2, 1:4] = v3

        # Assignin the current facet id to the coords array. This id is the
        # field that will be used to identify a single facet in the parsed
        # object.
        coords[base_id:base_id+3, 0] = fid

        # Assigning the current facet material to the mat list. This value
        # is repeated from base_id to base_id + 2 referent to all 3 vertices.
        mat[base_id] = temp[0]
        mat[base_id + 1] = temp[0]
        mat[base_id + 2] = temp[0]

        # Calculating the polygon area of the facet and assigning it to
        # the coords array.
        coords[base_id:base_id + 3, 4] = poly_area(np.vstack((v1, v2, v3)))

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

    # Creating the pandas Series and DataFrames from the material and coords
    # variables and then joining both in a final DataFrame (df).
    material = pd.Series(mat, name='material')
    df = pd.DataFrame(coords, columns=['facet', 'vx', 'vy', 'vz', 'area'])
    df = pd.concat([df, material], axis=1)

    return df, triangles


def containsAny(str, set):
    """
    Check whether 'str' contains ANY of the chars in 'set'
    http://code.activestate.com/recipes/65441-checking-whether-a-string-
    contains-a-set-of-chars/
    """

    return 1 in [c in str for c in set]
