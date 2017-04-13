# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 12:52:04 2017

@author: mathe
"""
import numpy as np
import pandas as pd
from itertools import ifilter, izip
import trimesh
from area import poly_area


def parse_wood(wood_obj):

    # Reading the object file into obj as a single string.
    with open(wood_obj) as f:
        obj = f.read()

    # Spliting the initial string into clusters by the keyword 'usemtl' which
    # means that every facet will be separated into a different string.
    # After splitting the string, filter all generated strings to keep only
    # those containing vertices information 'v'.
    facet_str = obj.split('usemtl ')
    facet_str = [i for i in facet_str if containsAny(i, 'v') is True]
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

    return coords, triangles


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

    foliage_vertices = []
    foliage_tri = []

    for c, t in izip(clone_params, transform_params):

#        foliage_vertices.append(c + np.dot(leaf_coord, t.reshape([3, 3])))
        foliage_vertices.append(c + np.dot(leaf_coord, t.reshape([3, 3]).T))

        try:
            trimax
        except:
            foliage_tri.append(leaf_tri)
        else:
            foliage_tri.append(trimax + leaf_tri)

        trimax = len(foliage_vertices) * leaf_coord.shape[0]

    foliage_vertices = np.concatenate(foliage_vertices, axis=0)

    foliage_tri = np.concatenate(foliage_tri, axis=0)

    return foliage_vertices, foliage_tri


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
        vt, st = create_cylinder(c)
        coords = np.vstack((coords, vt))
        triangles = np.vstack((triangles, st + np.max(triangles) + 1))

    for s in sph_str:
        vt, st = create_sphere(s)
        coords = np.vstack((coords, vt))
        triangles = np.vstack((triangles, st + np.max(triangles) + 1))

    return coords, triangles


def create_cylinder(cylinder_info):
    call_str = cylinder_info.split('\n')
    vert_str = list(ifilter(lambda line: 'v' in line, call_str))
    spawn_str = list(ifilter(lambda line: 'cyl' in line, call_str))

    v = np.asarray([j for i in vert_str for j in
                    i.split(' ')[1:]]).astype(float).reshape([2, 3])
    s = np.asarray([j for i in spawn_str for j in
                    i.split(' ')[1:]]).astype(float)

    h = np.linalg.norm(v[0, :] - v[1, :])

    cyl = trimesh.creation.cylinder(s[2], h, sections=16)
    c_v = cyl.vertices
    c_v[:, 2] = c_v[:, 2] + (h / 2)
    c_v = c_v + v[0, :]
    c_f = cyl.faces

    return c_v, c_f


def create_sphere(sphere_info):
    call_str = sphere_info.split('\n')
    center_str = list(ifilter(lambda line: 'v' in line, call_str))
    spawn_str = list(ifilter(lambda line: 'sph' in line, call_str))

    c = np.asarray([j for i in center_str for j in
                    i.split(' ')[1:]]).astype(float)
    s = np.asarray([j for i in spawn_str for j in
                    i.split(' ')[1:]]).astype(float)

    sph = trimesh.creation.uv_sphere(s[1], count=[16, 16])
    s_v = sph.vertices + c
    s_f = sph.faces

    return s_v, s_f


def containsAny(str, set):
    """
    Check whether 'str' contains ANY of the chars in 'set'
    http://code.activestate.com/recipes/65441-checking-whether-a-string-
    contains-a-set-of-chars/
    """

    return 1 in [c in str for c in set]


if __name__ == "__main__":


    foliage_obj = 'D:\Dropbox\PhD\Data\librat\Scenes\RAMI_IV_originals\HET09_full\ACPL_foliage.obj'
    leaf_obj = 'D:\Dropbox\PhD\Data\librat\Scenes\RAMI_IV_originals\HET09_full\ACPL_leaf.obj'
    wood_obj = 'D:\Dropbox\PhD\Data\librat\Scenes\RAMI_IV_originals\HET09_full\ACPL_wood.obj'
