# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 22:15:32 2017

@author: Matheus
"""
import numpy as np
from itertools import ifilter
import trimesh


def cylinder(cylinder_info):
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


def sphere(sphere_info):
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
