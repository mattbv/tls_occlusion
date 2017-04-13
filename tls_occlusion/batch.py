# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 22:41:19 2017

@author: mathe
"""
import numpy as np
import glob
import os
import pf1
import mayavi.mlab as mlab


def main(obj_file, pc_folder):

    files = glob.glob(pc_folder + os.sep + '*.csv')

    for f in files:

        filename = os.path.basename(f).split('.')[0]

        if os.path.isfile('results' + os.sep + filename +
                          'facets.csv') is False:
            print('Starting %s' % filename)

            facets_df, pc_df, tri = pf1.main(f, obj_file)

            facets_df.to_csv('results' + os.sep + filename + 'facets.csv')
            pc_df.to_csv('results' + os.sep + filename + 'points.csv')
            np.savetxt('results' + os.sep + filename + 'triangles.txt', tri)

            x = np.asarray(facets_df.vx).astype(float)
            y = np.asarray(facets_df.vy).astype(float)
            z = np.asarray(facets_df.vz).astype(float)
            nn = np.asarray(facets_df.n_points).astype(float)
            dens = np.asarray(facets_df.density).astype(float)

            mlab.triangular_mesh(x, y, z, tri, scalars=nn)
            mlab.colorbar()
            mlab.savefig('results' + os.sep + filename + '_n-points_.png',
                         size=(1920, 1080), magnification='auto')
            mlab.close()

            mlab.triangular_mesh(x, y, z, tri, scalars=dens)
            mlab.savefig('results' + os.sep + filename + '_density_.png',
                         size=(1920, 1080), magnification='auto')
            mlab.close()

    return


if __name__ == '__main__':
    facets_obj = 'data' + os.sep + '1002_etri_erecto_wt_b.ratObj'
    pc_folder = 'D:\\Dropbox\\PhD\\Data\\librat\\single_trees\\xyzm_1002'
    main(facets_obj, pc_folder)
