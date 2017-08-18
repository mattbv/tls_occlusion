# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 22:18:18 2017

@author: Matheus
"""


def containsAny(str, set):
    """
    Check whether 'str' contains ANY of the chars in 'set'
    http://code.activestate.com/recipes/65441-checking-whether-a-string-
    contains-a-set-of-chars/
    """

    return 1 in [c in str for c in set]
