# distutils: language = c++
# distutils: sources = C_code/gfs.cpp

# Cython interface file for wrapping the object
#
#

from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp.string cimport string

# c++ interface to cython
cdef extern from "C_code/gfs.h" namespace "gfs_manager_space":
  cdef cppclass gfsManager:
        gfsManager(vector[vector[int]], vector[int], string) except +
        int rows
        vector[string] labels
        set[string] labels_set
        vector[string] S
        vector[int] S_index
        map[int, vector[string]] F
        string strategy
        vector[int] greedyAlgorithm(int);

# creating a cython wrapper class
cdef class PygfsManager:
    cdef gfsManager *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, vector[vector[int]] dataset, vector[int] labels_input, strategy_arg):
        self.thisptr = new gfsManager(dataset, labels_input, strategy_arg)
    def __dealloc__(self):
        del self.thisptr
    def greedyAlgorithm(self, feat_card):
        return self.thisptr.greedyAlgorithm(feat_card)