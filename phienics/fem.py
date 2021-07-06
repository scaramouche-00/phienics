"""
.. module:: fem

This module contains a class for code-wide finite-element settings.

"""

# Copyright (C) Daniela Saadeh 2020
# This file is part of phi-enics

import dolfin as d

class Fem(object):
    """
    Base class with code-wide settings on the function spaces. 

    For further details (in particular, when
    continuous vs discontinuous function spaces are employed), 
    see the accompanying `paper <https://arxiv.org/abs/2011.07037>`_ .

    *Arguments*
        mesh 
            a :class:`mesh.Mesh` object
        func_cont
            choice of interpolating function family for continuous function spaces.
            Default: 'CG', i.e. continuous Lagrange
        func_disc
            choice of interpolating function family for discontinuous function spaces.
            Default: 'DG', i.e. discontinuous Lagrange
        func_degree
            degree of interpolating polynomials
    
    """

    def __init__( self, mesh, func_cont='CG', func_disc='DG', func_degree=4 ):
        """The constructor"""

        # mesh
        self.mesh = mesh

        # interpolating functions
        self.func_cont = func_cont
        self.func_disc = func_disc
        self.func_degree = func_degree

        # continuous function space for single scalar function
        self.Pn = d.FiniteElement( self.func_cont, d.interval, self.func_degree )
        self.S = d.FunctionSpace( self.mesh.mesh, self.Pn )

        # discontinuous function space for single scalar function
        self.dPn = d.FiniteElement( self.func_disc, d.interval, self.func_degree ) 
        self.dS = d.FunctionSpace( self.mesh.mesh, self.dPn )


