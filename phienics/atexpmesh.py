"""
.. module:: atexpmesh

This module contains an example mesh class (`arctan-exp mesh`, in the accompanying `paper <arXiv reference>_`)
with a nonlinear mesh that's finer around the source-vacuum transition and coarses everywhere else.

"""

# Copyright (C) Daniela Saadeh 2020
# This file is part of phi-enics

from mesh import Mesh

import scipy.optimize as sopt

import numpy as np
import dolfin as d





class ArcTanExpMesh(Mesh):
    r"""
    This class implements the Arctan-exp mesh described in detail at Sec. 3.4.1 of the accompanying paper
    <this_will_be_the_arXiv_link>_ . This mesh applies the transformation:

    .. math:: r = T(x) = \frac{2}{\pi} r_{\rm s} \arctan{(kx)} \exp{(a x^3 + bx)}

    to obtain a nonlinear radial mesh (:math:`r` coordinate) starting from a uniform mesh (:math:`x` coordinate).

    *Arguments*
        rs
            source radius (typically, 1.)
        r_min
            minimum radius (typically, 0.)
        r_max
            maximum radius box size (i.e. the simulation's 'infinity'). It must be large
            compared to the source radius and the Compton wavelength of the fields involved
        num_cells
            number of cells in the mesh
        too_fine
            this parameter is used to make sure points are not closer than machine precision
            allows to resolve. Points are required to be spaced at least
            :math:`\mathrm{too\_fine}\times\mathrm{DOLFIN\_EPS}`. See :func:`check_mesh`
        linear_refine
            after producing a non-uniform mesh, optionally slice cells in the 
            [linear\_start, linear\_stop] interval this many times (use 0 to skip this step)
        linear_start
            refine linearly from here
        linear_stop
            refine linearly up to here
        linear_cells
            if linear\_refine was chosen, how many cells were added by linear refinement
        r_rm
            optionally remove points around this radius using a transformation 
            :math:`r = A_{\rm rm}/2 \arctan{\left( k_rm x \right)}` (see section 3.4.3
            of the accompanying paper <this_will_be_the_arXiv_link>_ )
        A_rm
            transformation parameter; only valid if r_rm is a valid radiusx
        k_rm
            transformation parameter; only valid if r_rm is a valid radius
        k
            parameter :math:`k` in the :math:`T(x)` transformation
        a
            parameter :math:`a` in the :math:`T(x)` transformation
        b
            parameter :math:`b` in the :math:`T(x)` transformation

    """

    def __init__( self, rs=1., r_min=0., r_max=1e9,
                  num_cells=500, too_fine=1e4, Ntol=1e-8,
                  linear_refine=0, linear_start=None, linear_stop=None,
                  r_rm=None, A_rm=25., k_rm=20., adjust_transition=False,
                  k=25., a=5e-2, b=3e-2 ):
        """The constructor"""
        
        Mesh.__init__( self, rs, r_min, r_max, num_cells, too_fine, Ntol,
                       linear_refine, linear_start, linear_stop,
                       r_rm, A_rm, k_rm, adjust_transition )

        # params of the non-linear transform
        self.k = k
        self.a = a
        self.b = b
        
        # build the mesh
        self.mesh = None
        self.build_mesh()

        


    def check_params( self ):
        r""" 
        Check that the input parameters :math:`a,b,k` make sense: :math:`a` and :math:`b` must be positive
        or zero, but not simultaneously zero; :math:`k` must be strictly positive.

        """

        # a, b positive or zero, but not simultaneously zero
        a_b_check = ( np.sign([ self.a, self.b ]) >= 0. ).all() \
                    and not ( np.sign([ self.a, self.b ]) == 0. ).all()
        
        # k strictly positive
        k_check = ( self.k > 0. )
        
        all_fine = ( a_b_check and k_check )

        if not all_fine:
            message = "Invalid mesh input parameters ( k, a, b ). It must be: k>0 and a,b>=0 but (a,b) != (0,0)."
            raise ValueError(message)




    
        
    def baseline_transform( self ):
        r"""
        Defines the baseline trasformation

        .. math:: r = T(x) = \frac{2}{\pi} r_{\rm s} \arctan{(kx)} \exp{(a x^3 + bx)}

        together with its first and second derivatives, and an approximation for the inverse transformation
        at small and large radii.

        """

        k, a, b = self.k, self.a, self.b
        rs = self.rs

        # baseline transformation
        T = lambda x : rs * 2./np.pi * np.arctan( k*x ) * np.exp( a * x**3 + b * x )

        # derivatives of baseline transformation
        Tprime = lambda x : rs * 2./np.pi * np.exp( a * x**3 + b * x ) * \
                      ( k / ( 1. + (k*x)**2 ) + np.arctan( k*x ) * ( 3. * a * x**2 + b ) )
        
        Tprimeprime = lambda x : rs * 2./np.pi * np.exp( a * x**3 + b * x ) * ( -2. * k**3 * x / ( 1. + (k*x)**2 )**2 + \
                                      2. * k * ( 3.*a*x**2 + b ) / ( 1. + (k*x)**2 ) + 6.*a*x * np.arctan(k*x) + \
                                      ( 3.*a*x**2 + b )**2 * np.arctan(k*x) )

        # approximate inverse of baseline transformation for small r...
        small_r_Tm1 = lambda r : r / Tprime( 0. )

        # ... and large r
        try:
            # if a !=0, find inverse of exp term using Cardano's formula for 3rd degree equations
            p = b/a
            q = lambda r, rs : 1./a * ( np.log(rs) - np.log(r) )
            D = lambda r, rs : ( q(r,rs)/2. )**2 + ( p/3. )**3
            large_r_Tm1 =  lambda r : np.cbrt( -q(r,rs)/2. + np.sqrt(D(r,rs)) ) + np.cbrt( -q(r,rs)/2. - np.sqrt(D(r,rs)) )
        except ZeroDivisionError:
            # otherwise use simple log
            large_r_Tm1 =  lambda r : 1./b * ( np.log(r) - np.log(rs) )

        return T, Tprime, Tprimeprime, small_r_Tm1, large_r_Tm1


        
