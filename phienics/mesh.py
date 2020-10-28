"""
.. module:: mesh

This module contains basic functionalities for all mesh subclasses.

"""

# Copyright (C) Daniela Saadeh 2020
# This file is part of phi-enics

from copy import copy

import scipy.optimize as sopt

import numpy as np
import dolfin as d



class NoConvergence(Exception):
    pass

class FinalNoConvergence(Exception):
    pass

class PointsTooFine(Exception):
    pass

class NegativeDerivative(Exception):
    pass


class Mesh(object):
    r"""
    Base class for mesh classes.

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
            optionally decluster points around this radius, using a transformation 
            :math:`r = A_{\rm rm}/2 \arctan{\left( k_rm x \right)}` (see section 3.4.3
            of the accompanying paper <this_will_be_the_arXiv_link>_ )
        A_rm
            transformation parameter; only valid if r_rm is a valid radiusx
        k_rm
            transformation parameter; only valid if r_rm is a valid radius
        adjust_transition
            if `True`, radial coordinates are rescaled so that the minimum mesh spacing is placed exactly
            at :math:`r_s` - this may be useful when working with a step or top-hat source.
            Otherwise, the spacing is determined by the input parameters alone. `Default: False`.


    Inside a notebook, you can check the mesh parameters by using:
  
    .. code-block:: python

        yourmesh.__dict__
            

    .. note:: All distances are expressed in units of Mn as defined in 
              :class:`UV.UVSolver` or :class:`IR.IRSolver`. For the (most sensible) choice
              of :math:`M_n={R_s}^{-1}`, distances are in units of the source radius.

    .. note:: use :func:`utils.get_values` to output function values at mesh vertices, which takes care
              of this FEniCS bug:
              https://www.allanswered.com/post/jgvjw/fenics-plot-incorrectly-orders-coordinates-for-function-on-locally-refined-1d-mesh/.

    """

    def __init__( self, rs=1., r_min=0., r_max=1e9,
                  num_cells=500, too_fine=1e4, Ntol=1e-8,
                  linear_refine=0, linear_start=None, linear_stop=None,
                  r_rm=None, A_rm=25., k_rm=20., adjust_transition=False ):
        """The constructor"""

        # extrema, source radius, number of cells
        self.rs = rs
        self.r_min = r_min
        self.r_max = r_max
        self.num_cells = num_cells

        # minimum separation between points (units of Dolfin's machine precision)
        self.too_fine = too_fine

        # tolerance of Newton solvers
        self.Ntol = Ntol

        # optional linear refinement
        self.linear_refine = linear_refine
        self.linear_start = linear_start
        self.linear_stop = linear_stop
        self.linear_cells = 0

        # optional removal of points at r_rm
        self.r_rm = r_rm
        self.A_rm = A_rm
        self.k_rm = k_rm

        # place (or not) minimum distance exactly at rs
        self.adjust_transition = adjust_transition

        # useful quantities to store
        self.xs = None
        self.x_rm = None
        self.c = None

        # extrema of the starting mesh
        self.x_min = None
        self.x_max = None

        # transformation, derivatives of transformation, small- and large-r approximations of inverse transform
        self.T, self.Tprime, self.Tprimeprime = None, None, None
        self.small_r_Tm1, self.large_r_Tm1 = None, None

        # mesh object
        self.mesh = None




    def check_params( self ):
        pass


    def baseline_transform( self ):
        pass



    def check_mesh( self ):
        r"""
        Check that all points in the mesh are separated enough that machine precision 
        can tell them apart. Instead of sheer machine precision, I employ a safer 
        threshold too\_fine :math:`\times` DOLFIN\_EPS, where DOLFIN\_EPS is machine precision. 
        The quantity too\_fine is a mesh attribute.

        """

        # sort the coordinates first because, if linear refinement is used, the vertices will not be in order
        sorts = np.sort( self.mesh.coordinates(), axis=0 )
        distances = sorts[1:] - sorts[:-1]

        if min(distances) < self.too_fine * d.DOLFIN_EPS:
            message = "Some mesh points are too close for machine precision: " + \
                      "check your mesh parameters or lower mesh.too_fine."
            raise PointsTooFine, message
        

        
        

    def apply_linear_refinement( self ):
        r"""
        Refine the mesh linearly: slice all cells between a radius linear\_start
        and a radius linear\_stop in half, for a number of times specified by linear\_refine.

        """
        
        for i in range( self.linear_refine ):
            cells = d.cells( self.mesh )
            cell_markers = d.CellFunction( "bool", self.mesh )
            cell_markers.set_all(False)
            for cell in cells:
                # slice cell if (1) it's large enough to be sliced and be resolvable within machine precision
                # (2) its left vertex is within the range specified by the user
                left = cell.get_vertex_coordinates()[0]
                divide = ( cell.circumradius > self.too_fine * d.DOLFIN_EPS ) and \
                        ( self.linear_start < left < self.linear_stop )        
                if divide :
                    cell_markers[cell] = True
                    
            self.mesh = d.refine( self.mesh, cell_markers )

        # how many points were added by linear refinement (if any)? Store the info
        self.linear_cells = len( self.mesh.coordinates() ) - 1 - self.num_cells



        
    def check_derivative_is_positive( self, Tstar_prime, f=0.5, k=0.1, n=11 ):

        # if k_rm (declustering param) is negative, it may be that the derivative of the transform at x_rm
        # becomes briefly negative around x_rm. This can happen even if the distance between mesh vertices
        # ends up being positive. However, we need a monotonic transformation.
        # this trick uses two transformations of the x space to compute the derivative around x_rm more efficiently,
        # and verify its sign

        x_of_xi = lambda xi : np.sqrt( 1. + self.A_rm/2. ) * xi / abs(self.k_rm) + self.x_rm
        xi_of_y = lambda y : np.tan( k*y )
        y_of_xi = lambda xi : 1./k * np.arctan(xi)
        
        xi_min = -np.sqrt( (1.-f)/f )
        xi_max = np.sqrt( (1.-f)/f )

        y_min = y_of_xi( xi_min )
        y_max = y_of_xi( xi_max )

        y_arr = np.linspace( y_min, y_max, n )
        xi_arr = xi_of_y(y_arr)
        x_arr = x_of_xi(xi_arr)
        
        if np.any( Tstar_prime(x_arr) < 0. ):
            message = "   WARNING: the input declustering parameters give rise to a non-monotonic mesh transform; " + \
                      "please check your k_rm and consider passing a larger (i.e. less negative) value."
            raise NegativeDerivative, message


        

    def baseline_transform_w_adjusted_xs( self ):

        T, Tprime, Tprimeprime, small_r_Tm1, large_r_Tm1 = self.baseline_transform()

        # find minimum of first derivative of transformation
        xs_0 = small_r_Tm1(self.rs)
        F = lambda x : Tprimeprime(x)
        self.xs = sopt.newton(F,xs_0)

        # adjust transition so that minimum of first derivative sits at rs
        self.c = T(self.xs)/self.rs

        # update baseline transformation
        T_new = lambda x : T(x)/self.c
        Tprime_new = lambda x : Tprime(x)/self.c
        Tprimeprime_new = lambda x : Tprimeprime(x)/self.c
        small_r_Tm1_new = lambda r : small_r_Tm1(r*self.c)
        large_r_Tm1_new = lambda r : large_r_Tm1(r*self.c)

        return T_new, Tprime_new, Tprimeprime_new, small_r_Tm1_new, large_r_Tm1_new





    def transform_with_declustering( self ):
        """
        Declusters points at a specified radius r_rm. See Sec. 3.4.3 of the accompanying paper <this_will_be_the_arXiv_link>_
        for details.

        In order to enable this feature in custom mesh classes, the baseline trasformation, first and second
        derivatives must be defined in the child mesh classes.

        """

        # get baseline transformation
        T_bl, Tprime_bl, Tprimeprime_bl, small_r_Tm1_bl, large_r_Tm1_bl = self.baseline_transform()

        # define part of function that removes points at r_rm
        T_rm = lambda x, x_rm : self.A_rm / np.pi * np.arctan( self.k_rm * (x-x_rm) ) + 1. + self.A_rm/2.
        Tprime_rm = lambda x, x_rm : self.A_rm / np.pi * self.k_rm / ( 1. + (self.k_rm * (x-x_rm))**2 )
        Tprimeprime_rm = lambda x, x_rm : -2. * self.A_rm / np.pi * self.k_rm**3 * (x-x_rm) \
                         / ( 1. + (self.k_rm * (x-x_rm))**2 )**2

        # radial rescaling
        C = lambda x_rm : T_bl(x_rm) * ( 1. + self.A_rm / 2. ) / self.r_rm

        # define full function
        T = lambda x, x_rm : T_bl(x) * T_rm(x,x_rm) / C(x_rm)
        Tprime = lambda x, x_rm : ( Tprime_bl(x) * T_rm(x,x_rm) + T_bl(x) * Tprime_rm(x,x_rm) ) / C(x_rm)
        
        Tprimeprime = lambda x, x_rm : ( Tprimeprime_bl(x) * T_rm(x,x_rm) + 2. * Tprime_bl(x) * Tprime_rm(x,x_rm) + \
                                            T_bl(x) * Tprimeprime_rm(x,x_rm) ) / C(x_rm)
        
        # find xs and x_rm
        F1 = lambda xs, x_rm : T( xs, x_rm ) - self.rs
        F2 = lambda xs, x_rm : Tprimeprime( xs, x_rm )
        F = lambda x : np.array([ F1(*x.T), F2(*x.T) ])
        
        xs_0 = small_r_Tm1_bl( self.rs )

        try:
            x_rm_0 = large_r_Tm1_bl( self.r_rm / ( 1. + self.A_rm / 2. ) )
            guess = np.array([ xs_0, x_rm_0 ])
            solv = sopt.root( F, guess, tol=self.Ntol )
            if not solv.success:
                # sopt.root does not raise an exception automatically when it does not converge
                raise NoConvergence
        except NoConvergence:
            # try with a different initial guess for x_rm:
            x_rm_0 = small_r_Tm1_bl( self.r_rm / ( 1. + self.A_rm / 2. ) )
            guess = np.array([ xs_0, x_rm_0 ])
            solv = sopt.root( F, guess, tol=self.Ntol )
            if not solv.success:
                message = "The search for an updated transformation performing declustering failed."
                raise FinalNoConvergence, message
            
        self.xs, self.x_rm = solv.x
        self.c = C(self.x_rm)

        # update transformation and derivative
        Tstar = lambda x : T( x, self.x_rm )
        Tstar_prime = lambda x : Tprime( x, self.x_rm )
        if self.k_rm < 0.:
            self.check_derivative_is_positive( Tstar_prime )
        
        Tstar_primeprime = lambda x : Tprimeprime( x, self.x_rm )
        small_r_Tstar_m1 = small_r_Tm1_bl # keep same

        # adjust large-r approximation by asymptote of arctan
        large_r_Tstar_m1 = lambda r : large_r_Tm1_bl( r/(1. + self.A_rm) ) if self.k_rm >=0. \
                           else large_r_Tm1_bl(r)
       

        return Tstar, Tstar_prime, Tstar_primeprime, small_r_Tstar_m1, large_r_Tstar_m1        

        
        

    def build_mesh( self ):
        r"""
        Build the mesh:

        (1) check that the input parameters for the mesh are valid (the function performing the test must be
            defined in the child mesh class)
        (2) set the mesh transformation. Depending on whether `r\_rm=None` or `r\_rm=float`, the transformation
            will be the baseline transformation (as defined in the child mesh class)
            or its updated form declustering mesh points at `r\_rm`
        (3) create a starting linear mesh, in particular, compute its extrema by using the Newton method 
        (4) obtain a nonlinear mesh by applying the transformation defined at (2)
        (5) apply additional linear refinement if required
        (6) check that the obtained mesh points are not closer than the user-set tolerance of
            :math:`\mathrm{too\_fine}\times\mathrm{DOLFIN\_EPS}`.

        """

        # check input parameters are valid
        self.check_params()

        # define transformation - either baseline or with point removal at r_rm
        if self.r_rm is None:
            if self.adjust_transition:
                transform = self.baseline_transform_w_adjusted_xs()
            else:
                transform = self.baseline_transform()
        else:
            transform = self.transform_with_declustering()
        self.T, self.Tprime, self.Tprimeprime, self.small_r_Tm1, self.large_r_Tm1 = transform
        
        # extrema of starting mesh - find using the Newton method
        x_min_0 = self.small_r_Tm1( self.r_min ) # initial guess
        F_min = lambda x : self.T(x) - self.r_min
        self.x_min = sopt.newton( F_min, x_min_0, self.Tprime, tol=self.Ntol )
        x_max = self.large_r_Tm1( self.r_max )
        F_max = lambda x: self.T(x) - self.r_max
        self.x_max = sopt.newton( F_max, x_max, self.Tprime, tol=self.Ntol )
        
        # create starting linear mesh from x_min to x_max
        self.mesh = d.IntervalMesh( self.num_cells, self.x_min, self.x_max )
        x_mesh = self.mesh.coordinates()[:,0]
        
        # apply transform and get a refined radial mesh
        T = np.vectorize( self.T )
        refined_mesh = T( x_mesh )
        self.mesh.coordinates()[:,0] = refined_mesh[:]

        # apply optional linear refinement if required
        if self.linear_refine:
            self.apply_linear_refinement()

        # check that points are not closer than user-set tolerance
        self.check_mesh()




        
