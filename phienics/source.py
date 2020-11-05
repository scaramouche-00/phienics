"""
.. module source

This module contains the definition of several source profiles.

"""

# Copyright (C) Daniela Saadeh 2020
# This file is part of phi-enics


from dolfin import Expression, assemble, dx
from scipy.integrate import quad

import scipy.optimize as sopt
import dolfin as d
import numpy as np

import os
import glob


class Source(object):
    r"""
    Base class for (static, spherically symmetric) sources.
    
    *Arguments*
        fem
            a :class:`fem.Fem` instance
        Ms
            source mass (in units :math:`M_P`)
        Rs
            source radius (in units :math:`{M_P}^{-1}`)
        f
            (`float` or `None`) mass fraction to be enclosed within the source radius.
            For `f=None`, the characteristic scale of the source profile is used
        from\_lut
            use a :math:`f(\bar{t})` look-up table to obtain an initial guess for :math:`\bar{t}`, the radial rescaling to
            apply so that a mass fraction `f` is enclosed within the source radius
        lut\_dir
            path of the look-up table
        lut\_root
            root of the file names for the look-up table
        lut\_tmin
            if generating a look-up table, minimum :math:`\bar{t}` to use in the computation
        lut\_tmax
            if generating a look-up table, maximum :math:`\bar{t}` to use in the computation
        lut\_tnum
            if generating a look-up table, number of :math:`\bar{t}` (between `lut\_tmin` and `lut\_tmax`)
            for which `f` is computed
            
    """

    def __init__( self, fem, Ms=1e20, Rs=1e47, f=0.95,
                  from_lut=True, lut_dir='./lut/', lut_root='', lut_tmin=0.8, lut_tmax=1.2, lut_tnum=100 ):

        self.fem = fem
        self.Ms = Ms
        self.Rs = Rs

        # fraction of mass within rs
        self.f = f

        # initial guess for t
        self.t0 = None
        self.t = None
        
        self.lut = None
        self.t_array = None
        self.from_lut = from_lut
        self.lut_dir = lut_dir
        self.lut_root = lut_root
        self.lut_tmin = lut_tmin
        self.lut_tmax = lut_tmax
        self.lut_tnum = lut_tnum
        
        # fenics Expression with the definition of the source profile, in the code rescaled units
        self.rho = None



    def set_t0( self ):
        # get initial guess for t (units source radius)
        if self.from_lut:
            # in the look-up table, find closest value to user-set f, for the chosen w
            idx_t = np.where( self.lut < self.f )[0][0]
            # take average of two closest t values
            self.t0 = 0.5 * ( self.t_array[idx_t] + self.t_array[idx_t - 1] )
        else:
            self.t0 = 1.
    
    
    def dM_dr( self, r, t ):
        # dimensionless radial density
        pass
    
    def frac_mass_within_rs( self, t ):
        # fractional mass within rs, as a function of t
        return quad( self.dM_dr, 0., self.fem.mesh.rs, t )[0]
    
    
    def find_t( self ):
        # find t to obtain user-set fractional mass within rs
        if self.f is None:
            self.t = 1.
        else:
            self.set_t0()
            F = lambda t : self.frac_mass_within_rs(t) - self.f
            try:
                self.t = sopt.broyden1( F, self.t0 ).item()
            except sopt.nonlin.NoConvergence:
                # if the Broyden method fails, print warning and set f=None and t=1
                print('****************************************************************************')
                print('   WARNING: could not set f=%.3e. Setting f=None and t=1.' % self.f )
                print('****************************************************************************')
                self.f=None
                self.t=1.


                

    def set_lut( self ):
        # load or generate lut, if not present
        try:
            self.t_array = np.load( self.lut_dir + self.lut_root + 't_array.npy' )
            self.lut = np.load( self.lut_dir + self.lut_root + 'lut.npy' )
        except IOError:
            self.generate_lut()


    

    def generate_lut( self ):
        # generate look-up-table with the fractional mass within rs for different values of t
        print('Generating look-up table, this may take some time...')

        # create dir for the look-up table, if it doesn't exist already
        if not os.path.isdir(self.lut_dir):
            os.mkdir(self.lut_dir)

        # fill arrays for t and look-up table
        rs = self.fem.mesh.rs
        t_array = np.linspace( self.lut_tmin*rs, self.lut_tmax*rs, self.lut_tnum )
        lut = np.zeros_like( t_array )
        for j, t in enumerate(t_array):
            lut[j] = self.frac_mass_within_rs(t)

        # save for future use
        np.save( self.lut_dir + self.lut_root + 't_array.npy', t_array )
        np.save( self.lut_dir + self.lut_root + 'lut.npy',lut)

        # store for current use
        self.t_array = t_array
        self.lut = lut


    def clean_up( self ):
        # dump look-up arrays once the source is built
        if self.lut is not None:
            del self.lut
            del self.t_array
        


    def erase_lut( self ):

        for filename in glob.glob( self.lut_dir + self.lut_root + '*' ):
            os.remove( filename )


