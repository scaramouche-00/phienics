"""
.. module:: tophatsource

This module contains the class definitions of a smoothed top hat and step source profiles.

"""

# Copyright (C) Daniela Saadeh 2020
# This file is part of phi-enics

from source import Source

from dolfin import Expression
from scipy.integrate import quad

import mpmath as mp
import numpy as np

import os



class TopHatSource(Source):
    r"""
    This class defines a smoothed top-hat distribution:

    .. math:: \rho(r) =  \frac{M_S}{ 4 \pi(-2w^3)\textrm{Li}_3(-e^{\bar{t} r_S/w}) }
              \frac{1}{\exp{\frac{r - \bar{t} r_S}{ w }} + 1 } 

    where :math:`w` is the width of the transition, :math:`\textrm{Li}_3(x)` is the 
    polylogarithm function of order 3 and :math:`\bar{t}` is chosen, inside the code, so that :math:`r_S`
    includes a user-set fraction :math:`f` of the total mass.
    
    *Arguments*
        fem
            a :class:`fem.Fem` instance
        Ms
            source mass (in units :math:`M_P`)
        Rs
            source radius (in units :math:`{M_P}^{-1}`)
        f
            (`float` or `None`) mass fraction to be enclosed within the source radius.
            For `f=None`, :math:`\bar{t}=1` is used
        w
            width of the transition
        from\_lut
            use a :math:`f(w,\bar{t})` look-up table to obtain an initial guess for :math:`\bar{t}`,
            the radial rescaling to apply so that a mass fraction `f` is enclosed within the source radius
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
        lut\_wmin
            if generating a look-up table, minimum `w` to use in the computation
        lut\_wmax
            if generating a look-up table, maximum `w` to use in the computation
        lut\_wnum
            if generating a look-up table, number of `w` (between `lut\_wmin` and `lut\_wmax`)
            for which `f` is computed

        """

    def __init__( self, fem, Ms=1e20, Rs=1e47, f=0.95, w=0.02,
                  from_lut=True, lut_dir='./lut/', lut_root='top_hat_', lut_tmin=0.8, lut_tmax=1.2, lut_tnum=100,
                  lut_wmin=1e-4, lut_wmax=7e-1, lut_wnum=300 ):

        Source.__init__( self, fem, Ms, Rs, f, from_lut, lut_dir, lut_root, lut_tmin, lut_tmax, lut_tnum )

        self.w = w
        self.A = None 

        # look-up-table
        self.lut_wmin = lut_wmin
        self.lut_wmax = lut_wmax
        self.lut_wnum = lut_wnum

        if self.from_lut:
            self.set_lut()
            
        self.build_source()

        self.clean_up()

        


    def set_t0( self ):
        # initial guess for t
        
        if self.from_lut:
            # in look-up w array, find closest value to user-set w
            idx_w = np.where( self.w_array > self.w ) [0][0]
            # in the look-up table, find closest value to user-set f, for the chosen w
            # use it to identify closest t
            idx_t = np.where( self.lut[idx_w,:] < self.f )[0][0]
            # take average of two closest t values
            self.t0 = 0.5 * ( self.t_array[idx_t] + self.t_array[idx_t - 1] )
        else:
            self.t0 = 1.
        
        

    def polylog_term( self, t, w=None ):
        # the Fermi-Dirac integral used in the normalisation of the source profile

        # this syntax allows using this method inside generate_lut for other values of w
        if w is None:
            w = self.w

        mu = t*self.fem.mesh.rs/w
        
        # For very steep profiles/large mu, the argument of polylog becomes large and overflows:
        # use asymptotic series in this case.
        # The exact and asymptotic-series calculations match within machine precision for mu >~ 30
        if mu < 30.:
            return mp.fp.polylog( 3, -np.exp(mu) )
        else:
            return -1./6. * ( mu**3 + np.pi**2 * mu )


    def dM_dr( self, r, t, w=None ):
        # dimensionless radial mass density

        # this syntax allows using this method inside generate_lut for other values of w
        if w is None:
            w = self.w
        
        rs = self.fem.mesh.rs
        pl_term = self.polylog_term(t,w)
        return 1. / ( -2. * w**3 * pl_term ) / ( np.exp( (r - t*rs)/w )  + 1. ) * r**2



    def set_lut( self ):
        # load or generate lut, if not present
        
        try:
            self.w_array = np.load( self.lut_dir + self.lut_root + 'w_array.npy' )
            self.t_array = np.load( self.lut_dir + self.lut_root + 't_array.npy' )
            self.lut = np.load( self.lut_dir + self.lut_root + 'lut.npy' )
        except IOError:
            self.generate_lut()
    
    

    def generate_lut( self ):
        # generate look-up-table with the fractional mass within rs for different values of t and w
        
        print('Generating look-up table, this may take some time...')

        # create dir for the look-up table, if it doesn't exist already
        if not os.path.isdir(self.lut_dir):
            os.mkdir(self.lut_dir)

        # fill arrays for w, t and look-up table
        rs = self.fem.mesh.rs
        w_array = np.logspace( np.log10(self.lut_wmin), np.log10(self.lut_wmax), self.lut_wnum )
        t_array = np.linspace( self.lut_tmin*rs, self.lut_tmax*rs, self.lut_tnum )
        lut = np.zeros_like( np.outer( w_array, t_array ) )
        for i, w in enumerate(w_array):
            for j, t in enumerate(t_array):
                lut[i,j] = quad( self.dM_dr, 0., self.fem.mesh.rs, args=(t, w) )[0]

        # save for future use
        np.save( self.lut_dir + self.lut_root + 'w_array.npy', w_array )
        np.save( self.lut_dir + self.lut_root + 't_array.npy', t_array )
        np.save( self.lut_dir + self.lut_root + 'lut.npy',lut)

        # store for current use
        self.w_array = w_array
        self.t_array = t_array
        self.lut = lut

    
    def clean_up( self ):
        # dump look-up arrays once the source is built
        
        if self.lut is not None:
            del self.lut
            del self.t_array
            del self.w_array
            
        

    def build_source( self ):

        self.find_t()

        # set normalisation
        self.A = self.Ms / ( -8. * np.pi * self.w**3 * self.polylog_term(self.t) )
        
        self.rho = Expression(' A / ( exp( (x[0] - t * rs)/w ) + 1. )',
                              degree=self.fem.func_degree, A=self.A, t=self.t,
                              rs=self.fem.mesh.rs, w=self.w )






        
class StepSource(Source):
    r"""This class defines a step source profile:

    .. math:: \rho(r) = \frac{3 M_S}{4 \pi (\bar{t}r_S)^3} \Theta(\bar{t} r_S-r)

    where :math:`\Theta` is the step function and :math:`\bar{t}` is chosen so that :math:`r_S`
    encloses a user-set fraction :math:`f` of the total mass.
    
    *Arguments*
        fem
            a :class:`fem.Fem` instance
        Ms
            source mass (in units :math:`M_P`)
        Rs
            source radius (in units :math:`{M_P}^{-1}`)
        f
            (`float` or `None`) mass fraction to be enclosed within the source radius.
            For `f=None`, :math:`\bar{t}=1` is used

    """

    def __init__( self, fem, Ms=1e20, Rs=1e47, f=None ):  
        
        Source.__init__( self, fem, Ms, Rs, f )

        # height and width of the step - useful in plots (set within build_source)
        self.rho0 = None
        self.r_step = None
        
        self.rho = None
        self.build_source()


    def find_t( self ):
        
        if self.f is None:
            self.t = 1.
        else:
            self.t = 1. / np.cbrt( self.f )


    def build_source( self ):

        self.find_t()

        self.rho0 = 3. * self.Ms / ( 4. * np.pi * (self.t * self.fem.mesh.rs)**3 )
        self.r_step = self.t * self.fem.mesh.rs

        self.rho = Expression('x[0] < r_step ? rho0 : 0.',
                              degree=self.fem.func_degree,
                              rho0=self.rho0, r_step=self.r_step )



