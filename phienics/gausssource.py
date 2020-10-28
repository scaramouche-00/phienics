"""
.. module:: gausssource

This module contains the class definition of a Gaussian source profile.

"""

# Copyright (C) Daniela Saadeh 2020
# This file is part of phi-enics


from source import Source
from dolfin import Expression
import numpy as np



class GaussianSource(Source):
    r"""
    This class defines a Gaussian source:

    .. math:: \rho(r) = \frac{1}{\left( \sqrt{2\pi} \sigma \right)^3} 
             \exp{\left( -\frac{1}{2} \frac{r^2}{\sigma^2} \right)}

    where :math:`\sigma = \bar{t} r_S`, and where :math:`\bar{t}` is chosen, inside the code, so that :math:`r_S`
    includes a fraction :math:`f` of the total mass.
    
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

    """

    def __init__( self, fem, Ms=1e20, Rs=1e47, f=0.95,
                  from_lut=True, lut_dir='./lut/', lut_root='gauss_', lut_tmin=0.2, lut_tmax=1.3, lut_tnum=100 ):

        Source.__init__( self, fem, Ms, Rs, f, from_lut, lut_dir, lut_root, lut_tmin, lut_tmax, lut_tnum )

        # look-up table
        if self.from_lut:
            self.set_lut()
              
        self.rho = None
        self.build_source()
        
        self.clean_up()

        

    def dM_dr( self, r, t ):
        # dimensionless radial mass density
        sigma = t * self.fem.mesh.rs
        return 2. / ( np.sqrt(2.*np.pi) * sigma**3 ) * np.exp( -0.5 * (r/sigma)**2 ) * r**2


        

    def build_source( self ):

        self.find_t()

        sigma = self.t * self.fem.mesh.rs

        self.rho = Expression('Ms / pow( sqrt(2*pi) * sigma, 3 ) * exp( -0.5 * pow(x[0]/sigma,2) )',
                              Ms=self.Ms, sigma=sigma, degree=self.fem.func_degree )

        

            
