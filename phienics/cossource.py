"""
.. module:: cossource

This module contains the class definition of a truncated cosine source profile.

"""

# Copyright (C) Daniela Saadeh 2020
# This file is part of phi-enics


from phienics.source import Source
from dolfin import Expression
import numpy as np




class CosSource(Source):
    r"""
    This class defines a source profile with a truncated cosine:

    .. math:: \rho(r) = \begin{cases}
              \frac{3\pi M_S}{4 (\bar{t}r_S)^3(\pi^2-6)} 
              \left[\cos\left(\pi\frac{r}{\bar{t}r_S}\right)+1\right] & \text{if } r \leq \bar{t}\,r_S \\
              0 & \text{otherwise}
              \end{cases}

    where :math:`\bar{t}` is chosen, inside the code, so that :math:`r_S`
    encloses a fraction :math:`f` of the total mass.
    
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
                  from_lut=True, lut_dir='./lut/', lut_root='cos_', lut_tmin=0.8, lut_tmax=4., lut_tnum=100 ):
        
        Source.__init__( self, fem, Ms, Rs, f, from_lut, lut_dir, lut_root, lut_tmin, lut_tmax, lut_tnum )

        self.A = None
        self.t = None

        # look-up table
        if self.from_lut:
            self.set_lut()
        
        self.rho = None
        self.build_source()
        
        self.clean_up()
        


    def dM_dr( self, r, t ):
        # dimensionless radial mass density
        rs = self.fem.mesh.rs
        
        if r < t * rs:
            return 3. * np.pi**2 / ( (t*rs)**3 * ( np.pi**2 - 6. ) ) * \
                ( np.cos( np.pi * r / (t*rs) ) + 1. ) * r**2
        else:
            return 0.
        


    def build_source( self ):

        rs = self.fem.mesh.rs

        self.find_t()
        
        self.A = 3. * np.pi * self.Ms / ( 4. * (self.t*rs)**3 ) / ( np.pi**2 - 6. )

        self.rho = Expression( 'x[0] < (t*rs) ? A * ( cos( pi * x[0]/(t*rs) ) + 1. ) : 0.',
                                     degree=self.fem.func_degree, A=self.A, t=self.t, rs=rs )



