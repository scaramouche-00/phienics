"""
.. module:: gcakesource

This module contains the class definition of the 'Gaussian wedding cake' source profile.

"""

# Copyright (C) Daniela Saadeh 2020
# This file is part of phi-enics

from phienics.source import Source

from dolfin import Expression, assemble, dx
from scipy.integrate import quad

import numpy as np
            

class GCakeSource(Source):
    r"""
    This class defines a 'Gaussian wedding cake' source profile:

    .. math:: \rho(r) = \frac{M_S}{4 \pi (\bar{t}r_S)^3 X}
              \left[A_1 \exp{\left(-\frac{1}{2}\frac{r^2}{{\sigma_1}^2}\right)} 
              + A_2 \exp{\left(-\frac{1}{2}\frac{(r-\mu_2)^2}{{\sigma_2}^2}\right)}
              + A_3 \exp{\left(-\frac{1}{2}\frac{(r-\mu_3)^2}{{\sigma_3}^2}\right)}\right]

    with :math:`\mu_2=\bar{t}r_S/3`, :math:`\mu_3=2 \bar{t}r_S/3`,
    :math:`\sigma_1=\bar{t} r_S/9`, :math:`\sigma_2=\bar{t}r_S/7`, 
    :math:`\sigma_3=\bar{t} r_S/12`, and the dimensionless factors :math:`A_{1,2,3}`
    chosen so that :math:`\rho(0)=3/2\rho(\mu_2)=3\rho(\mu_3), A_3=1`. 

    The normalisation factor :math:`X` satisfies :math:`4 \pi \int_0^{\infty}\rho r^2 dr=M_S`
    and :math:`\bar{t}` is chosen, inside the code, so that :math:`r_S`
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
        K12
            ratio of heights of first and second peak
        K13
            ratio of heights of first and third peak
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

    def __init__( self, fem, Ms=1e20, Rs=1e47, f=0.95, K12=1.5, K13=3.,
                  from_lut=True, lut_dir='./lut/', lut_root='gcake_', lut_tmin=0.6, lut_tmax=3., lut_tnum=100 ):

        Source.__init__( self, fem, Ms, Rs, f, from_lut, lut_dir, lut_root, lut_tmin, lut_tmax, lut_tnum )

        # peak ratios
        self.K12 = K12
        self.K13 = K13
        
        # means and sigmas of the three Gaussians (set inside build_source)
        self.mu2 = None
        self.mu3 = None
        self.sigma1 = None
        self.sigma2 = None
        self.sigma3 = None

        # multiplicative constants for the three Gaussians (set inside build_source)
        self.A1, self.A2, self.A3 = None, None, None
        
        # look-up table
        if self.from_lut:
            self.set_lut()

        # normalisation (set within build_source)
        self.X = None

        self.rho = None
        self.build_source()

        self.clean_up()


        


    def set_lut( self ):
        # load or generate lut, if not present
        try:
            self.K_array = np.load( self.lut_dir + self.lut_root + 'K_array.npy' )
            self.t_array = np.load( self.lut_dir + self.lut_root + 't_array.npy' )
            self.lut = np.load( self.lut_dir + self.lut_root + 'lut.npy' )

            if (self.K12, self.K13) != tuple(self.K_array):
                self.generate_lut()
                
        except IOError:
            self.generate_lut()

        

    def mus_of_t( self, t ):
        # returns mu2 and mu3 as a function of t - this is used when determining t

        rs = self.fem.mesh.rs

        mu2 = lambda t : t * rs / 3.
        mu3 = lambda t : 2. * t * rs / 3.

        return mu2(t), mu3(t)


    def sigmas_of_t( self, t ):
        # returns sigma1, sigma2 and sigma3 as a function of t - this is used when determining t

        rs = self.fem.mesh.rs
        
        sigma1 = lambda t : t * rs / 9.
        sigma2 = lambda t : t * rs / 7.
        sigma3 = lambda t : t * rs / 12.

        return sigma1(t), sigma2(t), sigma3(t)
           

    def As_of_t( self, t ):
        # Solve a linear system of equations to determine A1/A3 and A2/A3 - only the ratios count.
        # A1/A3 and A2/A3 must be chosen so that K12 and K13 are the effective final peak height ratios.
        
        K12, K13 = self.K12, self.K13

        mu2, mu3 = self.mus_of_t( t )
        sigma1, sigma2, sigma3 = self.sigmas_of_t(t )
        
        # define the matrix a of the unknowns
        a11 = 1. - K12 * np.exp( -0.5 * (mu2/sigma1)**2 )
        a12 = np.exp( -0.5 * (mu2/sigma2)**2 ) - K12
        a21 = 1. - K13 * np.exp( -0.5 * (mu3/sigma1)**2 )
        a22 = np.exp( -0.5 * (mu2/sigma2)**2 ) - K13 * np.exp( -0.5 * ((mu3-mu2)/sigma2)**2 )
        
        # define the known term b
        b1 = - np.exp( -0.5 * (mu2/sigma3)**2 ) + K12 * np.exp( -0.5 * ((mu3-mu2)/sigma3)**2 )
        b2 = - np.exp( - 0.5 * (mu3/sigma3)**2 ) + K13
        
        # solve
        a = np.array([ [ a11, a12 ], [ a21, a22 ] ])
        b = np.array([ b1, b2 ])
        A13, A23 = np.linalg.solve( a, b )

        A3 = 1.
        A1 = A13 * A3
        A2 = A23 * A3

        return A1, A2, A3



    def frac_mass_within_rs( self, t ):
        
        rs = self.fem.mesh.rs
                        
        mu2, mu3 = self.mus_of_t( t )
        sigma1, sigma2, sigma3 = self.sigmas_of_t( t )
        A1, A2, A3 = self.As_of_t( t )
        
        # I split the integrand three ways as the computation is more accurate
        
        rho1_r2 = lambda r : A1 * np.exp( -0.5 * (r/sigma1)**2 ) * r**2
        rho2_r2 = lambda r : A2 * np.exp( -0.5 * ((r-mu2)/sigma2)**2 ) * r**2
        rho3_r2 = lambda r : A3 * np.exp( -0.5 * ((r-mu3)/sigma3)**2 ) * r**2
        
        X1 = quad( rho1_r2, 0., np.inf )[0]
        X2 = quad( rho2_r2, 0., np.inf )[0]
        X3 = quad( rho3_r2, 0., np.inf )[0]
        
        X = 4. * np.pi * ( X1 + X2 + X3 )
        
        norm_rho1_r2 = lambda r : rho1_r2(r) / X
        norm_rho2_r2 = lambda r : rho2_r2(r) / X
        norm_rho3_r2 = lambda r : rho3_r2(r) / X
        
        integr_mass_1 = quad( norm_rho1_r2, 0., rs )[0]
        integr_mass_2 = quad( norm_rho2_r2, 0., rs )[0]
        integr_mass_3 = quad( norm_rho3_r2, 0., rs )[0]
        
        total_integr_mass = 4. * np.pi * ( integr_mass_1 + integr_mass_2 + integr_mass_3 )

        return total_integr_mass
        


    def generate_lut( self ):

        self.K_array = np.array([ self.K12, self.K13 ])

        np.save( self.lut_dir + self.lut_root + 'K_array.npy', self.K_array  )

        Source.generate_lut( self )
    
    
    def clean_up( self ):
        # dump look-up arrays once the source is built
        if self.lut is not None:
            del self.lut
            del self.t_array
            del self.K_array
    


    def build_source( self ):

        self.find_t()

        # now that we have t, get corresponding mu2, mu3 and A1, A2, A3
        self.mu2, self.mu3 = self.mus_of_t( self.t )
        self.sigma1, self.sigma2, self.sigma3 = self.sigmas_of_t( self.t )
        self.A1, self.A2, self.A3 = self.As_of_t( self.t )   
        
        # define the three Gaussian components
        rho1 = Expression('A1 * exp( -0.5 * pow( x[0]/sigma1, 2 ))', A1=self.A1, sigma1=self.sigma1,
                                domain=self.fem.S, degree=self.fem.func_degree )
        rho2 = Expression('A2 * exp( -0.5 * pow( (x[0]-mu2)/sigma2, 2 ))', A2=self.A2, mu2=self.mu2, sigma2=self.sigma2,
                                domain=self.fem.S, degree=self.fem.func_degree )
        rho3 = Expression('A3 * exp( -0.5 * pow( (x[0]-mu3)/sigma3, 2 ))', A3=self.A3, mu3=self.mu3, sigma3=self.sigma3,
                                domain=self.fem.S, degree=self.fem.func_degree )
        
        # get normalisation
        r2 = Expression('pow(x[0],2)', degree=self.fem.func_degree ) # r^2
        self.X = ( assemble( rho1 * r2 * dx ) + assemble( rho2 * r2 * dx ) + assemble( rho3 * r2 * dx ) )

        self.rho = Expression('Ms / (4. * pi * pow(rs,3)) * (rho1 + rho2 + rho3) / X', Ms=self.Ms, rs=self.fem.mesh.rs,
                              rho1=rho1, rho2=rho2, rho3=rho3, X=self.X, degree=self.fem.func_degree )
