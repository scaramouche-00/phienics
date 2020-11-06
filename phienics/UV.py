
"""
.. module:: UV

This module contains the solver for the UV-complete theory described in Section VI of 
    De Rham et al <http://xxx.lanl.gov/abs/1702.08577>, plus a coupling to matter.

"""

# Copyright (C) Daniela Saadeh 2020
# This file is part of phi-enics


from phienics.solver import Solver
from phienics.utils import r2_norm, project

from dolfin import inner, grad, dx
from dolfin import Expression, Constant

from numpy import log10

import dolfin as d
import numpy as np





class UVFields(object):
    r"""
    Physical parameters of the UV-complete theory described in Section VI of 
    `De Rham et al <http://xxx.lanl.gov/abs/1702.08577>`_:

    .. math::

              &\Box\phi - m^2\phi - \alpha\Box H = \frac{\rho}{M_p}

              &\Box H - M^2 H - \alpha\Box\phi - \frac{\lambda}{3!} H^3 = 0

    plus an additional coupling to matter to study screening.

    :math:`\phi` is a cosmological light scalar field of mass :math:`m`, 
    :math:`H` is a self-interacting more massive scalar field of mass :math:`M`, :math:`\rho`
    is a massive source to which :math:`\phi` is coupled, :math:`\alpha` 
    is the coupling between the two scalar fields, :math:`\lambda` is the strength of the 
    self-coupling of :math:`H` and :math:`M_p` is the Planck mass.

    *Arguments*
        m
            mass of the light field :math:`m`
        M
            mass of the heavy field :math:`M`
        Mp
            Planck mass :math:`M_p`
        alpha
            coupling constant :math:`\alpha`
        lam
            self-coupling :math:`\lambda` (note that 'lambda' is a reserved word in Python, and can't
            be used as a variable name)

    """

    def __init__( self, m=1e-51, M=1e-48, Mp=1., alpha=0.4, lam=0.7 ):

        self.m = m
        self.M = M
        self.Mp = Mp

        self.alpha = alpha
        self.lam = lam

        # used in computation of On
        self.log10_On_coeff = lambda n : n*log10(self.lam/6.) - (6*n+2)*log10(self.M)
        self.On_oth_coeff = lambda n : self.alpha**(2*n+2) # alpha may be negative so it can't be included in the log
        # used in computation of Qn
        self.log10_Qn_coeff = lambda n : -(3*n-1)*log10(self.M)
        self.Qn_oth_coeff = lambda n : self.alpha**n






class UVSolver(Solver):
    r"""
    Solver for the UV-complete theory described in Section VI of 
    `De Rham et al <http://xxx.lanl.gov/abs/1702.08577>`_ (plus a coupling to matter). 
    Derives from :class:`solver.Solver`.

    For better accuracy in the computation of the fields' Laplacians, within the code
    the original system of equations is expressed as:

    .. math:: & Y - m^2 \phi - \alpha Z = \frac{\rho}{M_P} \\
              & Z - M^2 H - \alpha Y - \frac{\lambda}{6} H^3 = 0 \\
              & \nabla^2 \phi = Y \\
              & \nabla^2 H = Z 


    *Arguments*
        fem 
            a :class:`fem.Fem` instance
        source
            a :class:`source.Source` instance
        fields
            a :class:`UVFields` instance
        Mn
            distances are rescaled following :math:`\hat{r} = M_n r` inside the code. 
            Default: :math:`M_n={r_S}^{-1}`
        Mf1
            field rescaling for :math:`\phi`: :math:`\hat{\phi}=\phi/M_{f1}`. Default: :math:`M_s M_n / M_P`.
        Mf2
            field rescaling for :math:`H`: :math:`\hat{H}=H/M_{f2}`. Default: :math:`|\alpha| M_{f1}`.
        abs\_du\_tol
            if `criterion` is `'change'`, `abs\_du\_tol` 
            is the quantity :math:`\epsilon_{\rm abs}^{\rm (S)}` in Eq. :eq:`Eq_change_criterion`,
            i.e. the absolute tolerance on the change in the solution between two iterations
        rel\_du\_tol
            if `criterion` is `'change'`, `rel\_du\_tol` 
            is the quantity :math:`\epsilon_{\rm rel}^{\rm (S)}` in Eq. :eq:`Eq_change_criterion`,
            i.e. the relative tolerance on the change in the solution between two iterations
        abs\_res\_tol
            if `criterion` is `'residual'`, `abs\_du\_tol` 
            is the quantity :math:`\epsilon_{\rm abs}^{\rm (R)}` in Eq. :eq:`Eq_residual_criterion`,
            i.e. the absolute tolerance on the residuals at a given iteration
        rel\_res\_tol
            if `criterion` is `'residual'`, `rel\_du\_tol` 
            is the quantity :math:`\epsilon_{\rm rel}^{\rm (R)}` in Eq. :eq:`Eq_residual_criterion`,
            i.e. the relative tolerance on the residuals at a given iteration
        max\_iter
            number of maximum iterations allowed. If the solver does not converge in max\_iter iterations,
            it will stop and print a warning; however, it will still return the output and it will not throw an
            exception
        criterion
            `'change'` or `'residual'` (default: `'residual'`): convergence criterion
        norm\_change
            norm used in the evaluation of the `'change'` criterion (default: `'linf'`, i.e. maximum
            absolute value at nodes)
        norm\_res
            norm used in the evaluation of the `'residual'` criterion (default: `'linf'`, i.e. maximum
            absolute value at nodes)
        
    
    """

    
    def __init__( self, fem, source, fields,
                  Mn=None, Mf1=None, Mf2=None,
                  abs_du_tol=1e-8, rel_du_tol=1e-8, abs_res_tol=1e-10, rel_res_tol=1e-10,
                  max_iter=100, criterion='residual', norm_change='linf', norm_res='linf' ):
        
        """The constructor"""
        
        Solver.__init__( self, fem, source, fields, Mn, abs_du_tol, rel_du_tol,
                         abs_res_tol, rel_res_tol, max_iter, criterion, norm_change, norm_res )

        # vector finite element and function space for rescaled (phi, h, y, z)
        self.E = d.MixedElement([ self.fem.Pn, self.fem.Pn, self.fem.Pn, self.fem.Pn ])
        self.V = d.FunctionSpace( self.fem.mesh.mesh, self.E )
        # (discontinuous) vector finite element and function space for the four strong residuals
        self.dE = d.MixedElement([ self.fem.dPn, self.fem.dPn, self.fem.dPn, self.fem.dPn ])
        self.dV = d.FunctionSpace( self.fem.mesh.mesh, self.dE )

        # fields' rescalings
        if Mf1 is None:
            self.Mf1 = self.source.Ms * self.Mn / self.fields.Mp
        else:
            self.Mf1 = Mf1
            
        if Mf2 is None:
            if ( self.fields.alpha != 0. ):
                self.Mf2 = abs(self.fields.alpha) * self.Mf1
            else:
                self.Mf2 = self.Mf1
        else:
            self.Mf2 = Mf2

        # solution and field profiles (computed by the solver)
        self.u = None
        self.phi, self.h, self.y, self.z = None, None, None, None
        self.Phi, self.H, self.Y, self.Z = None, None, None, None

        # gradients of the scalar fields
        self.grad_phi = None # rescaled
        self.grad_Phi = None # physical
        self.grad_h = None
        self.grad_H = None

        # phi scalar force (physical units)
        self.force = None
        

    

    

    def get_Dirichlet_bc( self ):
        r"""
        Returns the Dirichlet boundary conditions for the UV theory:

        .. math:: \hat{\phi}(\hat{r}_{\rm max}) = \hat{H}(\hat{r}_{\rm max}) = 
                  \hat{Y}(\hat{r}_{\rm max}) = \hat{Z}(\hat{r}_{\rm max}) = 0
        
        The Neumann boundary conditions
        
        .. math:: \frac{d\hat{\phi}}{d\hat{r}}(0) = \frac{d\hat{H}}{d\hat{r}}(0) = 0

        are natural boundary conditions in the finite element method, and are implemented within
        the variational formulation.

        """

        # mixed systems require vector boundary conditions
        # see https://fenicsproject.org/pub/tutorial/html/._ftut1010.html#ftut1:reactionsystem

        # define values at infinity
        # for 'infinity', we use the last mesh point, i.e. r_max (i.e. mesh[-1]) 
        phiD = d.Constant( 0. )
        hD = d.Constant( 0. )
        yD = d.Constant( 0. )
        zD = d.Constant( 0. )

        # define 'infinity' boundary: the rightmost mesh point - within machine precision
        def boundary(x):
            return self.fem.mesh.r_max - x[0] < d.DOLFIN_EPS

        bc_phi = d.DirichletBC( self.V.sub(0), phiD, boundary, method='pointwise' )
        bc_h = d.DirichletBC( self.V.sub(1), hD, boundary, method='pointwise' )
        bc_y = d.DirichletBC( self.V.sub(2), yD, boundary, method='pointwise' )
        bc_z = d.DirichletBC( self.V.sub(3), zD, boundary, method='pointwise' )

        Dirichlet_bc = [ bc_phi, bc_h, bc_y, bc_z ]

        return Dirichlet_bc




    def initial_guess( self ):
        r"""
        Obtains an initial guess for the Newton solver.

        This is done by solving the equation of motion for :math:`\lambda=0`, which form a linear system.
        All other parameters are unchanged.

        """

        # cast params as constant functions so that, if they are set to 0, 
        # fenics still understand what it is integrating
        m, M, Mp = Constant( self.fields.m ), Constant( self.fields.M ), Constant( self.fields.Mp )
        alpha = Constant( self.fields.alpha )
        Mn, Mf1, Mf2 = Constant( self.Mn ), Constant( self.Mf1 ), Constant( self.Mf2 )
        
        # get the boundary conditions
        Dirichlet_bc = self.get_Dirichlet_bc()
        
        # create a vector (phi,h) with the two trial functions for the fields
        u = d.TrialFunction(self.V)
        # ... and split it into phi and h
        phi, h, y, z = d.split(u)
        
        # define test functions over the function space
        v1, v2, v3, v4 = d.TestFunctions(self.V)

        # r^2
        r2 = Expression('pow(x[0],2)', degree=self.fem.func_degree)

        # define bilinear form    
        # Eq.1
        a1 = y * v1 * r2 * dx - ( m / Mn )**2 * phi * v1 * r2 * dx \
             - alpha * ( Mf2/Mf1 ) * z * v1 * r2 * dx
        
        # Eq.2
        a2 = z * v2 * r2 * dx - ( M / Mn )**2 * h * v2 * r2 * dx \
             - alpha * ( Mf1/Mf2 ) * y * v2 * r2 * dx
        
        a3 = - inner( grad(phi), grad(v3) ) * r2 * dx - y * v3 * r2 * dx
        
        a4 = - inner( grad(h), grad(v4) ) * r2 * dx - z * v4 * r2 * dx
        
        # both equations
        a = a1 + a2 + a3 + a4
        
        # define linear form
        L = self.source.rho / Mp * Mn / Mf1 * v1 * r2 * dx

        # define a vector with the solution
        sol = d.Function(self.V)
        
        # solve linearised system
        pde = d.LinearVariationalProblem( a, L, sol, Dirichlet_bc )
        solver = d.LinearVariationalSolver( pde )
        solver.solve()

        return sol




    

    def weak_residual_form( self, sol ):
        r"""
        Computes the residual with respect to the weak form of the equations:

        .. math:: F = F_1 + F_2 + F_3 + F_4

        with

        .. math:: &F_1(\hat{\phi},\hat{H},\hat{Y},\hat{Z}) = \int\hat{Y} v_1 \hat{r}^2 d\hat{r} 
                  - \left(\frac{m}{M_n}\right)^2 \int\hat{\phi} v_1 \hat{r}^2 d\hat{r}
                  - \alpha \frac{M_{f2}}{M_{f1}} \int \hat{Z} v_1 \hat{r}^2 d\hat{r}
                  - \frac{M_n}{M_{f1}} \int \frac{\hat{\rho}}{M_p} v_1 \hat{r}^2 d\hat{r} \\

                  &F_2(\hat{\phi},\hat{H},\hat{Y},\hat{Z}) = \int \hat{Z} v_2 \hat{r}^2 d\hat{r} 
                  - \left( \frac{M}{M_n} \right)^2 \int \hat{H} v_2 \hat{r}^2 d\hat{r}
                  - \alpha \frac{M_{f1}}{M_{f2}} \int \hat{Y} v_2 \hat{r}^2 d\hat{r}
                  - \frac{\lambda}{6} \left( \frac{M_{f2}}{M_n} \right)^2 \int \hat{H}^3 v_2 \hat{r}^2 d\hat{r} \\

                  &F_3(\hat{\phi},\hat{H},\hat{Y},\hat{Z}) = - \int \hat{\nabla} \hat{\phi} \cdot 
                  \hat{\nabla} v_3 \hat{r}^2 d\hat{r} - \hat{Y} v_3 \hat{r}^2 d\hat{r} \\
        
                  &F_4(\hat{\phi},\hat{H},\hat{Y},\hat{Z}) = - \int \hat{\nabla} \hat{H} \cdot
                  \hat{\nabla} v_4 \hat{r}^2 d\hat{r} - \hat{Z} v_4 \hat{r}^2 d\hat{r}

        The weak residual is employed within :func:`solver.Solver.solve` to check convergence -
        also see :func:`solver.Solver.compute_errors`.

        *Parameters*
            sol
                the solution with respect to which the weak residual is computed.

        """
        
        # define test functions over the function space
        v1, v2, v3, v4 = d.TestFunctions(self.V)

        # cast params as constant functions so that, if they are set to 0, 
        # fenics still understand what it is integrating
        m, M, Mp = Constant( self.fields.m ), Constant( self.fields.M ), Constant( self.fields.Mp )
        alpha, lam = Constant( self.fields.alpha ), Constant( self.fields.lam )
        Mn, Mf1, Mf2 = Constant( self.Mn ), Constant( self.Mf1 ), Constant( self.Mf2 )

        # split solution into phi and h
        phi, h, y, z = d.split( sol )
        
        # r^2
        r2 = Expression('pow(x[0],2)', degree=self.fem.func_degree)

        # define the weak residual form

        # Eq 1
        F1 = y * v1 * r2 * dx - (m/Mn)**2 * phi * v1 * r2 * dx \
             - alpha * (Mf2/Mf1) * z * v1 * r2 * dx - self.source.rho / Mp * Mn / Mf1 * v1 * r2 * dx

        # Eq 2
        F2 = z * v2 * r2 * dx - (M/Mn)**2 * h * v2 * r2 * dx \
             - alpha * (Mf1/Mf2) *  y * v2 * r2 * dx - lam/6. * (Mf2/Mn)**2 * h**3 * v2 * r2 * dx

        # Eq 3
        F3 = - inner( grad(phi), grad(v3) ) * r2 * dx - y * v3 * r2 * dx

        # Eq 4
        F4 = - inner( grad(h), grad(v4) ) * r2 * dx - z * v4 * r2 * dx
        
        # all equations
        F = F1 + F2 + F3 + F4

        return F

        

        


    def strong_residual_form( self, sol, units ):
        r"""
        Computes the residual with respect to the strong form of the equations.

        The total residual is obtained by summing the residuals of all equations:


        .. math:: F = F_1 + F_2 + F_3 + F_4

        where, in dimensionless in-code units (`units='rescaled'`):
        
        .. math:: & F_1(\hat{\phi},\hat{H},\hat{Y},\hat{Z}) = \hat{Y}
                  - \left( \frac{m}{M_n} \right)^2\hat{\phi}
                  - \alpha \frac{M_{f2}}{M_{f1}} \hat{Z} - \frac{\hat{\rho}}{M_p}\frac{M_n}{M_{f1}}

                  & F_2(\hat{\phi},\hat{H},\hat{Y},\hat{Z}) = \hat{Z} 
                  - \left( \frac{M}{M_n} \right)^2 \hat{H}  
                  - \alpha \frac{M_{f1}}{M_{f2}} \hat{Y} 
                  - \frac{\lambda}{6} \left( \frac{M_{f2}}{M_n} \right)^2 \hat{H}^3

                  & F_3(\hat{\phi},\hat{H},\hat{Y},\hat{Z}) = \hat{\nabla}^2\hat{\phi} - \hat{Y}

                  & F_4(\hat{\phi},\hat{H},\hat{Y},\hat{Z}) = \hat{\nabla}^2\hat{H} - \hat{Z}


        and in physical units (`units='physical'`):

        .. math:: & F_1(\phi,H,Y,Z) = Y - m^2 \phi -\alpha H - \frac{\rho}{M_P}

                  & F_2(\phi,H,Y,Z) = Z - M^2 H - \alpha\phi - \frac{\lambda}{6} H^3

                  & F_3(\phi,H,Y,Z) = \nabla^2\phi - Y
        
                  & F_4(\phi,H,Y,Z) = \nabla^2 H - Z



        .. note:: In this function, the Laplacian :math:`\hat{\nabla}^2` is obtained by projecting
                  :math:`\frac{\partial^2}{\partial\hat{r}^2} + 2\frac{\partial}{\partial\hat{r}}`.
                  As such, it should not be used with interpolating polynomials of degree less than 2.

        Note that the weak residual in :func:`weak_residual_form` is just the scalar product
        of the strong residuals by test functions.

        *Parameters*
            sol
                the solution with respect to which the weak residual is computed.
            units
                `'rescaled'` (for the rescaled units used inside the code) or `'physical'`, for physical units

        """

        if units=='rescaled':
            resc_13 = 1.
            resc_24 = 1.
        elif units=='physical':
            resc_13 = self.Mn**2 * self.Mf1
            resc_24 = self.Mn**2 * self.Mf2
        else:
            message = "Invalid choice of units: valid choices are 'physical' or 'rescaled'."
            raise ValueError(message)

        # cast params as constant functions so that, if they are set to 0, 
        # fenics still understand what it is integrating        
        m, M, Mp = Constant( self.fields.m ), Constant( self.fields.M ), Constant( self.fields.Mp )
        alpha, lam = Constant( self.fields.alpha ), Constant( self.fields.lam )
        Mn, Mf1, Mf2 = Constant( self.Mn ), Constant( self.Mf1 ), Constant( self.Mf2 )

        # split solution in phi, h, y, z
        phi, h, y, z = d.split( sol )

        # initialise residual function
        F = d.Function( self.dV )

        # define r for use in the computation of the Laplacian
        r = Expression( 'x[0]', degree=self.fem.func_degree )
        
        # equation 1
        f1 = y - ( m/Mn )**2 * phi - alpha * ( Mf2/Mf1 ) * z - self.source.rho / Mp * Mn / Mf1
        f1 *= resc_13
        F1 = project( f1, self.fem.dS, self.fem.func_degree )
        # equation 2
        f2 = z - ( M/Mn )**2 * h - alpha * ( Mf1/Mf2 ) * y - lam / 6. * ( Mf2/Mn )**2 * h**3
        f2 *= resc_24
        F2 = project( f2, self.fem.dS, self.fem.func_degree )
        # equation 3  - I expand manually the Laplacian into 2/r df/dr + d2f/dr2
        f3 = Constant(2.)/r * phi.dx(0) + phi.dx(0).dx(0) - y
        f3 *= resc_13
        F3 = project( f3, self.fem.dS, self.fem.func_degree )
        # equation 4 
        f4 = Constant(2.)/r * h.dx(0) + h.dx(0).dx(0) - z
        f4 *= resc_24
        F4 = project( f4, self.fem.dS, self.fem.func_degree )

        # combine equations
        fa = d.FunctionAssigner( self.dV, [ self.fem.dS, self.fem.dS, self.fem.dS, self.fem.dS ])
        fa.assign(F, [F1, F2, F3, F4])

        return F




    def linear_solver( self, u_k ):
        r"""
        Solves the (linear) Newton iteration Eq. :eq:`Eq_linear_solver` for the UV theory.

        For the UV theory, the form of the Newton iteration is:

        .. math::  & \int \hat{Y} v_1 \hat{r}^2 d\hat{r}
                   - \left(\frac{m}{M_n}\right)^2 \int \hat{\phi} v_1 \hat{r}^2 d\hat{r}
                   - \alpha \int \hat{Z} v_1 \hat{r}^2 d\hat{r} + 

                   & + \int \hat{Z} v_2 \hat{r}^2 d\hat{r}
                   - \left(\frac{M}{M_n}\right)^2 \int \hat{H} v_2 \hat{r}^2 d\hat{r}
                   - \alpha \int \hat{Y} v_2 \hat{r}^2 d\hat{r} 
                   - \frac{\lambda}{2} \int {\hat{H}_k}^2 \hat{H} v_2 \hat{r}^2 d\hat{r} +

                   & - \int \hat{\nabla}\hat{\phi} \cdot \hat{\nabla} v_3 \hat{r}^2 d\hat{r}
                   - \int \hat{Y} v_3 \hat{r}^2 d\hat{r}
                   - \int \hat{\nabla}\hat{H} \cdot \hat{\nabla} v_4 \hat{r}^2 d\hat{r}
                   - \int \hat{Z} v_4 \hat{r}^2 d\hat{r} = 

                   & = \frac{M_n}{M_{f1}} \int \frac{\hat{\rho}}{M_P} v_1 \hat{r}^2 d\hat{r}
                   - \int \frac{\lambda}{3} {\hat{H}_k}^3 v_2 \hat{r}^2 d\hat{r}


        *Arguments*
            u_k  
                solution at the previous iteration

        """
        
        # get the boundary conditions
        Dirichlet_bc = self.get_Dirichlet_bc()
        
        # create a vector (phi,h) with the two trial functions for the fields
        u = d.TrialFunction(self.V)
        # ... and split it into phi and h
        phi, h, y, z = d.split(u)
        
        # define test functions over the function space
        v1, v2, v3, v4 = d.TestFunctions(self.V)
        
        # split solution at current iteration into phi_k, h_k, y_k, z_k
        phi_k, h_k, y_k, z_k = d.split(u_k)

        # cast params as constant functions so that, if they are set to 0, 
        # fenics still understand what it is integrating        
        m, M, Mp = Constant( self.fields.m ), Constant( self.fields.M ), Constant( self.fields.Mp )
        alpha, lam = Constant( self.fields.alpha ), Constant( self.fields.lam )
        Mn, Mf1, Mf2 = Constant( self.Mn ), Constant( self.Mf1 ), Constant( self.Mf2 )
        
        # r^2   
        r2 = Expression('pow(x[0],2)', degree=self.fem.func_degree)

        # define bilinear form    
        # Eq.1
        a1 = y * v1 * r2 * dx - ( m / Mn )**2 * phi * v1 * r2 * dx \
             - alpha * ( Mf2/Mf1 ) * z * v1 * r2 * dx
        
        # Eq.2
        a2 = z * v2 * r2 * dx - ( M / Mn )**2 * h * v2 * r2 * dx \
             - alpha * ( Mf1/Mf2 ) * y * v2 * r2 * dx \
             - lam/2. * ( Mf2 / Mn )**2 * h_k**2 * h * v2 * r2 * dx

        # Laplacian of phi
        a3 = - inner( grad(phi), grad(v3) ) * r2 * dx - y * v3 * r2 * dx

        # Laplacian of H
        a4 = - inner( grad(h), grad(v4) ) * r2 * dx - z * v4 * r2 * dx
        
        # all equations
        a = a1 + a2 + a3 + a4
        
        # define linear form
        L1 = self.source.rho / Mp * Mn / Mf1 * v1 * r2 * dx          # Eq.1
        L2 = - lam / 3. * ( Mf2 / Mn )**2 * h_k**3 * v2 * r2 * dx    # Eq.2
        L = L1 + L2                             # all equations
        
        # define a vector with the solution
        sol = d.Function(self.V)
        
        # solve linearised system
        pde = d.LinearVariationalProblem( a, L, sol, Dirichlet_bc )
        solver = d.LinearVariationalSolver( pde )
        solver.solve()

        return sol




    def compute_physics( self, sol ):
        """
        Computes field profiles and gradients in physical units, starting from the dimensionless
        profiles that are obtained solving the equations of motion with :ref:`solve`.
        Then computes the scalar force associated to :math:`\phi` (physical units only).

        """
        
        # solution and profiles - rescaled units
        self.u = sol
        self.phi, self.h, self.y, self.z = sol.split(deepcopy=True)
        
        # profiles - physical units
        self.Phi = d.Function( self.fem.S )
        self.H = d.Function( self.fem.S )
        self.Y = d.Function( self.fem.S )
        self.Z = d.Function( self.fem.S )
        self.Phi.vector()[:] = self.Mf1 * self.phi.vector()[:]
        self.H.vector()[:] = self.Mf2 * self.h.vector()[:]
        self.Y.vector()[:] = self.Mn**2 * self.Mf1 * self.y.vector()[:]
        self.Z.vector()[:] = self.Mn**2 * self.Mf2 * self.z.vector()[:]

        # gradients of the scalar fields
        self.grad_phi = self.grad( self.phi, 'rescaled' )
        self.grad_Phi = self.grad( self.Phi, 'physical' )
        self.grad_h = self.grad( self.h, 'rescaled' )
        self.grad_H = self.grad( self.H, 'physical' )

        # scalar force - physical
        self.force = self.scalar_force( self.Phi )                    








    def output_term( self, eqn=1, term='LHS', norm='none', units='rescaled', output_label=False ):
        r"""
        Outputs the different terms in the UV system of equations.

        The terms can be output in physical or dimensionless in-code units, by choosing either
        `units='physical'` or `units='rescaled'`.

        Possible choices of terms are listed in the tables below; recall that :math:`Y\equiv\nabla^2\phi`
        and :math:`Z\equiv\nabla^2 H`.

        - `eqn=1`:

        ==============   ===============================================================================================   ==============
         `units=`         `term='LHS'`                                                                                      `term='RHS'`
        ==============   ===============================================================================================   ==============
         `'rescaled'`     :math:`\hat{Y} - \left(\frac{m}{M_n}\right)^2\hat{\phi} - \frac{M_{f2}}{M_{f1}}\alpha\hat{Z}`             :math:`\frac{\hat{\rho}}{M_P}\frac{Mn}{M_{f1}}`
         `'physical'`     :math:`Y -m^2 \phi -\alpha Z`                                                                             :math:`\frac{\rho}{M_P}`
        ==============   ===============================================================================================   ==============



        ==============   =================   =================================================   =============================================   =================
         `units=`         `term=1`            `term=2`                                            `term=3`                                        `term=4`
        ==============   =================   =================================================   =============================================   =================
         `'rescaled'`     :math:`\hat{Y}`     :math:`-\left(\frac{m}{M_n}\right)^2\hat{\phi}`     :math:`-\frac{M_{f2}}{M_{f1}}\alpha\hat{Z}`     same as `'RHS'`
         `'physical'`     :math:`Y`           :math:`-m^2\phi`                                    :math:`-\alpha Z`                               same as `'RHS'`
        ==============   =================   =================================================   =============================================   =================


        
        - `eqn=2`:


        ==============   ===========================================================================================   ==========================================================================
         `units=`         `term='LHS'`                                                                                  `term='RHS'`
        ==============   ===========================================================================================   ==========================================================================
         `'rescaled'`    :math:`\hat{Z} -\left(\frac{M}{M_n}\right)^2\hat{H} - \alpha\frac{M_{f1}}{M_{f2}}\hat{Y}`      :math:`-\frac{\lambda}{6} \left( \frac{M_{f2}}{M_n} \right)^2 \hat{H}^3`
         `'physical'`     :math:`Z -M^2 H -\alpha Y`                                                                    :math:`-\frac{\lambda}{6} H^3`
        ==============   ===========================================================================================   ==========================================================================



        ==============   =================   =============================================   =============================================   =================
         `units=`         `term=1`            `term=2`                                            `term=3`                                        `term=4`
        ==============   =================   =============================================   =============================================   =================
         `'rescaled'`     :math:`\hat{Z}`     :math:`-\left(\frac{M}{Mn}\right)^2\hat{H}`     :math:`-\alpha\frac{M_{f1}}{M_{f2}}\hat{Y}`     same as `'RHS'`
         `'physical'`     :math:`Z`           :math:`-M^2 H`                                  :math:`-\alpha Y`                               same as `'RHS'`
        ==============   =================   =============================================   =============================================   =================


            Although the right-hand-side (RHS) of the equation of motion for :math:`H` is 0,
            this choice allows better comparison of the scale of the equation against the residuals.



        - `eqn=3`:

        ==============   ==========================================   ==================================   =================
         `units=`         `term='LHS'`                                 `term=1`                             `term=2`
        ==============   ==========================================   ==================================   =================
         `'rescaled'`     :math:`\hat{\nabla}^2\hat{\phi}-\hat{Y}`     :math:`\hat{\nabla}^2\hat{\phi}`     :math:`\hat{Y}`
         `'physical'`     :math:`\nabla^2\phi - Y`                     :math:`\nabla^2\phi`                 :math:`Y`
        ==============   ==========================================   ==================================   =================
     


        - `eqn=4`:

        ==============   =======================================   ==================================   =================
         `units=`         `term='LHS'`                              `term=1`                             `term=2`
        ==============   =======================================   ==================================   =================
         `'rescaled'`     :math:`\hat{\nabla}^2\hat{H}-\hat{Z}`     :math:`\hat{\nabla}^2\hat{H}`        :math:`\hat{Z}`
         `'physical'`     :math:`\nabla^2 H - Z`                    :math:`\nabla^2 H`                   :math:`Z`
        ==============   =======================================   ==================================   =================
     


        Equations (1) and (2) allow to look at the terms in the equations of motion,
        Eq. (3) and (4) enforce consistency relations :math:`Y = '\nabla^2\phi` and :math:`Z = \nabla^2 H`. 

        .. note:: In this method, the Laplacian :math:`\hat{\nabla}^2` is obtained by projecting
                  :math:`\frac{\partial^2}{\partial\hat{r}^2} + 2\frac{\partial}{\partial\hat{r}}`.
                  As such, it should not be used with interpolating polynomials of degree less than 2.
            
        *Parameters*
            eqn
                (`integer`) choice of equation
            term
                choice of `LHS`, `RHS` or a number; for `eqn=1,2`, valid terms range from 1 to 4, for
                `eqn=3,4`, valid terms range from 1 to 2
            norm 
                `'L2'`, `'linf'` or `'none'`. If `'L2'` or `'linf'`: compute the :math:`L_2` or
                :math:`\ell_{\infty}` norm of the residual; if `'none'`, return the full
                term over the box - as opposed to its norm.
            units
                `rescaled` (default) or `physical`; choice of units for the output
            output_label
                if `True`, output a string with a label for the term (which can be used, e.g. in plot legends)
            
        """
        
        # cast params as constant functions so that, if they are set to 0, FEniCS still understand
        # what is being integrated
        m, M, Mp =  Constant( self.fields.m ), Constant( self.fields.M ), Constant( self.fields.Mp )
        alpha, lam = Constant( self.fields.alpha ), Constant( self.fields.lam )
        Mn, Mf1, Mf2 = Constant( self.Mn ), Constant( self.Mf1 ), Constant( self.Mf2 )
        
        if units=='rescaled':
            resc_13 = 1.
            resc_24 = 1.
            str_phi, str_h, str_y, str_z = '\\hat{\\phi}', '\\hat{H}', '\\hat{Y}', '\\hat{Z}'
            str_nabla2 = '\\hat{\\nabla}^2'
            str_coup_z = '\\alpha \\frac{M_{f2}}{M_{f1}} \\hat{Z}'
            str_coup_y = '\\alpha \\frac{M_{f1}}{M_{f2}} \\hat{Y}'
            str_m2 = '\\left( \\frac{m}{M_n} \\right)^2'
            str_M2 = '\\left( \\frac{M}{M_n} \\right)^2'
            str_nl = '\\frac{\\lambda}{6} \\frac{M_{f2}}{M_n} \\hat{H}^3'
            str_rho = '\\frac{\hat{\\rho}}{M_p}\\frac{M_n}{M_{f1}}'
                
        elif units=='physical':
            resc_13 = self.Mn**2 * self.Mf1
            resc_24 = self.Mn**2 * self.Mf2
            str_phi, str_h, str_y, str_z = '\\phi', 'H', 'Y', 'Z'
            str_nabla2 = '\\nabla^2'
            str_coup_z = '\\alpha Z'
            str_coup_y = '\\alpha Y'
            str_m2 = 'm^2'
            str_M2 = 'M^2'
            str_nl = '\\frac{\\lambda}{6} \\hat{H}^3'
            str_rho = '\\frac{\\rho}{M_p}'
            
        else:
            message = "Invalid choice of units: valid choices are 'physical' or 'rescaled'."
            raise ValueError(message)
        
        # split solution in phi, h, y, z
        phi, h, y, z = self.phi, self.h, self.y, self.z
        
        # define r for use in the computation of the Laplacian
        r = Expression( 'x[0]', degree=self.fem.func_degree )
        
        if eqn==1:
            if term=='LHS':
                # I expand manually the Laplacian into 2/r df/dr + d2f/dr2
                Term = y - ( m/Mn )**2 * phi - alpha * ( Mf2/Mf1 ) * z
                label = r"$%s - %s%s - %s$" % ( str_y, str_m2, str_phi, str_coup_z )
            elif term=='RHS':
                Term =  self.source.rho / Mp * Mn / Mf1
                label = r"$%s$" % str_rho
            elif term==1:
                Term = y
                label = r"$%s$" % str_y
            elif term==2:
                Term = - ( m/Mn )**2 * phi
                label = r"$-%s%s$" % ( str_m2, str_phi )
            elif term==3:
                Term = - alpha * ( Mf2/Mf1 ) * z
                label = r"$-%s$" % str_coup_z
            elif term==4:
                Term =  self.source.rho / Mp * Mn / Mf1
                label = r"$%s$" % str_rho
            # rescale if needed to get physical units
            Term *= resc_13
        
            
        elif eqn==2:
            if term=='LHS':
                Term = z - ( M/Mn )**2 * h - alpha * ( Mf1/Mf2 ) * y
                label = r"$%s - %s%s -%s$" % ( str_z, str_M2, str_h, str_coup_y )
            elif term=='RHS':
                Term = lam / 6. * ( Mf2/Mn )**2 * h**3
                label = r"$%s$" % str_nl
            elif term==1:
                Term = z
                label = r"$%s$" % str_z
            elif term==2:
                Term = - ( M/Mn )**2 * h
                label = r"$%s%s$" % ( str_M2, str_h )
            elif term==3:
                Term = - alpha * ( Mf1/Mf2 ) * y
                label = r"$-%s$" % str_coup_y
            elif term==4:
                Term = lam / 6. * ( Mf2/Mn )**2 * h**3
                label = r"$%s$" % str_nl 
            # rescale if needed to get physical units
            Term *= resc_24

            
        elif eqn==3:
            # consistency of y = Del phi
            if term=='LHS':
                Term = Constant(2.)/r * phi.dx(0) + phi.dx(0).dx(0) - y
                label = r"$%s%s - %s$" % ( str_nabla2, str_phi, str_y )
            elif term==1:
                Term = Constant(2.)/r * phi.dx(0) + phi.dx(0).dx(0)
                label = r"$%s%s$" % ( str_nabla2, str_phi )
            elif term==2:
                Term = y
                label = r"$%s$" % str_y
            # rescale if needed to get physical units
            Term *= resc_13
            
            
        elif eqn==4:
            # consistency of z = Del H
            if term=='LHS':
                Term = Constant(2.)/r * h.dx(0) + h.dx(0).dx(0) - z
                label = r"$%s%s - %s$" % ( str_nabla2, str_h, str_z )
            elif term==1:
                Term = Constant(2.)/r * h.dx(0) + h.dx(0).dx(0)
                label = r"$%s%s$" % ( str_nabla2, str_h )
            elif term==2:
                Term = z
                label = r"$%s$" % str_z
            # rescale if needed to get physical units
            Term *= resc_24
            
            
        Term_func = project( Term, self.fem.dS, self.fem.func_degree )
        
        
        # 'none' = return function, not norm
        if norm=='none':
            result = Term_func
            # from here on return a norm. This nested if is to preserve the structure of the original
            # built-in FEniCS norm function
        elif norm=='linf':
            # infinity norm, i.e. max abs value at vertices    
            result = r2_norm( Term_func.vector(), self.fem.func_degree, norm_type=norm )
        else:
            result = r2_norm( Term_func, self.fem.func_degree, norm_type=norm )

        if output_label:
            return result, label
        else:
            return result

