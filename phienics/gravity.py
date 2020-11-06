"""
.. module:: gravity

This module contains a solver for the Poisson equation.

"""

# Copyright (C) Daniela Saadeh 2020
# This file is part of phi-enics

from phienics.solver import Solver
from phienics.utils import project

from dolfin import Expression, Constant, inner, grad, dx

import dolfin as d



class PoissonSolver(Solver):
    r"""
    Computes the Newtonian potential :math:`\Phi_N` by solving the Poisson equation:

    .. math:: \nabla^2 \Phi_N = \frac{\rho}{2 {M_P}^2}
              :label: Eq_Poisson

    where :math:`\rho` the source and :math:`M_P` the Planck mass.

    """

    def __init__( self, fem, source, fields=None, Mp=1.,
                  Mn=None, abs_du_tol=None, rel_du_tol=None, abs_res_tol=None, rel_res_tol=None,
                  max_iter=None, criterion=None, norm_change=None, norm_res=None ):


        # the nonlinear machinery is not needed here, but it's useful to use the functionalities of the Solver class
        Solver.__init__( self, fem, source, fields, Mn, abs_du_tol, rel_du_tol,
                         abs_res_tol, rel_res_tol, max_iter, criterion, norm_change, norm_res )

        # Planck mass
        self.Mp = Mp

        # Newtonian potential (physical units)
        self.PhiN = None

        # Newton force (physical units)
        self.Newton_force = None

        




    def get_Dirichlet_bc( self ):
        r"""
        Returns the Dirichlet boundary conditions for gravity (Eq. :eq:`Eq_Poisson`):

        .. math:: \Phi_N(\hat{r}_{\rm max}) = 0
        
        """
        
        # boundary conditions
        def boundary(x):
            return self.fem.mesh.r_max - x[0] < d.DOLFIN_EPS
        PhiND = d.Constant( 0. )
        Dirichlet_bc = d.DirichletBC( self.fem.S, PhiND, boundary, method='pointwise' )
        
        return Dirichlet_bc




    def solve( self ):
        """
        Overrides the base-class nonlinear solver with a linear solver for the Poisson
        equation Eq. :eq:`Eq_Poisson`.
        """
        
        Dirichlet_bc = self.get_Dirichlet_bc()
        
        # trial and test function
        u = d.TrialFunction( self.fem.S )
        v = d.TestFunction( self.fem.S )
        PhiN = d.Function( self.fem.S )
        
        # define equation
        Mn, Mp = Constant( self.Mn ), Constant( self.Mp )
        r2 = Expression( 'pow(x[0],2)', degree=self.fem.func_degree )
        a = - inner( grad(u), grad(v) ) * r2 * dx
        L = 0.5 * self.source.rho * Mn / Mp**2 * v * r2 * dx
        
        # solve
        eqn = d.LinearVariationalProblem( a, L, PhiN, Dirichlet_bc )
        solver = d.LinearVariationalSolver( eqn )
        solver.solve()

        # set potential and force
        self.PhiN = PhiN
        self.Newton_force = self.grad( -PhiN, radial_units='physical' )



    def strong_residual_form( self, sol, units ):
        r"""
        Computes the residual :math:`F` with respect to the strong form of the Poisson equation Eq. :eq:`Eq_Poisson`.

        In dimensionless in-code units (`units='rescaled'`) it is:

        .. math:: F = \hat{\nabla}^2 \Phi_N - \frac{\hat{\rho} M_n}{2 {M_P}^2}

        and in physical units (`units='physical'`):

        .. math:: F = \nabla^2 \Phi_N - \frac{\rho}{2 {M_P}^2}

        .. note:: In this function, the Laplacian :math:`\hat{\nabla}^2` is obtained by projecting
                  :math:`\frac{\partial^2}{\partial\hat{r}^2} + 2\frac{\partial}{\partial\hat{r}}`.
                  As such, it should not be used with interpolating polynomials of degree less than 2.

        *Parameters*
            sol
                the solution with respect to which the weak residual is computed.
            units
                `'rescaled'` (for the rescaled units used inside the code) or `'physical'`, for physical units

        """
        
        Mn, Mp = Constant( self.Mn ), Constant( self.Mp )
        
        if units=='rescaled':
            resc = 1.
        elif units=='physical':
            resc = Mn**2
        else:
            message = "Invalid choice of units: valid choices are 'physical' or 'rescaled'."
            raise ValueError(message)

        # define r for use in the computation of the Laplacian
        r = Expression( 'x[0]', degree=self.fem.func_degree )
        
        f = ( sol.dx(0).dx(0) + Constant(2.) / r * sol.dx(0) ) - 1. / (2 * Mp**2) * self.source.rho * Mn
        f *= resc

        F = project( f, self.fem.dS, self.fem.func_degree )

        return F





    def output_term( self, term='LHS', norm='none', units='rescaled', output_label=False ):
        r"""
        Outputs the left- and right-hand side of the Poisson equation Eq. :eq:`Eq_Poisson`.

        The terms can be output in physical or dimensionless in-code units, by choosing either
        `units='physical'` or `units='rescaled'`:

        ==============   ==============================   ===============================================
         `units=`         `term='LHS'`                    `term='RHS'`
        ==============   ==============================   ===============================================
         `'rescaled'`     :math:`\hat{\nabla}^2\Phi_N`    :math:`\frac{1}{2} \frac{\hat{\rho}}{M_P} M_n`
         `'physical'`     :math:`\nabla^2 \Phi_N`         :math:`\frac{\rho}{2 {M_P}^2}`
        ==============   ==============================   ===============================================

        where :math:`\Phi_N` is the Newtonian potential, :math:`\rho` the source and :math:`M_P` the 
        Planck mass.

        .. note:: In this method, the Laplacian :math:`\hat{\nabla}^2` is obtained by projecting
                  :math:`\frac{\partial^2}{\partial\hat{r}^2} + 2\frac{\partial}{\partial\hat{r}}`.
                  As such, it should not be used with interpolating polynomials of degree less than 2.
            
        *Parameters*
            term
                choice of `'LHS'` and `'RHS'` for the left- and right-hand side of the Poisson equation
            norm 
                `'L2'`, `'linf'` or `'none'`. If `'L2'` or `'linf'`: compute the :math:`L_2` or
                :math:`\ell_{\infty}` norm of the residual; if `'none'`, return the full
                term over the box - as opposed to its norm.
            units
                `rescaled` (default) or `physical`; choice of units for the output
            output_label
                if `True`, output a string with a label for the term (which can be used, e.g. in plot legends)
            
        """

        Mp, Mn =  Constant( self.Mp ), Constant( self.Mn )
        
        if units=='rescaled':
            resc = 1.
            str_nabla2 = '\\hat{\\nabla}^2'
            str_rho = '\\frac{1}{2} \\frac{\\hat{\\rho}}{{M_p}^2} M_n'
            
        elif units=='physical':
            resc = self.Mn**2
            str_nabla2 = '\\nabla^2'
            str_rho = '\\frac{\\rho}{2 {M_p}^2}'
            
        else:
            message = "Invalid choice of units: valid choices are 'physical' or 'rescaled'."
            raise ValueError(message)
            
        str_PhiN = '\\Phi_N'
        
        # define r for use in the computation of the Laplacian
        r = Expression( 'x[0]', degree=self.fem.func_degree )

        if term=='LHS':
            term = self.PhiN.dx(0).dx(0) + Constant(2.) / r * self.PhiN.dx(0)
            label = r"$%s %s$" % ( str_nabla2, str_PhiN )
        elif term=='RHS':
            term = 1. / (2 * Mp**2) * self.source.rho * Mn
            label = r"$%s$" % str_rho

        # rescale if needed to get physical units
        term *= resc

        term = project( term, self.fem.dS, self.fem.func_degree )

        # 'none' = return function, not norm
        if norm=='none':
            result = term
            # from here on return a norm. This nested if is to preserve the structure of the original
            # built-in FEniCS norm function
        elif norm=='linf':
            # infinity norm, i.e. max abs value at vertices    
            result = r2_norm( term.vector(), self.fem.func_degree, norm_type=norm )
        else:
            result = r2_norm( term, self.fem.func_degree, norm_type=norm )

        if output_label:
            return result, label
        else:
            return result
