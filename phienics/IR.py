"""
.. module:: IR

This module contains the solver for a theory of a single massive scalar field characterised by
higher-derivative operators.

"""

# Copyright (C) Daniela Saadeh 2020
# This file is part of phi-enics

from solver import Solver
from utils import r2_norm, project

from dolfin import inner, grad, dx
from dolfin import Expression, Constant

from numpy import log10

import numpy as np
import dolfin as d





class IRFields(object):
    r"""
    Physical parameters of the massive Galileon theory of equation of motion:

    .. math:: \Box\pi -m^2\pi - \frac{\epsilon}{\Lambda^{3n-1}} \Box\left( \Box\pi^n \right) = \frac{\rho}{M_P}
              :label: Eq_IR_theory

    Here :math:`\pi` is a massive scalar field of mass :math:`m`, :math:`\Lambda` is a cut-off scale with units of mass,
    and :math:`\rho` is the energy density of the source. The dimensionless parameter :math:`\epsilon` 
    regulates the strength of the nonlinear term which is expected to give rise to Vainshtein screening. 

    *Arguments*
        m
            mass of Galileon field :math:`\pi`
        Lambda
            cut-off scale of the theory
        Mp
            Planck mass :math:`M_P`
        epsilon
            nonlinear parameter :math:`\epsilon`
        n
            exponent :math:`n` in the theory

    """

    def __init__( self, m=1e-51, Lambda=1e-49, Mp=1., epsilon=0.02, n=3 ):
        self.m = m
        self.Lambda = Lambda
        self.Mp = Mp

        self.epsilon = epsilon

        self.n = n

        # used in the computation of On. I call the input param of these two functions 'k' to distinguish
        # from the IR param n
        self.log10_On_coeff = lambda k : - (6*k+2)*log10(self.Lambda)
        self.On_oth_coeff = lambda k : self.epsilon**k
        # usef in the computation of Qn
        self.log10_Qn_coeff = lambda k : - (3*k-1)*log10(self.Lambda)
        self.Qn_oth_coeff = lambda k : self.epsilon**k





        

class IRSolver(Solver):
    r"""
    Solver for the Galileon theory in Eq. :eq:`Eq_IR_theory`. Derives from :class:`solver.Solver`.

    For better accuracy in the computation of the field's Laplacian and of the nonlinear term,
    the equation of motion Eq. :eq:`Eq_IR_theory` is expressed within the code as:

    .. math:: & \nabla^2 \phi = Y \\
              & W = Y^n \\
              & Y - m^2 \phi - \frac{\epsilon}{\Lambda^{3n-1}} \nabla^2 W = \frac{\rho}{M_P}


    Two off-the-shelf choices are available for the initial guess:

    - `guess_choice='NL'` (default):
                          
                                    this choice assumes that the nonlinear term in the equation of 
                                    motion is dominant, obtaining the approximation

                                    .. math:: -\nabla^2 W_0 \approx \frac{\Lambda^{3n-1}}{\epsilon} \frac{\rho}{M_P}

                                    from which :math:`Y` is obtained as :math:`Y_0=\sqrt[n]{W_0}`, and
                                    finally :math:`\nabla^2\pi_0=Y_0`


    - `guess_choice='KG'`:

                          this choice assumes that the nonlinear term in the equation of motion is subdominant,
                          obtaning the approximation

                          .. math:: \nabla^2\pi - m^2 \pi \approx \frac{\rho}{M_P}

                          which is a linear system and can be solved as is.

    
    Two off-the-shelf choice of dimensionless units are similarly available for the scalar field :math:`\pi`,
    when the user does not wish to set it manually;
    having defined :math:`\hat{\pi}=\pi/M_{f1}`, one can choose:

    - `Mf1='NL'` (default):

                           with this choice, the code attempts to find an optimal rescaling for 
                           :math:`\pi` based on the assumption that the nonlinear term is dominant,
                           by setting

                           .. math:: M_{f1} = \left( \Lambda / M_n \right)^{(3n-1)/n} ( M_S / M_P)^{1/n} \, m

    - `Mf1='source'`:
                     
                     this choice results in a dimensionless source term that is :math:`\approx O(1)`,
                     by setting:
     
                     .. math:: M_{f1} = M_S / M_P M_n

    where :math:`M_S` is the source mass and, in both cases, :math:`M_n` is a unit of inverse distance 
    used in the definition of the in-code dimensionless radial distances: :math:`\hat{r}\equiv r M_n`
    (for the definition of other symbols, please see :class:`IRFields`).


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
            field rescaling for :math:`\pi`: :math:`\hat{\pi}=\pi/M_{f1}`; choice of `'NL'`, `'source'` 
            or a real number. Default:`Mf1='NL'`
        guess_choice
            choice of `NL` or `KG`. Default: `NL`
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
                  Mn=None, Mf1='NL', guess_choice='NL',
                  abs_du_tol=1e-8, rel_du_tol=1e-8, abs_res_tol=1e-10, rel_res_tol=1e-10,
                  max_iter=100, criterion='residual',
                  norm_change='linf', norm_res='linf' ):

        
        """The constructor"""
        
        Solver.__init__( self, fem, source, fields, Mn, abs_du_tol, rel_du_tol,
                         abs_res_tol, rel_res_tol, max_iter, criterion, norm_change, norm_res )
   
        # vector finite element and function space for rescaled (pi, w, y)
        self.E = d.MixedElement([ self.fem.Pn, self.fem.Pn, self.fem.Pn ])
        self.V = d.FunctionSpace( self.fem.mesh.mesh, self.E )
        # (discontinuous) vector finite element and function space for the three strong residuals
        self.dE = d.MixedElement([ self.fem.dPn, self.fem.dPn, self.fem.dPn ])
        self.dV = d.FunctionSpace( self.fem.mesh.mesh, self.dE )

        # field rescaling
        if Mf1=='NL':
            self.Mf1 = ( self.fields.Lambda / self.Mn )**(3.-1./self.fields.n) * \
            ( self.source.Ms / self.fields.Mp )**(1./self.fields.n) * self.fields.m
        elif Mf1=='source':
            self.Mf1 = self.source.Ms / self.fields.Mp * self.Mn
        else:
            self.Mf1 = Mf1
                
        # choice of initial guess
        self.guess_choice = guess_choice

        # solution and field profiles (computed by the solver)
        self.u = None
        self.pi, self.w, self.y = None, None, None # rescaled
        self.Pi, self.W, self.Y = None, None, None # physical

        # gradient of the scalar field
        self.grad_pi = None # rescaled
        self.grad_Pi = None # physical

        # scalar force (physical units)
        self.force = None


    


    def get_Dirichlet_bc( self ):
        r"""
        Returns the Dirichlet boundary conditions for the IR theory:

        .. math:: \hat{\pi}(\hat{r}_{\rm max}) = \hat{W}(\hat{r}_{\rm max}) = 
                  \hat{Y}(\hat{r}_{\rm max}) = 0
        
        The Neumann boundary conditions

        .. math:: \left\lbrace \frac{d\hat{\pi}}{d\hat{r}}(0) = 0;  
                  \frac{d\hat{W}}{d\hat{r}}(0) = \mathrm{finite} \right\rbrace

        are natural boundary conditions in the finite element method, and are implemented within
        the variational formulation.

        """

        # define values at infinity
        # for 'infinity', we use the last mesh point, i.e. r_max (i.e. mesh[-1]) 
        piD = d.Constant( 0. )
        wD = d.Constant( 0. )
        yD = d.Constant( 0. )

        # define 'infinity' boundary: the rightmost mesh point - within machine precision
        def boundary(x):
            return self.fem.mesh.r_max - x[0] < d.DOLFIN_EPS

        bc_pi = d.DirichletBC( self.V.sub(0), piD, boundary, method='pointwise' )
        bc_w = d.DirichletBC( self.V.sub(1), wD, boundary, method='pointwise' )
        bc_y = d.DirichletBC( self.V.sub(2), yD, boundary, method='pointwise' )

        Dirichlet_bc = [ bc_pi, bc_w, bc_y ]

        return Dirichlet_bc





    def initial_guess( self ):
        # Choose the initial guess from the options available

        if self.guess_choice=='KG':
            return self.KG_initial_guess()
        elif self.guess_choice=='NL':
            return self.NL_initial_guess()
    
    


    

    def KG_initial_guess( self ):
        r"""
        Obtains an initial guess for the Galileon equation of motion by assuming the nonlinear term is
        subdominant, i.e.:

        .. math:: \Box\pi - m^2\pi \approx \frac{\rho}{Mp}

        The initial guess is computed by first solving the system of equations:

        .. math:: & \hat{\nabla}^2\hat{\pi} = \hat{Y} \\
                  & \hat{Y} - \left( \frac{m}{M_n} \right)^2\pi = \frac{\hat{\rho}}{M_p}

        and then obtaining :math:`\hat{W}=\hat{Y}^n` by projection.

        """

        # define a function space for (pi, y) only
        piy_E =  d.MixedElement([ self.fem.Pn, self.fem.Pn ])
        piy_V = d.FunctionSpace( self.fem.mesh.mesh, piy_E )
        
        # get the boundary conditions for pi and y only
        piD, yD = Constant( 0. ), Constant( 0. )
        def boundary(x):
            return self.fem.mesh.r_max - x[0] < d.DOLFIN_EPS
        bc_pi = d.DirichletBC( piy_V.sub(0), piD, boundary, method='pointwise' )
        bc_y = d.DirichletBC( piy_V.sub(1), yD, boundary, method='pointwise' )
        Dirichlet_bc = [ bc_pi, bc_y ]
        
        # Trial functions for pi and y
        u = d.TrialFunction( piy_V )
        pi, y = d.split( u )
        
        # test functions for the two equations
        v1, v3 = d.TestFunctions( piy_V )
        
        # cast params as constant functions so that, if they are set to 0, FEniCS still understand
        # what is being integrated
        m, Mp, Mn, Mf1 = Constant( self.fields.m ), Constant( self.fields.Mp ), Constant( self.Mn ), Constant( self.Mf1 )
        n = self.fields.n
        
        # r^2  
        r2 = Expression( 'pow(x[0],2)', degree=self.fem.func_degree )

        # bilinear form
        a1 = - inner( grad(pi), grad(v1) ) * r2 * dx - y * v1 * r2 * dx
        a3 = y * v3 * r2 * dx - ( m / Mn )**2 * pi * v3 * r2 * dx
        a = a1 + a3
        
        # linear form (L1=0)
        L3 = self.source.rho / Mp * Mn / Mf1 * v3 * r2 * dx
        L = L3
        
        # solve system
        sol = d.Function( piy_V )
        pde = d.LinearVariationalProblem( a, L, sol, Dirichlet_bc )
        solver = d.LinearVariationalSolver( pde )
        print 'Getting KG initial guess...'
        solver.solve()

        # split solution into pi and y - cast as independent functions, not components of a vector function
        pi, y = sol.split( deepcopy=True )

        # obtain w by projecting y**n
        w = y**n
        w = project( w, self.fem.S, self.fem.func_degree )

        # and now pack pi, w, y into one function...
        guess = d.Function( self.V )
        # this syntax is because, even though pi and y are effectively defined on fem.S, from fenics point
        # of view, they are obtained as splits of a function
        fa = d.FunctionAssigner( self.V, [ pi.function_space(), self.fem.S, y.function_space() ])
        fa.assign( guess, [pi, w, y] )
        
        return guess




    def NL_initial_guess( self, y_method='vector' ):
        r"""
        Obtains an initial guess for the Galileon equation of motion by assuming the nonlinear term is
        dominant, i.e.:

        .. math:: -\frac{\epsilon}{\Lambda^{3n-1}}\nabla^2(\nabla^2\pi^n) \approx \frac{\rho}{M_P}

        The initial guess is computed by first solving the Poisson equation:

        .. math:: -\hat{\nabla}\hat{W} =\left( \frac{\Lambda}{M_n} \right)^{3n-1}
                  \left(\frac{M_n}{M_{f1}}\right)^n \frac{\hat{\rho}}{\epsilon M_P}

        and then obtaining :math:`\hat{Y}=\sqrt[n]{\hat{W}}` through one of three methods explained below.
        Finally, :math:`\hat{\pi}` is computed by solving the Poisson equation

        .. math:: \hat{\nabla}\hat{\pi} = \hat{Y}.

        The main methods to obtain :math:`\hat{Y}` from :math:`\hat{Z}` are interpolation and projection:
        the standard FEniCS implementation for both can be chosen by setting `y\_method='interpolate'` 
        and `y\_method='project'`.

        A third method (`y\_method='vector'`, default), formally identical to interpolation,
        consists in assigning :math:`\hat{Y}`'s value at all nodes through e.g.:

        .. code-block:: python
          
            y.vector().set_local( np.sqrt( np.abs( w.vector().get_local() ) ) )

        However, because of differences in the implementations of :math:`\sqrt[n]{\cdot}` called by
        the two methods, the latter generally gives better results compared to `'interpolate'`.

        *Arguments*
            y_method
                `'vector'` (default), `'interpolate'` or `'project'`

        """
            
        # cast params as constant functions so that, if they are set to 0, FEniCS still understand
        # what is being integrated
        m, Lambda, Mp =  Constant( self.fields.m ), Constant( self.fields.Lambda ), Constant( self.fields.Mp )
        epsilon = Constant( self.fields.epsilon )
        Mn, Mf1 = Constant( self.Mn ), Constant( self.Mf1 )
        n = self.fields.n
        
        # get the boundary conditions for w only
        def boundary( x ):
            return self.fem.mesh.r_max - x[0] < d.DOLFIN_EPS
        wD = Constant( 0. )
        w_Dirichlet_bc = d.DirichletBC( self.fem.S, wD, boundary, method='pointwise' )
        
        # define trial and test function
        w_ = d.TrialFunction( self.fem.S )
        v_ = d.TestFunction( self.fem.S )
        
        # for the measure  
        r2 = Expression( 'pow(x[0],2)', degree=self.fem.func_degree )

        # bilinear and linear forms
        w_a = inner( grad(w_), grad(v_) ) * r2 * dx
        w_L = ( Lambda/Mn )**(3*n-1) * ( Mn/Mf1 )**n / epsilon * self.source.rho / Mp * v_ * r2 * dx

        # define a function for the solution
        w = d.Function( self.fem.S )

        # solve
        w_pde = d.LinearVariationalProblem( w_a, w_L, w, w_Dirichlet_bc )
        w_solver = d.LinearVariationalSolver( w_pde )
        print 'Getting NL initial guess...'
        w_solver.solve()

        # now we have w. we can obtain y by projection or interpolation
        if y_method=='interpolate':
            # I use the functions sqrt and cbrt because they're more precise than pow(w,1/n)
            if n==2:
                code = "sqrt(fabs(w))"
            elif n==3:
                code = "cbrt(w)"
            else:
                if n%2==0: # even power
                    code = "pow(fabs(w),1./n)"
                else: # odd power
                    code = "pow(w,1./n)"
            y_expression = Expression( code, w=w, n=n, degree=self.fem.func_degree )
            y = d.interpolate( y_expression, self.fem.S )

        elif y_method=='vector':
            # this should formally be identical to 'interpolate', but it's a workaround to this
            # potential FEniCS bug which occurs in the previous code block:
            # https://bitbucket.org/fenics-project/dolfin/issues/1079/interpolated-expression-gives-wrong-result
            y = d.Function( self.fem.S )
            if n==2:
                y.vector().set_local( np.sqrt( np.abs( w.vector().get_local() ) ) )
            elif n==3:
                y.vector().set_local( np.cbrt( np.abs( w.vector().get_local() ) ) )
            else:
                if n%2==0: # even power
                    y.vector().set_local( np.abs( w.vector().get_local() )**(1./self.fields.n) )
                else: # odd power
                    y.vector().set_local( w.vector().get_local()**(1./self.fields.n) )

        elif y_method=='project': 
            y = w**(1./n)
            y = project( y, self.fem.S, self.fem.func_degree )
        
        # we obtain pi by solving Del pi = y
        piD = Constant( 0. )
        pi_Dirichlet_bc = d.DirichletBC( self.fem.S, piD, boundary, method='pointwise' )
        pi_ = d.TrialFunction( self.fem.S )
        v_ = d.TestFunction( self.fem.S )
        pi_a = - inner( grad(pi_), grad(v_) ) * r2 * dx
        pi_L = y * v_ * r2 * dx
        pi = d.Function( self.fem.S )   
        pi_pde = d.LinearVariationalProblem( pi_a, pi_L, pi, pi_Dirichlet_bc )
        pi_solver = d.LinearVariationalSolver( pi_pde )
        pi_solver.solve()

        # now let's pack pi, w, y into a single function
        guess = d.Function( self.V )
        fa = d.FunctionAssigner( self.V, [ self.fem.S, self.fem.S, self.fem.S ])
        fa.assign( guess, [pi, w, y] )

        return guess



    
        

    def weak_residual_form( self, sol ):
        r"""
        Computes the residual with respect to the weak form of the equations:

        .. math:: F = F_1 + F_2 + F_3
        
        with
        
        .. math:: F_1(\hat{\pi},\hat{W},\hat{Y}) & = - \int\hat{\nabla}\hat{\pi}\hat{\nabla}v_1 \hat{r}^2 d\hat{r}
                  - \int \hat{Y} v_1 \hat{r}^2 d\hat{r} \\

                  F_2(\hat{\pi},\hat{W},\hat{Y}) & = \int \hat{W} v_2 \hat{r}^2 d\hat{r} 
                  - \int \hat{Y}^n v_2 \hat{r}^2 d\hat{r} \\

                  F_3(\hat{\pi},\hat{W},\hat{Y}) & = \int \hat{Y} v_3 \hat{r}^2 d\hat{r} 
                  - \int \left( \frac{m}{M_n} \right)^2 \hat{\pi} v_3 \hat{r}^2 d\hat{r} + 

                  & + \epsilon \left( \frac{M_n}{\Lambda} \right)^{3n-1}
                  \left(\frac{M_{f1}}{M_n}\right)^{n-1} \int\hat{\nabla} \hat{W} \hat{\nabla} v_3 \hat{r}^2 d\hat{r} 
                  - \int \frac{\hat{\rho}}{M_p}\frac{M_n}{M_{f1}} v_3 \hat{r}^2 d\hat{r}

        The weak residual is employed within :func:`solver.Solver.solve` to check convergence -
        also see :func:`solver.Solver.compute_errors`.

        *Parameters*
            sol
                the solution with respect to which the weak residual is computed.


        """
        
        # cast params as constant functions so that, if they are set to 0, FEniCS still understand
        # what is being integrated
        m, Lambda, Mp =  Constant( self.fields.m ), Constant( self.fields.Lambda ), Constant( self.fields.Mp )
        n = self.fields.n
        epsilon = Constant( self.fields.epsilon )
        Mn, Mf1 = Constant( self.Mn ), Constant( self.Mf1 )
        
        # test functions
        v1, v2, v3 = d.TestFunctions( self.V )
        
        # split solution into pi, w, y
        pi, w, y = d.split(sol)
        
        # r^2
        r2 = Expression( 'pow(x[0],2)', degree=self.fem.func_degree )
        
        # define the weak residual form
        F1 = - inner( grad(pi), grad(v1)) * r2 * dx - y * v1 * r2 * dx
        F2 = w * v2 * r2 * dx - y**n * v2 * r2 * dx
        F3 = y * v3 * r2 * dx - (m/Mn)**2 * pi * v3 * r2 * dx + \
             epsilon * ( Mn / Lambda )**(3*n-1) * ( Mf1 / Mn )**(n-1) * inner( grad(w), grad(v3) ) * r2 * dx \
             - self.source.rho / Mp * Mn / Mf1 * v3 * r2 * dx
        
        F = F1 + F2 + F3
        
        return F




    def strong_residual_form( self, sol, units ):
        r"""
        Computes the residual with respect to the strong form of the equations.

        The total residual is obtained by summing the residuals of all equations:

        .. math:: F = F_1 + F_2 + F_3

        where, in dimensionless in-code units (`units='rescaled'`):

        .. math:: & F_1(\hat{\pi},\hat{W},\hat{Y}) = \hat{\nabla}^2 \hat{\pi} - \hat{Y}

                  & F_2(\hat{\pi},\hat{W},\hat{Y}) = \hat{W} - \hat{Y}^n

                  & F_3(\hat{\pi},\hat{W},\hat{Y}) = \hat{Y} - \left( \frac{m}{M_n} \right)^2 \hat{\pi}
                  - \epsilon \left( \frac{M_n}{\Lambda} \right)^{3n-1}
                  \left(\frac{M_{f1}}{M_n}\right)^{n-1} \hat{\nabla}^2 \hat{W}
                  - \frac{\hat{\rho}}{M_P} \frac{M_n}{M_{f1}}


        and in physical units (`units='physical'`):

        .. math:: & F_1(\pi,W,Y) = \nabla^2\pi - Y

                  & F_2(\pi,W,Y) = W - Y^n

                  & F_3(\pi,W,Y) = Y - m^2 \pi - \frac{\epsilon}{\Lambda^{3n-1}} \nabla^2 W - \frac{\rho}{M_P}



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
            resc_1, resc_2, resc_3 = 1., 1., 1.
        elif units=='physical':
            resc_1 = self.Mn**2 * self.Mf1
            resc_2 = ( self.Mn**2 * self.Mf1 )**self.fields.n
            resc_3 = self.Mn**2 * self.Mf1
            print '********************************************************************************************************************'
            print '   WARNING: residuals of equation 2 may hit the minimum representable number, consider using rescaled units instead'
            print '********************************************************************************************************************'
        else:
            message = "Invalid choice of units: valid choices are 'physical' or 'rescaled'."
            raise ValueError, message
        
        
        # cast params as constant functions so that, if they are set to 0, FEniCS still understand
        # what is being integrated
        m, Lambda, Mp =  Constant( self.fields.m ), Constant( self.fields.Lambda ), Constant( self.fields.Mp )
        epsilon = Constant( self.fields.epsilon )
        Mn, Mf1 = Constant( self.Mn ), Constant( self.Mf1 )
        n = self.fields.n

        # split solution into pi, w, y
        pi, w, y = d.split(sol)

        # initialise residual function
        F = d.Function( self.dV )

        # define r for use in the computation of the Laplacian
        r = Expression( 'x[0]', degree=self.fem.func_degree )

        # equation 1
        f1 = pi.dx(0).dx(0) + Constant(2.)/r * pi.dx(0) - y
        f1 *= Constant( resc_1 )
        F1 = project( f1, self.fem.dS, self.fem.func_degree )
        # equation 2
        f2 = w - y**n
        f2 *= Constant( resc_2 )
        F2 = project( f2, self.fem.dS, self.fem.func_degree )  
        # equation 3 
        f3 = y - ( m/Mn )**2 * pi \
             - epsilon * ( Mn / Lambda )**(3*n-1) * ( Mf1 / Mn )**(n-1) * (  w.dx(0).dx(0) + Constant(2.)/r * w.dx(0) ) \
             - self.source.rho / Mp * Mn / Mf1
        f3 *= Constant( resc_3 )
        F3 = project( f3, self.fem.dS, self.fem.func_degree )

        # combine equations
        fa = d.FunctionAssigner( self.dV, [ self.fem.dS, self.fem.dS, self.fem.dS ])
        fa.assign(F, [F1, F2, F3])

        return F






    def linear_solver( self, u_k ):
        r"""
        Solves the (linear) Newton iteration Eq. :eq:`Eq_linear_solver` for the IR theory.

        For the IR theory, the form of the Newton iterations is:

        .. math:: & - \int \hat{\nabla}\hat{\pi} \cdot \hat{\nabla} v_1 \hat{r}^2 d\hat{r}
                   - \int \hat{Y} v_1 \hat{r}^2 d\hat{r} +

                  & + \int \hat{W} v_2 \hat{r}^2 d\hat{r} -n \int {\hat{Y}_k}^{n-1} \hat{Y} v_2 \hat{r}^2 d\hat{r} +

                  & + \int \hat{Y} v_3 \hat{r}^2 d\hat{r} 
                  - \int \left( \frac{m}{M_n} \right)^2 \hat{\pi} v_3 \hat{r}^2 d\hat{r}
                  + \epsilon \left( \frac{M_n}{\Lambda} \right)^{3n-1} \left( \frac{M_{f1}}{M_n} \right)^{n-1}
                  \int \nabla\hat{W} \cdot \nabla v_3 \hat{r}^2 d\hat{r}

                  & = (1-n) \int {\hat{Y}_k}^n v_2 \hat{r}^2 d\hat{r}
                  + \int \frac{\hat{\rho}}{M_P} \frac{M_n}{M_{f1}} v_3 \hat{r}^2 d\hat{r}

        *Arguments*
            u_k  
                solution at the previous iteration

        """
        
        # get the boundary conditions
        Dirichlet_bc = self.get_Dirichlet_bc()
        
        # create a vector (pi,w,y) with the three trial functions for the fields
        u = d.TrialFunction( self.V )
        # ... and split it into pi, w, y
        pi, w, y = d.split( u )
        
        # define test functions over the function space
        v1, v2, v3 = d.TestFunctions( self.V )
        
        # split solution at current iteration into pi_k, w_k, y_k - this is only really useful for y_k
        pi_k, w_k, y_k = d.split( u_k )
        
        # cast params as constant functions so that, if they are set to 0, FEniCS still understand
        # what is being integrated
        m, Lambda, Mp =  Constant( self.fields.m ), Constant( self.fields.Lambda ), Constant( self.fields.Mp )
        epsilon = Constant( self.fields.epsilon )
        Mn, Mf1 = Constant( self.Mn ), Constant( self.Mf1 )
        n = self.fields.n
        
        # r^2   
        r2 = Expression('pow(x[0],2)', degree=self.fem.func_degree)
        
        # define bilinear form
        a1 = - inner( grad(pi), grad(v1) ) * r2 * dx - y * v1 * r2 * dx    
        a2 = w * v2 * r2 * dx - n * y_k**(n-1) * y * v2 * r2 * dx
        a3 = y * v3 * r2 * dx - ( m / Mn )**2 * pi * v3 * r2 * dx + \
             epsilon * ( Mn / Lambda )**(3*n-1) * ( Mf1 / Mn )**(n-1) * inner( grad(w), grad(v3) ) * r2 * dx
        
        a = a1 + a2 + a3
        
        # define linear form
        # we have L1 = 0.
        L2 = (1-n) * y_k**n * v2 * r2 * dx
        L3 = self.source.rho / Mp * Mn / Mf1 * v3 * r2 * dx

        L = L2 + L3

        # define a vector with the solution
        sol = d.Function( self.V )
            
        # solve linearised system
        pde = d.LinearVariationalProblem( a, L, sol, Dirichlet_bc )
        solver = d.LinearVariationalSolver( pde )
        solver.solve()

        return sol
    


        

    def compute_physics( self, sol ):
        
        # solution and profiles - rescaled units
        self.u = sol
        self.pi, self.w, self.y = sol.split(deepcopy=True)
        
        # profiles - physical units
        self.Pi = d.Function( self.fem.S )
        self.W = d.Function( self.fem.S )
        self.Y = d.Function( self.fem.S )
        self.Pi.vector()[:] = self.Mf1 * self.pi.vector()[:]
        self.W.vector()[:] = (self.Mn**2 * self.Mf1)**self.fields.n * self.w.vector()[:]
        self.Y.vector()[:] = self.Mn**2 * self.Mf1 * self.y.vector()[:]

        # gradient of the scalar field
        self.grad_pi = self.grad( self.pi, 'rescaled' )
        self.grad_Pi = self.grad( self.Pi, 'physical' )

        # scalar force - physical
        self.force = self.scalar_force( self.Pi )
        
        




        

    def output_term( self, eqn=1, term='LHS', norm='none', units='rescaled', output_label=False ):
        r"""
        Outputs the different terms in the IR system of equations.

        The terms can be output in physical or dimensionless in-code units, by choosing either
        `units='physical'` or `units='rescaled'`.

        Possible choices of terms are listed in the tables below; recall that :math:`Y\equiv\nabla^2\phi`
        and :math:`W \equiv Y^n`.

        - `eqn=1`:

        ==============   ===========================================   =================================   =================
         `units=`         `term='LHS'`                                  `term=1`                            `term=2`
        ==============   ===========================================   =================================   =================
         `'rescaled'`     :math:`\hat{Y} - \hat{\nabla}^2\hat{\pi}`     :math:`\hat{\nabla}^2\hat{\pi}`     :math:`\hat{Y}`
         `'physical'`     :math:`Y - \nabla^2\pi`                       :math:`\nabla^2\pi`                 :math:`Y`
        ==============   ===========================================   =================================   =================


        - `eqn=2`:

        ==============   =============================   ===================   =================
         `units=`         `term='LHS'`                    `term=1`              `term=2`
        ==============   =============================   ===================   =================
         `'rescaled'`     :math:`\hat{W} - \hat{Y}^n`     :math:`\hat{Y}^n`     :math:`\hat{W}`
         `'physical'`     :math:`W - Y^n`                 :math:`Y^n`           :math:`W`
        ==============   =============================   ===================   =================



        - `eqn=3`:


        ==============   ============================================================================================================================================================================   =================================================
         `units=`         `term='LHS'`                                                                                                                                                                   `term='RHS'`
        ==============   ============================================================================================================================================================================   =================================================
         `'rescaled'`     :math:`\hat{Y} -\left(\frac{m}{M_n}\right)^2\hat{\pi} - \epsilon \left( \frac{M_n}{\Lambda} \right)^{3n-1} \left( \frac{M_{f1}}{M_n} \right)^{n-1} \hat{\nabla}^2 \hat{W}`     :math:`\frac{\hat{\rho}}{M_P}\frac{Mn}{M_{f1}}`
         `'physical'`     :math:`Y - m^2 \pi - \frac{\epsilon}{\Lambda^{3n-1}} \nabla^2 W`                                                                                                               :math:`\frac{\rho}{M_P}`
        ==============   ============================================================================================================================================================================   =================================================


        ==============   =================   ================================================   =============================================================================================================================   =================
         `units=`         `term=1`            `term=2`                                           `term=3`                                                                                                                        `term=4`
        ==============   =================   ================================================   =============================================================================================================================   =================
         `'rescaled'`     :math:`\hat{Y}`     :math:`-\left(\frac{m}{M_n}\right)^2\hat{\pi}`     :math:`- \epsilon \left( \frac{M_n}{\Lambda} \right)^{3n-1} \left( \frac{M_{f1}}{M_n} \right)^{n-1} \hat{\nabla}^2 \hat{W}`     same as `'RHS'`
         `'physical'`     :math:`Y`           :math:`-m^2 \pi`                                   :math:`- \frac{\epsilon}{\Lambda^{3n-1}} \nabla^2 W`                                                                            same as `'RHS'`
        ==============   =================   ================================================   =============================================================================================================================   =================




        Equation (3) allows to look at the terms in the equation of motion,
        Eq. (1) and (2) enforce consistency relations :math:`Y = '\nabla^2\pi` and :math:`W=Y^n`. 

        .. note:: In this method, the Laplacian :math:`\hat{\nabla}^2` is obtained by projecting
                  :math:`\frac{\partial^2}{\partial\hat{r}^2} + 2\frac{\partial}{\partial\hat{r}}`.
                  As such, it should not be used with interpolating polynomials of degree less than 2.
            
        *Parameters*
            eqn
                (`integer`) choice of equation
            term
                choice of term: for `eqn=1,2`, valid terms range are `'LHS', 1, 2`; for
                `eqn=3`, valid terms are `'LHS', 'RHS'` and integers from 1 to 4
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
        m, Lambda, Mp =  Constant( self.fields.m ), Constant( self.fields.Lambda ), Constant( self.fields.Mp )
        epsilon = Constant( self.fields.epsilon )
        Mn, Mf1 = Constant( self.Mn ), Constant( self.Mf1 )
        n = self.fields.n
        
        if units=='rescaled':
            resc_1, resc_2, resc_3 = 1., 1., 1.
            str_pi, str_w, str_y = '\\hat{\pi}', '\\hat{W}', '\\hat{Y}'
            str_nabla2 = '\\hat{\\nabla}^2'
            str_m2 = '\\left( \\frac{m}{M_n} \\right)^2'
            str_nl = '\\epsilon \\left( \\frac{M_n}{\\Lambda} \\right)^{%d} \\left( \\frac{M_{f1}}{M_n} \\right)^{%d} \\hat{\\nabla}^2 \\hat{W}' % (3*n-1,n-1)
            str_rho = '\\frac{\hat{\\rho}}{M_p}\\frac{M_n}{M_{f1}}'
            
        elif units=='physical':
            resc_1 = self.Mn**2 * self.Mf1
            resc_2 = ( self.Mn**2 * self.Mf1 )**self.fields.n
            resc_3 = self.Mn**2 * self.Mf1
            print '********************************************************************************************************************'
            print '   WARNING: numbers in equation 2 may hit the minimum representable number, consider using rescaled units instead'
            print '********************************************************************************************************************'
            str_pi, str_w, str_y = '\\pi', 'W', 'Y'
            str_nabla2 = '\\nabla^2'
            str_m2 = 'm^2'
            str_nl = '\\frac{\\epsilon}{\\Lambda^{%d}} \\nabla^2 W' % (3*n-1)
            str_rho = '\\frac{\\rho}{M_p}'
            
        else:
            message = "Invalid choice of units: valid choices are 'physical' or 'rescaled'."
            raise ValueError, message
        
        pi, w, y = self.pi, self.w, self.y
        
        # define r for use in the computation of the Laplacian
        r = Expression( 'x[0]', degree=self.fem.func_degree )
        
        if eqn==1:
            if term=='LHS':
                Term = y - ( pi.dx(0).dx(0) + Constant(2.)/r * pi.dx(0) )
                label = r"$%s - %s%s$" % ( str_y, str_nabla2, str_pi )
            elif term==1:
                Term = ( pi.dx(0).dx(0) + Constant(2.)/r * pi.dx(0) )
                label = r"$%s%s$" % ( str_nabla2, str_pi )
            elif term==2:
                Term = y
                label = r"$%s$" % str_y
            # rescale if needed to get physical units
            Term *= resc_1


        elif eqn==2:
            if term=='LHS':
                Term = w - y**n
                label = r"$%s - %s^{%d}$" % (str_w, str_y, n)
            elif term==1:
                Term = y**n
                label = r"$%s^{%d}$" % (str_y, n)
            elif term==2:
                Term = w
                label = r"$%s$" % str_w
            # rescale if needed to get physical units
            Term *= resc_2
        
        
        elif eqn==3:
            if term=='LHS':
                Term = y - ( m/Mn )**2 * pi - \
                       epsilon * ( Mn / Lambda )**(3*n-1) * ( Mf1 / Mn )**(n-1) * ( w.dx(0).dx(0) + Constant(2.)/r * w.dx(0) )
                label = r"$%s - %s%s - %s$" % ( str_y, str_m2, str_pi, str_nl )
            elif term=='RHS':
                Term = self.source.rho / Mp * Mn / Mf1
                label = r"$%s$" % str_rho
            elif term==1:
                Term = y
                label = r"$%s$" % str_y
            elif term==2:
                Term = - ( m/Mn )**2 * pi
                label = r"$-%s %s$" % ( str_m2, str_pi )
            elif term==3:
                Term = - epsilon * ( Mn / Lambda )**(3*n-1) * ( Mf1 / Mn )**(n-1) * ( w.dx(0).dx(0) + Constant(2.)/r * w.dx(0) )
                label = r"$-%s$" % str_nl
            elif term==4:
                Term = self.source.rho / Mp * Mn / Mf1
                label = r"$%s$" % str_rho
            # rescale if needed to get physical units
            Term *= resc_3
        

        Term = project( Term, self.fem.dS, self.fem.func_degree )
    

        # 'none' = return function, not norm
        if norm=='none':
            result = Term
            # from here on return a norm. This nested if is to preserve the structure of the original
            # built-in FEniCS norm function
        elif norm=='linf':
            # infinity norm, i.e. max abs value at vertices    
            result = r2_norm( Term.vector(), self.fem.func_degree, norm_type=norm )
        else:
            result = r2_norm( Term, self.fem.func_degree, norm_type=norm )

        if output_label:
            return result, label
        else:
            return result
