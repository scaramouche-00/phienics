"""
.. module:: solver

This module contains basic functionalities for solvers of any theory of screening.

"""

# Copyright (C) Daniela Saadeh 2020
# This file is part of phi-enics


from phienics.utils import r2_norm, project

from dolfin import Expression, dx, Constant, inner, grad

from scipy.special import binom
from numpy import log10

import dolfin as d
import numpy as np



class Solver(object):
    r""" 
    Base class for solvers of models of screening.

    This solver implements the Newton method given the equation for the iterations: if :math:`\mathcal{F}(u)=0`
    is the original nonlinear equation (or system of equations), this solver iterates:

    .. math:: 0 = \mathcal{F}[u^{(k)}] + \int d{\bf x} \frac{\delta \mathcal{F}}{\delta u({\bf x})}[u^{(k)}]
                 \left(u_{\rm true}-u^{(k)}\right)
       :label: Eq_linear_solver

    at every iteration :math:`k`, until convergence. Eq. :eq:`Eq_linear_solver` must be supplied 
    as :func:`linear_solver`. For further details see the `paper <arxiv_link>_`.

    Two convergence tests are available:

    1) `'residual'` (default):
    
    .. math:: || \mathcal{F}[u^{(k)}] || \leq \epsilon_{\rm rel}^{\rm (R)}
              || \mathcal{F}[u^{(0)}] || + \epsilon_{\rm abs}^{\rm (R)}
       :label: Eq_residual_criterion

    2) `'change'`, i.e. change in the solution:

    .. math:: || u^{(k)}-u^{(k-1)} || \leq \epsilon_{\rm rel}^{\rm (S)} || u^{(0)} ||
              + \epsilon_{\rm abs}^{\rm (S)}
       :label: Eq_change_criterion

    where :math:`\epsilon_{\rm rel}^{\rm (\cdot)}` and :math:`\epsilon_{\rm abs}^{\rm (\cdot)}` are
    relative and absolute tolerances, :math:`u^{(0)}` is the initial guess, and :math:`|| \cdot ||` is some norm 
    on the solution space.


    *Arguments*
        fem
            a :class:`fem.Fem` instance
        source
            a :class:`source.Source` instance
        fields
            a :class:`UV.UVFields` or :class:`IR.IRFields` instance
        Mn
            distances are rescaled following :math:`\hat{r} = M_n r` inside the code. 
            Default: :math:`M_n={r_S}^{-1}`
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
                  Mn=None, abs_du_tol=1e-8, rel_du_tol=1e-8, abs_res_tol=1e-10, rel_res_tol=1e-10,
                  max_iter=100, criterion='residual', norm_change='linf', norm_res='linf' ):
        
        """The constructor"""

        # finite element properties, including the mesh
        self.fem = fem
        # function space for the system of equations if any
        self.V = None
        
        # source properties
        self.source = source

        # parameters of the theory, including masses, coupling and self-coupling
        self.fields = fields

        # rescalings (in units Planck mass)
        # distance rescaling
        self.Mn = Mn
        if self.Mn is None:
            self.Mn = self.fem.mesh.rs / self.source.Rs
                
        # solver properties
        # tolerance of the non-linear solver - change in solution (du)
        self.abs_du_tol = abs_du_tol
        self.rel_du_tol = rel_du_tol
        # tolerance of the non-linear solver - residual (F)
        self.abs_res_tol = abs_res_tol
        self.rel_res_tol = rel_res_tol   
        # maximum number of iterations
        self.max_iter = max_iter     
        # choice of convergence criterion and norm
        self.criterion = criterion
        self.norm_change = norm_change
        self.norm_res = norm_res

        # flag: has it converged?
        self.converged = False
        # iteration counter
        self.i = 0
       
        # absolute and relative error at every iteration
        # change in solution
        self.abs_du = np.zeros( self.max_iter )
        self.rel_du = np.zeros( self.max_iter )
        # residual
        self.abs_res = np.zeros( self.max_iter )
        self.rel_res = np.zeros( self.max_iter )
        
        # initial residual and norm of solution at initial iteration - used
        # for relative change and residual; set in the update_errors function
        self.u0_norm = None
        self.F0_norm = None




    def get_Dirichlet_bc( self ):
        pass


    def initial_guess( self ):
        pass


    def weak_residual_form( self ):
        pass
    

    def strong_residual_form( self ):
        pass
    

    def linear_solver( self ):
        pass


    def compute_physics( self ):
        pass
    

    def strong_residual( self, sol, units='rescaled', norm='linf' ):
        r"""Returns the residual, either as a function on the domain or as a norm of that function.

        The former is just a wrapper around :func:`strong_residual_form`, which is determined by the model
        (see e.g. :func:`UV.UVSolver.strong_residual_form`); the latter computes a choice of norm defined
        by the parameter `norm`.

        *Arguments*
            sol
                the solution :math:`u` on which the residual :math:`\mathcal{F}[u]` is to be computed
            units
                'rescaled' (for the rescaled units used inside the code) or 'physical', for physical units
            norm 
                `'L2'`, `'linf'` or `'none'`. If `'L2'` or `'linf'`: compute the :math:`L_2` or
                :math:`\ell_{\infty}` norm of the residual; if `'none'`, return the full
                function - as opposed to its norm.


        """

        F = self.strong_residual_form( sol, units )

        # 'none' = return function, not norm
        if norm=='none':
            result = F
        # from here on return a norm. This nested if is to preserve the structure of the original
        # built-in FEniCS norm function
        elif norm=='linf':
            # infinity norm, i.e. max abs value at vertices
            result = r2_norm( F.vector(), self.fem.func_degree, norm_type=norm )
        else:
            result = r2_norm( F, self.fem.func_degree, norm_type=norm )

        return result




    def compute_errors( self, du_k, u_k ):
        r"""
        Computes measures of error at a given iteration.
        
        With reference to Eq. :eq:`Eq_residual_criterion` and Eq. :eq:`Eq_change_criterion`, 
        this function computes the norm of the residual :math:`|| \mathcal{F}[u^{(k)}] ||` and 
        the norm of the change in the solution :math:`|| u^{(k)}-u^{(k-1)} ||` at one given iteration.

        *Parameters*
            du_k
                difference in the solution over the whole domain at this iteration. 
            u_k
                solution at this iteration, for the computation of the residual :math:`||F(u_k)||`

        """
        
        # compute residual of solution at this iteration - assemble form into a vector
        F = d.assemble( self.weak_residual_form( u_k ) )
        
        # ... and compute norm. L2 is note yet implemented
        if self.norm_res=='L2':
            message = "L2 norm not implemented for residuals. Please use a different norm ('l2' or 'linf')"
            raise NotImplementedError(message)

        # the built-in norm function is fine because it's computing the linf or Euclidean norm here
        F_norm = d.norm( F, norm_type=self.norm_res )
        
        # now compute norm of change in the solution
        # this nested if is to preserve the structure of the original built-in FEniCS norm function 
        # within the modified r2_norm function
        if self.norm_change=='L2':
            # if you are here, you are computing the norm of a function object, as an integral
            # over the whole domain, and you need the r^2 factor in the measure (i.e. r^2 dr )
            du_norm = r2_norm( du_k, self.fem.func_degree, norm_type=self.norm_change )
            
        else:
            # if you are here, you are computing the norm of a vector. For this, the built-in norm function is
            # sufficient. The norm is either linf (max abs value at vertices) or l2 (Euclidean norm)
            du_norm = d.norm( du_k.vector(), norm_type=self.norm_change )

        return du_norm, F_norm




    def solve( self ):
        """Solves the non-linear equation iteratively using the Newton's method, starting from 
           a linear solver as in Eq. :eq:`Eq_linear_solver`.

        """

        u_k = self.initial_guess()     
        abs_du = d.Function( self.V )

        while (not self.converged) and self.i < self.max_iter:

            # get solution at this iteration from linear solver
            sol = self.linear_solver( u_k )
            
            # if this is the initial iteration, store the norm of the initial solution and initial residual 
            # for future computation of relative change and residual
            if self.i == 0:
                # the first 'sol' passed as input takes the place of what is normally the variation in the solution
                self.u0_norm, self.F0_norm = self.compute_errors( sol, sol )

            # compute and store change in the solution
            abs_du.vector()[:] = sol.vector()[:] - u_k.vector()[:]
            self.abs_du[self.i], self.abs_res[self.i] = self.compute_errors( abs_du, sol )
            # compute and store relative errors
            self.rel_du[self.i] = self.abs_du[self.i] / self.u0_norm
            self.rel_res[self.i] = self.abs_res[self.i] / self.F0_norm


            # write report: to keep output legible only write tolerance for the criterion that's effectively working
            if self.criterion=='residual':
                print('Non-linear solver, iteration %d\tabs_du = %.1e\trel_du = %.1e\t' \
                      % (self.i, self.abs_du[self.i], self.rel_du[self.i] ) )
                print('abs_res = %.1e (tol = %.1e)\trel_res = %.1e (tol = %.1e)' \
                    % ( self.abs_res[self.i], self.abs_res_tol, self.rel_res[self.i], self.rel_res_tol ))

            else:
                print('Non-linear solver, iteration %d\tabs_du = %.1e (tol = %.1e)\trel_du = %.1e (tol=%.1e)\t' \
                      % (self.i, self.abs_du[self.i], self.abs_du_tol, self.rel_du[self.i], self.rel_du_tol ) )
                print('abs_res = %.1e\trel_res = %.1e' \
                    % ( self.abs_res[self.i], self.abs_res_tol, self.rel_res[self.i], self.rel_res_tol ))


            # check convergence
            if self.criterion=='residual':
                self.converged = ( self.abs_res[self.i] < self.rel_res_tol * self.F0_norm ) \
                or ( self.abs_res[self.i] < self.abs_res_tol )
                
            else:
                self.converged = ( self.abs_du[self.i] < self.rel_du_tol * self.u0_norm ) \
                or ( self.abs_du[self.i] < self.abs_du_tol )

                
            # if maximum number of iterations has been reached without converging, throw a warning
            if ( self.i+1 == self.max_iter and ( not self.converged ) ):
                print("*******************************************************************************")
                print("   WARNING: the solver hasn't converged in the maximum number of iterations")
                print("*******************************************************************************")
            
            # update for next iteration
            self.i += 1
            u_k.assign(sol)



        # use the obtained solution to compute field profiles, gradients and the scalar force
        self.compute_physics( sol )






    def grad( self, field, radial_units ):
        r"""
        Returns the gradient of the scalar field in input.

        The gradient is computed by projecting the radial derivative of the field onto the discontinuous
        function space specified in :class:`solver.fem`.
        
        *Arguments*
            field
                the field of which you want to compute the gradient
            radial_units
                'physical' or 'rescaled': the former returns the field's gradient with respect to physical distances
                (units :math:`{M_p}^{-1}`), the latter returns the gradient with respect to the dimensionless rescaled 
                distances used within the code
        
        """
    
        if radial_units=='physical':
            grad_ = Constant(self.Mn) * field.dx(0)
        elif radial_units=='rescaled':
            grad_ = field.dx(0)
        else:
            message = "Invalid choice of radial units: valid choices are 'physical' or 'rescaled'."
            raise ValueError(message)
    
        grad_ = project( grad_, self.fem.dS, self.fem.func_degree )

        return grad_


    

    def scalar_force( self, field ):     
        r"""
        Returns the magnitude of the scalar force associated to the input field, per unit mass (units :math:`M_p`):
        
        .. math :: F_{\varphi} = \frac{\nabla\varphi}{M_P}
        
        if :math:`\varphi` is the input field.
        
        *Arguments*
        field
            the field associated to the scalar force
        
        """

        grad = self.grad( field, 'physical' )
        force = - grad / Constant(self.fields.Mp)
        force = project( force, self.fem.dS, self.fem.func_degree )
        
        return force



    


    def On( self, n, method='derivative', rescale=1., output_rescaled_op=False ):
        r"""
        Computes the operators:
        
        .. math:: O_n = (-1)^{n+1} \frac{2n+2}{2n+1} \binom{3n}{n} \alpha^{2n+1} \lambda^n
                  \nabla^2 \left((\nabla^2 \varphi)^{2n+1}\right) M^{-6n-2}
        
        associated to a field :math:`\varphi`.

        In the examples provided within :math:`\varphi\mathrm{enics}`, :math:`\varphi` is the the UV field :math:`\phi`
        or the IR field :math:`\pi`.

        The operators are computed starting from the field's rescaled Laplacian
        :math:`y\equiv\hat{\nabla}^2\hat{\varphi}`, so the field's equation of motion must have been solved 
        prior to invoking this method.

        For the method used to compute :math:`\nabla^2 \left((\nabla^2 \varphi)^{2n+1}\right)`, see the documentation
        of :func:`Qn`.

        *Arguments*
            n   
                order of the operator :math:`O_n`
            method
                `'derivative'`, `'auxiliary'` or `'gradient'` - see func:`Qn`
            rescale
                (optional) temporary auxiliary variable used to rescale the Laplacian during the computation of
                :math:`\nabla^2 \left((\nabla^2 \varphi)^{2n+1}\right)`, to prevent hitting the maximum/minimum 
                representable number
            output_rescaled_op
                `True` to obtain the rescaled operator 
                :math:`\hat{O}_n \equiv \hat{\nabla}^2 \left((\hat{\nabla}^2 \hat{\varphi})^{2n+1}\right)`
                alongside the physical operator :math:`O_n`, in a tuple :math:`(\hat{O}_n, O_n)`;
                `False` otherwise. Default: `False`.
        
        """

        # copy the Laplacian
        if self.y is None:
            message = "The Laplacian doesn't seem to have been computed. Please run solve() first."
            raise ValueError(message)
        y = d.Function( self.fem.S )
        y.assign( self.y )
        y.vector()[:] *= rescale # rescale for better precision (undone later)
        
        
        # we can obtain On in three ways
        if method=='derivative':
            # expand Del( (y^(2n+1) ) and project
            r = Expression( 'x[0]', degree=self.fem.func_degree )
            On_1 = (2.*n+1.) * y**(2*n) * ( y.dx(0).dx(0) + d.Constant(2.) / r * y.dx(0) )
            On_2 = (2.*n+1.) * (2.*n) * y**(2*n-1) * y.dx(0)**2
            On = On_1 + On_2
            On = project( On, self.fem.dS, self.fem.func_degree )
            
            
        elif method=='auxiliary':
            # project W=y^(2n+1) and solve On = Del(W)
            W = y**(2*n+1)
            W = project( W, self.fem.dS, self.fem.func_degree )
            
            On_ = d.TrialFunction( self.fem.dS )
            v_ = d.TestFunction( self.fem.dS )
            
            r2 = Expression( 'pow(x[0],2)', degree=self.fem.func_degree )
            On_a = On_ * v_ * r2 * dx
            On_L = - inner( grad(W), grad(v_) ) * r2 * dx
            
            On = d.Function( self.fem.dS )
            
            On_pde = d.LinearVariationalProblem( On_a, On_L, On )
            On_solver = d.LinearVariationalSolver( On_pde )
            On_solver.solve()
            
            
        elif method=='gradient':
            # expand Del( y^(2n+1) ) within the weak formulation, then solve the system
            On_ = d.TrialFunction( self.fem.dS )
            v_ = d.TestFunction( self.fem.dS )
        
            
            r2 = Expression( 'pow(x[0],2)', degree=self.fem.func_degree )
            On_a = On_ * v_ * r2 * dx
            On_L = - (2*n+1) * y**(2*n) * inner( grad(y), grad(v_) ) * r2 * dx
            
            On = d.Function( self.fem.dS )
            
            On_pde = d.LinearVariationalProblem( On_a, On_L, On )
            On_solver = d.LinearVariationalSolver( On_pde )
            On_solver.solve()
        

        # for the physical operator, unrescale and multiply by the theory-specific remaining terms  
        Mn, Mf1 = self.Mn, self.Mf1
        log10_On_coeff, On_oth_coeff = self.fields.log10_On_coeff(n), self.fields.On_oth_coeff(n)
        log10_R = log10_On_coeff + (4*n+4) * log10(Mn) + (2*n+1) * ( log10(Mf1) - log10( rescale ) )
        C_On = (-1)**(n+1) * binom(3*n,n) / (2.*n+1.) * On_oth_coeff * 10.**log10_R
        
        phys_On = d.Function( self.fem.dS )
        phys_On.vector()[:] = C_On * On.vector()[:]
        
        if output_rescaled_op:
            return On, phys_On
        else:
            return phys_On






    def Qn( self, n, method='derivative', rescale=1., output_rescaled_op=False ):
        r"""
        Computes the operators:
        
        .. math:: Q_n = \frac{\alpha^n}{M^{3n-1}} \nabla^2( ( \nabla^2\varphi )^n )
        
        associated to a field :math:`\varphi`.

        In the examples provided within :math:`\varphi\mathrm{enics}`,
        :math:`\varphi` is the the UV field :math:`\phi` or the IR field :math:`\pi`.
 
        The key to computing :math:`Q_n` is obtaining the rescaled operator 
        :math:`\hat{Q}_n \equiv \hat{\nabla}^2 \left((\hat{\nabla}^2 \hat{\varphi})^{n}\right)`.
        The Laplacian :math:`y\equiv\hat{\nabla}^2\hat{\varphi}` is obtained from the solution
        to the equation of motion; :math:`\hat{Q}_n` can then be computed in three different ways,
        specified by the `method` option:

        * `derivative` (default):
           the rescaled operator is decomposed as:

           .. math:: \hat{\nabla}^2( y^{n} ) = n y^{n-1} \hat{\nabla}^2 y 
                  + n (n-1) y^{n-2} \hat{\nabla} y \cdot \hat{\nabla} y

           before being projected on the discontinuous function space specified in the `fem` instance;

        * `auxiliary`:
           :math:`w \equiv y^n` is projected onto the discontinuous function space specified in the `fem`
           instance, then the code solves for the linear system :math:`\hat{Q}_n = \nabla^2(w)`;

        * `gradient`:
           for the UV and IR theories supplied as :math:`\varphi\mathrm{enics}` examples,
           the weak formulation of the equation :math:`\hat{Q}_n = \hat{\nabla}^2( y^n )` is

           .. math:: \int \hat{Q}_n v \hat{r}^2 d\hat{r} = \int \hat{\nabla}^2 ( y^n ) v \hat{r}^2 d\hat{r} = 
                     - \int \hat{\nabla}( y^n ) \hat{\nabla} v \hat{r}^2 d\hat{r} =
                     - n \int y^{n-1} \hat{\nabla}y \hat{\nabla}v \hat{r}^2 d\hat{r}

           where :math:`v` is a test function, so that the :math:`\hat{Q}_n` operator can be obtained by solving

           .. math:: \int \hat{Q}_n v \hat{r}^2 d\hat{r} =
                     - n \int y^{n-1} \hat{\nabla}y \hat{\nabla}v \hat{r}^2 d\hat{r}

           which is the case for the `gradient` option.

        Further details are given in `the paper <this will be the arXiv ref>_`.

        .. note:: The :math:`Q_n` operators are computed starting from the field's rescaled Laplacian
                  :math:`y\equiv\hat{\nabla}^2\hat{\varphi}`, so the method :func:`solve()` (solving the
                  field's equation of motion) must have been called before calling this method.

        *Arguments*
            n
                order of the operator :math:`Q_n`
            method
                `'derivative'`, `'auxiliary'` or `'gradient'`
            rescale
                (optional) temporary auxiliary rescaling for :math:`Q_n`, used in tests or to prevent hitting the 
                maximum/minimum representable number
            output_rescaled_op
                `True` to obtain the rescaled operator 
                :math:`\hat{Q}_n \equiv \hat{\nabla}^2 \left((\hat{\nabla}^2 \hat{\varphi})^{2n+1}\right)`
                alongside the physical operator :math:`Q_n`, in a tuple :math:`(\hat{O}_n, O_n)`;
                `False` otherwise. Default: `False`.

        """

        # copy the Laplacian
        if self.y is None:
            message = "The Laplacian doesn't seem to have been computed. Please run solve() first."
            raise ValueError(message)
        y = d.Function( self.y.function_space() )
        y.assign( self.y )
        y.vector()[:] *= rescale # rescale for better precision (undone later)
        
        # we can obtain Qn in three ways
        if method=='derivative':
            # expand Del( (y^(2n+1) ) and project
            r = Expression( 'x[0]', degree=self.fem.func_degree )
            Qn_1 = n * y**(n-1) * ( y.dx(0).dx(0) + d.Constant(2.) / r * y.dx(0) )
            Qn_2 = n * (n-1.) * y**(n-2) * y.dx(0)**2
            Qn = Qn_1 + Qn_2
            Qn = project( Qn, self.fem.dS, self.fem.func_degree )
            
            
        elif method=='auxiliary':
            # project w=y^n and solve Qn = Del(w)
            w = y**n
            w = project( w, self.fem.dS, self.fem.func_degree )
            
            Qn_ = d.TrialFunction( self.fem.dS )
            v_ = d.TestFunction( self.fem.dS )
            
            r2 = Expression( 'pow(x[0],2)', degree=self.fem.func_degree )
            Qn_a = Qn_ * v_ * r2 * dx
            Qn_L = - inner( grad(w), grad(v_) ) * r2 * dx
            
            Qn = d.Function( self.fem.dS )
            
            Qn_pde = d.LinearVariationalProblem( Qn_a, Qn_L, Qn )
            Qn_solver = d.LinearVariationalSolver( Qn_pde )
            Qn_solver.solve()

            
        elif method=='gradient':
            # expand Del( y^n ) within the weak formulation, then solve the system
            Qn_ = d.TrialFunction( self.fem.dS )
            v_ = d.TestFunction( self.fem.dS )
            
            
            r2 = Expression( 'pow(x[0],2)', degree=self.fem.func_degree )
            Qn_a = Qn_ * v_ * r2 * dx
            Qn_L = - n * y**(n-1) * inner( grad(y), grad(v_) ) * r2 * dx
            
            Qn = d.Function( self.fem.dS )
            
            Qn_pde = d.LinearVariationalProblem( Qn_a, Qn_L, Qn )
            Qn_solver = d.LinearVariationalSolver( Qn_pde )
            Qn_solver.solve()
        
            
        # for the physical operator, unrescale and multiply by the theory-specific remaining terms  
        log10_Qn_coeff, Qn_oth_coeff = self.fields.log10_Qn_coeff(n), self.fields.Qn_oth_coeff(n)
        # compute log and then exp to avoid hitting the maximum/minimum representable number
        log10_R = log10_Qn_coeff + (2*n+2) * log10(self.Mn) + n * ( log10(self.Mf1) - log10( rescale ) )

        phys_Qn = d.Function( self.fem.dS )
        phys_Qn.vector()[:] = Qn_oth_coeff * 10.**log10_R * Qn.vector()[:]
        
        if output_rescaled_op:
            return Qn, phys_Qn
        else:
            return phys_Qn






