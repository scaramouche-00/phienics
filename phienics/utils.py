"""
.. module:: utils

This module modifies FEniCS's built-in norm, errornorm and project functions 
to adopt the :math:`r^2 dr` measure instead of the :math:`dx` measure. 

Most of the code is identical to the original one - Copyright (C) Anders Logg:
`<https://bitbucket.org/fenics-project/dolfin/src/master/python/dolfin/fem/norms.py>`_

Additionally, I define a :func:`get_values` function to fix a FEniCS bug that was present 
in version 2017.2.0: because of the bug, 1D cell meshes were sometimes not 
connected correctly in plots.

"""

# Copyright (C) Daniela Saadeh 2020
# This file is part of phi-enics


# import of original norm and errornorm functions (minus projection)
from six import string_types
import ufl
from ufl import inner, grad, div, curl, dx, FiniteElement, VectorElement, Coefficient
from math import sqrt

import dolfin.cpp as cpp

from dolfin.cpp import GenericVector, GenericFunction, Function, Mesh, error, Vector
from dolfin.fem.interpolation import interpolate


# same imports as in original projection file (minus multimesh)
from dolfin.functions.function import *
from dolfin.functions.expression import *
from dolfin.functions.functionspace import *
from dolfin.fem.assembling import *

from dolfin import Expression


# mine
import numpy as np




def r2_norm(v, func_degree=None, norm_type="L2", mesh=None):
    """
    This function is a modification of FEniCS's built-in norm function that adopts the :math:`r^2dr`
    measure as opposed to the standard Cartesian :math:`dx` measure.

    For documentation and usage, see the 
    `original module <https://bitbucket.org/fenics-project/dolfin/src/master/python/dolfin/fem/norms.py>`_.

    .. note:: Note the extra argument func_degree: this is used to interpolate the :math:`r^2` 
              Expression to the same degree as used in the definition of the Trial and Test function
              spaces.

    .. note:: This modified function also implements this bug fix that was not in the 
              2017.2.0 release:
              <https://bitbucket.org/fenics-project/dolfin/commits/c438724fa5d7f19504d4e0c48695e17b774b2c2d>_

    """

    if not isinstance(v, (GenericVector, GenericFunction)):
        cpp.dolfin_error("norms.py",
                         "compute norm",
                         "expected a GenericVector or GenericFunction")

    # Check arguments
    if not isinstance(norm_type, string_types):
        cpp.dolfin_error("norms.py",
                         "compute norm",
                         "Norm type must be a string, not " +
                         str(type(norm_type)))
    if mesh is not None and not isinstance(mesh, cpp.Mesh):
        cpp.dolfin_error("norms.py",
                         "compute norm",
                         "Expecting a Mesh, not " + str(type(mesh)))

    # Get mesh from function
    if isinstance(v, Function) and mesh is None:
        mesh = v.function_space().mesh()

    # Define integration measure and domain
    dx = ufl.dx(mesh)

    # Select norm type
    if isinstance(v, GenericVector):
        return v.norm(norm_type.lower())
    
    elif (isinstance(v, Coefficient) and isinstance(v, GenericFunction)):
        # DS: HERE IS WHERE I MODIFY
        r2 = Expression('pow(x[0],2)', degree=func_degree)
    
        if norm_type.lower() == "l2":
            M = v**2 * r2 * dx
        elif norm_type.lower() == "h1":
            M = (v**2 + grad(v)**2) * r2 * dx
        elif norm_type.lower() == "h10":
            M = grad(v)**2 * r2 * dx
        elif norm_type.lower() == "hdiv":
            M = (v**2 + div(v)**2) * r2 * dx
        elif norm_type.lower() == "hdiv0":
            M = div(v)**2 * r2 * dx
        elif norm_type.lower() == "hcurl":
            M = (v**2 + curl(v)**2) * r2 * dx
        elif norm_type.lower() == "hcurl0":
            M = curl(v)**2 * r2 * dx
        else:
            cpp.dolfin_error("norms.py",
                             "compute norm",
                             "Unknown norm type (\"%s\") for functions"
                             % str(norm_type))
    else:
        cpp.dolfin_error("norms.py",
                         "compute norm",
                         "Unknown object type. Must be a vector or a function")

    # DS CHANGED: applied this bug fix:
    # https://bitbucket.org/fenics-project/dolfin/diff/site-packages/dolfin/fem/norms.py?diff2=c438724fa5d7&at=jan/general-discrete-gradient
    # Assemble value
    # r = assemble(M, form_compiler_parameters={"representation": "quadrature"})
    r = assemble(M)

    # Check value
    if r < 0.0:
        cpp.dolfin_error("norms.py",
                         "compute norm",
                         "Square of norm is negative, might be a round-off error")
    elif r == 0.0:
        return 0.0
    else:
        return sqrt(r)






    

def r2_errornorm(u, uh, func_degree=None, norm_type="l2", degree_rise=3, mesh=None ):
    """
    This function is a modification of FEniCS's built-in errornorm function that adopts the :math:`r^2dr`
    measure as opposed to the standard Cartesian :math:`dx` measure.

    For documentation and usage, see the 
    `original module <https://bitbucket.org/fenics-project/dolfin/src/master/python/dolfin/fem/norms.py>`_.

    .. note:: Note the extra argument func_degree: this is used to interpolate the :math:`r^2` 
              Expression to the same degree as used in the definition of the Trial and Test function
              spaces.

    """

    # Check argument
    if not isinstance(u, cpp.GenericFunction):
        cpp.dolfin_error("norms.py",
                         "compute error norm",
                         "Expecting a Function or Expression for u")
    if not isinstance(uh, cpp.Function):
        cpp.dolfin_error("norms.py",
                         "compute error norm",
                         "Expecting a Function for uh")

    # Get mesh
    if isinstance(u, Function) and mesh is None:
        mesh = u.function_space().mesh()
    if isinstance(uh, Function) and mesh is None:
        mesh = uh.function_space().mesh()
    if mesh is None:
        cpp.dolfin_error("norms.py",
                         "compute error norm",
                         "Missing mesh")

    # Get rank
    if not u.ufl_shape == uh.ufl_shape:
        cpp.dolfin_error("norms.py",
                         "compute error norm",
                         "Value shapes don't match")
    shape = u.ufl_shape
    rank = len(shape)

    # Check that uh is associated with a finite element
    if uh.ufl_element().degree() is None:
        cpp.dolfin_error("norms.py",
                         "compute error norm",
                         "Function uh must have a finite element")

    # Degree for interpolation space. Raise degree with respect to uh.
    degree = uh.ufl_element().degree() + degree_rise

    # Check degree of 'exact' solution u
    degree_u = u.ufl_element().degree()
    if degree_u is not None and degree_u < degree:
        cpp.warning("Degree of exact solution may be inadequate for accurate result in errornorm.")

    # Create function space
    if rank == 0:
        V = FunctionSpace(mesh, "Discontinuous Lagrange", degree)
    elif rank == 1:
        V = VectorFunctionSpace(mesh, "Discontinuous Lagrange", degree,
                                dim=shape[0])
    elif rank > 1:
        V = TensorFunctionSpace(mesh, "Discontinuous Lagrange", degree,
                                shape=shape)

    # Interpolate functions into finite element space
    pi_u = interpolate(u, V)
    pi_uh = interpolate(uh, V)

    # Compute the difference
    e = Function(V)
    e.assign(pi_u)
    e.vector().axpy(-1.0, pi_uh.vector())

    # Compute norm
    return r2_norm(e, func_degree=func_degree, norm_type=norm_type, mesh=mesh )






def project(v, V=None, func_degree=None, bcs=None, mesh=None,
            function=None,
            solver_type="lu",
            preconditioner_type="default",
            form_compiler_parameters=None):
    """
    This function is a modification of FEniCS's built-in project function that adopts the :math:`r^2dr`
    measure as opposed to the standard Cartesian :math:`dx` measure.

    For documentation and usage, see the 
    `original module <https://bitbucket.org/fenics-project/dolfin/src/master/python/dolfin/fem/projection.py>`_.

    .. note:: Note the extra argument func_degree: this is used to interpolate the :math:`r^2` 
              Expression to the same degree as used in the definition of the Trial and Test function
              spaces.

    """

    # Try figuring out a function space if not specified
    if V is None:
        # Create function space based on Expression element if trying
        # to project an Expression
        if isinstance(v, Expression):
            # FIXME: Add handling of cpp.MultiMesh
            if mesh is not None and isinstance(mesh, cpp.Mesh):
                V = FunctionSpace(mesh, v.ufl_element())
            else:
                cpp.dolfin_error("projection.py",
                                 "perform projection",
                                 "Expected a mesh when projecting an Expression")
        else:
            # Otherwise try extracting function space from expression
            V = _extract_function_space(v, mesh)

    # Check arguments
    if not isinstance(V, (FunctionSpace)):
        cpp.dolfin_error("projection.py",
                         "compute projection",
                         "Illegal function space for projection, not a FunctionSpace:  " +
                         str(v))


    # Ensure we have a mesh and attach to measure
    if mesh is None:
        mesh = V.mesh()
    dx = ufl.dx(mesh)

    # Define variational problem for projection
    
    # DS: HERE IS WHERE I MODIFY
    r2 = Expression('pow(x[0],2)', degree=func_degree)
    
    w = TestFunction(V)
    Pv = TrialFunction(V)
    a = ufl.inner(w, Pv) * r2 * dx
    L = ufl.inner(w, v) * r2 * dx

    # Assemble linear system
    A, b = assemble_system(a, L, bcs=bcs,
                           form_compiler_parameters=form_compiler_parameters)

    # Solve linear system for projection
    if function is None:
        function = Function(V)
    cpp.la_solve(A, function.vector(), b, solver_type, preconditioner_type)

    return function






def get_values( function, output_mesh=False ):
    """
    This function implements a workaround for a bug in the 2017.2.0 version of FEniCS that 
    can result in cells not being properly connected in 1D meshes:
    <https://www.allanswered.com/post/jgvjw/fenics-plot-incorrectly-orders-coordinates-for-function-on-locally-refined-1d-mesh/>_
    
    This function substitutes the built-in compute_vertex_values() method.
    

    *Arguments*
        function
            the function you want to evaluate
        output_mesh
            (option) whether to return the ordered mesh values as well (default: False)

             
    *Example of usage*
    
        .. code-block:: python
    
            f_values = get_values( function )
            r, f_values = get_values( function, output_mesh=True )
    

    """

    mesh = function.function_space().mesh()

    r_values = mesh.coordinates()[:,0]
    X = list(r_values)
    
    unordered_values = function.compute_vertex_values()
    ordered_values = np.array([ _f for _,_f in sorted( zip(X, unordered_values) ) ])

    r_values = np.sort(r_values)

    if output_mesh:
        return r_values, ordered_values
    else:
        return ordered_values

    
