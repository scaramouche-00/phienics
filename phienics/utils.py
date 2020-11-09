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
import ufl
from ufl import inner, grad, div, curl, dx, FiniteElement, VectorElement, Coefficient
from math import sqrt

import dolfin.cpp as cpp

from dolfin.cpp.la import GenericVector, Vector
from dolfin.function.function import Function
from dolfin.cpp.mesh import Mesh
from dolfin.fem.interpolation import interpolate
from dolfin.function.functionspace import (FunctionSpace,
                                           VectorFunctionSpace,
                                           TensorFunctionSpace)

from dolfin.fem.projection import _extract_function_space


# mine
from dolfin import Expression, TestFunction, TrialFunction
from dolfin import assemble_system, assemble
import numpy as np




def r2_norm(v, func_degree=None, norm_type="L2", mesh=None):
    """
    This function is a modification of FEniCS's built-in norm function that adopts the :math:`r^2dr`
    measure as opposed to the standard Cartesian :math:`dx` measure.

    For documentation and usage, see the 
    original module <https://bitbucket.org/fenics-project/dolfin/src/master/python/dolfin/fem/norms.py>_.

    .. note:: Note the extra argument func_degree: this is used to interpolate the :math:`r^2` 
              Expression to the same degree as used in the definition of the Trial and Test function
              spaces.

    """


    # Get mesh from function
    if isinstance(v, cpp.function.Function) and mesh is None:
        mesh = v.function_space().mesh()

    # Define integration measure and domain
    dx = ufl.dx(mesh)

    # Select norm type
    if isinstance(v, GenericVector):
        return v.norm(norm_type.lower())
    
    elif (isinstance(v, Coefficient) and isinstance(v, Function)):
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
            raise ValueError("Unknown norm type {}".format(str(norm_type)))
    else:
        raise TypeError("Do not know how to compute norm of {}".format(str(v)))

    # Assemble value and return
    return sqrt(assemble(M))






    

def r2_errornorm(u, uh, norm_type="l2", degree_rise=3, mesh=None ):
    """
    This function is a modification of FEniCS's built-in errornorm function that adopts the :math:`r^2dr`
    measure as opposed to the standard Cartesian :math:`dx` measure.

    For documentation and usage, see the 
    original module <https://bitbucket.org/fenics-project/dolfin/src/master/python/dolfin/fem/norms.py>_.

    """


    # Get mesh
    if isinstance(u, cpp.function.Function) and mesh is None:
        mesh = u.function_space().mesh()
    if isinstance(uh, cpp.function.Function) and mesh is None:
        mesh = uh.function_space().mesh()
    # if isinstance(uh, MultiMeshFunction) and mesh is None:
    #     mesh = uh.function_space().multimesh()
    if hasattr(uh, "_cpp_object") and mesh is None:
        mesh = uh._cpp_object.function_space().mesh()
    if hasattr(u, "_cpp_object") and mesh is None:
        mesh = u._cpp_object.function_space().mesh()
    if mesh is None:
        raise RuntimeError("Cannot compute error norm. Missing mesh.")

    # Get rank
    if not u.ufl_shape == uh.ufl_shape:
        raise RuntimeError("Cannot compute error norm. Value shapes do not match.")
    
    shape = u.ufl_shape
    rank = len(shape)

    # Check that uh is associated with a finite element
    if uh.ufl_element().degree() is None:
        raise RuntimeError("Cannot compute error norm. Function uh must have a finite element.")

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
    return r2_norm(e, func_degree=degree, norm_type=norm_type, mesh=mesh )






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
            if mesh is not None and isinstance(mesh, cpp.mesh.Mesh):
                V = FunctionSpace(mesh, v.ufl_element())
            # else:
            #     cpp.dolfin_error("projection.py",
            #                      "perform projection",
            #                      "Expected a mesh when projecting an Expression")
        else:
            # Otherwise try extracting function space from expression
            V = _extract_function_space(v, mesh)



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
    cpp.la.solve(A, function.vector(), b, solver_type, preconditioner_type)

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

    
