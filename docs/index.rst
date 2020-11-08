.. phi-enics documentation master file, created by
   sphinx-quickstart on Wed Mar 25 22:05:28 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to phi-enics's documentation!
******************************************

.. toctree::
   :maxdepth: 2
   :caption: Contents:








===================
How to run the code
===================

This code works with the FEniCS library, so you will first need to have FEniCS installed.

.. note:: This version of code runs with Python 2.7 and FEniCS 2017.2.0.

I recommend running FEniCS with docker, as described in these instructions - you will need the points
‘Quickstart’ (1), ‘Introduction’ (2) and ‘Running Jupyter notebooks’ (5):
`<http://fenics.readthedocs.io/projects/containers/en/latest/>`_

.. note:: If you are a Windows user, please make sure to use Docker Toolbox (not Docker), as stated in the link above.
 
Once docker and the FEniCS library are setup, phi-enics can be run using the jupyter notebooks provided
(remember to follow point (5) of the link above).
 
To play with the code, you can start by running the UV_main.ipynb and IR_main.ipynb notebooks provided,
which compute the field profiles, gradients and operators for given model parameters.

 
===============   
Code components
===============

For the physics/computational details behind the code components, please see the accompanying
paper <this_will_be_the_arXiv_link>_.


Mesh and other FEM details
==========================


Base class
----------

.. autoclass:: phienics.mesh.Mesh
   :members:

ArcTanExpMesh
-------------

.. autoclass:: phienics.atexpmesh.ArcTanExpMesh
   :members:


ArcTanPowerLawMesh
------------------

.. autoclass:: phienics.atplmesh.ArcTanPowerLawMesh
   :members:


Other FEM details
-----------------

.. autoclass:: phienics.fem.Fem
   :members:




Source
======

Important mathematical details on the source profiles are given in Sec. 2 of the accompanying paper
<this_will_be_the_arXiv_link>_ .

.. note:: For all source classes, the source mass must be expressed in units of :math:`M_P`,
	  and the source radius in units :math:`{M_P}^{-1}`.

.. autoclass:: phienics.source.Source
   :members:

.. autoclass:: phienics.tophatsource.TopHatSource
   :members:

.. autoclass:: phienics.tophatsource.StepSource
   :members:

.. autoclass:: phienics.cossource.CosSource
   :members:

.. autoclass:: phienics.gausssource.GaussianSource
   :members:

.. autoclass:: phienics.gcakesource.GCakeSource
   :members:


Solvers
=======

.. note:: Throughout, hatted symbols indicate quantities expressed in in-code dimensionless units:
	  e.g. :math:`\hat{\phi}` indicates the rescaled :ref:`UV` field :math:`\phi/M_{f1}`,
	  where :math:`M_{f1}` is some mass.

	  

Base class
----------

.. autoclass:: phienics.solver.Solver
   :members:


      
.. _UV:
UV theory
---------

.. autoclass:: phienics.UV.UVFields
   :members:


.. autoclass:: phienics.V.UVSolver
   :members:


      
IR theory
---------


.. autoclass:: phienics.IR.IRFields
   :members:


.. autoclass:: phienics.IR.IRSolver
   :members:



Gravity
-------

.. autoclass:: phienics.gravity.PoissonSolver
   :members:

      

Utilities
=========
.. automodule:: phienics.utils
   :members:


