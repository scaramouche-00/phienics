====================
Theories and solvers
====================


.. note:: Throughout, hatted symbols indicate quantities expressed in in-code dimensionless units:
	  e.g. :math:`\hat{\phi}` indicates the rescaled UV field :math:`\phi/M_{f1}`,
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


.. autoclass:: phienics.UV.UVSolver
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


