===================
How to run the code
===================

:math:`\phi` enics supports both Python 2.7 and Python 3: however, support for Python 2.7 will be discontinued. The Python 2.7 version runs with FEniCS 2017.2.0, whereas the Python 3.x version runs the latest version (2019.1.0 at the time of writing).

Clone the :math:`\phi` enics repository:

``git clone https://github.com/scaramouche-00/phienics.git``

Navigate to the directory of the repository. Checkout the ``python2.7`` branch for Python 2.7 (i.e. enter ``git checkout python2.7``), or stay in the master branch for Python 3.x (i.e. do nothing).

If you use Anaconda, create a new environment, using the ``phienics_env.yml`` file provided:

``conda env create -f phienics_env.yml``

Activate the environment:

``conda activate phienics``

Install the code, using the ``setup.py`` file provided:

``python setup.py install``

Now you can run the code. To play with it, you can start by running the ``UV_main.ipynb`` and ``IR_main.ipynb`` notebooks provided.
