# phienics

Accurate screening with the finite element method

## Warning

This README is a work in progress. phienics is being actively developed and may contain bugs or instabilities! I strongly recommend getting in touch with me if you're planning to use the code so I can help support you. 

## Super quickstart

This code works with Python 2.7 and the FEniCS library, version 2017.2.0.

I recommend running FEniCS in a docker container. Start by installing docker, by following the instructions at [docker.com](https://docs.docker.com/engine/getstarted/step_one/).

<bf> Note: </bf> If you are a Windows user, please make sure to use Docker Toolbox (not Docker).

Familiarise yourself with running FEniCS in Docker, by following the 'Quickstart' (1), 'Introduction' (2) and 'Running Jupyter notebooks' points [here][http://fenics.readthedocs.io/projects/containers/en/latest/]

To pull the 2017.2.0 version of FEniCS, specify:
`quay.io/fenicsproject/stable:2017.2.0`
in the commands above, as opposed to `quay.io/fenicsproject/stable`

Clone or download the Astronomaly repository:<br>
`git clone https://github.com/scaramouche-00/phienics`

Navigate to the phienics folder. To play with the code, you can start by running the UV_main.ipynb and IR_main.ipynb notebooks provided, which compute the field profiles, gradients and operators for two case-study theories.




