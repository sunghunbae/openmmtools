from setuptools import setup
if __name__ == '__main__':
    setup(
        install_requires=[
            'openmm>=8',
            'mdtraj',
            'netCDF4',
            'mpi4py',
            'pymbar',
            'jax',
            'numba',
            'scipy',
            'numpy<=1.25',
            'ase',
            'matplotlib',
            'rdkit',
            'Pillow',
            ],
        package_data={
            },
        scripts=[]
    )
