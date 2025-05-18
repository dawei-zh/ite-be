from setuptools import setup, find_packages

setup(
    name = 'imaginary-time-optimization',
    version = '1.0',
    #description = 'Solving classical optimization problems by imaginary time evolution based method',
    #author = 'dawei-zh',
    #author_email = 'daweiz@usc.edu',
    python_requires = '>=3.9.12',
    packages=find_packages(),
    install_requires = [
        'numpy==1.26.4',
        'networkx==3.2.1',
        'qiskit==1.2.0',
        'qiskit-aer==0.14.2'
    ],
    classifiers = [
        'Programming Language :: Python :: 3.9',
    ]
)