""" Setup config
"""

from setuptools import setup

setup(
    name='sleep_analysis',
    version='0.1',
    author='Andrew Berger',
    author_email='a5b@stanford.edu',
    description='Project package for sleep data analysis',
    url='https://github.com/ClandininLab/sleep_analysis',
    packages=['sleep_analysis'],
    python_requires='>=3.5',
    install_requires=[
        'matplotlib',
        'pandas',
        'seaborn',
        'pingouin'
    ],
    extras_require={
        'test': ['nose', 'hypothesis']
    }
)
