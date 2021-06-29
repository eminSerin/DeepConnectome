from setuptools import setup

setup(
    name='DeepConnectome',
    version='0.1',
    packages=['bin', 'docs', 'tests', 'deepconnectome', 'deepconnectome.io', 'deepconnectome.viz',
              'deepconnectome.utils', 'deepconnectome.models', 'deepconnectome.selection',
              'deepconnectome.preprocessing'],
    url='',
    license='GPL-3.0',
    author='Emin Serin',
    author_email='emin.serin@charite.de',
    description='A python toolbox to make connectome-based predictions using BrainNetCNN. '
)
