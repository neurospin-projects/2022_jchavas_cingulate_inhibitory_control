import os
from setuptools import setup, find_packages

setup(
    name='name',
    version='0.0.1',
    packages=find_packages(
        exclude=['tests*', 'notebooks*']),
    license='CeCILL license version 2',
    description='Deep learning models '
                'to analyze anterior cingulate sulcus patterns',
    long_description=open('README.rst').read(),
    install_requires=['pandas',
                      'scipy',
		      'psutil',
		      'orca',
                      'matplotlib',
                      'torch',
                      'tqdm',
                      'torchvision',
                      'torch-summary',
                      'hydra',
                      'hydra.core',
                      'dataclasses',
                      'OmegaConf',
                      'sklearn',
                      'scikit-image',
                      'pytorch-lightning',
                      'lightly',
                      'toolz',
		              'ipykernel',
                      'kaleido',
                      'pytorch_ssim',
                      'seaborn',
                      'statsmodels',
                      'umap-learn'
                      ],
    extras_require={"anatomist": ['deep_folding @ \
                        git+https://git@github.com/neurospin/deep_folding',
                      ],
    },
    url='',
    author='',
    author_email=''
)
