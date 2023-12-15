# %%
from setuptools import setup, find_packages

# %%
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

# %%
setup(name='parnet',
      version='0.1.0',
      description='A PyTorch extension of the RBPNet model as described in Horlacher et al. (2023), DOI: https://doi.org/10.1186/s13059-023-03015-7.',
      url='http://github.com/mhorlacher/parnet',
      author='Marc Horlacher',
      author_email='marc.horlacher@gmail.com',
      license='MIT',
      install_requires=requirements,
      packages=find_packages(),
      include_package_data=True,
      entry_points = {
            'console_scripts': [
                  'parnet=parnet.__main__:main',
            ],
      },
      zip_safe=False)