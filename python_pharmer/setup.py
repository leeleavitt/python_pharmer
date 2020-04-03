import setuptools 

setuptools.setup(name='python_pharmer',
      version='1',
      description='Extracting ND2 video data',
      author='Lee Leavitt, Rishi Alluri',
      author_email='lee.leavitt.u@gmail.com',
      packages=setuptools.find_packages(),
      package_data={'':['models/*.h5']}
      )

