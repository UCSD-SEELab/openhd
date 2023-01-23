from setuptools import setup, find_packages


if __name__ == '__main__':
    setup(name='openhd',
            python_requires='>=3.7, <3.9',
            version='0.1',
            description='OpenHD for HD computing acceleration on GPGPU',
            url='https://github.com/UCSD-SEELab/openhd',
            author='SEELAB @ UCSD + CELL @ DGIST',
            author_email='openhd.mailing@gmail.com',
            license='MIT',
            packages=find_packages(),
            include_package_data=True,
            install_requires=[
                'numpy',
                'astor',
                'sympy==1.2',
                'pyimage',
                'pydot', # Graphviz has to be manually installed
                'scikit-cuda', # Graphviz has to be manually installed
                ],
            zip_safe=False)
