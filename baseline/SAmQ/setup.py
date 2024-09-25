from distutils.core import setup

setup(
    name='SAmQ',
    version='0.1.0',
    author='Sinong Geng',
    author_email='sgeng@cs.princeton.edu',
    packages=['SAmQ'],
    url='https://github.com/gengsinong/SAmQ.git',
    license='MIT LICENSE',
    description='State aggregation minimizing Q error for DDM estimation',
    long_description=open('README.md').read(),
    install_requires=[
        "torch==1.10.0",
        "tensorboard==2.10.0",
        "psutil==5.9.2",
        "protobuf==3.13.0",
        "dowel==0.0.3",
        "tqdm==4.62.3",
        "wandb==0.13.4",
        "protobuf==3.19.4",
    ]
)
