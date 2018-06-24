from setuptools import find_packages
from setuptools import setup


version = '0.0.0'


setup(
    name='chainer_epic_kitchens',
    version=version,
    packages=find_packages(),
    install_requires=open('requirements.txt').readlines(),
    description='',
    long_description=open('README.md').read(),
    author='Shingo Kitagawa',
    author_email='shingogo.5511@gmail.com',
    url='https://github.com/knorth55/chainer-epic-kitchens',
    license='MIT',
    keywords='machine-learning',
)
