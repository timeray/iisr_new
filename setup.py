try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Data processing for Irkutsk Incoherent Scatter Radar',
    'author': 'Setov Artem',
    'url': '',
    'download_url': 'Where to download it.',
    'author_email': 'artemsetov@gmail.com',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['iisr'],
    'scripts': [],
    'name': 'iisr'
}

setup(**config, install_requires=['numpy', 'bitstring'])
