from distutils.core import setup


setup(
    name='UNICORNd3mWrapper',
    version='1.0.0',
    description='UNsupervised Image Clustering with Object Recognition Network primitive',
    packages=['UNICORNd3mWrapper'],
    keywords=['d3m_primitive'],
    install_requires=[
        'pandas >= 0.22.0, < 0.23.0',
        'numpy >= 1.13.3',
        'nk_unicorn >= 1.0.0'
    ],
    dependency_links=[
        "git+https://github.com/NewKnowledge/nk_unicorn@c40ac04dddf4bfa955607f572722d784541690c6#egg=nk_unicorn-1.0.0"
    ],
    entry_points={
        'd3m.primitives': [
            'distil.unicorn = UNICORNd3mWrapper:unicorn'
        ],
    }
)