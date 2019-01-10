from distutils.core import setup


setup(
    name='UNICORNd3mWrapper',
    version='1.1.0',
    description='UNsupervised Image Clustering with Object Recognition Network primitive',
    packages=['UNICORNd3mWrapper'],
    keywords=['d3m_primitive'],
    install_requires=[
        'pandas >= 0.22.0, < 0.23.0',
        'numpy >= 1.13.3',
        'd3m_unicorn >= 1.0.0'
    ],
    dependency_links=[
        "git+https://github.com/NewKnowledge/d3m_unicorn@924713009fbbc8d7b4c564f1106d46d6cda595c5#egg=d3m_unicorn-1.0.0"
    ],
    entry_points={
        'd3m.primitives': [
            'distil.unicorn = UNICORNd3mWrapper:unicorn'
        ],
    }
)
