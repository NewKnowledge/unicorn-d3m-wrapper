from distutils.core import setup


setup(
    name='UNICORNd3mWrapper',
    version='1.1.0',
    description='UNsupervised Image Clustering with Object Recognition Network primitive',
    packages=['UNICORNd3mWrapper'],
    keywords=['d3m_primitive'],
    install_requires=[
        'pandas == 0.23.4',
        'numpy >= 1.15.4',
        'd3m_unicorn >= 1.0.0'
    ],
    dependency_links=[
        "git+https://github.com/NewKnowledge/d3m_unicorn@464aa4cdf765dc27bcf1056d827d1ed8c24836ec#egg=d3m_unicorn-1.0.0"
    ],
    entry_points={
        'd3m.primitives': [
            'digital_image_processing.unicorn.Unicorn = UNICORNd3mWrapper:unicorn'
        ],
    }
)
