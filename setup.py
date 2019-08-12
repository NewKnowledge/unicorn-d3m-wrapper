from distutils.core import setup


setup(
    name='UNICORNd3mWrapper',
    version='1.1.1',
    description='UNsupervised Image Clustering with Object Recognition Network primitive',
    packages=['UNICORNd3mWrapper'],
    keywords=['d3m_primitive'],
    install_requires=[
        'pandas == 0.23.4',
        'numpy >= 1.15.4',
        'scikit-learn == 0.20.3',        
        'd3m_unicorn @ git+https://github.com/NewKnowledge/d3m_unicorn@90826ef95a5e50932bae85537211d7bb715d02ae#egg=d3m_unicorn-1.0.0'
    ],
    entry_points={
        'd3m.primitives': [
            'digital_image_processing.unicorn.Unicorn = UNICORNd3mWrapper:unicorn'
        ],
    }
)
