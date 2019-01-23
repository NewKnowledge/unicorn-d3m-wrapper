# Unicorn (UNsupervised Image Clustering with Object Recognition Network) D3M Wrapper

Wrapper of the Unicorn library into D3M infrastructure. All code is written in Python 3.5 and must be run in 3.5 or greater.

The base library can be found here: https://github.com/NewKnowledge/d3m_unicorn.

## Install

pip3 install -e git+https://github.com/NewKnowledge/unicorn-d3m-wrapper.git#egg=UNICORN3mWrapper --process-dependency-links

## Output

The output is a dataframe with 5 columns. The first column is an index, the second column is the filepath or URI of an image, the third column is a bounding box around the detected object, the fourth column is a label for the detected object, and the fifth column is a clustering assignment for the image. 


## Available Functions

#### produce
Produce image object recognition and object classification predictions for an image provided as a URI or filepath. The input is a pandas dataframe where a column is a pd.Series of image paths/URLs. The output is described above. 
