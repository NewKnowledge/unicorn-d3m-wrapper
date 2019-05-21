import os
import sys
import typing
import numpy as np
import pandas as pd

from d3m_unicorn import *

from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base
from common_primitives import utils as utils_cp, dataset_to_dataframe as DatasetToDataFrame
from common_primitives import denormalize

from keras import backend as K

__author__ = 'Distil'
__version__ = '1.1.0'
__contact__ = 'mailto:nklabs@newknowledge.io'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    target_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=(),
        max_size=sys.maxsize,
        min_size=0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='names of columns with image paths'
    )

    output_labels = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=(),
        max_size=sys.maxsize,
        min_size=0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='desired names for croc output columns'
    )


class unicorn(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
            Produce image object classification predictions and OCR for an
            image provided as an URI or filepath

        Parameters
        ----------
        inputs : pandas dataframe where a column is a pd.Series of image paths/URLs

        Returns
        -------
        output : A dataframe with image labels/classifications/cluster assignments
        """
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "475c26dc-eb2e-43d3-acdb-159b80d9f099",
        'version': __version__,
        'name': "unicorn",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Image Clustering', 'fast fourier transfom', 'Image'],
        'source': {
            'name': __author__,
            'contact': __contact__,
            'uris': [
                # Unstructured URIs.
                "https://github.com/NewKnowledge/unicorn-d3m-wrapper",
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
        "installation": [
            {
                "type": "PIP",
                "package_uri": "git+https://github.com/NewKnowledge/d3m_unicorn.git@c53240153cb6afc016adf3df569b86e4afe20bcd#egg=d3m_unicorn"
            },
            {
                "type": "PIP",
                "package_uri": "git+https://github.com/NewKnowledge/unicorn-d3m-wrapper.git@{git_commit}#egg=UNICORNd3mWrapper".format(
                    git_commit=utils.current_git_commit(os.path.dirname(__file__))
                ),
            },
                                  {
            "type": "TGZ",
            "key": "croc_weights",
            "file_uri": "http://public.datadrivendiscovery.org/croc.tar.gz",
            "file_digest":"0be3e8ab1568ec8225b173112f4270d665fb9ea253093cd9ea98c412c9053c92"
        },
        ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.digital_image_processing.unicorn.Unicorn',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        "algorithm_types": [
            metadata_base.PrimitiveAlgorithmType.MULTILABEL_CLASSIFICATION # TODO
        ],
        "primitive_family": metadata_base.PrimitiveFamily.DIGITAL_IMAGE_PROCESSING
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, volumes: typing.Dict[str,str]=None)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed,  volumes=volumes)
        
        self.volumes = volumes

    def _get_column_base_path(self, inputs: Inputs, column_name: str) -> str:
        # fetches the base path associated with a column given a name if it exists
        column_metadata = inputs.metadata.query((metadata_base.ALL_ELEMENTS,))
        if not column_metadata or len(column_metadata) == 0:
            return None

        num_cols = column_metadata['dimension']['length']
        for i in range(0, num_cols):
            col_data = inputs.metadata.query((metadata_base.ALL_ELEMENTS, i))
            if col_data['name'] == column_name and 'location_base_uris' in col_data:
                return col_data['location_base_uris'][0]

        return None

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
            Produce image object classification predictions and OCR for an
            image provided as an URI or filepath

        Parameters
        ----------
        inputs : pandas dataframe where a column is a pd.Series of image paths/URLs

        Returns
        -------
        output : A dataframe with image labels/classifications/cluster assignments
        """
    
        target_columns = self.hyperparams['target_columns']
        output_labels = self.hyperparams['output_labels']
        
        imagepath_df = inputs
        image_analyzer = Unicorn(weights_path=self.volumes["croc_weights"]+"/inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

        for i, ith_column in enumerate(target_columns):
            # initialize an empty dataframe
            result_df = pd.DataFrame()
            output_label = output_labels[i]


            # get the base uri from the column metadata and remove the the
            # scheme portion
            base_path = self._get_column_base_path(inputs, ith_column)
            if base_path:
                base_path = base_path.split('://')[1]

            # update the paths with the base if necessary
            col_paths = imagepath_df.loc[:, ith_column]
            if base_path:
                for i in range(0, len(col_paths)):
                    col_paths[i] = os.path.join(base_path, col_paths[i])

            result_df = image_analyzer.cluster_images(col_paths)
            imagepath_df = imagepath_df.iloc[:, 0]
       
            imagepath_df = pd.concat(
                [imagepath_df.reset_index(drop=True), result_df], axis=1)
            imagepath_df.columns = ['d3mIndex', 'image', 'label']
            

        K.clear_session()
        
        # create metadata for the unicorn output dataframe
        unicorn_df = d3m_DataFrame(imagepath_df)
        # first column (d3mIndex)
        col_dict = dict(unicorn_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type("1")
        col_dict['name'] = 'd3mIndex'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        unicorn_df.metadata = unicorn_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)
        # second column (image)
        col_dict = dict(unicorn_df.metadata.query((metadata_base.ALL_ELEMENTS, 1)))
        col_dict['structural_type'] = type("it is a string")
        col_dict['name'] = 'image'
        col_dict['semantic_types'] = ('http://schema.org/Text', 'https://metadata.datadrivendiscovery.org/types/Attribute')
        unicorn_df.metadata = unicorn_df.metadata.update((metadata_base.ALL_ELEMENTS, 1), col_dict)
        # third column (label)
        col_dict = dict(unicorn_df.metadata.query((metadata_base.ALL_ELEMENTS, 2)))
        col_dict['structural_type'] = type("1")
        col_dict['name'] = 'label'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute')
        unicorn_df.metadata = unicorn_df.metadata.update((metadata_base.ALL_ELEMENTS, 2), col_dict)
        
        return CallResult(unicorn_df)


if __name__ == '__main__':
    volumes = {} # d3m large primitive architecture dictionary of large files
    volumes["croc_weights"]= '/home/croc_weights' # location of extracted required files archive
    client = unicorn(
        hyperparams={
            'target_columns': ['filename'],
            'output_labels': ['label']}, volumes=volumes)
    hyperparams_class = denormalize.DenormalizePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    denorm = denormalize.DenormalizePrimitive(hyperparams = hyperparams_class.defaults())
    input_dataset = denorm.produce(inputs = container.Dataset.load("file:///home/datasets/seed_datasets_current/124_188_usps/TRAIN/dataset_TRAIN/datasetDoc.json")).value 
    ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = {"dataframe_resource":"learningData"})
    df = d3m_DataFrame(ds2df_client.produce(inputs = input_dataset).value) 
    result = client.produce(inputs=df)
    print(result.value)
