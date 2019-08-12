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
__version__ = '1.1.1'
__contact__ = 'mailto:nklabs@newknowledge.io'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    image_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=(),
        max_size=sys.maxsize,
        min_size=0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='names of columns with image paths'
    )

class unicorn(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
        """
            Produce image classification predictions by clustering an Inception model
            finetuned on all columns of images in the dataset (assumption = single column of target labels)

            Parameters
            ----------
            inputs : d3m dataframe with columns of image paths and optional labels

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
        self.image_analyzer = Unicorn(weights_path=self.volumes["croc_weights"]+"/inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
            Produce image object classification predictions

            Parameters
            ----------
            inputs : d3m dataframe with columns of image paths and optional labels

            Returns
            -------
            output : A dataframe with image labels/classifications/cluster assignments
        """

        # cluster each column of images separately (bc producing separate set of labels for each column)
        
        result_df = d3m_DataFrame(pd.DataFrame())
        for idx, col in enumerate(self.hyperparams['image_columns']):
            base_path = self._get_column_base_path(inputs, col).split('://')[1]
            image_paths = [os.path.join(base_path, c) for c in inputs[col]])

            # instantiate Inception model and produce cluster labels
            image_analyzer = Unicorn(weights_path=self.volumes["croc_weights"]+"/inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
            result_df[col + '_cluster_labels'] = image_analyzer.cluster_images(image_paths)['pred_class']

            col_dict = dict(result_df.metadata.query((metadata_base.ALL_ELEMENTS, idx)))
            col_dict['structural_type'] = type(1)
            col_dict['name'] = col + '_cluster_labels'
            col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute', 'https://metadata.datadrivendiscovery.org/types/CategoricalData')
            result_df.metadata = result_df.metadata.update((metadata_base.ALL_ELEMENTS, idx), col_dict)
        
        # create metadata for whole dictionary
        df_dict = dict(result_df.metadata.query((metadata_base.ALL_ELEMENTS, )))
        df_dict_1 = dict(result_df.metadata.query((metadata_base.ALL_ELEMENTS, ))) 
        df_dict['dimension'] = df_dict_1
        df_dict_1['name'] = 'columns'
        df_dict_1['semantic_types'] = ('https://metadata.datadrivendiscovery.org/types/TabularColumn',)
        df_dict_1['length'] = len(self.hyperparams['image_columns'])       
        result_df.metadata = result_df.metadata.update((metadata_base.ALL_ELEMENTS,), df_dict)

        return CallResult(utils_cp.append_columns(inputs, result_df))

# if __name__ == '__main__':
#     volumes = {} # d3m large primitive architecture dictionary of large files
#     volumes["croc_weights"]= '/home/croc_weights' # location of extracted required files archive
#     client = unicorn(
#         hyperparams={
#             'target_columns': ['filename'],
#             'output_labels': ['label']}, volumes=volumes)
#     hyperparams_class = denormalize.DenormalizePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
#     denorm = denormalize.DenormalizePrimitive(hyperparams = hyperparams_class.defaults())
#     input_dataset = denorm.produce(inputs = container.Dataset.load("file:///home/datasets/seed_datasets_current/124_188_usps/TRAIN/dataset_TRAIN/datasetDoc.json")).value 
#     ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = {"dataframe_resource":"learningData"})
#     df = d3m_DataFrame(ds2df_client.produce(inputs = input_dataset).value) 
#     result = client.produce(inputs=df)
#     print(result.value)
