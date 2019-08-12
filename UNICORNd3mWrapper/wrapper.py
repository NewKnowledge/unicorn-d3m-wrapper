import os
import sys
import typing
import numpy as np
import pandas as pd

from d3m_unicorn import *

from d3m.primitive_interfaces.base import CallResult, PrimitiveBase

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base
from common_primitives import utils as utils_cp, dataset_to_dataframe as DatasetToDataFrame
from common_primitives import denormalize
from sklearn.preprocessing import LabelEncoder

from keras import backend as K

__author__ = 'Distil'
__version__ = '1.1.0'
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
    batch_size = hyperparams.UniformInt(
        lower = 1, 
        upper = 256,
        upper_inclusive=True, 
        default = 32, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'batch size'
    )
    last_layer_epochs = hyperparams.UniformInt(
        lower = 1, 
        upper = sys.maxsize,
        default = 100, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'number of epochs'
    )
    last_block_epochs = hyperparams.UniformInt(
        lower = 1, 
        upper = sys.maxsize,
        default = 10, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'number of epochs'
    )
    include_class_weights = hyperparams.UniformBool(
        default = True, 
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="whether to include class weights in finetuning of Inception model")


class unicorn(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
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
            # {
            # "type": "TGZ",
            # "key": "croc_weights",
            # "file_uri": "http://public.datadrivendiscovery.org/croc.tar.gz",
            # "file_digest":"0be3e8ab1568ec8225b173112f4270d665fb9ea253093cd9ea98c412c9053c92"
            # },
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
        
        self.image_analyzer = Unicorn(weights_path=self.volumes["croc_weights"]+"/inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
        self.image_paths = None
        self.image_labels = None
        self.class_weights = None
        self.targets = None

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

    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params: Params) -> None:
        self.params = params

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        '''
            Sets primitive's training data
            
            Parameters
            ----------
            inputs: column(s) of image paths
            outputs: labels from dataframe's target column
        '''

        # create single list of image paths from all target image columns
        self.image_paths = []
        for col in self.hyperparams['image_columns']:
            # get the base uri from the column metadata and remove the the scheme portion
            base_path = self._get_column_base_path(inputs, col).split('://')[1]
            self.image_paths.extend([os.path.join(base_path, c) for c in inputs[col]])

        # broadcast image labels for each column of images
        self.targets = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        if not len(self.targets):
            self.targets = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/Target')
        if not len(self.targets):
            self.targets = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')

        # assert that the number of image columns is less than or equal to the number of target columns in the df
        assert len(self.hyperparams['image_columns']) <= len(self.targets), "List of image columns cannot be longer than list of target columns"

        # train label encoder
        self.encoder = LabelEncoder().fit(inputs.iloc[:,self.targets[0]])
        self.image_labels = self.encoder.transform(np.repeat(inputs.iloc[:,self.targets[0]], len(self.hyperparams['image_columns'])))

        # calculate class weights for target labels if desired
        if self.hyperparams['include_class_weights']:
            self.class_weights = dict(inputs.iloc[:,targets[0]].value_counts())
            if len(self.hyperparams['image_columns']) > 1:
                self.class_weights.update((k, v * len(self.hyperparams['image_columns'])) for k, v in self.class_weights.items())

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        '''
            Trains a single Inception model on all columns of image paths using dataframe's target column
        '''
        self.finetune_model = self.image_analyzer.finetune(
            self.image_paths, 
            self.image_labels,
            nclasses = len(self.encoder.classes_),
            batch_size = self.hyperparams['batch_size'],
            last_layer_epochs = self.hyperparams['last_layer_epochs'],
            last_block_epochs = self.hyperparams['last_block_epochs'],
            class_weight = self.class_weights)

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
    
        # get metadata labels for primary key and target label columns
        key = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        col_names = [inputs.metadata.query_column(key[0])['name']]
        target_names = [inputs.metadata.query_column(idx)['name'] for idx in self.targets]

        # create output dataframe
        result_df = d3m_DataFrame(pd.DataFrame(columns=col_names.extend(target_names)))
        result_df[key] = inputs[key]
        col_dict = dict(result_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type(1)
        col_dict['name'] = key
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        result_df.metadata = result_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)

        for idx, col in enumerate(self.hyperparams['image_columns']):
            base_path = self._get_column_base_path(inputs, col).split('://')[1]
            image_paths = [os.path.join(base_path, c) for c in inputs[col]]
            preds = self.image_analyzer.finetuned_predict(image_paths, batch_size = self.hyperparams['batch_size'])
            result_df[target_names[idx]] = self.encoder.inverse_transform(np.argmax(preds))
            col_dict = dict(result_df.metadata.query((metadata_base.ALL_ELEMENTS, idx+1)))
            col_dict['structural_type'] = type(1)
            col_dict['name'] = target_names[idx]
            col_dict['semantic_types'] = ('http://schema.org/Integer', 
                                        'https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 
                                        'https://metadata.datadrivendiscovery.org/types/TrueTarget', 
                                        'https://metadata.datadrivendiscovery.org/types/Target')
            result_df.metadata = result_df.metadata.update((metadata_base.ALL_ELEMENTS, idx+1), col_dict)
        
        return CallResult(result_df)

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
