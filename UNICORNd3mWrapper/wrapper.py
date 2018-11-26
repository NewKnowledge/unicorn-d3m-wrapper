import os
import sys
import typing
import numpy as np
import pandas as pd

from d3m_unicorn import *

from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.metadata import hyperparams, base as metadata_base, params

from keras import backend as K

__author__ = 'Distil'
__version__ = '1.0.0'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame


class Params(params.Params):
    pass


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


class unicorn(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "'475c26dc-eb2e-43d3-acdb-159b80d9f099'",
        'version': __version__,
        'name': "unicorn",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Image Clustering', 'fast fourier transfom', 'Image'],
        'source': {
            'name': __author__,
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
                "package_uri": "git+https://github.com/NewKnowledge/d3m_unicorn.git@97b24ce39c3a26c1d753104c80012c352efd6920#egg=d3m_unicorn"
            },
            {
                "type": "PIP",
                "package_uri": "git+https://github.com/NewKnowledge/unicorn-d3m-wrapper.git@{git_commit}#egg=UNICORNd3mWrapper".format(
                    git_commit=utils.current_git_commit(os.path.dirname(__file__))
                ),
            }
        ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.distil.unicorn',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        "algorithm_types": [
            metadata_base.PrimitiveAlgorithmType.MULTILABEL_CLASSIFICATION # TODO
        ],
        "primitive_family": metadata_base.PrimitiveFamily.DIGITAL_IMAGE_PROCESSING
    })

    def __init__(self, *, hyperparams: Hyperparams)-> None:
        super().__init__(hyperparams=hyperparams)

    def fit(self) -> None:
        pass

    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params: Params) -> None:
        self.params = params

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        pass

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

    def produce(self, *, inputs: Inputs) -> CallResult[Outputs]:
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
        image_analyzer = Unicorn()

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

            imagepath_df = pd.concat(
                [imagepath_df.reset_index(drop=True), result_df], axis=1)

        K.clear_session()
        return CallResult(imagepath_df)


if __name__ == '__main__':
    client = unicorn(
        hyperparams={
            'target_columns': ['test_column'],
            'output_labels': ['test_column_prefix']})
    imagepath_df = pd.DataFrame(
        pd.Series(['http://i0.kym-cdn.com/photos/images/facebook/001/253/011/0b1.jpg',
                   'http://i0.kym-cdn.com/photos/images/facebook/001/253/011/0b1.jpg']))
    imagepath_df.columns = ['test_column']
    result = client.produce(inputs=imagepath_df)
    print(result.head)
