from d3m import index
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

# Creating pipeline
pipeline_description = Pipeline(context=Context.TESTING)
pipeline_description.add_input(name='inputs')

# Step 0: Denormalize primitive
step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.denormalize.Common'))
step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
step_0.add_output('produce')
pipeline_description.add_step(step_0)

# Step 1: dataset_to_dataframe
step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_1.add_hyperparameter(name='dataframe_resource', argument_type= ArgumentType.VALUE, data='learningData')
step_1.add_output('produce')
pipeline_description.add_step(step_1)

# Step 2: Unicorn primitive
step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.digital_image_processing.unicorn.Unicorn'))
step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_2.add_hyperparameter(name='image_columns', argument_type= ArgumentType.VALUE, data=['image'])
step_2.add_output('produce')
pipeline_description.add_step(step_2)

# Step 3: imputer -> imputes null values based on mean of column
step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_cleaning.imputer.SKlearn'))
step_3.add_hyperparameter(name='return_result', argument_type=ArgumentType.VALUE,data='replace')
step_3.add_hyperparameter(name='use_semantic_types', argument_type=ArgumentType.VALUE,data=True)
pipeline_description.add_step(step_3)

# Step 4,5,6: Distil ensemble random forest classifier
step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'))
step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
step_4.add_output('produce')
pipeline_description.add_step(step_4)

# extract target columns
step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'))
step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
step_5.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,data=('https://metadata.datadrivendiscovery.org/types/Target',))
step_5.add_output('produce')
pipeline_description.add_step(step_5)

# extract feature columns
step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.learner.random_forest.DistilEnsembleForest'))
step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
step_6.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
step_6.add_output('produce')
pipeline_description.add_step(step_6)

# Final Output
pipeline_description.add_output(name='output predictions', data_reference='steps.6.produce')

# Output json pipeline
blob = pipeline_description.to_json()
filename = blob[8:44] + '.json'
with open(filename, 'w') as outfile:
    outfile.write(blob)

# output dataset metafile (from command line argument)
metafile = blob[8:44] + '.meta'
dataset = sys.argv[1]
with open(metafile, 'w') as outfile:
    outfile.write('{')
    outfile.write(f'"problem": "{dataset}_problem",')
    outfile.write(f'"full_inputs": ["{dataset}_dataset"],')
    outfile.write(f'"train_inputs": ["{dataset}_dataset_TRAIN"],')
    outfile.write(f'"test_inputs": ["{dataset}_dataset_TEST"],')
    outfile.write(f'"score_inputs": ["{dataset}_dataset_SCORE"]')
    outfile.write('}')
