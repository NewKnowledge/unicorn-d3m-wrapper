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
step_2.add_hyperparameter(name='target_columns', argument_type= ArgumentType.VALUE, data=['filename'])
step_2.add_hyperparameter(name='output_labels', argument_type= ArgumentType.VALUE, data=['label'])
step_2.add_output('produce')
pipeline_description.add_step(step_2)

# Final Output
pipeline_description.add_output(name='output predictions', data_reference='steps.2.produce')

# Output to JSON
with open('pipeline.json', 'w') as outfile:
    outfile.write(pipeline_description.to_json())
