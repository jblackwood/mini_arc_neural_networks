Ok now i want to make a change where I encode the arc task as a knowledge graph into a rank 3 tensor and the encoder does tucker decomposition on the rank 3 tensor.

write a function to transform a (batch_size, 50) tensor for an arc task into a (batch_size, num_subject, num_relation, num_object) size tensor that encodes the binary facts of the input and output grid as a knowledge graph.

for (subject, relation, object) facts of the knowledge base:
      - for each cell, include it as a subject and have relations for it's row, column, and value. So (cell_0, row, 0), (cell_0, column, 0). The row and column should share the same object dimension, but the cell value should be a unique object dimension since the cell values represent colors not integer values.
      - every integer value should be a subject with it's relation for the value below it. So (1, above, 0), (2, above, 1), (3, above, 2), etc.
      - each cell should have a relation indicating if it's in the input or output grid. So (cell_0, grid_type, input_grid) (cell_26, grid_type, output_grid).

write a unit test file for the function to ensure it works correctly. 

for the task embedding. reshape it into a small rank 3 tensor