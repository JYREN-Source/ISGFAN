Please refer to the "requirements" file for the environment setup and dependencies needed for this project.

"main.py" contains the model training code, including dynamic loss weighting, forward propagation data flow, and other training-related components. Simply run this file to start the training process.

"test.py" contains the model testing code and includes visualization generation such as t-SNE plots and confusion matrices.

The "Data" folder includes operations for noise addition and dataset partitioning. It also contains data from two working conditions of the CWRU dataset for convenient experimentation. Run "main.py" to get started.

The "losses" folder contains implementations of: Gradient Reversal Layer, Improved Orthogonal Loss, Focal Domain Loss (with subdomain attention mechanism)

The "Model" folder contains the structure of each model module.

The "visualization" folder stores visualization images. The "result" foldercontains a pre-trained model that can be directly used for testing
