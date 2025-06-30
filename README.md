# VisionLanguageActionModel

This code follows recent research of hierarchical reinforcement learning through the GPT framework

This implementation includes: 
A detectron2 maskrcnn with fine tuning on the RoboSeg dataset for instance segmentation of robot manipulators, fitted with data handling and augmentation through randomized augmentations. This part represents image features for the VLA

Includes a from scratch implementation of a GPT in PyTorch for robot manipulator inference of multimodal image and natural language input for task execution, trained on the CALVIN dataset.

The GPT's encoder is trained on a latent representation vector of a hierarchical action plan, with the decoder being trained through imitation learning of expert trajectories.
