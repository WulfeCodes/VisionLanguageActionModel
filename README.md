# VisionLanguageActionModel

This code follows recent research of hierarchical reinforcement learning through the GPT framework

This implementation includes: 
A detectron2 maskrcnn with fine tuning on the RoboSeg dataset for instance segmentation of robot manipulators, fitted with data handling and augmentation through randomized augmentations. This part represents image features for the VLA

Includes a from scratch implementation of a GPT in PyTorch for robot manipulator inference of multimodal image and natural language input for task execution, trained on the CALVIN dataset.

The GPT's encoder is trained on a latent representation vector of a hierarchical action plan, with the decoder being trained through imitation learning of expert trajectories.

## Implementation Steps
If you wish to train the instance segmentation model yourself call the fine_tune() method, simply just pass in the RoboSeg dataset path. Whose download can be found from the data.zip of: https://huggingface.co/datasets/michaelyuanqwq/roboseg/tree/main

otherwise download you can download the maskrcnn .pth from: https://drive.google.com/file/d/1o1XqOvLJh1OiGtflw7ojzJjsT9pZr7bW/view?usp=sharing 

To load pass in the weight_directory and weight_file into the Segmentation_Model constructor.

As for the Transformer, if you'd like to train call the Transformer.train() method and pass in the respective '/calvin_debug_dataset/training/lang_annotations/auto_lang_ann.npy' along with the criterion, optimizer, and segmentation object.

To load, call the .load() and pass in the path to your weights, you can download the current version of weights from: https://drive.google.com/file/d/1jJ4-qj-IjgE94lGttyYFZ4nuhOdcG1Xh/view?usp=sharing

