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


## Some details:

For training, the transformer passes the image and token into their relative embedding space. The tokens is passed into the roberta-base token embedding, with the image being passed into a linear layer. Both of which are seperated by a learned seperating token. 
<img width="992" height="417" alt="image" src="https://github.com/user-attachments/assets/4a830fb3-c4a4-4547-96d7-fe94dfd9fc1d" />

Once the src(multimodal input) are made, source masks are created from their token size dimension size. The MM input are then projected into linear layers and split into heads. Attention computation follows, with the einsum operation with the src mask before the softmax. To complete the block, the outputs are sent to to subsequent 2 linear layers with recurrent concatenation and normalization. To make the dimensions accurate with the CALVIN embedding dim, the output linear layer is sent through another linear layer to be of the same size (384). 

The Encoder computation is only done once per episode, but N times per the one computation depending on the stacked length of your recursion. 

The decoder then follows taking in the proper mask, encoder output, and target input (the expert trajectories). In practice, the expert trajectories are iteratively concatenated with the current trajectory while the last index of the output being trained on the next step. This creates a left shift behavior. 

The Decoder implementation details follows standard encoder-decoder architecture. The decoder takes in matrix of current trajectories as Q,V,K, computes attention with the target mask. Sends a recurrent concatenated input of the attention and x into the linear layer, is then normalized and cross-attention is computed. 

<img width="575" height="232" alt="image" src="https://github.com/user-attachments/assets/c8a58892-0939-4376-9ce0-fb5f0eee5cbe" />

The same transformer block operation follows, with the encoder having linear layer projections of the action dim size, and the heads being split, attention with the target mask, a recurrent MLP, and dropout.
<img width="638" height="249" alt="image" src="https://github.com/user-attachments/assets/c1424055-5886-4d02-97b6-7d4bf0e28a1d" />

For proper action inference an end token is learned after each episode. 

