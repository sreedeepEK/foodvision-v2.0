### Fine-Tuning Vision Transformer(ViT) using LORA

In this repository, I fine-tune a Vision Transformer (ViT) model using the Food101 dataset. 

- **Dataset Preparation**:
  - Load the Food101 dataset, which contains 101 different food categories.
  - Split the dataset into training and testing sets.

- **Image Preprocessing**:
  - Use the `AutoImageProcessor` from Hugging Face to resize and normalize images to the ViT's required input size.
  - Implement a preprocessing pipeline that includes resizing, center cropping, and normalization.

- **Model Configuration**:
  - Load a pre-trained ViT model from Hugging Face’s model hub.
  - Configure the model for training using LoRA (Low-Rank Adaptation) to fine-tune only a subset of the parameters.

- **Training Setup**:
  - Define training arguments, including the learning rate, batch size, and evaluation strategy.
  - Specify the number of epochs for training.

- **Training Execution**:
  - Utilize the Hugging Face `Trainer` to manage the training loop.
  - Monitor training progress and evaluate the model's performance on the test dataset after each epoch.

- **Model Saving**:
  - Save the fine-tuned model to disk for later use in inference tasks.

- **Model Evaluation**:
  - Compute and print the evaluation accuracy to assess the model's performance.
 
### Requirements

To run this project, you will need:

- Python 3.7 or higher
- Required libraries (install via pip):
  
  ```bash
  pip install transformers accelerate evaluate datasets peft torch torchvision
  ```

#### Inference

Once the model is fine-tuned, you can use it to classify new food images. The inference process involves loading the fine-tuned model, preprocessing the input image, and obtaining the predicted class label.
