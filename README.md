# Neural Network for English-to-Hindi Transliteration
## Introduction
This project focuses on building an Encoder-Decoder model with attention for transliteration of English text to Hindi. The goal is to create a neural network model that can accurately convert English words into their corresponding Hindi counterparts. The model is implemented using PyTorch and makes use of the attention mechanism to improve the translation performance.

## Dataset
The dataset used for training and evaluation is a collection of English-Hindi word pairs. Each word pair consists of an English word and its transliterated Hindi word. The dataset is prepared specifically for the transliteration task and is available in a suitable format for training the model.

## Instructions
### Prerequisites
 - Python 3.9 or higher
 - PyTorch
 - NumPy
 - Matplotlib
 - scikit-learn

### Setup
#### Clone the repository:
```
git clone https://github.com/your_username/encoder-decoder-transliteration.git
```
Change into the project directory:
```
cd encoder-decoder-transliteration
```
Install the required dependencies:
```
pip install -r requirements.txt
```
#### Training the Model
To train the Encoder-Decoder model for transliteration, follow these steps:

You may run the train.py help script to view the available options:
```
python train.py --help
```

Set the appropriate options for the available params and run the training script.

#### Monitor the training progress:

During training, the script will display the loss and accuracy metrics for each batch and log them to Weights & Biases (wandb).
You can monitor the training progress and visualize the loss and accuracy curves using the generated wandb report.
Evaluate the model:

After training, you can evaluate the trained model on a separate validation set or test set using the calc_accuracy function.
Modify the evaluation code in the calc_accuracy function to suit your evaluation requirements.
Save and use the model:

Once you are satisfied with the training results, you can save the trained model using the torch.save function.
The saved model can be loaded and used for transliteration tasks in a separate script or application.

## Conclusion
The Encoder-Decoder model with attention implemented in this project provides a solution for transliterating English text to Hindi. By training the model on a suitable dataset and fine-tuning the hyperparameters, accurate transliterations can be obtained. The provided instructions guide you through the process of training the model and using it for transliteration tasks. Feel free to explore and experiment with different settings to improve the model's performance.