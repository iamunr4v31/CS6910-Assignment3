import torch
import torch.nn as nn
from utils import *
import argparse
from model import EncoderDecoder
train_data = TransliterationDataLoader("../aksharantar_sampled/hin/hin_train.csv")
test_data = TransliterationDataLoader("../aksharantar_sampled/hin/hin_test.csv")
val_data = TransliterationDataLoader("../aksharantar_sampled/hin/hin_valid.csv")

def calc_accuracy(net, Data, device = 'cpu'):
    """
    Calculates the accuracy of the given network model on the provided data.

    Args:
        net (nn.Module): The network model.
        Data (list): The data to evaluate the model on.
        device (str, optional): The device to run the model on. Defaults to 'cpu'.

    Returns:
        float: The accuracy of the model on the data.
    """
    net = net.eval()#.to('cpu')
    predictions = []
    accuracy = 0
    for i in range(len(Data)):
        data = Data[i]
        eng, hindi = data[0],data[1]
        gt = gt_rep(hindi, hindi_alpha2index, device_gpu)
        outputs = infer(net, eng, gt.shape[0], device_gpu)
        correct = 0
        for index, out in enumerate(outputs):
            val, indices = out.topk(1)
            hindi_pos = indices.tolist()[0]
            if hindi_pos[0] == gt[index][0]:
                correct += 1      
        accuracy += correct/gt.shape[0]
    accuracy /= len(Data)
    return accuracy

def train_batch(net, opt, criterion, batch_size, device = 'cpu', teacher_force = False):
    """
    Trains the network model for a single batch of data.

    Args:
        net (nn.Module): The network model.
        opt (torch.optim.Optimizer): The optimizer for updating the model's parameters.
        criterion (nn.Module): The loss criterion for training.
        batch_size (int): The size of the batch.
        device (str, optional): The device to run the model on. Defaults to 'cpu'.
        teacher_force (bool, optional): Whether to use teacher forcing during training. Defaults to False.

    Returns:
        tuple: A tuple containing the average loss and accuracy for the batch.
    """
    net.train().to(device)
    opt.zero_grad()
    eng_batch, hindi_batch = train_data.get_batch(batch_size)
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    accuracy = 0
    for i in range(batch_size):
        correct = 0
        input = word_rep(eng_batch[i], eng_alpha2index, device)
        gt = gt_rep(hindi_batch[i], hindi_alpha2index, device)
        outputs = net(input, gt.shape[0], device, ground_truth = gt if teacher_force else None)
        
        for index, output in enumerate(outputs):
            loss = criterion(output, gt[index]) / batch_size
            loss.backward(retain_graph = True)
            total_loss += loss
            
            val, indices = output.topk(1)
            hindi_pos = indices.tolist()[0]
            if hindi_pos[0] == gt[index][0]:
                correct += 1
            accuracy += correct/gt.shape[0]
    accuracy /= batch_size
    average_loss = total_loss / batch_size   
    opt.step()
    return average_loss, accuracy

def validate_batch(net, criterion, batch_size, device='cpu'):
    """
    Evaluates the network model on a single batch of validation data.

    Args:
        net (nn.Module): The network model.
        criterion (nn.Module): The loss criterion for evaluation.
        batch_size (int): The size of the batch.
        device (str, optional): The device to run the model on. Defaults to 'cpu'.

    Returns:
        tuple: A tuple containing the average loss and accuracy for the batch.
    """
    net.eval().to(device)
    eng_batch, hindi_batch = val_data.get_batch(batch_size)

    total_loss = 0
    total_correct = 0
    total_samples = 0
    accuracy = 0

    with torch.no_grad():
        for i in range(batch_size):
            correct  =0
            input = word_rep(eng_batch[i], eng_alpha2index, device)
            gt = gt_rep(hindi_batch[i], hindi_alpha2index, device)
            outputs = net(input, gt.shape[0], device)

            for index, output in enumerate(outputs):
                loss = criterion(output, gt[index]) / batch_size
                total_loss += loss.item()

                val, indices = output.topk(1)
            hindi_pos = indices.tolist()[0]
            if hindi_pos[0] == gt[index][0]:
                correct += 1
            accuracy += correct/gt.shape[0]
    accuracy /= batch_size
    average_loss = total_loss / batch_size
    return average_loss, accuracy

def train_setup(net, lr = 0.01, n_batches = 100, batch_size = 10, momentum = 0.9, display_freq=5, device = 'cpu', model_name = "model"):
    """
    Sets up and trains the network model for a given number of batches.

    Args:
        net (nn.Module): The network model.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.01.
        n_batches (int, optional): The number of batches to train for. Defaults to 100.
        batch_size (int, optional): The size of the batches. Defaults to 10.
        momentum (float, optional): The momentum factor for the optimizer. Defaults to 0.9.
        display_freq (int, optional): The frequency of displaying training progress. Defaults to 5.
        device (str, optional): The device to run the model on. Defaults to 'cpu'.
        model_name (str, optional): The name of the model for saving. Defaults to "model".

    Returns:
        numpy.ndarray: An array containing the losses for each batch.
    """
    net.train()
    net = net.to(device)
    criterion = nn.NLLLoss(ignore_index = -1)
    opt = optim.Adam(net.parameters(), lr=lr)
    teacher_force_upto = n_batches//3
    
    loss_arr = np.zeros(n_batches + 1)
    
    for i in range(n_batches):
        loss, accuracy = train_batch(net, opt, criterion, batch_size, device = device, teacher_force = i<teacher_force_upto)
        wandb.log({
            "train_loss": loss,
            "train_acc_per_sample": accuracy,
        }) 
        
    val_loss, _ = validate_batch(net, criterion, batch_size, device)
    val_accuracy = calc_accuracy(net, val_data)
    wandb.log({"val_loss": val_loss, "val_acc": val_accuracy})
            
    torch.save(net, f'./{model_name}.pt')
    return loss_arr

def train(args):
    """
    Trains the Encoder-Decoder model using the provided arguments.

    Args:
        args (argparse.Namespace): The command-line arguments.

    Returns:
        None
    """
    wandb.init(project=args.project_name, entity=args.entity_name)
    # wandb.run.name = f"Embedding Size:{wandb.config.embedding_size} | n_layers:{wandb.config.n_layers} | hidden_size:{wandb.config.hidden_size} | cell_type:{wandb.config.cell_type} | bidirectional:{wandb.config.bidirectional} | dropout:{wandb.config.dropout} | beam_width:{wandb.config.beam_width}"
    # config = wandb.config
    net = EncoderDecoder(input_size=len(eng_alpha2index), hidden_size=256, output_size=len(hindi_alpha2index), cell_type=args.cell_type, bidirectional=args.bidirectional,
                                                    beam_width=args.beam_width, embedding_size=args.embedding_size, num_layers=args.n_layers)
    
    train_setup(net, lr=0.001, n_batches=50, batch_size = 512, display_freq=5, device = device_gpu, model_name=wandb.run.name)

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Encoder-Decoder Model Argument Parser')

    # Add the argument options
    parser.add_argument("--project_name", type=str, default="CS6910-Assignment-3", help="Name of the project")
    parser.add_argument("--entity_name", type=str, default="cs20m059", help="Name of the entity")
    parser.add_argument('--embedding_size', type=int, default=256,
                        help='Size of the embedding layer')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of layers in the RNN')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Size of the hidden state of the RNN')
    parser.add_argument('--cell_type', type=str, default="gru",
                        choices=['lstm', 'gru', 'rnn'], help='Type of RNN cell')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Use bidirectional RNN')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability')
    parser.add_argument('--beam_width', type=int, default=2,
                        help='Beam width for beam search')

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function
    train(args)