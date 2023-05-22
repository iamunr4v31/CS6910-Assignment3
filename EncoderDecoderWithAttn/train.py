from utils import *
from model import EncoderDecoderWithAttention
import matplotlib.pyplot as plt
import argparse

train_data = TransliterationDataLoader("./aksharantar_sampled/hin/hin_train.csv")
test_data = TransliterationDataLoader("./aksharantar_sampled/hin/hin_test.csv")
val_data = TransliterationDataLoader("./aksharantar_sampled/hin/hin_valid.csv")

model = EncoderDecoderWithAttention(len(eng_alpha2index), 256, len(hindi_alpha2index), verbose=True)

def train_batch(model, opt, criterion, batch_size, device='cpu', teacher_force=False):
    """
    Performs training on a single batch of data.

    Args:
        model (EncoderDecoderWithAttention): The model to train.
        opt (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss criterion.
        batch_size (int): The batch size.
        device (str): The device to perform the computations on.
        teacher_force (bool): Whether to use teacher forcing during training.

    Returns:
        float: The average loss per batch.
        float: The accuracy of the predictions.
        ndarray: The attention matrices.
    """
    model.train().to(device)
    opt.zero_grad()
    eng_batch, hindi_batch = train_data.get_batch(batch_size)

    total_loss = 0
    accuracy = 0
    attention_matrices = []
    
    for i in range(batch_size):
        input = word_rep(eng_batch[i], eng_alpha2index, device)
        gt = gt_rep(hindi_batch[i], hindi_alpha2index, device)
        outputs, attention_matrix = model(input, gt.shape[0], device, ground_truth=gt if teacher_force else None)
        
        correct = 0
        for index, output in enumerate(outputs):
            loss = criterion(output, gt[index]) / batch_size
            loss.backward(retain_graph=True)
            total_loss += loss

            val, indices = output.topk(1)
            hindi_pos = indices.tolist()[0]
            
            if hindi_pos[0] == gt[index][0]:
                correct += 1
        
        accuracy += correct / gt.shape[0]
        attention_matrices.append(attention_matrix)

    accuracy /= batch_size
    opt.step()

    return total_loss.cpu().detach().numpy() / batch_size, accuracy, np.array(attention_matrices)


def calc_accuracy(model, Data, device='cpu'):
    """
    Calculates the accuracy of the model on the given data.

    Args:
        model (EncoderDecoderWithAttention): The model to evaluate.
        Data (list): The data to evaluate the model on.
        device (str): The device to perform the computations on.

    Returns:
        float: The accuracy of the model on the data.
    """
    model = model.eval()
    predictions = []
    accuracy = 0
    
    for i in range(len(Data)):
        data = Data[i]
        eng, hindi = data[0], data[1]
        gt = gt_rep(hindi, hindi_alpha2index, device_gpu)
        outputs = infer(model, eng, gt.shape[0], device_gpu)
        
        correct = 0
        for index, out in enumerate(outputs):
            val, indices = out.topk(1)
            hindi_pos = indices.tolist()[0]
            
            if hindi_pos[0] == gt[index][0]:
                correct += 1
        
        accuracy += correct / gt.shape[0]
    
    accuracy /= len(Data)
    return accuracy


def train_setup(model, lr=0.01, n_batches=100, batch_size=10, momentum=0.9, display_freq=5, device='cpu', model_name='model'):
    """
    Performs the setup for training the model.

    Args:
        model (EncoderDecoderWithAttention): The model to train.
        lr (float): The learning rate for the optimizer.
        n_batches (int): The number of training batches.
        batch_size (int): The batch size.
        momentum (float): The momentum for the optimizer.
        display_freq (int): The frequency of displaying the training progress.
        device (str): The device to perform the computations on.
        name (str): The name of the model for saving.

    Returns:
        None
    """
    log = {}
    model = model.to(device)
    criterion = nn.NLLLoss(ignore_index=-1)
    opt = optim.Adam(model.parameters(), lr=lr)
    teacher_force_upto = n_batches // 3

    for i in range(n_batches):
        loss, accuracy, attention_matrix = train_batch(model, opt, criterion, batch_size, device=device, teacher_force=i < teacher_force_upto)

        log['loss'] = loss
        log['acc'] = accuracy

        val_acc = calc_accuracy(model, val_data)
        log['val_acc'] = val_acc

        if i == n_batches - 1:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(attention_matrix, cmap='hot', interpolation='nearest')
            ax.set_xlabel('Source')
            ax.set_ylabel('Target')
            plt.close(fig)
            log['attention'] = wandb.Image(fig)

        wandb.log(log)

    torch.save(model, f"{model_name}.pt")
    
def train(args):
    """
    Trains the Encoder-Decoder model using the provided arguments.

    Args:
        args (argparse.Namespace): The command-line arguments.

    Returns:
        None
    """
    model = EncoderDecoderWithAttention(len(eng_alpha2index), args.hidden_dim, len(hindi_alpha2index))
    wandb.init(project="Assignment 3", entity="iamunr4v31")
    wandb.run.name = 'Attention Model'
    train_setup(model, lr=0.001, n_batches=2000, batch_size = 64, display_freq=5, device = device_gpu, model_name=f"{args.model_name}.pt")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension')
    parser.add_argument('--model-name', type=str, default='Attention Model', help='model name')
    args = parser.parse_args()
    train(args)