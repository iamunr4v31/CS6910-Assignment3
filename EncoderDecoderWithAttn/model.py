import torch.nn as nn
from utils import *

class EncoderDecoderWithAttention(nn.Module):
    """
    A sequence-to-sequence model with attention mechanism.

    Args:
        input_size (int): The size of the input vocabulary.
        hidden_size (int): The size of the hidden state of the RNN.
        output_size (int): The size of the output vocabulary.
        verbose (bool): Whether to print intermediate shapes during the forward pass.

    Attributes:
        hidden_size (int): The size of the hidden state of the RNN.
        output_size (int): The size of the output vocabulary.
        encoder_rnn_cell (nn.GRU): The GRU cell used for the encoder.
        decoder_rnn_cell (nn.GRU): The GRU cell used for the decoder.
        h2o (nn.Linear): Linear layer mapping hidden state to output size.
        softmax (nn.LogSoftmax): Log softmax activation function.
        U (nn.Linear): Linear layer for attention mechanism.
        W (nn.Linear): Linear layer for attention mechanism.
        attn (nn.Linear): Linear layer for attention mechanism.
        out2hidden (nn.Linear): Linear layer mapping output size to hidden size.
        verbose (bool): Whether to print intermediate shapes during the forward pass.

    Methods:
        forward(input, max_output_chars=MAX_OUTPUT_CHARS, device='cpu', ground_truth=None):
            Performs forward pass of the encoder-decoder model with attention mechanism.

    Returns:
        outputs (List[Tensor]): The list of output tensors at each timestep.
        attention_matrices (ndarray): The attention matrices at each timestep.
    """
    def __init__(self, input_size, hidden_size, output_size, verbose=False):
        super(EncoderDecoderWithAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.encoder_rnn_cell = nn.GRU(input_size, hidden_size)
        self.decoder_rnn_cell = nn.GRU(hidden_size*2, hidden_size)
        
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        
        self.U = nn.Linear(self.hidden_size, self.hidden_size)
        self.W = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size, 1)
        self.out2hidden = nn.Linear(self.output_size, self.hidden_size)   
        
        self.verbose = verbose
        
    def forward(self, input, max_output_chars = MAX_OUTPUT_CHARS, device = 'cpu', ground_truth = None):
        attention_matrices = [] 
        # encoder
        encoder_outputs, hidden = self.encoder_rnn_cell(input)
        encoder_outputs = encoder_outputs.view(-1, self.hidden_size)
        
        if self.verbose:
            print('Encoder output', encoder_outputs.shape)
        
        # decoder
        decoder_state = hidden
        decoder_input = torch.zeros(1, 1, self.output_size).to(device)
        
        outputs = []
        U = self.U(encoder_outputs)
        
        if self.verbose:
            print('Decoder state', decoder_state.shape)
            print('Decoder intermediate input', decoder_input.shape)
            print('U * Encoder output', U.shape)
        
        for i in range(max_output_chars):
            
            W = self.W(decoder_state.view(1, -1).repeat(encoder_outputs.shape[0], 1))
            V = self.attn(torch.tanh(U + W))
            attn_weights = F.softmax(V.view(1, -1), dim = 1) 
            
            if self.verbose:
                print('W * Decoder state', W.shape)
                print('V', V.shape)
                print('Attn', attn_weights.shape)

            attention_matrices.append(attn_weights.cpu().detach().numpy())
            
            attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
            
            embedding = self.out2hidden(decoder_input)
            decoder_input = torch.cat((embedding[0], attn_applied[0]), 1).unsqueeze(0)
            
            if self.verbose:
                print('Attn LC', attn_applied.shape)
                print('Decoder input', decoder_input.shape)
                
            out, decoder_state = self.decoder_rnn_cell(decoder_input, decoder_state)
            
            if self.verbose:
                print('Decoder intermediate output', out.shape)
                
            out = self.h2o(decoder_state)
            out = self.softmax(out)
            outputs.append(out.view(1, -1))
            
            if self.verbose:
                print('Decoder output', out.shape)
                self.verbose = False
            
            max_idx = torch.argmax(out, 2, keepdim=True)
            if not ground_truth is None:
                max_idx = ground_truth[i].reshape(1, 1, 1)
            one_hot = torch.zeros(out.shape, device=device)
            one_hot.scatter_(2, max_idx, 1) 
            
            decoder_input = one_hot.detach()
            
        return outputs,np.array(attention_matrices)