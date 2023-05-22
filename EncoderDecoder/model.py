import torch.nn as nn
from utils import *

class EncoderDecoder(nn.Module):
    """
    Encoder-Decoder model for transliteration.

    Args:
        input_size (int): Size of the input vocabulary.
        hidden_size (int): Size of the hidden state in the RNN cells.
        output_size (int): Size of the output vocabulary.
        cell_type (str): Type of RNN cell to use ('gru', 'rnn', or 'lstm').
        bidirectional (bool): Whether to use bidirectional RNNs.
        num_layers (int): Number of layers in the RNN cells.
        embedding_size (int): Size of the input embedding.
        beam_width (int): Width of the beam for beam search.
        verbose (bool): Whether to print intermediate outputs during forward pass.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        cell_type='gru',
        bidirectional=False,
        num_layers=1,
        embedding_size=256,
        beam_width=5,
        verbose=False,
    ):
        super(EncoderDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.beam_width = beam_width
        self.cell_type = cell_type
        self.embedding = nn.Embedding(input_size, embedding_size)

        if self.cell_type == 'gru':
            self.encoder_rnn_cell = nn.GRU(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
            )
            self.decoder_rnn_cell = nn.GRU(
                input_size=output_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
            )
        elif self.cell_type == 'rnn':
            self.encoder_rnn_cell = nn.RNN(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
            )
            self.decoder_rnn_cell = nn.RNN(
                input_size=output_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
            )
        elif self.cell_type == 'lstm':
            self.encoder_rnn_cell = nn.LSTM(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
            )
            self.decoder_rnn_cell = nn.LSTM(
                input_size=output_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
            )

        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.verbose = verbose

    def forward(
        self,
        input,
        max_output_chars=MAX_OUTPUT_CHARS,
        device='cpu',
        ground_truth=None,
    ):
        """
        Forward pass of the Encoder-Decoder model.

        Args:
            input (torch.Tensor): Input tensor.
            max_output_chars (int): Maximum number of output characters.
            device (str): Device to perform computations on.
            ground_truth (torch.Tensor): Ground truth tensor for teacher forcing.

        Returns:
            List[torch.Tensor]: List of output tensors.
        """
        # Encoder
        input = input.long()
        embedded_input = self.embedding(input)
        embedded_input = embedded_input.view(
            (embedded_input.shape[0] * embedded_input.shape[2], 1, embedded_input.shape[3])
        )

        if self.cell_type == 'lstm':
            out, (hidden, _) = self.encoder_rnn_cell(embedded_input)
        else:
            out, hidden = self.encoder_rnn_cell(embedded_input)

        if self.verbose:
            print('Encoder input', input.shape)
            print('Encoder output', out.shape)
            print('Encoder hidden', hidden.shape)

        # Decoder
        decoder_state = hidden
        decoder_input = torch.zeros(1, 1, self.output_size).to(device)
        decoder_state_ = torch.zeros_like(decoder_state).to(device)
        outputs = []

        if self.verbose:
            print('Decoder state', decoder_state.shape)
            print('Decoder input', decoder_input.shape)

        for i in range(max_output_chars):
            if self.cell_type == 'lstm':
                out, (decoder_state, _) = self.decoder_rnn_cell(decoder_input, (decoder_state, decoder_state_))
            else:
                out, decoder_state = self.decoder_rnn_cell(decoder_input, decoder_state)

        if self.verbose:
            print('Decoder intermediate output', out.shape)

        out = self.h2o(decoder_state)
        out = self.softmax(out)
        outputs.append(out.view(1, -1))

        if self.verbose:
            print('Decoder output', out.shape)
            self.verbose = False

        if ground_truth is not None:
            max_idx = ground_truth[i].reshape(1, 1, 1)
        else:
            topk_probs, topk_indices = out.topk(self.beam_width, dim=2)
            topk_probs = topk_probs.view(1, -1)
            topk_indices = topk_indices.view(1, -1)
            topk_probs[torch.isnan(topk_probs)] = 0
            topk_probs[torch.isinf(topk_probs)] = 0
            topk_probs /= topk_probs.sum()
            selected_indices = torch.multinomial(topk_probs, 1)
            max_idx = topk_indices[0][selected_indices[0]].reshape(1, 1, 1)

        one_hot = torch.FloatTensor(out.shape).to(device)
        one_hot.zero_()
        one_hot.scatter_(2, max_idx, 1)

        decoder_input = one_hot.detach()

        return outputs
