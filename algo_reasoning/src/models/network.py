import torch
import torch.nn as nn
from inspect import signature
from loguru import logger

from .encoder import Encoder
from .decoder import Decoder
from .processor import PGN
from algo_reasoning.src.data.data import CLRSData


def stack_hints(hints):
    return {k: torch.stack([hint[k] for hint in hints], dim=-1) for k in hints[0]} if hints else {}

class EncodeProcessDecode(torch.nn.Module):
    def __init__(self, 
                 algorithms, 
                 hidden_dim=128, 
                 nb_nodes=16, 
                 batch_size=32,
                 msg_passing_steps=3, 
                 use_lstm=False, 
                 hint_loss_weight=0.1, 
                 dropout_prob=0.1,
                 encode_hints=True):
        super().__init__()
        self.msg_passing_steps = msg_passing_steps
        self.nb_nodes = nb_nodes
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.use_lstm = use_lstm
        self.processor = PGN(hidden_dim, hidden_dim)
        self.encoders = {}
        self.decoders = {}
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(dropout_prob)
        for algorithm in algorithms:
            self.encoders[algorithm] = Encoder(algorithm, batch_size=batch_size, encode_hints=encode_hints, nb_nodes=nb_nodes, hidden_dim=hidden_dim)
            self.decoders[algorithm] = Decoder(algorithm, hidden_dim=hidden_dim, nb_nodes=nb_nodes, no_hint=hint_loss_weight == 0.0)

        if use_lstm:
            self.lstm = nn.LSTM(hidden_dim, hidden_dim)
    
    def _one_step_prediction(self, batch, hidden, hint_step=None, lstm_state=None):
        algorithm = batch.algorithm
        node_fts, edge_fts, graph_fts, adj_mat = self.encoders[algorithm](batch, hint_step=hint_step)

        nxt_hidden = hidden

        for _ in range(self.msg_passing_steps):
            nxt_hidden, nxt_edge = self.processor(
                node_fts,
                edge_fts,
                graph_fts,
                nxt_hidden,
                adj_mat
            )
        
        if self.dropout_prob > 0.0:
            nxt_hidden = self.dropout(nxt_hidden)

        if self.use_lstm:
            nxt_hidden, nxt_lstm_state = self.lstm(hidden, lstm_state)
        else:
            nxt_lstm_state = None
        
        output_pred = self.decoders[algorithm](nxt_hidden, nxt_edge)

        return output_pred, nxt_hidden, nxt_lstm_state
    
    def forward(self, batch):
        algorithm = batch.algorithm
        hidden = torch.zeros(self.batch_size, self.nb_nodes, self.hidden_dim)
        lstm_state = None
    
        max_len = batch.length.to(torch.int).max().item()

        output_pred, hidden, lstm_state = self._one_step_prediction(batch, hidden, lstm_state=lstm_state)
        hints = output_pred.hints
        for step in range(max_len):
            output_step, hidden, lstm_state = self._one_step_prediction(batch, hidden, hint_step=step, lstm_state=lstm_state)
            hints.concat(output_step.hints)
        
        return CLRSData(inputs=batch.inputs, hints=hints, length=max_len, outputs=output_step.outputs, algorithm=algorithm)
