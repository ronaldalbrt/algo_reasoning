import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder
from .processor import PGN, MPNN
from algo_reasoning.src.data import AlgorithmicData, AlgorithmicOutput
from algo_reasoning.src.specs import SPECS, CATEGORIES_DIMENSIONS, Type


class EncodeProcessDecode(torch.nn.Module):
    def __init__(self, 
                algorithms, 
                hidden_dim=128, 
                msg_passing_steps=3, 
                use_lstm=False, 
                dropout_prob=0.0,
                teacher_force_prob=0.0,
                encode_hints=True,
                decode_hints=True,
                soft_hints=True,
                freeze_processor=False,
                pretrained_processor=None,
                seed=None):
        super().__init__()
        self.msg_passing_steps = msg_passing_steps
        self.hidden_dim = hidden_dim
        self.soft_hints = soft_hints
        self.use_lstm = use_lstm
        self.teacher_force_prob = teacher_force_prob
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self._generator = torch.Generator() 
        if seed is not None: 
            self._generator = self._generator.manual_seed(seed)

        for algorithm in algorithms:
            self.encoders[algorithm] = Encoder(algorithm, encode_hints=encode_hints, hidden_dim=hidden_dim, soft_hints=self.soft_hints)
            self.decoders[algorithm] = Decoder(algorithm, hidden_dim=hidden_dim, decode_hints=decode_hints)

        if pretrained_processor is None:
            self.processor = MPNN(hidden_dim, hidden_dim)
        else:
            self.processor = pretrained_processor

        if freeze_processor:
            for p in self.processor.parameters():
                p.requires_grad = False
        
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(dropout_prob)
        
        if use_lstm:
            self.lstm = nn.LSTM(hidden_dim, hidden_dim)

    def process_hints(self, hints, algorithm, batch_nb_nodes=16):
        for key, value in hints.items():
            _, _, type_ = SPECS[algorithm][key]
        
            if type_ == Type.POINTER:
                new_value = torch.softmax(value, dim=-1) if self.soft_hints else torch.argmax(value, dim=-1)

            elif type_ == Type.CATEGORICAL:
                new_value = torch.softmax(value, dim=-1) if self.soft_hints else F.one_hot(torch.argmax(value, dim=-1), CATEGORIES_DIMENSIONS[algorithm][key]).float()
            
            elif type_ == Type.MASK_ONE:
                new_value = torch.softmax(value, dim=-1) if self.soft_hints else F.one_hot(torch.argmax(value, dim=-1), batch_nb_nodes).float()
            
            elif type_ == Type.MASK:
                new_value = torch.sigmoid(value) if self.soft_hints else torch.sigmoid(value).long().float()
            else:
                new_value = value

            hints[key] = new_value

        return hints
    
    def _one_step_prediction(self, batch, hidden, hints=None, hint_step=None, lstm_state=None):
        algorithm = batch.algorithm
        nb_nodes = batch.inputs.pos.shape[1]
        
        if hints is not None:
            if self.training and self.teacher_force_prob > torch.rand(1, generator=self._generator).item():
                hints = batch.hints
            else:
                hints = self.process_hints(hints.clone(), algorithm=algorithm, batch_nb_nodes=nb_nodes)

        node_fts, edge_fts, graph_fts, adj_mat = self.encoders[algorithm](batch, hints=hints, hint_step=hint_step)

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
        
        output_pred = self.decoders[algorithm](nxt_hidden, nxt_edge, graph_fts)

        return output_pred, nxt_hidden, nxt_lstm_state
    
    def forward(self, batch):
        algorithm = batch.algorithm
        if batch.inputs.batch is not None:
            batch_size = len(batch.inputs.batch)
        else:
            batch = batch.unsqueeze(0)
            batch_size = 1
        nb_nodes = batch.inputs.pos.shape[1]
        device = batch.inputs.pos.device
        
        hidden = torch.zeros(batch_size, nb_nodes, self.hidden_dim, device=device)
        lstm_state = None
    
        max_len = (batch.max_length - 1).long().item()

        output_pred, hidden, lstm_state = self._one_step_prediction(batch, hidden, lstm_state=lstm_state)
        hints = output_pred.hints
        hidden_embeddings = [hidden]
        if max_len > 0:
            for step in range(max_len):
                output_step, hidden, lstm_state = self._one_step_prediction(batch, hidden, hints=hints, hint_step=step, lstm_state=lstm_state)
                hidden_embeddings.append(hidden)
                hints = hints.concat(output_step.hints)
        else:
            output_step, hidden, lstm_state = self._one_step_prediction(batch, hidden, hints=hints, hint_step=0, lstm_state=lstm_state)
            hidden_embeddings.append(hidden)

        output = AlgorithmicData(inputs=batch.inputs, hints=hints, length=torch.tensor([max_len]*batch_size), outputs=output_step.outputs, algorithm=algorithm).to(device)
        hidden_embeddings = torch.stack(hidden_embeddings, dim=1)

        return AlgorithmicOutput(output=output, hidden_embeddings=hidden_embeddings)