import torch
import torch_geometric.nn as nn
from inspect import signature
from loguru import logger

from .encoder import Encoder
from .decoder import Decoder, grab_outputs, output_mask
from .processor import Processor
from algo_reasoning.utils.utils import stack_hidden

def stack_hints(hints):
    return {k: torch.stack([hint[k] for hint in hints], dim=-1) for k in hints[0]} if hints else {}

HIDDEN_DIM = 128
GRU_ENABLED = False
MSG_PASSING_STEPS = 3
HINT_LOSS_WEIGHT = 0.0
USE_LAST_HIDDEN = False

class EncodeProcessDecode(torch.nn.Module):
    def __init__(self, algorithms, hidden_dim=128, msg_passing_steps=3, gru_enabled=False, hint_loss_weight=0.0, use_last_hidden=False):
        super().__init__()
        self.cfg = cfg
        self.processor = Processor()
        self.encoders = {}
        for algorithm in algorithms:
            self.encoders[algorithm] = Encoder(algorithm, hidden_dim=hidden_dim)

        decoder_input = self.cfg.MODEL.HIDDEN_DIM*3 if self.cfg.MODEL.DECODER_USE_LAST_HIDDEN else self.cfg.MODEL.HIDDEN_DIM*2
        #self.decoder = Decoder(specs, decoder_input, no_hint=self.cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT == 0.0)
        #logger.debug(f"Decoder: {self.cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT == 0.0}")

        if self.processor.has_edge_weight():
            self.edge_weight_name = "edge_weight"
        elif self.processor.has_edge_attr():
            self.edge_weight_name = "edge_attr"


        #if self.cfg.MODEL.GRU.ENABLE:
        #    self.gru = torch.nn.GRUCell(self.cfg.MODEL.HIDDEN_DIM, self.cfg.MODEL.HIDDEN_DIM)
        if GRU_ENABLED:
            self.gru = torch.nn.GRUCell(HIDDEN_DIM, HIDDEN_DIM)
        
    def process_weights(self, batch):
        if self.edge_weight_name == "edge_attr":
            return batch.weights.unsqueeze(-1).type(torch.float32)
        else:
            return batch.weights
        
    def forward(self, batch):
        algorithm = batch.algorithm
        input_hidden = self.encoders[algorithm](batch)
        max_len = batch.length.max().item()
        hints = []
        output = None

        # Process for length
        hidden = input_hidden
        for step in range(max_len):
            last_hidden = hidden
            #for _ in range(self.cfg.MODEL.MSG_PASSING_STEPS):
            for _ in range(MSG_PASSING_STEPS):
                hidden = self.processor(input_hidden, hidden, last_hidden, **{self.edge_weight_name: self.process_weights(batch) for _ in range(1) if hasattr(batch, 'weights') })
                #if self.cfg.MODEL.GRU.ENABLE:
                if GRU_ENABLED:
                    hidden = self.gru(hidden, last_hidden)
            if self.training and self.cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT > 0.0:
                hints.append(self.decoder(stack_hidden(input_hidden, hidden, last_hidden, self.cfg.MODEL.DECODER_USE_LAST_HIDDEN), batch, 'hints'))

            # Check if output needs to be constructed
            if (batch.length == step+1).sum() > 0:
                # Decode outputs
                #if self.training and self.cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT > 0.0:
                if self.training and HINT_LOSS_WEIGHT > 0.0:
                    # The last hint is the output, no need to decode again, its the same decoder
                    output_step = grab_outputs(hints[-1], batch)
                else:
                    #output_step = self.decoder(stack_hidden(input_hidden, hidden, last_hidden, self.cfg.MODEL.DECODER_USE_LAST_HIDDEN), batch, 'outputs')
                    output_step = stack_hidden(input_hidden, hidden, last_hidden, USE_LAST_HIDDEN)
                
                # Mask output
                mask = output_mask(batch, step)   
                if output is None:
                    output = {k: output_step[k]*mask[k] for k in output_step}
                else:
                    for k in output_step:
                        output[k][mask[k]] = output_step[k][mask[k]]

        hints = stack_hints(hints)

        return output, hints, hidden

