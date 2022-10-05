
import logging
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from stanza.models.classifiers.data import SentimentDatum
from stanza.models.classifiers.utils import ModelType, build_output_layers
import stanza.models.constituency.trainer as constituency_trainer

logger = logging.getLogger('stanza')
tlogger = logging.getLogger('stanza.classifiers.trainer')

class ConstituencyClassifier(nn.Module):
    def __init__(self, constituency_parser, labels, args):
        super(ConstituencyClassifier, self).__init__()
        self.labels = labels
        # we build a separate config out of the args so that we can easily save it in torch
        self.config = SimpleNamespace(fc_shapes = args.fc_shapes,
                                      dropout = args.dropout,
                                      num_classes = len(labels),
                                      constituency_backprop = args.constituency_backprop,
                                      constituency_batch_norm = args.constituency_batch_norm,
                                      constituency_top_layer = args.constituency_top_layer,
                                      constituency_use_words = args.constituency_use_words,
                                      bert_model = args.bert_model,
                                      model_type = ModelType.CONSTITUENCY)

        self.constituency_parser = constituency_parser

        # word_lstm:         hidden_size * num_tree_lstm_layers * 2 (start & end)
        # transition_stack:  transition_hidden_size
        # constituent_stack: hidden_size
        self.hidden_size = self.constituency_parser.hidden_size + self.constituency_parser.transition_hidden_size
        if self.config.constituency_use_words:
            self.hidden_size += self.constituency_parser.hidden_size * self.constituency_parser.num_tree_lstm_layers * 2

        # maybe have batch_norm, maybe use Identity
        if self.config.constituency_batch_norm:
            self.input_norm = nn.BatchNorm1d(self.hidden_size)
        else:
            self.input_norm = nn.Identity()

        self.fc_input_size = self.hidden_size
        self.fc_layers = build_output_layers(self.fc_input_size, self.config.fc_shapes, self.config.num_classes)
        self.dropout = nn.Dropout(self.config.dropout)

    def log_configuration(self):
        tlogger.info("Backprop into parser: %s", self.config.constituency_backprop)
        tlogger.info("Batch norm: %s", self.config.constituency_batch_norm)
        tlogger.info("Use start & end words: %s", self.config.constituency_use_words)
        tlogger.info("Intermediate layers: %s", self.config.fc_shapes)

    def log_norms(self):
        lines = ["NORMS FOR MODEL PARAMTERS"]
        lines.extend(self.constituency_parser.get_norms())
        for name, param in self.named_parameters():
            if param.requires_grad and not name.startswith('constituency_parser.'):
                lines.append("%s %.6g" % (name, torch.norm(param).item()))
        logger.info("\n".join(lines))


    def forward(self, inputs):
        # assume all pieces are on the same device
        device = next(self.parameters()).device

        inputs = [x.constituency if isinstance(x, SentimentDatum) else x for x in inputs]

        if self.config.constituency_backprop:
            states = constituency_trainer.analyze_trees(self.constituency_parser, inputs, use_tqdm=False)
        else:
            with torch.no_grad():
                states = constituency_trainer.analyze_trees(self.constituency_parser, inputs, use_tqdm=False)

        constituent_lists = [x.constituents for x in states]
        states = [x.state for x in states]

        word_begin_hx = torch.stack([state.word_queue[0].hx for state in states])
        word_end_hx = torch.stack([state.word_queue[state.word_position].hx for state in states])
        transition_hx = torch.stack([self.constituency_parser.transition_stack.output(state.transitions) for state in states])
        # go down one layer to get the embedding off the top of the S, not the ROOT
        # (in terms of the typical treebank)
        # the idea being that the ROOT has no additional information
        # and may even have 0s for the embedding in certain circumstances,
        # such as after learning UNTIED_MAX long enough
        if self.config.constituency_top_layer:
            constituent_hx = torch.stack([self.constituency_parser.constituent_stack.output(state.constituents) for state in states])
        else:
            constituent_hx = torch.cat([constituents[-2].tree_hx for constituents in constituent_lists], axis=0)

        if self.config.constituency_use_words:
            previous_layer = torch.cat((word_begin_hx, word_end_hx, transition_hx, constituent_hx), axis=1)
        else:
            previous_layer = torch.cat((transition_hx, constituent_hx), axis=1)
        previous_layer = self.input_norm(previous_layer)
        previous_layer = self.dropout(previous_layer)
        for fc in self.fc_layers[:-1]:
            # relu cause many neuron die
            previous_layer = self.dropout(F.gelu(fc(previous_layer)))
        out = self.fc_layers[-1](previous_layer)
        return out

    def get_params(self, skip_modules=True):
        model_state = self.state_dict()
        # skip all of the constituency parameters here -
        # we will add them by calling the model's get_params()
        skipped = [k for k in model_state.keys() if k.startswith("constituency_parser.")]
        for k in skipped:
            del model_state[k]

        constituency_state = self.constituency_parser.get_params(skip_modules)

        params = {
            'model':         model_state,
            'constituency':  constituency_state,
            'config':        self.config,
            'labels':        self.labels,
        }
        return params

