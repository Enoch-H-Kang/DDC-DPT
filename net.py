import torch
import torch.nn as nn
import transformers
transformers.set_seed(0)
from transformers import GPT2Config, GPT2Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Transformer(nn.Module):
    """Transformer class."""

    def __init__(self, config):
        super(Transformer, self).__init__()

        self.config = config
        self.test = config['test']
        self.horizon = self.config['horizon']
        self.n_embd = self.config['n_embd']
        self.n_layer = self.config['n_layer']
        self.n_head = self.config['n_head']
        self.state_dim = 1
        self.action_dim = 1
        self.dropout = self.config['dropout']

        config = GPT2Config(
            n_positions=4 * (1 + self.horizon),
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=4,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(config) # Model

        self.embed_transition = nn.Linear(
            2 * self.state_dim + self.action_dim, self.n_embd)
        
        self.pred_q_values = nn.Linear(self.n_embd, 2)

    def forward(self, x):
        query_states = x['query_states'][:, None] # (batch_size, 1, state_dim). #None and unsqueeze are equivalent
        zeros = x['zeros'][:, None] # (batch_size, 1, state_dim+1)
        
        state_seq = torch.cat([query_states, x['context_states']], dim=1) # (batch_size, 1+horizon, state_dim)
        
        action_seq = torch.cat(
            [zeros[:, :, 1], x['context_actions']], dim=1) # (batch_size, 1+horizon, action_dim)
        next_state_seq = torch.cat(
            [zeros[:, :, 1], x['context_next_states']], dim=1) # (batch_size, 1+horizon, state_dim)
        
        
        state_seq = state_seq.unsqueeze(2) # (batch_size, 1+horizon, 1, state_dim)
        action_seq = action_seq.unsqueeze(2) # (batch_size, 1+horizon, 1, action_dim)
        next_state_seq = next_state_seq.unsqueeze(2) # (batch_size, 1+horizon, 1, state_dim)
       

        seq = torch.cat(
            [state_seq, action_seq, next_state_seq], dim=2) 
        stacked_inputs = self.embed_transition(seq)
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        preds = self.pred_q_values(transformer_outputs['last_hidden_state'])
        if self.test:
            return preds[:, -1, :]
        return preds[:, 1:, :]

