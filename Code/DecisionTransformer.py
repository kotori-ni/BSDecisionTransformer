import torch
import torch.nn as nn

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, n_layers=3, n_heads=4, max_len=30):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.action_embedding = nn.Embedding(action_dim, hidden_dim)
        self.rtg_embedding = nn.Linear(1, hidden_dim)
        self.timestep_embedding = nn.Embedding(max_len, hidden_dim)
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=n_layers)
        
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, states, actions, rtgs, timesteps):
        batch_size, seq_len = states.size(0), states.size(1)
        
        state_emb = self.state_embedding(states)
        action_emb = self.action_embedding(actions)
        rtg_emb = self.rtg_embedding(rtgs.unsqueeze(-1))
        timestep_emb = self.timestep_embedding(timesteps)
        
        inputs = torch.stack((rtg_emb, state_emb, action_emb), dim=2).reshape(batch_size, seq_len * 3, self.hidden_dim)
        inputs = inputs + timestep_emb.repeat(1, 3, 1)
        
        mask = self._generate_square_subsequent_mask(seq_len * 3).to(states.device)
        output = self.transformer(inputs.transpose(0, 1), mask=mask).transpose(0, 1)
        
        action_token_states = output[:, 1::3]
        action_pred = self.action_head(action_token_states)
        
        return action_pred
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask