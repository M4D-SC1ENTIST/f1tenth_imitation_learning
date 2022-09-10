import torch
from torch import nn, optim
import torch.nn.functional as F


class AgentPolicyMLPEIL(nn.Module):
    """
    The AgentPolicy consists of a two-layer MLP.
    """
    def __init__(self, observ_dim, hidden_dim, action_dim, lr, device):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(observ_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.mlp.to(device)
        self.optimizer = optim.Adam(self.mlp.parameters(), lr)
        self.loss_function = EILLoss()
        # self.loss_function = nn.MSELoss()
        self.device = device

    
    def forward(self, observ_tensor: torch.FloatTensor):
        return self.mlp(observ_tensor)

    def train(self, good_observs, good_actions, bad_observs, bad_actions, interv_observs, interv_actions):
        """
        Trains the agent given a batch of observations and expert-labeled actions.
        
        Returns: the loss.
        """
        good_observ_tensor = torch.as_tensor(good_observs, dtype=torch.float32, device=self.device)
        good_action_tensor = torch.as_tensor(good_actions, dtype=torch.float32, device=self.device)
        bad_observ_tensor = torch.as_tensor(bad_observs, dtype=torch.float32, device=self.device)
        bad_action_tensor = torch.as_tensor(bad_actions, dtype=torch.float32, device=self.device)
        interv_observ_tensor = torch.as_tensor(interv_observs, dtype=torch.float32, device=self.device)
        interv_action_tensor = torch.as_tensor(interv_actions, dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        
        good_pred_action = self(good_observ_tensor)
        bad_pred_action = self(bad_observ_tensor)
        interv_pred_action = self(interv_observ_tensor)

        loss = self.loss_function(good_action_tensor, good_pred_action, bad_action_tensor, bad_pred_action, interv_action_tensor, interv_pred_action)
        # loss = self.loss_function(interv_action_tensor, interv_pred_action)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_action(self, observ):
        """
        Predicts an action given an observation.
        """
        observ_tensor = torch.as_tensor(observ, dtype=torch.float32, device=self.device)
        action_tensor = self(observ_tensor)
        return action_tensor.detach().cpu().numpy()



class EILLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super(EILLoss, self).__init__()
        self.reduction = reduction

    def forward(self, good_expert_act, good_agent_act, bad_expert_act, bad_agent_act, interv_expert_act, interv_agent_act, eta=0.8):
        good_loss = F.mse_loss(good_agent_act, good_expert_act, reduction=self.reduction)
        bad_loss = F.mse_loss(bad_agent_act, bad_expert_act, reduction=self.reduction)
        interv_loss = F.mse_loss(interv_agent_act, interv_expert_act, reduction=self.reduction)

        return good_loss + bad_loss + eta * interv_loss