class NNUE_AlphaZero(nn.Module):
    def __init__(self, pretrained_nnue: NNUE, num_moves=4672, freeze_backbone=True):
        super().__init__()
        self.backbone = pretrained_nnue
        self.trunk = nn.Linear(32, 128)
        self.policy_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_moves)
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, lr_scale=0.1):
        for param in self.backbone.parameters():
            param.requires_grad = True
        return [
            {'params': self.backbone.parameters(),  'lr': 1e-4 * lr_scale},
            {'params': self.trunk.parameters(),      'lr': 1e-4},
            {'params': self.policy_head.parameters(),'lr': 1e-4},
            {'params': self.value_head.parameters(), 'lr': 1e-4},
        ]

    def forward(self, w_acc, b_acc, legal_move_mask=None):
        w_processed = torch.matmul(w_acc, self.backbone.input_weights) + self.backbone.input_bias
        b_processed = torch.matmul(b_acc, self.backbone.input_weights) + self.backbone.input_bias
        x = torch.cat([w_processed, b_processed], dim=1)         
        x = self.backbone.clipped_relu(x)
        x = self.backbone.clipped_relu(self.backbone.l2(x))  
        x = self.backbone.clipped_relu(self.backbone.l3(x)) 

        trunk = F.relu(self.trunk(x))                  

        policy_logits = self.policy_head(trunk)         
        if legal_move_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_move_mask, -1e9)
        policy = F.softmax(policy_logits, dim=-1)

        value = self.value_head(trunk)

        return policy, value

    def refresh_accumulator(self, active_features):
        return self.backbone.refresh_accumulator(active_features)

    def update_accumulator(self, accumulator, added, removed):
        return self.backbone.update_accumulator(accumulator, added, removed)
nnue = NNUE(input_size=40960)
nnue.load_state_dict(torch.load('your_nnue.pt'))
model = NNUE_AlphaZero(nnue, num_moves=4672, freeze_backbone=True)
def alphazero_loss(policy_pred, value_pred, policy_target, value_target, l2_lambda=1e-4):
   policy_loss = -(policy_target * torch.log(policy_pred + 1e-8)).sum(dim=-1).mean()
   value_loss = F.mse_loss(value_pred.squeeze(-1), value_target.squeeze(-1))
   l2_loss= sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
   return policy_loss + value_loss+l2_lambda*l2_loss


model = NNUE_AlphaZero(nnue, num_moves=4672, freeze_backbone=True)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)
phase1_steps=10000
phase2_steps=10000
for step in range(phase1_steps):
    policy, value = model(w_acc, b_acc, legal_move_mask)                  # these must come from your self-play / replay buffer, e.g.:
                                                                          # w_acc, b_acc         → halfkp_extractor.board_to_halfkp(board)
                                                                          # target_policy        → mcts visit count distribution (4672,)
                                                                          # target_value         → game outcome +1/0/-1
                                                                          # legal_move_mask      → halfkp_extractor.get_legal_moves(board)
    loss = alphazero_loss(policy, value, target_policy, target_value)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


param_groups = model.unfreeze_backbone(lr_scale=0.1)
optimizer = torch.optim.Adam(param_groups) 

for step in range(phase2_steps):
    policy, value = model(w_acc, b_acc, legal_move_mask)
    loss = alphazero_loss(policy, value, target_policy, target_value)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
