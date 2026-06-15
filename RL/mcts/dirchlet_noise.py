
import numpy as np
import math
import chess
import torch
import numpy as np
from RL.chess_env.features import HalfKPExtractor
halfkp_extractor = HalfKPExtractor()def dirichlets_loss(policy:torch.Tensor,alpha: float=0.3,epsilon: float=0.25) -> torch.Tensor:
  noise=torch.distributions.Dirichlet(torch.full_like(policy,alpha)).sample()
  return (1-epsilon)*policy+epsilon*noise
