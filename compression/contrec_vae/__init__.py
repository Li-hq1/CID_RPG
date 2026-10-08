"""ContRec's SigmaVae for continuous item features (reconstruction + anti-collapse)."""

from .config import VAEConfig
from .model import SigmaVae, configure_loss

__all__ = ["SigmaVae", "VAEConfig", "configure_loss"]
