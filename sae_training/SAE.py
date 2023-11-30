
#%%
"""Most of this is just copied over from Arthur's code and slightly simplified:
https://github.com/ArthurConmy/sae/blob/main/sae/model.py
"""
from typing import Literal

import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from transformer_lens.hook_points import HookedRootModule, HookPoint


#%%
# TODO make sure that W_dec stays unit norm during training
class SAE(HookedRootModule):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg.d_in
        if not isinstance(self.d_in, int):
            raise ValueError(
                f"d_in must be an int but was {self.d_in=}; {type(self.d_in)=}"
            )
        self.d_sae = cfg.d_sae
        self.dtype = cfg.dtype
        self.device = cfg.device

        # NOTE: if using resampling neurons method, you must ensure that we initialise the weights in the order W_enc, b_enc, W_dec, b_dec
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_in, self.d_sae, dtype=self.dtype, device=self.device)
            )   
        )
        self.b_enc = nn.Parameter(
            torch.zeros(self.d_sae, dtype=self.dtype, device=self.device)
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_sae, self.d_in, dtype=self.dtype, device=self.device)
            )
        )

        with torch.no_grad():
            # Anthropic normalize this to have unit columns
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

        self.b_dec = nn.Parameter(
            torch.zeros(self.d_in, dtype=self.dtype, device=self.device)
        )

        self.hook_sae_in = HookPoint()
        self.hook_hidden_pre = HookPoint()
        self.hook_hidden_post = HookPoint()
        self.hook_sae_out = HookPoint()

        self.setup()  # Required for `HookedRootModule`s

    def forward(self, x, return_mode: Literal["sae_out", "hidden_post", "both"]="both"):
        sae_in = self.hook_sae_in(
            x - self.b_dec
        )  # Remove encoder bias as per Anthropic

        hidden_pre = self.hook_hidden_pre(
            einops.einsum(
                sae_in,
                self.W_enc,
                "... d_in, d_in d_sae -> ... d_sae",
            )
            + self.b_enc
        )
        hidden_post = self.hook_hidden_post(torch.nn.functional.relu(hidden_pre))

        sae_out = self.hook_sae_out(
            einops.einsum(
                hidden_post,
                self.W_dec,
                "... d_sae, d_sae d_in -> ... d_in",
            )
            + self.b_dec
        )

        if return_mode == "sae_out":
            return sae_out
        elif return_mode == "hidden_post":
            return hidden_post
        elif return_mode == "both":
            return sae_out, hidden_post
        else:
            raise ValueError(f"Unexpected {return_mode=}")

    @torch.no_grad()
    def resample_neurons(
        self,
        x: Float[Tensor, "batch_size n_hidden"],
        frac_active_in_window: Float[Tensor, "window n_hidden_ae"],
        neuron_resample_scale: float,
    ) -> None:
        '''
        Resamples neurons that have been dead for `dead_neuron_window` steps, according to `frac_active`.
        '''
        sae_out = self.forward(x, return_mode="sae_out")
        per_token_l2_loss = (sae_out - x).pow(2).sum(dim=-1).squeeze()

        # Find the dead neurons in this instance. If all neurons are alive, continue
        is_dead = (frac_active_in_window.sum(0) < 1e-8)
        dead_neurons = torch.nonzero(is_dead).squeeze(-1)
        alive_neurons = torch.nonzero(~is_dead).squeeze(-1)
        n_dead = dead_neurons.numel()
        
        if n_dead == 0:
            return # If there are no dead neurons, we don't need to resample neurons
        
        # Compute L2 loss for each element in the batch
        # TODO: Check whether we need to go through more batches as features get sparse to find high l2 loss examples. 
        if per_token_l2_loss.max() < 1e-6:
            return  # If we have zero reconstruction loss, we don't need to resample neurons
        
        # Draw `n_hidden_ae` samples from [0, 1, ..., batch_size-1], with probabilities proportional to l2_loss
        distn = Categorical(probs = per_token_l2_loss / per_token_l2_loss.sum())
        replacement_indices = distn.sample((n_dead,)) # shape [n_dead]

        # Index into the batch of hidden activations to get our replacement values
        replacement_values = (x - self.b_dec)[replacement_indices] # shape [n_dead n_input_ae]

        # Get the norm of alive neurons (or 1.0 if there are no alive neurons)
        W_enc_norm_alive_mean = 1.0 if len(alive_neurons) == 0 else self.W_enc[:, alive_neurons].norm(dim=0).mean().item()
        
        # Use this to renormalize the replacement values
        replacement_values = (replacement_values / (replacement_values.norm(dim=1, keepdim=True) + 1e-8)) * W_enc_norm_alive_mean * neuron_resample_scale

        # Lastly, set the new weights & biases
        self.W_enc.data[:, dead_neurons] = replacement_values.T.squeeze(1)
        self.b_enc.data[dead_neurons] = 0.0
