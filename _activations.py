import direction_utils
import generation_utils
from neural_controllers import NeuralController
    
def add(hidden_state, control_vec, control_coef):
    return hidden_state + control_vec * control_coef

def proj(hidden_state, control_vec, control_coef):
    # TODO: Change probed layer, otherwise norm grows with depth due to residual stream
    norm = torch.linalg.norm(hidden_state, axis=-1, keepdim=True)
    proj = hidden_state @ control_vec.mT
    proj = torch.where(proj > norm, 0, norm - proj)
    hidden_state = hidden_state + control_coef * control_vec * proj
    norm2 = torch.linalg.norm(hidden_state, axis=-1, keepdim=True)
    return hidden_state / norm2 * norm

def cos(hidden_state, control_vec, control_coef):
    # TODO: Change probed layer, otherwise norm grows with depth due to residual stream
    norm = torch.linalg.norm(hidden_state, axis=-1, keepdim=True)
    G = control_coef * norm
    X = hidden_state
    v = control_vec
    c = G - X @ v.mT
    return X + c * v

class Activation(NeuralController):
    def __init__(self, *args, **kwargs):
        NeuralController.__init__(self, *args, **kwargs)

    def get_hidden_states(self, prompts):
        return direction_utils.get_hidden_states(
            prompts, 
            self.model, 
            self.tokenizer, 
            self.hidden_layers, 
            self.hyperparams['forward_batch_size']
        )

    def generate(self, prompt, layers_to_control=[], control_coef=0, steer_func=add, **kwargs):
        return self._controlled_generate(prompt, layers_to_control, control_coef, steer_func, **kwargs)
        
    def _controlled_generate(self, prompt, layers_to_control, control_coef, steer_func, **kwargs):
        self.steer_func = steer_func
        self._activations = {}
        self.activations = {}
        self._projections = {}
        self.projections = {}
        
        parent_hook_model = generation_utils.hook_model
        generation_utils.hook_model = self.hook_model
        out = super()._controlled_generate(prompt, layers_to_control, control_coef, **kwargs)
        generation_utils.hook_model = parent_hook_model
        
        self._activations = {k: self._activations[k] for k in self.hidden_layers}
        self.activations = {k: self.activations[k] for k in self.hidden_layers}
        self._projections = {k: self._projections[k] for k in self.hidden_layers}
        self.projections = {k: self.projections[k] for k in self.hidden_layers}
        
        return out

    def hook_model(self, model, directions, layers_to_control, control_coef, component_idx=0):
        hooks = {}
        for layer_idx in self.hidden_layers:
            control_vec = directions[layer_idx][component_idx]
            if len(control_vec.shape)==1:
                control_vec = control_vec.reshape(1,1,-1)
            _control_coef = control_coef if layer_idx in layers_to_control else 0

            def block_hook(
                module, input, output, 
                control_vec=control_vec, control_coef=_control_coef, layer_idx=layer_idx, rep_token=-1
            ):
                """
                note that module, input are unused, but are
                required by torch.
                """ 
            
                new_output = output[0]
                control_vec = control_vec.to(dtype=new_output.dtype, device=new_output.device)
            
                # Save rep_token token activations before steering
                if layer_idx not in self._activations:
                    self._activations[layer_idx] = []
                self._activations[layer_idx].append(new_output[:, rep_token, :].to('cpu'))
            
                # Save projection of rep_token activations onto control_vec before steering
                if layer_idx not in self._projections:
                    self._projections[layer_idx] = []
                self._projections[layer_idx].append((new_output[:, rep_token, :] @ control_vec.mT).to('cpu'))
                
                # Steer activation/hidden_state
                new_output = steer_func(new_output, control_vec, control_coef)
            
                # Save rep_token token activations after steering
                if layer_idx not in self.activations:
                    self.activations[layer_idx] = []
                self.activations[layer_idx].append(new_output[:, rep_token, :].to('cpu'))

                # Save projection of rep_token activations onto control_vec after steering
                if layer_idx not in self.projections:
                    self.projections[layer_idx] = []
                self.projections[layer_idx].append((new_output[:, rep_token, :] @ control_vec.mT).to('cpu'))
                
                if isinstance(output, tuple):
                    new_output = (new_output,) + output[1:] 
                    
                return new_output
                      
            block = model.model.layers[layer_idx]
            hook_handle = block.register_forward_hook(block_hook)
            hooks[layer_idx] = hook_handle
        
        return hooks

def test_activations(activations):
    shapes = [x.shape for v in activations.values() for x in v]
    assert all([shapes[0] == shape for shape in shapes])

def test(_self, prompt: str):
    hidden_states = _self.get_hidden_states(prompt)
    output_text = activator.generate("hi", max_new_tokens=1, do_sample=False)

    for i in _self.hidden_layers:
        assert (_self.activations[i][-1] == hidden_states[i]).all(), i

    for i in _self.hidden_layers:
        assert (_self.activations[i][-1] == _self._activations[i][-1]).all(), i

    # Annoying... https://discuss.pytorch.org/t/different-result-with-the-same-input-data-in-eval-and-no-grad-model/165651/3
    # TODO: investigate why model() gives different activations than generate() for layer_index=-1

############################################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch

def describe(activation, projections=None):
    x = activation
    x = x.flatten().float()
    stats = {}
    stats["count"] = len(x)
    stats[">0"] = len(x[x > 0])
    stats["<0"] = len(x[x < 0])
    stats["mean"] = x.mean().item()
    stats["std"] = x.std().item()
    stats["min"] = x.min().item()
    stats["25%"] = x.quantile(0.25).item()
    stats["50%"] = x.quantile(0.50).item()
    stats["75%"] = x.quantile(0.75).item()
    stats["max"] = x.max().item()
    stats["l2"] = torch.linalg.norm(x).item()
    if projections is not None:
        v = projections.flatten().float()
        stats["proj"] = v.item()
    return stats

def is_equal(A, B):
    return torch.isclose(A @ B.T, torch.linalg.norm(A).square())

def cov(activations):
    return torch.cat(activations).cov()

def corrcoef(activations):
    return torch.cat(activations).corrcoef()

def cosine_sim(activations_A, activations_B):
    A = torch.cat(activations_A)
    B = torch.cat(activations_B)
    inner = A @ B.T
    normA = torch.linalg.norm(A, axis=1)
    normB = torch.linalg.norm(B, axis=1)
    outer = torch.outer(normA, normB)
    return inner / outer
    
def create_heatmap(matrix, labels, title, xlabel="$Layer_i$", ylabel="$Layer_i$", figsize=(6, 6)):
    m, n = matrix.shape
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap='viridis')
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(m), labels, rotation=90)
    ax.set_yticks(np.arange(n), labels)
    plt.tight_layout()
    plt.show()