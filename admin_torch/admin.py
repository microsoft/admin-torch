# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import math

class OmegaResidual(torch.nn.Module):
    """
    Residual connection module with shortcut connection rescaling.

    Parameters
    ---------- 
    init_value: ``float``, required.
        The initialization value of the shortcut connection rescalar, omega. 
    as_parameter: ``bool``, optional (default = False).
        Whether to set the rescalar as trainable parameter. Note that, when set as trainable
        parameters, the rescalar would be set as a vector (similar to the weight vector in layer 
        norm), and the embed_dim input is required.  
    embed_dim: ``int``, optional (default = None).
        The hidden state dimension of the shortcut connection. This field is required and only used 
        when ``as_parameter == True``. 
    """
    
    def __init__(self, init_value, as_parameter=False, embed_dim=None):
        super().__init__()
        if as_parameter:
            assert embed_dim is not None, 'embed_dim is required when as_parameter is set as True'
            self.omega = torch.nn.Parameter(torch.ones(embed_dim))
            self.omega.data.fill_(init_value)
            self.forward = self.forward_omega
        else:            
            self.register_buffer('omega', torch.FloatTensor([init_value]))
            if 1.0 == init_value:
                self.forward = self.forward_original
            else:
                self.forward = self.forward_omega
    
    def forward(self, x, f_x):
        """
        Calculate x * omega + f_x. The output shape would be same with the input shape. 

        When omega is set to be a constant 1 (``as buffer`` and ``O(n)`` output change), the 
        ``OmegaResidual`` would downgrade to the ordinary residual module and x + f_x  would be 
        calculated instead.  
        """
        raise NotImplementedError("Placeholder forward function used in OmegaResidual")

    def forward_original(self, x, f_x):
        return x + f_x

    def forward_omega(self, x, f_x):
        return x * self.omega + f_x

def calculate_init(
        num_res_layers,
        output_change_scale='O(logn)',
    ) -> int:
    r"""
    Calculate initialization for omega.

    Parameters
    ---------- 
    num_res_layers: ``int``, required.
        The total number of residual layers. Typical n-layer Transformer encoder has 2n residual layers. 
    output_change_scale: ``str``, optional (default = ``'O(logn)'``).
        The desired output change scale at initialization. Only ``'O(n)'``, ``'O(logn)'`` / ``'default'``, 
        and ``'O(1)'`` are supported. 

    Returns
    -------
    int: It would return the initialization value.
    """
    if 'O(logn)' == output_change_scale or 'default' == output_change_scale:
        omega_value = (num_res_layers + 1) / math.log(num_res_layers + 1) - 1
    elif 'O(n)' == output_change_scale:
        omega_value = 1.
    else:
        assert 'O(1)' == output_change_scale, \
            'only O(n), O(logn), and O(1) output changes are supported.'
        omega_value = num_res_layers
    return omega_value ** 0.5

def as_module(
        num_res_layers,
        output_change_scale='default',
        as_parameter=False,
        embed_dim=None
    ) -> OmegaResidual:
    r"""
    Calculate initialization for omega and return a residual module with the initialized omega. 

    Parameters
    ----------
    num_res_layers: ``int``, required.
        The total number of residual layers. Typical n-layer Transformer encoder has 2n residual layers. 
    output_change_scale: ``str``, optional (default = ``'O(logn)'``).
        The desired output change scale at initialization. Only ``'O(n)'``, ``'O(logn)'`` / ``'default'``, 
        and ``'O(1)'`` are supported. 
    as_parameter: ``bool``, optional (default = False).
        Whether to set the rescalar as trainable parameter. Note that, when set as trainable
        parameters, the rescalar would be set as a vector (similar to the weight vector in
        layer norm), and the embed_dim input is required.  
    embed_dim: ``int``, optional (default = None).
        The hidden state dimension of the shortcut connection. This field is required and only
        used when as_parameter == True. 
        
    Returns
    -------
    admin_torch.OmegaResidual: It would return a ``OmegaResidual`` module with the properly initialized omega inside.

    Example
    -------
    
    .. highlight:: python
    .. code-block:: python

        import torch.nn as nn
        import admin_torch

        class TransformerEncoderLayer(nn.Module):

            def __init__(self, cfg):
                super().__init__()
                
                num_layer =  2 * cfg.encoder_layers # number of residual layers

                self.attn = nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads)
                self.residual_attn = admin_torch.as_module(num_layer) 
                self.ln_attn = nn.LayerNorm(cfg.embed_dim)
    
                self.ffn = nn.Sequential(
                    nn.Linear(cfg.embed_dim, cfg.feedforward_dim),
                    nn.ReLU(),
                    nn.Linear(cfg.feedforward_dim)
                )
                self.residual_ffn = admin_torch.as_module(num_layer) 
                self.ln_ffn = nn.LayerNorm(cfg.embed_dim)
            
            def forward(self, x):

                f_x, _ = self.attn(x)
                x = self.residual_attn(x, f_x)
                x = self.ln_attn(x)

                f_x = self.ffn(x)
                x = self.residual_ffn(x, f_x)
                x = self.ln_ffn(x)

                return x
    """
    omega_value = calculate_init(num_res_layers, output_change_scale)
    return OmegaResidual(omega_value, as_parameter=as_parameter, embed_dim=embed_dim)

def as_buffer(
        network, 
        buffer_name,
        num_res_layers,
        output_change_scale='default',
    ) -> None:
    r"""
    Calculate initialization for omega and *register* omega as a buffer (not trainable).

    Parameters
    ----------
    network: ``torch.nn.Module``, required.
        The ``torch.nn.Module`` contains the residual network. This is where the omega would 
        be registered to.   
    buffer_name: ``str``, required.
        The name of omega (as buffer). The omega can be accessed in the network, using the
        given name.
    num_res_layers: ``int``, required.
        The total number of residual layers. Typical n-layer Transformer encoder has 2n residual layers. 
    output_change_scale: ``str``, optional (default = ``'O(logn)'``).
        The desired output change scale at initialization. Only ``'O(n)'``, ``'O(logn)'`` / ``'default'``, 
        and ``'O(1)'`` are supported. 
        
    Returns
    -------
    None: No returns. The initialized omega would be registered as a buffer within `network`. 

    Example
    -------
    
    .. highlight:: python
    .. code-block:: python

        import torch.nn as nn
        import admin_torch

        class TransformerEncoderLayer(nn.Module):

            def __init__(self, cfg):
                super().__init__()
                
                num_layer =  2 * cfg.encoder_layers # number of residual layers

                self.attn = nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads)
                admin_torch.as_buffer(self, 'attn_omega', num_layer)
                self.ln_attn = nn.LayerNorm(cfg.embed_dim)
    
                self.ffn = nn.Sequential(
                    nn.Linear(cfg.embed_dim, cfg.feedforward_dim),
                    nn.ReLU(),
                    nn.Linear(cfg.feedforward_dim)
                )
                admin_torch.as_buffer(self, 'ffn_omega', num_layer)
                self.ln_ffn = nn.LayerNorm(cfg.embed_dim)
            
            def forward(self, x):

                f_x, _ = self.attn(x)
                x = x * self.attn_omega + f_x
                x = self.ln_attn(x)

                f_x = self.ffn(x)
                x = x * self.ffn_omega + f_x
                x = self.ln_ffn(x)

                return x
    """
    assert isinstance(network, torch.nn.Module), \
        'the input network has to be a torch.nn.Module object'
    omega_value = calculate_init(num_res_layers, output_change_scale)
    network.register_buffer(buffer_name, torch.FloatTensor([omega_value]))

def as_parameter(
        network, 
        parameter_name,
        num_res_layers,
        embed_dim,
        output_change_scale='default',
    ) -> None:
    r"""
    Calculate initialization for omega and *register* omega as a parameter (trainable).

    Parameters
    ----------
    network: ``torch.nn.Module``, required.
        The ``torch.nn.Module`` contains the residual network. This is where the omega would 
        be registered to.   
    parameter_name: ``str``, required.
        The name of omega (as parameter). The omega can be accessed in the network, using the
        given name.
    num_res_layers: ``int``, required.
        The total number of residual layers. Typical n-layer Transformer encoder has 2n residual layers. 
    embed_dim: ``int``, required.
        The hidden state dimension of the shortcut connection. 
    output_change_scale: ``str``, optional (default = ``'O(logn)'``).
        The desired output change scale at initialization. Only ``'O(n)'``, ``'O(logn)'`` / ``'default'``, 
        and ``'O(1)'`` are supported. 
        
    Returns
    -------
    None: No returns. The initialized omega would be registered as a parameter within `network`. 
        
    Example
    -------
    
    .. highlight:: python
    .. code-block:: python

        import torch.nn as nn
        import admin_torch

        class TransformerEncoderLayer(nn.Module):

            def __init__(self, cfg):
                super().__init__()

                num_layer =  2 * cfg.encoder_layers # number of residual layers

                self.attn = nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads)
                admin_torch.as_parameter(self, 'attn_omega', num_layer, cfg.embed_dim) 
                self.ln_attn = nn.LayerNorm(cfg.embed_dim)
    
                self.ffn = nn.Sequential(
                    nn.Linear(cfg.embed_dim, cfg.feedforward_dim),
                    nn.ReLU(),
                    nn.Linear(cfg.feedforward_dim)
                )
                admin_torch.as_parameter(self, 'ffn_omega', num_layer, cfg.embed_dim) 
                self.ln_ffn = nn.LayerNorm(cfg.embed_dim)
            
            def forward(self, x):

                f_x, _ = self.attn(x)
                x = x * self.attn_omega + f_x
                x = self.ln_attn(x)

                f_x = self.ffn(x)
                x = x * self.ffn_omega + f_x
                x = self.ln_ffn(x)

                return x
    """
    omega_vector = torch.ones(embed_dim)
    omega_vector.data.fill_(calculate_init(num_res_layers, output_change_scale))
    network.register_parameter(parameter_name,torch.nn.Parameter(omega_vector))
