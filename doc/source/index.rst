.. Admin-Torch documentation file.

:github_url: https://github.com/LiyuanLucasLiu/Admin

*************************
Admin-Torch documentation
*************************

A plug-in-and-play PyTorch wrapper for `Adaptive model initialization (Admin)`__.

For a neural network f, input x, randomly initialized weight w, we describe its stability (
``output_change_scale``) as

.. math:: E[|f(x, w) - f(x, w + \delta)|_2^2], \mbox{where } \delta \mbox{ is a random perturbation.}

In `our study`__, we show that, an original N-layer Transformer's ``output_change_scale`` is ``O(n)``, 
which unstabilizes its training. Admin stabilize Transformer's training by regulating this scale to 
``O(logn)`` and ``O(1)``. We keep ``O(logn)`` as the ``default`` setting, which can handle most scenarios.
In need of additional stability, set ``output_change_scale`` to ``O(1)`` instead. 

__ https://arxiv.org/abs/2004.08249
__ https://arxiv.org/abs/2004.08249


admin_torch\.as_module()
===============================
.. autofunction:: admin_torch.as_module

   
admin_torch\.as_parameter()
===============================
.. autofunction:: admin_torch.as_parameter

   
admin_torch\.as_buffer()
===============================
.. autofunction:: admin_torch.as_buffer


admin_torch\.OmegaResidual
================================
.. autoclass:: admin_torch.OmegaResidual
	:members:
