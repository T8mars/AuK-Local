# AuK in ComfyUI

The ComfyUI integration is maintained and distributed as the independent
[AuK · T8star-Aix native node package](https://github.com/T8mars/Comfyui-Auk-T8).

Install it through ComfyUI Manager or clone it into `ComfyUI/custom_nodes`.
Follow that repository's README for dependencies, model layout, and workflows.
The native nodes load models from `ComfyUI/models/auk` and do not connect to
AuK Local, port 7860, or a service token.

AuK Local remains a standalone light-themed workstation with its own runtime
and models. Neither product needs the other to start or generate audio.
