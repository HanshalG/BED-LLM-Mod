"""Animals (20 Questions) BED environment.

The prompts have always lived in the top-level :mod:`prompts` module;
this subpackage re-exports them under the same public name for symmetry with
``environments.location_finding.prompts``.

To use the prompts:

.. code-block:: python

    from environments.animals import prompts

    sys_msg = prompts.belief_distribution_system_prompt()
"""

from . import prompts
from .env import AnimalsBEDEnvironment

__all__ = ["prompts", "AnimalsBEDEnvironment"]
