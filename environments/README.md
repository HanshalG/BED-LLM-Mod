# BED environments

Each environment implements :class:`core.environment.Environment` and registers
with :mod:`core.defaults`.

## Adding a new environment

1. Subclass :class:`core.environment.Environment` with types ``S, H, A, O``.
2. Implement simulation (:meth:`sample_hidden_state`, :meth:`observe`).
3. Implement belief refresh (:meth:`initial_belief_state`, :meth:`update_belief_state`)
   using ``core.BeliefState[H]`` internally and externally.
4. Implement :meth:`generate_candidate_actions` for EIG / strategy methods.
5. Provide likelihood:
   - **Binary LLM:** mixin :mod:`core.llm_likelihood` + :meth:`build_likelihood_messages`.
   - **Continuous:** :meth:`predictive_means`, optional
     :meth:`belief_state_for_eig_scoring`, and optional
     :meth:`score_continuous_forward_search_depth2_batched` for LLM depth-2.
6. Read task-specific options from ``config.environment`` and validate them in
   :meth:`validate_config`.
7. Implement :meth:`summarize_run` for task-specific metric names; :mod:`main`
   saves all metric series generically.
8. Register in :mod:`core.defaults`.

See :mod:`environments.minimal_binary` for a minimal worked example.
