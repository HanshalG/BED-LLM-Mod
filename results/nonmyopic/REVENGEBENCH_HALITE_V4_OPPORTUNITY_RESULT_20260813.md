# RevengeBench Halite V4 opportunity result

Date: 2026-08-13

Status: **failed closed: frozen probe is not a native Halite I policy**

OpenRouter calls/cost: **0 / $0**

## Result

The SDK-correct V4 feeder passes its intended diagonal check on all completed
cells: paired engine replays are byte-exact and true-hypothesis self-distance is
zero. The arena nevertheless fails before a complete likelihood family.

For hidden hypothesis 0, selected probe 2, and seed 0, both fresh arms terminate
after one move. Their complete `.hlt` files are byte-identical. The player-name
field for probe 2 is `0 0 0`, because the frozen source emits an action triple
during the Halite initialization handshake rather than a bot name.

Source inspection confirms the selected policy implements a different square
matrix protocol: it reads one dimension, then repeated player tags and raw
cell triples. Halite I requires player tag, width, height, production map,
run-length encoded owner map, strength map, and a one-line bot name before the
turn loop. The selected probe therefore compiles as C but is not a valid native
Halite I policy.

The frozen gate requires every selected probe to compile, complete paired runs,
and produce at least three valid target decisions. This cell has one. Replacing,
wrapping, or translating the probe after selection would change its strategy and
the prospectively frozen intervention set. Halite V4 is thus a literal gate
failure, not a horizon result.
