<p align="center">
  <strong>UNIDROID</strong><br>
  Challenge submission and simulator
</p>

<h1 align="center">brain-comm</h1>

<p align="center">
  The link between a Unidroid challenge submission and the simulator.
</p>

```
challenge submission  ──brain-comm──  simulator
                      serve · check · smoke
```

Model-agnostic. Messages are `hello`, `begin_episode`, `act`, `end_episode`, `close`, and `error`.

## Submission

```
agent.yaml      # entry (module:Class), horizon, action_dim, weight hashes
brain/          # pure Python, importable from this directory
weights/        # files whose sha256 is listed in agent.yaml
train/          # stored for review, never executed
```

## Commands

```bash
brain-comm serve --submission ./examples/zero_brain --socket-dir /tmp/brain --replicas 1
brain-comm check --submission ./examples/zero_brain
brain-comm smoke --socket /tmp/brain/brain-0.sock
```

`serve` loads the submission and speaks the protocol. `check` validates it before a run. `smoke` confirms a live replica.

A brain image may set `BRAIN_COMM_CHECK=module:function`. That function returns an error string, or `None` when the submission is acceptable.

## License

BSD-3-Clause. Copyright (c) 2026 Unidroid. See [LICENSE](LICENSE).
