I created this repository during my review of `LieGroups.jl` to check whether it is possible to reproduce an example from the `navlie` package: https://decargroup.github.io/navlie/_build/html/tutorial/batch.html. I consolidated it into one script: batch_slam_example.py.

To run the code in this repository, you need to have uv installed on your machine: https://docs.astral.sh/uv/getting-started/installation/.
To reproduce the example, please run:

```bash
uv sync
uv run python batch_slam_example.py
```

That's it.
