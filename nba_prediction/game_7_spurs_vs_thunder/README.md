# Game 7 Spurs vs Thunder

NBA Game 7 market simulation and theoretical paper-trading optimizer.

Run:

```bash
python3 main.py
```

Force full NBA game-log refresh:

```bash
REFRESH_NBA_DATA=1 python3 main.py
```

The model estimates win probability, regulation result, handicaps, totals, expected value, and selected theoretical portfolio allocations. Gurobi is used if installed and licensed; otherwise the project falls back to a greedy optimizer.

This is educational analytics only, not betting advice.
