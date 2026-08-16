import json

with open('strategy_simulation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i in range(0, min(5, len(nb.get('cells', [])))):
    c = nb['cells'][i]
    print(f"\n{'='*20} Cell {i} ({c.get('cell_type')}) {'='*20}")
    print(''.join(c.get('source', [])))
