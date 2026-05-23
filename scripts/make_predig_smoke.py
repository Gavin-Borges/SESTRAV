import pandas as pd

df = pd.read_csv("results/external_predig_input_recombinant.csv")
df.head(50).to_csv("results/external_tool_outputs/predig_smoke_input.csv", index=False)
print("wrote smoke input", len(df.head(50)))
