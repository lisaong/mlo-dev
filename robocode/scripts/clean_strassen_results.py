# https://intelligentdevices.visualstudio.com/ELL/_workitems/edit/2910
import pandas as pd
import sys

SEP = ', '

# Usage: python clean_strassen_results.py path/to/results__float.csv
df = pd.read_csv(sys.argv[1], sep=SEP)
mask = (df.duration < 1)
print(mask)

df.loc[mask, df.columns[3:]] = -1.
print(df)

df.to_csv(sys.argv[1], index=False, float_format='%.6f')