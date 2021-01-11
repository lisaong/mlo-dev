import pandas as pd
import csv
import sys
import os
import glob

SEP = ', '

# Usage: python clean_transformer_results.py parent_dir

def fix(filename):
    df = pd.read_csv(filename, sep=SEP, dtype=str, engine='python')

    if df.columns[0] == 'sequenceLength':
        print(f'Fixing: {filename}...')
        # insert the numHeads column
        df_new = pd.concat([df.loc[:, df.columns[:6]],
            pd.DataFrame({'numHeadsTemp': [12, 12, 12, 12, 16, 16, 16, 16, 20, 20, 20, 20,
            25, 25, 25, 25]}), 
            df.loc[:, df.columns[6:]]], axis=1)

        df_new.columns = ['batchSize', 'sequenceLength', 'd_m', 'd_ff', 'd_k',
            'd_v', 'numHeads', 'duration', 'variance', 'ops', 'flops']

        print(df_new)

        df_new.to_csv(f'{filename}.tmp', index=False, quoting=csv.QUOTE_NONE)
        
        with open(f'{filename}', 'w') as wf:
            with open(f'{filename}.tmp', 'r') as rf:
                data = rf.read()
                wf.write(data.replace(",", ", "))
        
        os.remove(f'{filename}.tmp')        

for results_file in glob.glob(f'{sys.argv[1]}/**/results__float.csv', recursive=True):
    print(results_file)
    fix(results_file)
