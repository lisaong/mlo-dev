#!/usr/bin/python3
import fire

import matplotlib.pyplot as plt
import pandas as pd

def plot(csv, x, y, output):
    "Usage: plot result.csv xcol ycol output.png"
    df = pd.read_csv(csv)
    df.plot(x=x, y=y, marker='x');
    plt.grid()
    plt.savefig(output)

if __name__ == '__main__':
    fire.Fire()