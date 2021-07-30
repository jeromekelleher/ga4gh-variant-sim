"""
OLD SCRIPT - keeping here until all the relevant bits have been used.


Plot the scaling properties of the sgkit file format using
msprime simulations.
"""
import time
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tskit
import msprime
import click
import tsinfer


def write_tsinfer(ts, filename):

    with tsinfer.SampleData(path=filename, sequence_length=ts.sequence_length) as sd:
        for v in ts.variants():
            sd.add_site(
                v.site.position,
                v.genotypes,
                v.alleles,
            )
    print(sd)


@click.command()
@click.option("--sequence-length", type=int, default=100)
def run(sequence_length):

    for k in range(3, 8):
        n = 10 ** k
        print("======")
        print("n = ", n)
        print("======")
        before = time.perf_counter()
        ts = msprime.sim_ancestry(
            n,
            population_size=10_000,
            recombination_rate=1e-8,
            sequence_length=sequence_length * 10 ** 6,
            random_seed=42,
        )
        duration = time.perf_counter() - before
        print(f"Ran ancestry simulation for n={n} in {duration:.2f} seconds")
        before = time.perf_counter()
        ts = msprime.sim_mutations(ts, rate=1e-8)
        duration = time.perf_counter() - before
        print(f"Ran mutation simulation in {duration:.2f} seconds")
        prefix = f"k={k}_L={sequence_length}"
        ts.dump(f"{prefix}.trees")
        before = time.perf_counter()
        write_tsinfer(ts, f"{prefix}.samples")
        duration = time.perf_counter() - before
        print(f"Converted to tsinfer format in {duration:.2f} seconds")


def process():
    data = []
    gigabyte = 1024 ** 3
    for tree_file in pathlib.Path(".").glob("*.trees"):
        ts = tskit.load(tree_file)
        samples_file = tree_file.with_suffix(".samples")
        uncompressed = ts.num_sites * ts.num_samples / gigabyte
        data.append(
            {
                "n": ts.num_samples // 2,
                "L": ts.sequence_length,
                "uncompressed": uncompressed,
                "variants": ts.num_sites,
                "ts_size": tree_file.stat().st_size / gigabyte,
                "zarr_size": samples_file.stat().st_size / gigabyte,
            }
        )
    df = pd.DataFrame(data)
    df.to_csv("data.csv")


def plot():
    df = pd.read_csv("data.csv")
    df = df[df.L == 100000000.0]

    fig, ax = plt.subplots(1, 1)
    plt.loglog(df["n"], df["ts_size"], ".-", label="tskit")
    plt.loglog(df["n"], df["zarr_size"], ".-", label="zarr")
    plt.loglog(df["n"], df["uncompressed"], ".-", label="uncompressed")
    plt.xlabel("Sample size (diploid)")
    plt.ylabel("File size (GiB)")
    plt.title("File sizes for 100Mb of simulated ~human genotype data")

    xytext = (18.0, 0)
    largest_n = np.array(df.n)[-1]
    largest_value = np.array(df.ts_size)[-1]
    ax.annotate(
        f"{largest_value:.2f}",
        textcoords="offset points",
        xytext=xytext,
        xy=(largest_n, largest_value),
        xycoords="data",
    )

    largest_value = int(np.array(df.zarr_size)[-1])
    ax.annotate(
        f"{largest_value:d}",
        textcoords="offset points",
        xytext=xytext,
        xy=(largest_n, largest_value),
        xycoords="data",
    )

    largest_value = int(np.array(df.uncompressed)[-1])
    ax.annotate(
        f"{largest_value:d}",
        textcoords="offset points",
        xytext=xytext,
        xy=(largest_n, largest_value),
        xycoords="data",
    )

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xscale("log")
    ax2.set_xticks(df["n"])
    ax2.set_xticklabels([str(m) for m in df["variants"]])
    ax2.set_xlabel("Number of variants")

    ax.legend()  # [l1, l2, l3])
    plt.tight_layout()
    plt.savefig("zarr-scaling.png")


if __name__ == "__main__":
    # run()
    # process()

    plot()
