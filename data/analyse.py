"""
Analyse the generated files and plot benchmarks.
"""
import pathlib
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tskit
import click
import sgkit as sg


def du(path):
    result = subprocess.run(
        f"du -s {path}", shell=True, check=True, capture_output=True
    )
    # du returns results in KiB
    return int(result.stdout.split()[0]) * 1024


@click.command()
@click.option(
    "-L",
    "--sequence-length",
    type=int,
    default=10,
    help="The length of the simulated chromosome in megabases",
)
def process_files(sequence_length):
    pattern = f"*_L{sequence_length}.trees"
    outfile = f"L_{sequence_length}.csv"
    data = []
    for ts_path in pathlib.Path(".").glob(pattern):
        print(ts_path)
        ts = tskit.load(ts_path)
        sav_path = ts_path.with_suffix(".sav")
        vcf_path = ts_path.with_suffix(".vcf.gz")
        sg_path = ts_path.with_suffix(".sgz")
        ds = sg.load_dataset(sg_path)
        assert ts.num_individuals == ds.samples.shape[0]
        assert ts.num_sites == ds.variant_position.shape[0]
        # print(ds)
        data.append(
            {
                "num_samples": ts.num_individuals,
                "num_sites": ts.num_sites,
                "tsk_size": to_GiB(ts_path.stat().st_size),
                "vcfgz_size": to_GiB(vcf_path.stat().st_size),
                "sav_size": to_GiB(sav_path.stat().st_size),
                "sgkit_size": to_GiB(du(sg_path)),
            }
        )
        df = pd.DataFrame(data)
        df.to_csv(outfile)
    print(df)


@click.command()
@click.argument("datafile", type=click.File("r"))
def plot(datafile):
    df = pd.read_csv(datafile)
    print(df)

    fig, ax = plt.subplots(1, 1)
    plt.loglog(df["num_samples"], df["tsk_size"], ".-", label="tskit")
    plt.loglog(df["num_samples"], df["vcfgz_size"], ".-", label="vcf.gz")
    plt.loglog(df["num_samples"], df["sav_size"], ".-", label="sav")
    plt.loglog(df["num_samples"], df["sgkit_size"], ".-", label="sgkit")
    plt.xlabel("Sample size (diploid)")
    plt.ylabel("File size (GiB)")
    plt.title("File sizes for 10Mb of simulated ~human genotype data")

    xytext = (18.0, 0)
    largest_n = np.array(df.num_samples)[-1]
    largest_value = np.array(df.tsk_size)[-1]
    ax.annotate(
        f"{largest_value:.2f}",
        textcoords="offset points",
        xytext=xytext,
        xy=(largest_n, largest_value),
        xycoords="data",
    )

    largest_value = int(np.array(df.sav_size)[-1])
    ax.annotate(
        f"{largest_value:d}",
        textcoords="offset points",
        xytext=xytext,
        xy=(largest_n, largest_value),
        xycoords="data",
    )

    largest_value = int(np.array(df.vcfgz_size)[-1])
    ax.annotate(
        f"{largest_value:d}",
        textcoords="offset points",
        xytext=xytext,
        xy=(largest_n, largest_value),
        xycoords="data",
    )

    largest_value = int(np.array(df.sgkit_size)[-1])
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
    ax2.set_xticks(df["num_samples"])
    ax2.set_xticklabels([str(m) for m in df["num_sites"]])
    ax2.set_xlabel("Number of variants")

    ax.legend()  # [l1, l2, l3])
    plt.tight_layout()
    plt.savefig("scaling.png")




@click.group()
def cli():
    pass


cli.add_command(process_files)
cli.add_command(plot)


if __name__ == "__main__":
    cli()
