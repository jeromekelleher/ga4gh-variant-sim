"""
Analyse the generated files and plot benchmarks.
"""
import pathlib
import subprocess

import humanize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tskit
import click
import sgkit as sg


def du(path):
    result = subprocess.run(
        f"du -s --bytes {path}", shell=True, check=True, capture_output=True
    )
    return int(result.stdout.split()[0])

def to_GiB(num_bytes):
    return num_bytes / (1024 ** 3)


@click.command()
@click.argument("source_pattern", type=str)
@click.argument("output", type=click.Path())
@click.option("-s", "--suffix", default="")
def process_files(source_pattern, output, suffix):
    data = []
    for ts_path in pathlib.Path().glob(source_pattern):
        ts = tskit.load(ts_path)
        click.echo(f"{ts_path} n={ts.num_individuals}, m={ts.num_sites}")
        stem = ts_path.stem + suffix
        sav_path = pathlib.Path(stem + ".sav")
        vcf_path = pathlib.Path(stem + ".vcf.gz")
        sg_path = pathlib.Path(stem + ".sgz")
        ds = sg.load_dataset(sg_path)
        assert ts.num_individuals == ds.samples.shape[0]
        assert ts.num_sites == ds.variant_position.shape[0]
        assert np.array_equal(ds.variant_position, ts.tables.sites.position.astype(int))
        data.append(
            {
                "sequence_length": ts.sequence_length / 10**6,
                "num_samples": ts.num_individuals,
                "num_sites": ts.num_sites,
                "tsk_size": to_GiB(ts_path.stat().st_size),
                "vcfgz_size": to_GiB(vcf_path.stat().st_size),
                "sav_size": to_GiB(sav_path.stat().st_size),
                "sgkit_size": to_GiB(du(sg_path)),
            }
        )

    df = pd.DataFrame(data)
    df.to_csv(output)


@click.command()
@click.argument("datafile", type=click.File("r"))
@click.argument("output", type=click.Path())
def plot(datafile, output):
    df = pd.read_csv(datafile)
    print(df)
    L = df.sequence_length.unique()[0]

    K = df["vcfgz_size"]

    fig, ax = plt.subplots(1, 1)
    plt.semilogx(df["num_samples"], df["vcfgz_size"] / K, ".-", label="vcf.gz")
    plt.semilogx(df["num_samples"], df["sav_size"] / K, ".-", label="sav")
    plt.semilogx(df["num_samples"], df["sgkit_size"] / K, ".-", label="sgkit")
    plt.xlabel("Sample size (diploid)")
    plt.ylabel("File size relative to vcf.gz")
    plt.title(f"File sizes for {L}Mb of simulated ~human genotype data")

    for n, size in zip(df["num_samples"], df["vcfgz_size"]):
        print(n, size)
        ax.annotate(
            f"{humanize.naturalsize(size * 1024**3, binary=True)}",
            textcoords="offset points",
            xytext=(-15, -15),
            xy=(n, 1),
            xycoords="data",
        )

    # xytext = (18.0, 0)
    # largest_n = np.array(df.num_samples)[-1]
    # largest_value = np.array(df.tsk_size)[-1]
    # ax.annotate(
    #     f"{largest_value:.2f}",
    #     textcoords="offset points",
    #     xytext=xytext,
    #     xy=(largest_n, largest_value),
    #     xycoords="data",
    # )

    # largest_value = int(np.array(df.sav_size)[-1])
    # ax.annotate(
    #     f"{largest_value:d}",
    #     textcoords="offset points",
    #     xytext=xytext,
    #     xy=(largest_n, largest_value / K[-1]),
    #     xycoords="data",
    # )

    # largest_value = int(np.array(df.vcfgz_size)[-1])
    # ax.annotate(
    #     f"{largest_value:d}",
    #     textcoords="offset points",
    #     xytext=xytext,
    #     xy=(largest_n, largest_value),
    #     xycoords="data",
    # )

    # largest_value = int(np.array(df.sgkit_size)[-1])
    # ax.annotate(
    #     f"{largest_value:d}",
    #     textcoords="offset points",
    #     xytext=xytext,
    #     xy=(largest_n, largest_value),
    #     xycoords="data",
    # )

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xscale("log")
    ax2.set_xticks(df["num_samples"])
    ax2.set_xticklabels([str(m) for m in df["num_sites"]])
    ax2.set_xlabel("Number of variants")

    ax.legend()
    plt.tight_layout()
    plt.savefig(output)




@click.group()
def cli():
    pass


cli.add_command(process_files)
cli.add_command(plot)


if __name__ == "__main__":
    cli()
