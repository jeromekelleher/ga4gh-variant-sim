import click
import msprime
import tskit
import tsinfer


@click.command()
@click.argument("sample_size", type=int)
@click.argument("outfile", type=click.File("wb"))
@click.option(
    "-L",
    "--sequence-length",
    type=int,
    default=100,
    help="The length of the simulated chromosome in megabases",
)
def generate_trees(sample_size, outfile, sequence_length):
    """
    Simulates the trees and mutations for the specified diploid sample size.
    """
    # Using very basic human-like model here. We can add more elaborate
    # models, using stdpopsim.
    ts = msprime.sim_ancestry(
        sample_size,
        population_size=10_000,
        recombination_rate=1e-8,
        sequence_length=sequence_length * 10 ** 6,
        random_seed=42,
    )
    ts = msprime.sim_mutations(ts, rate=1e-8)
    ts.dump(outfile)


@click.command()
@click.argument("infile", type=click.File("rb"))
@click.argument("outfile", type=click.Path())
def trees_to_tsinfer_zarr(infile, outfile):
    """
    Convert the input tskit trees file to a tsinfer format zarr.
    """
    ts = tskit.load(infile)
    with tsinfer.SampleData(path=outfile, sequence_length=ts.sequence_length) as sd:
        for ind in ts.individuals():
            sd.add_individual(ploidy=len(ind.nodes))
        with click.progressbar(ts.variants()) as bar:
            for var in bar:
                sd.add_site(var.site.position, var.genotypes, var.alleles)
    print(sd)


@click.command()
@click.argument("infile", type=click.File("rb"))
@click.argument("outfile", type=click.File("w"))
def trees_to_vcf(infile, outfile):
    """
    Convert the input tskit trees file to vcf.
    """
    ts = tskit.load(infile)
    # If we want to write out extra fields, we'll need to write our
    # own code to output the VCF becauase tskit doesn't support this.
    ts.write_vcf(outfile)


@click.command()
@click.argument("infile", type=click.File("rb"))
# TODO presumably this must be a directory for sgkit zarrs?
@click.argument("outfile", type=click.Path())
def trees_to_sgkit_zarr(infile, outfile):
    """
    Convert the input tskit trees file to vcf.
    """
    pass


@click.group()
def cli():
    pass


cli.add_command(generate_trees)
cli.add_command(trees_to_vcf)
cli.add_command(trees_to_tsinfer_zarr)
cli.add_command(trees_to_sgkit_zarr)


if __name__ == "__main__":
    cli()
