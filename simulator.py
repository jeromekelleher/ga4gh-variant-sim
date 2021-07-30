import click
import itertools
import msprime
import numpy as np
import sgkit as sg
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
@click.option(
    "--ploidy",
    type=int,
    default=2
)
@click.option(
    "--contig-id",
    type=str,
    default="1",
    help="The ID of the contig in the output."
)
@click.option(
    "--max-alt-alleles",
    type=int,
    default=1,
    help="The number of alt alleles in the output. Sites with more or less are truncated or padded to this length."
)
@click.option(
    "--variant-chunk-size",
    type=int,
    default=10_000,
    help="The size of chunks in the variants dimension."
)
@click.option(
    "--sample-chunk-size",
    type=int,
    default=1_000,
    help="The size of chunks in the samples dimension."
)
def trees_to_sgkit_zarr(infile, outfile, ploidy, contig_id, max_alt_alleles, variant_chunk_size, sample_chunk_size):
    """
    Convert the input tskit trees file to vcf.
    """
    ts = tskit.load(infile)

    if ts.num_samples % ploidy != 0:
        raise ValueError("Sample size must be divisible by ploidy")
    num_individuals = ts.num_samples // ploidy
    individual_names = [f"tsk_{j}" for j in range(num_individuals)]

    samples = ts.samples()
    tables = ts.dump_tables()

    # TODO: is there some way of finding max_alleles from ts?
    max_alleles = max_alt_alleles + 1

    offset = 0
    first_variants_chunk = True
    for variants_chunk in chunks_iterator(ts.variants(samples=samples), variant_chunk_size):
        alleles = []
        genotypes = []
        for var in variants_chunk:
            site_alleles = var.alleles
            if len(site_alleles) > max_alleles:
                site_alleles = site_alleles[:max_alleles]
            elif len(site_alleles) < max_alleles:
                site_alleles = site_alleles + ([""] * (max_alleles - len(site_alleles)))
            alleles.append(site_alleles)
            genotypes.append(var.genotypes)

        alleles = np.array(alleles).astype("O")
        genotypes = np.expand_dims(genotypes, axis=2)
        genotypes = genotypes.reshape(-1, num_individuals, ploidy)

        n_variants_in_chunk = len(genotypes)

        variant_id = np.full((n_variants_in_chunk), fill_value=".", dtype="O")
        variant_id_mask = variant_id == "."

        ds = sg.create_genotype_call_dataset(
            variant_contig_names=[contig_id],
            variant_contig=np.zeros(n_variants_in_chunk, dtype="i1"),
            # TODO: should this be i8?
            variant_position=tables.sites.position[
                offset : offset + n_variants_in_chunk
            ].astype("i4"),
            variant_allele=alleles,
            sample_id=np.array(individual_names).astype("U"),
            call_genotype=genotypes,
            variant_id=variant_id,
        )
        ds["variant_id_mask"] = (
            ["variants"],
            variant_id_mask,
        )

        if first_variants_chunk:
            # Enforce uniform chunks in the variants dimension
            # Also chunk in the samples direction
            chunk_sizes = dict(variants=variant_chunk_size, samples=sample_chunk_size)
            from numcodecs import Blosc, PackBits
            compressor = Blosc(cname="zstd", clevel=7, shuffle=Blosc.AUTOSHUFFLE)
            encoding = {}
            for var in ds.data_vars:
                var_chunks = tuple(
                    chunk_sizes.get(dim, size)
                    for (dim, size) in zip(ds[var].dims, ds[var].shape)
                )
                encoding[var] = dict(chunks=var_chunks, compressor=compressor)
                if ds[var].dtype.kind == "b":
                    encoding[var]["filters"] = [PackBits()]
            ds.to_zarr(outfile, mode="w", encoding=encoding)
            first_variants_chunk = False
        else:
            # Append along the variants dimension
            ds.to_zarr(outfile, append_dim="variants")

        offset += n_variants_in_chunk


# Based on https://dev.to/orenovadia/solution-chunked-iterator-python-riddle-3ple
def chunks_iterator(iterator, n):
    """
    Convert an iterator into an iterator of iterators, where the inner iterators
    each return `n` items, except the last, which may return fewer.

    For the special case of an empty iterator, an iterator of an empty iterator is
    returned.
    """

    empty_iterator = True
    for first in iterator:  # take one item out (exits loop if `iterator` is empty)
        empty_iterator = False
        rest_of_chunk = itertools.islice(iterator, 0, n - 1)
        yield itertools.chain([first], rest_of_chunk)  # concatenate the first item back
    if empty_iterator:
        yield iter([])


@click.group()
def cli():
    pass


cli.add_command(generate_trees)
cli.add_command(trees_to_vcf)
cli.add_command(trees_to_tsinfer_zarr)
cli.add_command(trees_to_sgkit_zarr)


if __name__ == "__main__":
    cli()
