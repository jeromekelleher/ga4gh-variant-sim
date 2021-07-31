import tempfile
import pathlib
import itertools

import pytest
import msprime
import numpy as np
import sgkit as sg
import click.testing
import cyvcf2

import simulator


class TestSgkitRoundTrip:
    def verify(self, ts, variant_chunk_size=10, sample_chunk_size=10):
        with tempfile.TemporaryDirectory() as tmpdir:
            ts_path = pathlib.Path(tmpdir) / "ts.trees"
            sg_path = pathlib.Path(tmpdir) / "sgkit.zarr"
            ts.dump(ts_path)
            runner = click.testing.CliRunner()
            result = runner.invoke(
                simulator.trees_to_sgkit_zarr,
                [
                    str(ts_path),
                    str(sg_path),
                    "--variant-chunk-size",
                    variant_chunk_size,
                    "--sample-chunk-size",
                    sample_chunk_size,
                ],
            )
            assert result.exit_code == 0
            ds = sg.load_dataset(sg_path)
            assert ts.num_individuals == ds.samples.shape[0]
            assert ts.num_sites == ds.variant_position.shape[0]

            genotypes = np.array(ds.call_genotype).reshape(
                (ts.num_sites, ts.num_samples)
            )
            assert np.array_equal(genotypes, ts.genotype_matrix())
            assert np.all(ts.tables.sites.position == ds.variant_position)
            # TODO assert more stuff.

    @pytest.mark.parametrize("n", [2, 5, 100])
    def test_short_genome(self, n):
        ts = msprime.sim_ancestry(n, sequence_length=10, random_seed=234)
        ts = msprime.sim_mutations(ts, 0.1, random_seed=2345)
        assert ts.num_mutations > 0
        self.verify(ts)

    @pytest.mark.parametrize("n", [2, 5, 100])
    def test_long_genome(self, n):
        ts = msprime.sim_ancestry(n, sequence_length=10000, random_seed=234)
        ts = msprime.sim_mutations(ts, 0.01, random_seed=2345)
        assert ts.num_mutations > 0
        self.verify(ts, variant_chunk_size=1000)


class TestVcfRoundTrip:
    def verify(self, ts):
        with tempfile.TemporaryDirectory() as tmpdir:
            ts_path = pathlib.Path(tmpdir) / "ts.trees"
            vcf_path = pathlib.Path(tmpdir) / "out.vcf"
            ts.dump(ts_path)
            runner = click.testing.CliRunner()
            result = runner.invoke(
                simulator.trees_to_vcf,
                [
                    str(ts_path),
                    str(vcf_path),
                ],
            )
            assert result.exit_code == 0

            vcf = cyvcf2.VCF(vcf_path)
            for vcf_var, tsk_var in itertools.zip_longest(vcf, ts.variants()):
                assert vcf_var.REF == tsk_var.alleles[0]
                offset = 0
                for genotype in vcf_var.genotypes:
                    assert genotype[2]  # phased
                    assert np.all(
                        genotype[:2] == tsk_var.genotypes[offset : offset + 2]
                    )
                    offset += 2

    @pytest.mark.parametrize("n", [2, 5, 100])
    def test_short_genome(self, n):
        ts = msprime.sim_ancestry(n, sequence_length=10, random_seed=234)
        ts = msprime.sim_mutations(ts, 0.1, random_seed=2345)
        assert ts.num_mutations > 0
        self.verify(ts)

    @pytest.mark.parametrize("n", [2, 5, 100])
    def test_long_genome(self, n):
        ts = msprime.sim_ancestry(n, sequence_length=10000, random_seed=234)
        ts = msprime.sim_mutations(ts, 0.01, random_seed=2345)
        assert ts.num_mutations > 0
        self.verify(ts)
