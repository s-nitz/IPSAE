import os

import pytest
import pandas as pd
import numpy as np

from src.ipsae import single_ipsae, run_from_zip


def _load_dunbrack_output(filename_prefix):
    byres_df = pd.read_csv(f"{filename_prefix}_byres.txt", sep="\s+", index_col=0)

    chainchain_df = pd.read_csv(f"{filename_prefix}.txt", sep="\s+", index_col=[0, 1])

    return byres_df, chainchain_df


# run the dunbrack examples
def _ipsae_aurka_tpx2():
    # (poolnum, modelnum, pdb_path, pae_path, summary_file)
    paths = (
        0,
        0,
        "test/Example/fold_aurka_0_tpx2_0_model_0.cif",
        "test/Example/fold_aurka_0_tpx2_0_full_data_0.json",
    )
    single_ipsae(paths, 10, 10, "test/aurka_tpx2/")
    res_df, cc_df = _load_dunbrack_output(
        "test/Example/fold_aurka_0_tpx2_0_model_0_10_10"
    )

    ccasym = cc_df.loc[cc_df["Type"] == "asym"]

    cc_ipsae = pd.read_csv("test/aurka_tpx2/0/0/asym/ipsae_d0res.csv", index_col=0)
    assert np.allclose(
        cc_ipsae.values[:2, :2],  # dunbrack's original formulation can't handle ligands
        ccasym["ipSAE"].unstack(),
        equal_nan=True,
    )


def test_aurka_tpx2():
    # run IPSAE on the pair
    _ipsae_aurka_tpx2()
    # convert dunbrack outputs to my output format


def _ipsae_raf1_ksr1_mek1_9f755():
    # (poolnum, modelnum, pdb_path, pae_path, summary_file)
    paths = (
        0,
        0,
        "test/Example/RAF1_KSR1_MEK1_9f755_unrelaxed_alphafold2_multimer_v3_model_1_seed_000.pdb",
        "test/Example/RAF1_KSR1_MEK1_9f755_scores_alphafold2_multimer_v3_model_1_seed_000.json.gz",
    )
    single_ipsae(paths, 15, 15, "test/raf1_ksr1_mek1_9f755/")
    res_df, cc_df = _load_dunbrack_output(
        "test/Example/RAF1_KSR1_MEK1_9f755_unrelaxed_alphafold2_multimer_v3_model_1_seed_000_15_15"
    )

    ccasym = cc_df.loc[cc_df["Type"] == "asym"]

    cc_ipsae = pd.read_csv(
        "test/raf1_ksr1_mek1_9f755/0/0/asym/ipsae_d0res.csv", index_col=0
    )
    assert np.allclose(
        cc_ipsae.values,  # dunbrack's original formulation can't handle ligands
        ccasym["ipSAE"].unstack(),
        equal_nan=True,
    )


def test_raf1_ksr1_mek1_9f755():
    _ipsae_raf1_ksr1_mek1_9f755()

    # load outputs
    # "test/raf1_ksr1_mek1_9f755/0/0/max/ipsae_d0res.csv"


# run my example
# def _ipsae_mtb_essess_fullzip():
#     run_from_zip(
#         "test/mtb_essess/inputs.zip",
#         10,
#         10,
#         "test/mtb_essess/outputs/",
#     )


# def test_mtb_essess_fullzip():
#     # this takes too long, make a different zipfile
#     _ipsae_mtb_essess_fullzip()
