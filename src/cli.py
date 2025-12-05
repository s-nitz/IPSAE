import logging
import os
import zipfile
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from glob import glob
import re
import argparse

from .ipsae import ipsae, setup, configure


def write_outputs(output_dir, chain_asym, chain_max):
    logging.info("Writing chain-level outputs")
    asym_output = os.path.join(output_dir, "asym")
    max_output = os.path.join(output_dir, "max")
    os.makedirs(asym_output, exist_ok=True)
    os.makedirs(max_output, exist_ok=True)
    for metric, df in chain_asym.items():
        df.to_csv(os.path.join(asym_output, f"{metric}.csv"))
    for metric, df in chain_max.items():
        df.to_csv(os.path.join(max_output, f"{metric}.csv"))


def _zipsae(
    paths,
    pae_cutoff,
    dist_cutoff,
    zipfile_dir,
    output_dir,
):
    (
        poolnum,
        modelnum,
        pdb_path,
        pae_path,
    ) = paths
    with zipfile.ZipFile(zipfile_dir) as archive:
        algorithm, protein_file_type, _, _, _, _ = configure(
            pdb_path,
            pae_path,
            pae_cutoff,
            dist_cutoff,
        )
        with archive.open(pdb_path, "r") as pdb_file, archive.open(
            pae_path, "r"
        ) as pae_file:
            residues, chains, distances, pae_matrix, cb_plddt = setup(
                pdb_file,
                pae_file,
                protein_file_type,
                algorithm,
            )
            logging.info(f"{zipfile_dir} :: setup success")
        chain_asym, chain_max = ipsae(
            residues, chains, distances, pae_matrix, cb_plddt, pae_cutoff
        )
        logging.info(f"{zipfile_dir} :: writing outputs")
        write_outputs(
            os.path.join(output_dir, poolnum, modelnum), chain_asym, chain_max
        )


def single_ipsae(
    paths,
    pae_cutoff,
    dist_cutoff,
    output_dir,
):
    poolnum, modelnum, pdb_path, pae_path = paths
    logging.info(f"Running ipSAE on file {pdb_path}")
    algorithm, protein_file_type, _, _, _, _ = configure(
        pdb_path,
        pae_path,
        pae_cutoff,
        dist_cutoff,
    )

    with open(pdb_path, "r") as pdb_file, open(pae_path, "r") as pae_file:

        residues, chains, distances, pae_matrix, cb_plddt = setup(
            pdb_file,
            pae_file,
            protein_file_type,
            algorithm,
        )

    chain_asym, chain_max = ipsae(
        residues, chains, distances, pae_matrix, cb_plddt, pae_cutoff
    )

    write_outputs(
        os.path.join(output_dir, str(poolnum), str(modelnum)), chain_asym, chain_max
    )


def run_from_zip(
    zipfile_dir,
    pae_cutoff,
    dist_cutoff,
    output_dir,
    num_workers=12,
    file_prefix="fold_mtb_essentials_allvall",
    resume=False,
):
    with zipfile.ZipFile(zipfile_dir) as archive:
        inputs = []
        for filename in archive.namelist():
            if match := re.search(
                rf"{file_prefix}_(\d+)_summary_confidences_(\d).json", filename
            ):
                dirname = os.path.dirname(filename)  # mtb_full_1171
                poolnum, modelnum = match.groups()
                # logging.info(f"running on pool {poolnum} model {modelnum}")
                if resume:
                    existing_path = os.path.join(
                        output_dir, poolnum, modelnum, "max", "*.csv"
                    )
                    if len(glob(existing_path)) >= 4:
                        logging.info(
                            f"Pool {poolnum} model {modelnum} already complete, skipping."
                        )
                        continue
                pdb_path = os.path.join(
                    # mtb_full_1171/fold_mtb_full_1171_model_0.cif
                    dirname,
                    f"{file_prefix}_{poolnum}_model_{modelnum}.cif",
                )
                pae_path = os.path.join(
                    dirname,
                    f"{file_prefix}_{poolnum}_full_data_{modelnum}.json",
                )
                # inputs = (poolnum, modelnum, pdb_path, pae_path)
                # _zipsae(inputs, pae_cutoff, dist_cutoff, zipfile_dir, output_dir)
                # break
                inputs.append(
                    (
                        poolnum,
                        modelnum,
                        pdb_path,
                        pae_path,
                    )
                )
    logging.info(f"preparing to run {len(inputs)} files on {num_workers} CPUs.")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # enumerate pdb and pae files
        configured_zipsae = partial(
            _zipsae,
            pae_cutoff=pae_cutoff,
            dist_cutoff=dist_cutoff,
            zipfile_dir=zipfile_dir,
            output_dir=output_dir,
        )

        executor.map(configured_zipsae, inputs)


def run_from_folder(
    input_dir,
    pae_cutoff,
    dist_cutoff,
    output_dir,
    num_workers=12,
    file_prefix="fold_mtb_essentials_allvall",
):
    inputs = []
    for filename in glob(
        os.path.join(input_dir, "**/*confidence*.json"), recursive=True
    ):
        if match := re.search(
            r"(\d+)_summary_confidences_(\d).json", filename
        ):  # (pooled) af3 server style
            dirname = os.path.dirname(filename)
            poolnum, modelnum = match.groups()
            pdb_path = os.path.join(
                dirname, f"{file_prefix}_{poolnum}_model_{modelnum}.cif"
            )
            pae_path = os.path.join(
                dirname, f"{file_prefix}_{poolnum}_full_data_{modelnum}.json"
            )
            inputs.append(
                (
                    poolnum,
                    modelnum,
                    pdb_path,
                    pae_path,
                )
            )
        elif match := re.search(
            r"(\w+)_summary_confidences.json", filename
        ):  # af3 local style (?)
            dirname = os.path.dirname(filename)
            zinc_id = match.groups()[0]
            pdb_path = os.path.join(dirname, f"{zinc_id}_model.cif")
            pae_path = os.path.join(dirname, f"{zinc_id}_confidences.json")
            inputs.append(
                (
                    zinc_id,
                    0,
                    pdb_path,
                    pae_path,
                )
            )
        elif match := re.search(
            r"confidence_(\w+)_model_(\d+).json", filename
        ):  # boltz2 style
            dirname = os.path.dirname(filename)
            zinc_id, modelnum = match.groups()
            pdb_path = os.path.join(dirname, f"{zinc_id}_model_{modelnum}.cif")
            pae_path = os.path.join(dirname, f"pae_{zinc_id}_model_{modelnum}.npz")
            inputs.append(
                (
                    zinc_id,
                    modelnum,
                    pdb_path,
                    pae_path,
                )
            )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # enumerate pdb and pae files
        configured_single_ipsae = partial(
            single_ipsae,
            pae_cutoff=pae_cutoff,
            dist_cutoff=dist_cutoff,
            output_dir=output_dir,
        )

        executor.map(configured_single_ipsae, inputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("pae_cutoff", type=float)
    parser.add_argument("dist_cutoff", type=float)
    parser.add_argument("output_dir")
    parser.add_argument(
        "-p", "--file-prefix", type=str, default="fold_mtb_essentials_allvall"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-w", "--workers", type=int)
    parser.add_argument("-d", "--dry-run", action="store_true")
    parser.add_argument("-r", "--resume", action="store_true")

    args = parser.parse_args()
    loglevel = logging.INFO if args.verbose else logging.WARN
    logging.basicConfig(format="%(asctime)s::%(message)s", level=loglevel)

    if ".zip" in args.input_dir:
        if args.dry_run:
            print(f"Would run on zipfile {args.input_dir}.")
        else:
            # logging.info(
            #     f"Running ipSAE on all outputs stored in zipfile {args.input_dir}"
            # )
            run_from_zip(
                args.input_dir,
                args.pae_cutoff,
                args.dist_cutoff,
                args.output_dir,
                num_workers=args.workers,
                resume=args.resume,
                file_prefix=args.file_prefix,
            )
    else:  # assumes you're running from a directory
        logging.info(
            f"Running ipSAE on all outputs stored in directory {args.input_dir}"
        )
        run_from_folder(
            args.input_dir,
            args.pae_cutoff,
            args.dist_cutoff,
            args.output_dir,
            num_workers=args.workers,
            file_prefix=args.file_prefix,
        )


if __name__ == "__main__":
    main()
