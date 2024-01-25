from setuptools import setup, find_packages

setup(
    name="mlff",
    version="1.0",
    description="Build Neural Networks for Force Fields with JAX",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "clu",
        # "jax == 0.4.8",
        "flax",
        "jaxopt",
        "jraph",
        "optax",
        "orbax-checkpoint",
        "portpicker",
        # 'tensorflow',
        "scikit-learn",
        "ase",
        "tqdm",
        "wandb",
        "pyyaml",
        "pytest",
        "h5py",
        "ml_collections"
    ],
    include_package_data=True,
    package_data={"": ["sph_ops/cgmatrix.npz",
                       "sph_ops/u_matrix.pickle"]},
    entry_points={
        "console_scripts": [
            "evaluate=mlff.cAPI.mlff_eval:evaluate",
            "train=mlff.cAPI.mlff_train:train",
            "run_md=mlff.cAPI.mlff_md:run_md",
            "run_relaxation=mlff.cAPI.mlff_structure_relaxation:run_relaxation",
            "analyse_md=mlff.cAPI.mlff_analyse:analyse_md",
            "train_so3krates=mlff.cAPI.mlff_train_so3krates:train_so3krates",
            "train_so3kratACE=mlff.cAPI.mlff_train_so3kratace:train_so3kratace",
            "trajectory_to_xyz=mlff.cAPI.mlff_postprocessing:trajectory_to_xyz",
            "to_mlff_input=mlff.cAPI.mlff_input_processing:to_mlff_input",
            "train_so3krates_sparse=mlff.CLI.run_training:train_so3krates_sparse",
            "evaluate_so3krates_sparse=mlff.CLI.run_evaluation:evaluate_so3krates_sparse",
            "evaluate_so3krates_sparse_on=mlff.CLI.run_evaluation_on:evaluate_so3krates_sparse_on"
        ],
    },
)
