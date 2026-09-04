import os, time

# cublas reductions vary between runs, must be set before cublas initializes
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

torch.use_deterministic_algorithms(True, warn_only=True)

from .preprocess_utils.partfield_official.run_PF import PFInferenceModel
from .preprocess_utils.PartField_pipeline import PF_pipeline
from .preprocess_utils.manifold import fix_mesh_trimesh
from .preprocess_utils.merge_V_obj import load_mesh_and_merge

def preprocess(mesh_path, pf_model=None, output_path=None, save_tree_file=False, save_processed_mesh=False, sample_on_faces=10, sample_batch_size=100_000, merge_vertices_epsilon=1e-7):
    stem, _ = os.path.splitext(os.path.basename(mesh_path))
    if output_path is None:
        output_path = mesh_path
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(os.path.dirname(output_path), stem)
        # shutil.copy(mesh_path, output_path)
        

    preprocess_times = {}

    load_time = time.perf_counter()
    mesh = load_mesh_and_merge(mesh_path, epsilon=merge_vertices_epsilon)
    load_time = time.perf_counter() - load_time
    fix_time = time.perf_counter()
    mesh = fix_mesh_trimesh(mesh)   
    fix_time = time.perf_counter() - fix_time
    export_time = time.perf_counter()

    export_path = os.path.join(output_path, f"{stem}.obj")
    if save_processed_mesh:
        # Export mesh with adapted extension
        mesh.export(export_path)

    export_time = time.perf_counter() - export_time
    tree_dict = {}
    if pf_model is None:
        pf_model = PFInferenceModel(device="cuda")
    
    pf_time = time.perf_counter()

    bin_path = os.path.join(output_path, "bin")
    if save_tree_file:
        os.makedirs(bin_path, exist_ok=True)


    binary_file_path, tree_dict = PF_pipeline(
        pf_model=pf_model,
        mesh_path=export_path,
        mesh=mesh,
        output_path=bin_path,
        save_binary=save_tree_file,
        sample_on_faces=sample_on_faces,
        sample_batch_size=sample_batch_size,
    )
    pf_time = time.perf_counter() - pf_time

    preprocess_times["load"] = load_time
    preprocess_times["fix"] = fix_time
    preprocess_times["export"] = export_time
    preprocess_times["pf"] = pf_time
    
    return mesh, binary_file_path, tree_dict, preprocess_times
