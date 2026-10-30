"""The list of all capabilities.

Capabilities register themselves when imported. This list is the only place that
names them, so no manifest and no discovery mechanism is needed.
"""

from __future__ import annotations

import importlib
import sys

from alambiq.core import registry

MODULES = (
    "alambiq.capabilities.characterize.protein_inspect",
    "alambiq.capabilities.characterize.protein_search",
    "alambiq.capabilities.characterize.sequence_inspect_msa",
    "alambiq.capabilities.characterize.genome_inspect_context",
    "alambiq.capabilities.characterize.protein_match_profiles",
    "alambiq.capabilities.characterize.sequence_generate_msa",
    "alambiq.capabilities.characterize.protein_annotate_function",
    "alambiq.capabilities.characterize.protein_call_substitutions",
    "alambiq.capabilities.characterize.protein_embed",
    "alambiq.capabilities.characterize.protein_score_substitutions",
    "alambiq.capabilities.characterize.protein_predict_tracks",
    "alambiq.capabilities.characterize.protein_complete",
    "alambiq.capabilities.characterize.protein_predict_stability_change",
    "alambiq.capabilities.characterize.structure_predict_biomolecular_complex",
    "alambiq.capabilities.characterize.protein_explore_embeddings",
    "alambiq.capabilities.characterize.enzyme_retrieve_candidates",
    "alambiq.capabilities.characterize.enzyme_characterize_candidates",
    "alambiq.capabilities.investigate.molecule_compute_quantum_properties",
    "alambiq.capabilities.investigate.molecule_predict_admet",
    "alambiq.capabilities.investigate.reaction_inspect",
    "alambiq.capabilities.investigate.reaction_estimate_thermodynamics",
    "alambiq.capabilities.investigate.reaction_explore_routes",
    "alambiq.capabilities.investigate.reaction_analyze_enzyme_panel",
    "alambiq.capabilities.investigate.structure_compare",
    "alambiq.capabilities.investigate.structure_prepare_protein_system",
    "alambiq.capabilities.investigate.simulation_run_molecular_dynamics",
    "alambiq.capabilities.investigate.simulation_run_metadynamics",
    "alambiq.capabilities.investigate.simulation_inspect_trajectory",
    "alambiq.capabilities.investigate.simulation_compare_replica_structures",
    "alambiq.capabilities.investigate.simulation_compare_scalar_couplings",
    "alambiq.capabilities.investigate.simulation_analyze_binding_free_energy",
    "alambiq.capabilities.investigate.simulation_run_relative_free_energy",
    "alambiq.capabilities.investigate.structure_prepare_relative_binding_inputs",
    "alambiq.capabilities.investigate.simulation_evaluate_qmmm_state",
    "alambiq.capabilities.investigate.simulation_run_qmmm_dynamics",
    "alambiq.capabilities.investigate.structure_prepare_reactive_coordinates",
    "alambiq.capabilities.investigate.structure_prepare_finite_environment",
    "alambiq.capabilities.investigate.structure_prepare_protein_ligand_system",
    "alambiq.capabilities.investigate.simulation_analyze_reaction_free_energy",
    "alambiq.capabilities.investigate.structure_prepare_predicted_complex_for_dynamics",
    "alambiq.capabilities.investigate.structure_detect_binding_pockets",
    "alambiq.capabilities.measure.assay_inspect",
    "alambiq.capabilities.measure.assay_fit",
    "alambiq.capabilities.measure.assay_balance",
    "alambiq.capabilities.operate.define",
    "alambiq.capabilities.operate.propagate",
    "alambiq.capabilities.operate.simulate",
    "alambiq.capabilities.operate.fit",
    "alambiq.capabilities.operate.compare",
    "alambiq.capabilities.operate.summarize",
    "alambiq.capabilities.operate.resources",
    "alambiq.capabilities.engineer.candidate_fit",
    "alambiq.capabilities.engineer.candidate_select",
    "alambiq.capabilities.engineer.variant_propose",
    "alambiq.capabilities.engineer.construct_compose",
    "alambiq.capabilities.engineer.construct_assemble",
    "alambiq.capabilities.engineer.coding_optimize",
)


def load() -> None:
    """Import every capability module, registering what it declares.

    Importing an already-imported module does not run its decorators again, so
    a module whose registration has been cleared is reloaded rather than
    skipped.
    """
    for name in MODULES:
        if registry.is_loaded(name):
            continue
        if name in sys.modules:
            importlib.reload(sys.modules[name])
        else:
            importlib.import_module(name)
        registry.mark_loaded(name)
