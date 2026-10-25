"""Data-only System and State loading for native OpenMM runners."""

import re
import xml.etree.ElementTree as ET

_DATA_TYPES = frozenset(
    """
    System State ATMForce AmoebaGeneralizedKirkwoodForce AmoebaMultipoleForce
    AmoebaTorsionTorsionForce AmoebaVdwForce AmoebaWcaDispersionForce
    AndersenThermostat CMAPTorsionForce CMMotionRemover ConstantPotentialForce
    Continuous1DFunction Continuous2DFunction Continuous3DFunction CustomAngleForce
    CustomBondForce CustomCVForce CustomCentroidBondForce CustomCompoundBondForce
    CustomExternalForce CustomGBForce CustomHbondForce CustomManyParticleForce
    CustomNonbondedForce CustomTorsionForce CustomVolumeForce Discrete1DFunction
    Discrete2DFunction Discrete3DFunction DrudeForce GBSAOBCForce GayBerneForce
    HarmonicAngleForce HarmonicBondForce HippoNonbondedForce LCPOForce
    LocalCoordinatesSite MonteCarloAnisotropicBarostat MonteCarloBarostat
    MonteCarloFlexibleBarostat MonteCarloMembraneBarostat NonbondedForce
    OrientationRestraintForce OutOfPlaneSite PeriodicTorsionForce RBTorsionForce
    RGForce RMSDForce RPMDMonteCarloBarostat SymmetrySite ThreeParticleAverageSite
    TwoParticleAverageSite
    """.split()
)


class _DataTree(ET.TreeBuilder):
    def doctype(self, *args):
        raise ValueError("A document type declaration is present; serialized OpenMM inputs must be "
            "self-contained XML data")


def deserialize_data(text, expected_type, deserialize):
    """Check the complete data tree before entering a native deserialization proxy."""
    if expected_type not in {"System", "State"} or not isinstance(text, str):
        raise ValueError("The input must be an XML string with a System or State root")
    try:
        tree = ET.fromstring(text, parser=ET.XMLParser(target=_DataTree()))
    except ET.ParseError as exc:
        raise ValueError(
            "The input is not well-formed XML; a complete OpenMM data document is required"
        ) from exc
    if tree.tag != expected_type or tree.get("type") != expected_type:
        raise ValueError(
            f"The root element and its native type must both be {expected_type}"
        )
    for node in tree.iter():
        kind = node.get("type")
        if (
            kind is not None
            and kind not in _DATA_TYPES
            and re.fullmatch(r"-?[0-9]+", kind) is None
        ):
            raise ValueError(
                f"Serialized input contains the native type {kind}, which is not an allowed "
                "native data type"
            )
    return deserialize(text)
