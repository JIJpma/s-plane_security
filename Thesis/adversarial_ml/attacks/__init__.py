from adversarial_ml.attacks.fgsm import FGSM
from adversarial_ml.attacks.pgd import PGD

ATTACK_REGISTRY: dict[str, type] = {
    "fgsm": FGSM,
    "pgd": PGD,
}


def get_attack(name: str, **kwargs):
    """Instantiate an attack by name.  Extra kwargs are forwarded to the constructor."""
    cls = ATTACK_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown attack '{name}'. Choose from: {list(ATTACK_REGISTRY)}")
    return cls(**kwargs)
