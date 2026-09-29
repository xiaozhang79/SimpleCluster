"""Model registration for the three supported video backbones."""

from lmms_eval.models.registry_v2 import ModelManifest, ModelRegistryV2


AVAILABLE_SIMPLE_MODELS = {
    "llava_onevision": "Llava_OneVision",
    "llava_vid": "LlavaVid",
    "internvl3": "InternVL3",
}
AVAILABLE_CHAT_TEMPLATE_MODELS = {}
MODEL_ALIASES = {}

MODEL_REGISTRY_V2 = ModelRegistryV2()
for model_id, class_name in AVAILABLE_SIMPLE_MODELS.items():
    MODEL_REGISTRY_V2.register_manifest(
        ModelManifest(
            model_id=model_id,
            simple_class_path=f"lmms_eval.models.simple.{model_id}.{class_name}",
        )
    )

from simplecluster.registry import get_manifests

for manifest in get_manifests():
    MODEL_REGISTRY_V2.register_manifest(manifest, overwrite=True)
    AVAILABLE_SIMPLE_MODELS[manifest.model_id] = manifest.simple_class_path.rsplit(".", 1)[-1]

AVAILABLE_MODELS = {
    model_id: (manifest.simple_class_path or manifest.chat_class_path).rsplit(".", 1)[-1]
    for model_id in MODEL_REGISTRY_V2.list_canonical_model_ids()
    for manifest in [MODEL_REGISTRY_V2.get_manifest(model_id)]
}


def list_available_models(include_aliases: bool = False) -> list[str]:
    if include_aliases:
        return MODEL_REGISTRY_V2.list_model_names()
    return MODEL_REGISTRY_V2.list_canonical_model_ids()


def get_model_manifest(model_name: str) -> ModelManifest:
    return MODEL_REGISTRY_V2.get_manifest(model_name)


def get_model(model_name: str, force_simple: bool = False) -> type:
    return MODEL_REGISTRY_V2.get_model_class(model_name, force_simple=force_simple)
