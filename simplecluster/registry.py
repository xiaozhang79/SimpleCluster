from lmms_eval.models.registry_v2 import ModelManifest


def get_manifests() -> list[ModelManifest]:
    return [
        ModelManifest(
            model_id="llava_onevision_simplecluster",
            simple_class_path="simplecluster.llava_onevision.LlavaOneVisionSimpleCluster",
        ),
        ModelManifest(
            model_id="llava_video_simplecluster",
            simple_class_path="simplecluster.llava_video.LlavaVideoSimpleCluster",
        ),
        ModelManifest(
            model_id="internvl3_simplecluster",
            simple_class_path="simplecluster.internvl3.InternVL3SimpleCluster",
        ),
    ]
