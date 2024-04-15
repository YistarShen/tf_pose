from engine.registry import Registry

model_test = Registry('model_test', locations=['lib.pkg_sample'], show_log=True ) 

MODELS  = Registry('models', locations=['lib.models'], show_log=True)

LAYERS = Registry('layers', locations=['lib.layers'], show_log=True)

MODULES = Registry('layers', locations=['lib.models.modules'], show_log=True)

LOSSES =  Registry('loss', locations=['lib.losses'], show_log=True)

METRICS =  Registry('metric', locations=['lib.metrics'], show_log=True)

DATASETS = Registry('tfds_pipeline', locations=['lib.datasets'], show_log=True)

TRANSFORMS = Registry('transforms', locations=['lib.datasets.transforms'], show_log=True)

CODECS = Registry('codec', locations=['lib.codecs'])

INFER_PROC = Registry('inference_proc', locations=['lib.infer'], show_log=True)

INFER_API = Registry('inference_api', locations=['apis'], show_log=True)
