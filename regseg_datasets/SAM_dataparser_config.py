from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from regseg_datasets.SAM_dataparser import SAMDataParserConfig

sam_dataparser = DataParserSpecification(
    config=SAMDataParserConfig(),
)
