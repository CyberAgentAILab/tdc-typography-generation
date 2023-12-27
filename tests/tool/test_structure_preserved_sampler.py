

from typography_generation.__main__ import (
    get_prefix_lists,
    get_sampling_config,
)
from typography_generation.tools.structure_preserved_sampler import (
    StructurePreservedSampler,
)


def test_sample_iter(bartconfig, bartconfigdataset_test, bartmodel) -> None:
    prefix_list_object = get_prefix_lists(bartconfig)
    sampling_config = get_sampling_config(bartconfig)

    gpu = False
    save_dir = "job"
    sampler = StructurePreservedSampler(
        bartmodel,
        gpu,
        save_dir,
        bartconfigdataset_test,
        prefix_list_object,
        sampling_config,
        debug=True,
    )
    sampler.sample()
