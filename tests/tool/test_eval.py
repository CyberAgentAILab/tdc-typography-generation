import pytest
from typography_generation.__main__ import (
    get_global_config,
    get_model_config_input,
    get_prefix_lists,
)
from typography_generation.config.default import (
    get_font_config,
)
from typography_generation.io.build_dataset import build_test_dataset
from typography_generation.model.model import create_model
from typography_generation.tools.evaluator import Evaluator


def test_eval_iter(bartconfig, bartconfigdataset_test, bartmodel) -> None:
    prefix_list_object = get_prefix_lists(bartconfig)

    gpu = False
    save_dir = "job"
    evaluator = Evaluator(
        bartmodel,
        gpu,
        save_dir,
        bartconfigdataset_test,
        prefix_list_object,
        debug=True,
    )
    evaluator.eval_model()


@pytest.mark.parametrize(
    "data_dir, save_dir, config_name, elementembeddingflagconfigname, canvasembeddingflagconfigname, elementpredictionflagconfigname",
    [
        [
            "data",
            "job",
            "bart",
            "text_element_embedding_flag_config/wofontsize",
            "canvas_embedding_flag_config/canvas_detail_given",
            "text_element_prediction_flag_config/rawfontsize",
        ],
    ],
)
def test_flag_config_eval(
    data_dir: str,
    save_dir: str,
    config_name: str,
    elementembeddingflagconfigname: str,
    canvasembeddingflagconfigname: str,
    elementpredictionflagconfigname,
) -> None:
    global_config_input = {}
    global_config_input["data_dir"] = data_dir
    global_config_input["model_name"] = config_name
    global_config_input["test_config_name"] = "test_config"
    global_config_input["model_config_name"] = "model_config"
    global_config_input[
        "elementembeddingflag_config_name"
    ] = elementembeddingflagconfigname
    global_config_input[
        "canvasembeddingflag_config_name"
    ] = canvasembeddingflagconfigname
    global_config_input[
        "elementpredictionflag_config_name"
    ] = elementpredictionflagconfigname

    config = get_global_config(**global_config_input)
    model_name, model_kwargs = get_model_config_input(config)

    model = create_model(
        model_name,
        **model_kwargs,
    )
    # import torch
    # model.load_state_dict(torch.load("job/weight.pth", map_location="cpu"))
    prefix_list_object = get_prefix_lists(config)
    font_config = get_font_config(config)

    dataset = build_test_dataset(
        data_dir,
        prefix_list_object,
        font_config,
        debug=True,
    )

    gpu = False
    evaluator = Evaluator(
        model,
        gpu,
        save_dir,
        dataset,
        prefix_list_object,
        debug=True,
    )
    evaluator.eval_model()
