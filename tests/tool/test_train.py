import pytest
from typography_generation.__main__ import (
    get_global_config,
    get_model_config_input,
    get_prefix_lists,
)
from typography_generation.config.default import (
    get_font_config,
)
from typography_generation.io.build_dataset import (
    build_test_dataset,
    build_train_dataset,
)
from typography_generation.io.data_object import ModelInput
from typography_generation.model.model import create_model
from typography_generation.tools.loss import LossFunc
from typography_generation.tools.train import Trainer


def test_compute_loss(bartconfig, bartconfigdataloader, bartmodel) -> None:
    prefix_list_object = get_prefix_lists(bartconfig)

    tmp = bartconfigdataloader.__iter__()
    design_context_list, model_input_batchdata, _, _ = next(tmp)
    model_inputs = ModelInput(design_context_list, model_input_batchdata, gpu=False)
    outputs = bartmodel(model_inputs)
    prediction_config_element = bartconfig.text_element_prediction_attribute_config
    loss = LossFunc(
        bartmodel.model_name,
        prefix_list_object.target,
        prediction_config_element,
        gpu=False,
    )
    loss(model_inputs, outputs, training=True)


def test_train_iter(
    bartconfig, bartconfigdataset, bartconfigdataset_val, bartmodel
) -> None:
    save_dir = "job"
    optimizer_option = bartconfig.train_config.optimizer
    prefix_list_object = get_prefix_lists(bartconfig)
    prediction_config_element = bartconfig.text_element_prediction_attribute_config
    trainer = Trainer(
        bartmodel,
        False,
        save_dir,
        bartconfigdataset,
        bartconfigdataset_val,
        prefix_list_object,
        prediction_config_element,
        optimizer_option=optimizer_option,
        debug=True,
        epochs=1,
    )
    trainer.train_model()


@pytest.mark.parametrize(
    "data_dir, config_name",
    [["data", "mfc"], ["data", "canvasvae"]],
)
def test_train_config(data_dir: str, config_name: str) -> None:
    config = get_global_config(data_dir, config_name)
    model_name, model_kwargs = get_model_config_input(config)

    model = create_model(
        model_name,
        **model_kwargs,
    )
    font_config = get_font_config(config)
    optimizer_option = config.train_config.optimizer
    prefix_list_object = get_prefix_lists(config)
    prediction_config_element = config.text_element_prediction_attribute_config

    dataset, dataset_val = build_train_dataset(
        data_dir,
        prefix_list_object,
        font_config,
        debug=True,
    )

    save_dir = "job"
    trainer = Trainer(
        model,
        False,
        save_dir,
        dataset,
        dataset_val,
        prefix_list_object,
        prediction_config_element,
        optimizer_option=optimizer_option,
        debug=True,
        epochs=1,
    )
    trainer.train_model()


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
def test_flag_config_train(
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
    prefix_list_object = get_prefix_lists(config)
    font_config = get_font_config(config)
    prediction_config_element = config.text_element_prediction_attribute_config
    optimizer_option = config.train_config.optimizer

    dataset, dataset_val = build_train_dataset(
        data_dir,
        prefix_list_object,
        font_config,
        debug=True,
    )

    gpu = False
    trainer = Trainer(
        model,
        False,
        save_dir,
        dataset,
        dataset_val,
        prefix_list_object,
        prediction_config_element,
        optimizer_option=optimizer_option,
        debug=True,
        epochs=1,
    )
    trainer.train_model()
