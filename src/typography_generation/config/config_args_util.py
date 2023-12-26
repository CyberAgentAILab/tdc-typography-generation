from typing import Any, Dict, Tuple

import omegaconf
from logzero import logger

from typography_generation.config.base_config_object import GlobalConfig
from typography_generation.config.default import (
    build_config,
    get_bindata,
    get_datapreprocess_config,
    get_font_config,
    get_model_input_prefix_list,
    get_target_prefix_list,
)
from typography_generation.io.data_object import (
    BinsData,
    DataPreprocessConfig,
    FontConfig,
    PrefixListObject,
    SamplingConfig,
)


def args2add_data_inputs(args: Any) -> Tuple:
    add_data_inputs = {}
    add_data_inputs["data_dir"] = args.datadir
    add_data_inputs["global_config_input"] = get_global_config_input(args.datadir, args)
    add_data_inputs["gpu"] = args.gpu
    add_data_inputs["weight"] = args.weight
    add_data_inputs["jobdir"] = args.jobdir
    add_data_inputs["debug"] = args.debug
    return add_data_inputs


def get_conf(yaml_file: str) -> omegaconf:
    logger.info(f"load {yaml_file}")
    conf = omegaconf.OmegaConf.load(yaml_file)
    return conf


def get_global_config(
    data_dir: str,
    model_name: str,
    test_config_name: str = "test_config",
    model_config_name: str = "model_config",
    elementembeddingflag_config_name: str = "text_element_embedding_flag_config",
    elementpredictionflag_config_name: str = "text_element_prediction_flag_config",
    canvasembeddingflag_config_name: str = "canvas_embedding_flag_config",
    elementembeddingatt_config_name: str = "text_element_embedding_attribute_config",
    elementpredictionatt_config_name: str = "text_element_prediction_attribute_config",
    canvasembeddingatt_config_name: str = "canvas_embedding_attribute_config",
) -> GlobalConfig:
    metainfo_conf = get_conf(f"{data_dir}/config/{model_name}/metainfo.yaml")
    data_conf = get_conf(f"{data_dir}/config/{model_name}/data_config.yaml")
    model_conf = get_conf(f"{data_dir}/config/{model_name}/{model_config_name}.yaml")
    train_conf = get_conf(f"{data_dir}/config/{model_name}/train_config.yaml")
    test_conf = get_conf(f"{data_dir}/config/{model_name}/{test_config_name}.yaml")
    text_element_emb_flag_conf = get_conf(
        f"{data_dir}/config/{model_name}/{elementembeddingflag_config_name}.yaml"
    )
    canvas_emb_flag_conf = get_conf(
        f"{data_dir}/config/{model_name}/{canvasembeddingflag_config_name}.yaml"
    )
    text_element_pred_flag_conf = get_conf(
        f"{data_dir}/config/{model_name}/{elementpredictionflag_config_name}.yaml"
    )
    text_element_emb_attribute_conf = get_conf(
        f"{data_dir}/config/{model_name}/{elementembeddingatt_config_name}.yaml"
    )
    canvas_emb_attribute_conf = get_conf(
        f"{data_dir}/config/{model_name}/{canvasembeddingatt_config_name}.yaml"
    )
    text_element_pred_attribute_conf = get_conf(
        f"{data_dir}/config/{model_name}/{elementpredictionatt_config_name}.yaml"
    )
    config = build_config(
        metainfo_conf,
        data_conf,
        model_conf,
        train_conf,
        test_conf,
        text_element_emb_flag_conf,
        canvas_emb_flag_conf,
        text_element_pred_flag_conf,
        text_element_emb_attribute_conf,
        canvas_emb_attribute_conf,
        text_element_pred_attribute_conf,
    )
    return config


def get_data_config(
    config: GlobalConfig,
) -> Tuple[BinsData, FontConfig, DataPreprocessConfig]:
    bin_data = get_bindata(config)
    font_config = get_font_config(config)
    data_preprocess_config = get_datapreprocess_config(config)
    return bin_data, font_config, data_preprocess_config


def get_sampling_config(
    config: GlobalConfig,
) -> SamplingConfig:
    sampling_param = config.test_config.sampling_param
    sampling_param_geometry = config.test_config.sampling_param_geometry
    sampling_param_semantic = config.test_config.sampling_param_semantic
    sampling_num = config.test_config.sampling_num
    return SamplingConfig(
        sampling_param, sampling_param_geometry, sampling_param_semantic, sampling_num
    )


def get_global_config_input(data_dir: str, args: Any) -> Dict:
    global_config_input = {}
    global_config_input["data_dir"] = data_dir
    global_config_input["model_name"] = args.configname
    global_config_input["test_config_name"] = args.testconfigname
    global_config_input["model_config_name"] = args.modelconfigname
    global_config_input[
        "elementembeddingflag_config_name"
    ] = args.elementembeddingflagconfigname
    global_config_input[
        "canvasembeddingflag_config_name"
    ] = args.canvasembeddingflagconfigname
    global_config_input[
        "elementpredictionflag_config_name"
    ] = args.elementpredictionflagconfigname

    return global_config_input


def get_model_config_input(config: GlobalConfig) -> Tuple[str, Dict]:
    model_kwargs = {}

    model_kwargs["prefix_list_element"] = get_target_prefix_list(
        config.text_element_embedding_attribute_config
    )
    model_kwargs["prefix_list_canvas"] = get_target_prefix_list(
        config.canvas_embedding_attribute_config
    )
    model_kwargs["prefix_list_target"] = get_target_prefix_list(
        config.text_element_prediction_attribute_config
    )
    model_kwargs[
        "embedding_config_element"
    ] = config.text_element_embedding_attribute_config
    model_kwargs["embedding_config_canvas"] = config.canvas_embedding_attribute_config
    model_kwargs[
        "prediction_config_element"
    ] = config.text_element_prediction_attribute_config
    model_kwargs["d_model"] = config.model_config.d_model
    model_kwargs["n_head"] = config.model_config.n_head
    model_kwargs["dropout"] = config.model_config.dropout
    model_kwargs["num_encoder_layers"] = config.model_config.num_encoder_layers
    model_kwargs["num_decoder_layers"] = config.model_config.num_decoder_layers
    model_kwargs["seq_length"] = config.model_config.seq_length
    model_kwargs["std_ratio"] = config.model_config.std_ratio
    model_kwargs["bypass"] = config.model_config.bypass

    model_name = config.meta_info.model_name
    return model_name, model_kwargs


def get_train_config_input(config: GlobalConfig, debug: bool) -> Dict:
    train_kwargs = {}
    if debug is True:
        train_kwargs["epochs"] = 1
        train_kwargs["batch_size"] = 2
    else:
        train_kwargs["epochs"] = config.train_config.epochs
        train_kwargs["batch_size"] = config.train_config.batch_size
    train_kwargs["save_epoch"] = config.train_config.save_epoch
    train_kwargs["num_worker"] = config.train_config.num_worker
    train_kwargs["learning_rate"] = config.train_config.learning_rate
    train_kwargs["show_interval"] = config.train_config.show_interval
    train_kwargs["optimizer_option"] = config.train_config.optimizer
    train_kwargs["weight_decay"] = config.train_config.weight_decay
    train_kwargs["debug"] = debug
    return train_kwargs


def get_prefix_lists(config: GlobalConfig) -> PrefixListObject:
    prefix_list_textelement = get_target_prefix_list(
        config.text_element_embedding_attribute_config
    )
    prefix_list_canvas = get_target_prefix_list(
        config.canvas_embedding_attribute_config
    )
    prefix_list_model_input = get_model_input_prefix_list(config)
    prefix_list_target = get_target_prefix_list(
        config.text_element_prediction_attribute_config
    )
    prefix_list_object = PrefixListObject(
        prefix_list_textelement,
        prefix_list_canvas,
        prefix_list_model_input,
        prefix_list_target,
    )
    return prefix_list_object
