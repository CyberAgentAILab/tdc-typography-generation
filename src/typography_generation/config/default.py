from typing import Any, List, Union

from logzero import logger
from omegaconf import OmegaConf

from typography_generation.config.attribute_config import (
    CanvasContextEmbeddingAttributeConfig,
    TextElementContextEmbeddingAttributeConfig,
    TextElementContextPredictionAttributeConfig)
from typography_generation.config.base_config_object import (
    CanvasEmbeddingFlag, DataConfig, GlobalConfig, MetaInfo, ModelConfig,
    TestConfig, TextElementEmbeddingFlag, TextElementPredictionTargetFlag,
    TrainConfig)
from typography_generation.io.data_object import (BinsData,
                                                  DataPreprocessConfig,
                                                  FontConfig)


def get_bindata(config: GlobalConfig) -> BinsData:
    bin_data = BinsData(
        config.data_config.color_bin,
        config.data_config.small_spatio_bin,
        config.data_config.large_spatio_bin,
    )
    return bin_data


def get_font_config(config: GlobalConfig) -> FontConfig:
    font_config = FontConfig(
        config.data_config.font_num,
        config.data_config.font_emb_type,
        config.data_config.font_emb_weight,
        config.data_config.font_emb_dim,
        config.data_config.font_emb_name,
    )
    return font_config


def get_datapreprocess_config(config: GlobalConfig) -> DataPreprocessConfig:
    datapreprocess_config = DataPreprocessConfig(
        config.data_config.order_type,
        config.data_config.seq_length,
    )
    return datapreprocess_config


def get_target_prefix_list(tar_cls: Any) -> List:
    all_text_prefix_list = dir(tar_cls)
    target_prefix_list = []
    for prefix in all_text_prefix_list:
        elm = getattr(tar_cls, prefix)
        if elm.flag is True:
            target_prefix_list.append(prefix)
    return target_prefix_list


def get_model_input_prefix_list(conf: GlobalConfig) -> List:
    input_prefix_list = []
    input_prefix_list += get_target_prefix_list(
        conf.text_element_embedding_attribute_config
    )
    input_prefix_list += get_target_prefix_list(conf.canvas_embedding_attribute_config)
    input_prefix_list += get_target_prefix_list(
        conf.text_element_prediction_attribute_config
    )
    if "canvas_text_num" in input_prefix_list:
        pass
    else:
        input_prefix_list.append("canvas_text_num")
    return list(set(input_prefix_list))


def plusone_num_embeddings(config: GlobalConfig) -> None:
    prefix_lists = dir(config.text_element_embedding_attribute_config)
    for prefix in prefix_lists:
        elm = getattr(config.text_element_embedding_attribute_config, f"{prefix}")
        if elm.emb_layer == "nn.Embedding":
            elm.emb_layer_kwargs["num_embeddings"] = (
                int(elm.emb_layer_kwargs["num_embeddings"]) + 1
            )


def show_class_attributes(_class: Any):
    class_dict = _class.__dict__["_content"]
    logger.info(f"{class_dict}")


def build_config(
    metainfo_conf: MetaInfo,
    data_conf: DataConfig,
    model_conf: ModelConfig,
    train_conf: TrainConfig,
    test_conf: TestConfig,
    text_element_emb_flag_conf: TextElementEmbeddingFlag,
    canvas_emb_flag_conf: CanvasEmbeddingFlag,
    text_element_pred_flag_conf: TextElementPredictionTargetFlag,
    text_element_emb_attribute_conf: TextElementContextEmbeddingAttributeConfig,
    canvas_emb_attribute_conf: CanvasContextEmbeddingAttributeConfig,
    text_element_pred_attribute_conf: TextElementContextPredictionAttributeConfig,
) -> Union[Any, GlobalConfig]:
    conf = OmegaConf.structured(GlobalConfig)
    show_class_attributes(text_element_emb_flag_conf)
    show_class_attributes(canvas_emb_flag_conf)
    show_class_attributes(text_element_pred_flag_conf)
    conf = OmegaConf.merge(
        conf,
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
    plusone_num_embeddings(conf)
    return conf
