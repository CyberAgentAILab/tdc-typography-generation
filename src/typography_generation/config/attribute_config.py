from dataclasses import dataclass
from typing import Any, Union


@dataclass
class EmbeddingConfig:
    flag: bool = True
    inp_space: int = 256
    input_prefix: str = "prefix"
    emb_layer: Union[str, None] = "nn.Embedding"
    emb_layer_kwargs: Any = None
    specific_build: Union[str, None] = None
    specific_func: Union[str, None] = None


@dataclass
class PredictionConfig:
    flag: bool = True
    out_dim: int = 256
    layer: Union[str, None] = "nn.Linear"
    loss_type: str = "cre"
    loss_weight: float = 1.0
    ignore_label: int = -1
    decode_format: str = "cl"
    att_type: str = "semantic"


@dataclass
class TextElementContextPredictionAttributeConfig:
    text_font: PredictionConfig = PredictionConfig(
        "${text_element_prediction_flag.text_font}",
        "${data_config.font_num}",
        "nn.Linear",
        "cre",
        1.0,
        -1,
        "cl",
        "semantic",
    )
    text_font_emb: PredictionConfig = PredictionConfig(
        "${text_element_prediction_flag.text_font_emb}",
        "${data_config.font_emb_dim}",
        "nn.Linear",
        "mfc_gan",
        1.0,
        -10000,
        "emb",
        "semantic",
    )
    text_font_size: PredictionConfig = PredictionConfig(
        "${text_element_prediction_flag.text_font_size}",
        "${data_config.small_spatio_bin}",
        "nn.Linear",
        "cre",
        1.0,
        -1,
        "cl",
        "geometry",
    )
    text_font_size_raw: PredictionConfig = PredictionConfig(
        "${text_element_prediction_flag.text_font_size_raw}",
        "1",
        "nn.Linear",
        "l1",
        1.0,
        -1,
        "scalar",
        "geometry",
    )
    text_font_color: PredictionConfig = PredictionConfig(
        "${text_element_prediction_flag.text_font_color}",
        "${data_config.color_bin}",
        "nn.Linear",
        "cre",
        1.0,
        -1,
        "cl",
        "semantic",
    )
    text_angle: PredictionConfig = PredictionConfig(
        "${text_element_prediction_flag.text_angle}",
        "${data_config.small_spatio_bin}",
        "nn.Linear",
        "cre",
        1.0,
        -1,
        "cl",
        "geometry",
    )
    text_letter_spacing: PredictionConfig = PredictionConfig(
        "${text_element_prediction_flag.text_letter_spacing}",
        "${data_config.small_spatio_bin}",
        "nn.Linear",
        "cre",
        1.0,
        -1,
        "cl",
        "geometry",
    )
    text_line_height_scale: PredictionConfig = PredictionConfig(
        "${text_element_prediction_flag.text_line_height_scale}",
        "${data_config.small_spatio_bin}",
        "nn.Linear",
        "cre",
        1.0,
        -1,
        "cl",
        "geometry",
    )
    text_line_height_size: PredictionConfig = PredictionConfig(
        "${text_element_prediction_flag.text_line_height_size}",
        "${data_config.small_spatio_bin}",
        "nn.Linear",
        "cre",
        1.0,
        -1,
        "cl",
        "geometry",
    )
    text_capitalize: PredictionConfig = PredictionConfig(
        "${text_element_prediction_flag.text_capitalize}",
        2,
        "nn.Linear",
        "cre",
        1.0,
        -1,
        "cl",
        "semantic",
    )
    text_align_type: PredictionConfig = PredictionConfig(
        "${text_element_prediction_flag.text_align_type}",
        3,
        "nn.Linear",
        "cre",
        1.0,
        -1,
        "cl",
        "semantic",
    )
    text_center_y: PredictionConfig = PredictionConfig(
        "${text_element_prediction_flag.text_center_y}",
        "${data_config.large_spatio_bin}",
        "nn.Linear",
        "cre",
        1.0,
        -1,
        "cl",
        "geometry",
    )
    text_center_x: PredictionConfig = PredictionConfig(
        "${text_element_prediction_flag.text_center_x}",
        "${data_config.large_spatio_bin}",
        "nn.Linear",
        "cre",
        1.0,
        -1,
        "cl",
        "geometry",
    )
    text_height: PredictionConfig = PredictionConfig(
        "${text_element_prediction_flag.text_height}",
        "${data_config.small_spatio_bin}",
        "nn.Linear",
        "cre",
        1.0,
        -1,
        "cl",
        "geometry",
    )
    text_width: PredictionConfig = PredictionConfig(
        "${text_element_prediction_flag.text_width}",
        "${data_config.large_spatio_bin}",
        "nn.Linear",
        "cre",
        1.0,
        -1,
        "cl",
        "geometry",
    )


@dataclass
class TextElementContextEmbeddingAttributeConfig:
    text_font: EmbeddingConfig = EmbeddingConfig(
        "${text_element_embedding_flag.text_font}",
        "${data_config.font_num}",
        "text_font",
        "nn.Embedding",
        {
            "num_embeddings": "${data_config.font_num}",
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    text_font_size: EmbeddingConfig = EmbeddingConfig(
        "${text_element_embedding_flag.text_font_size}",
        "${data_config.small_spatio_bin}",
        "text_font_size",
        "nn.Embedding",
        {
            "num_embeddings": "${data_config.small_spatio_bin}",
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    text_font_size_raw: EmbeddingConfig = EmbeddingConfig(
        "${text_element_embedding_flag.text_font_size_raw}",
        "${data_config.small_spatio_bin}",
        "text_font_size",
        "nn.Embedding",
        {
            "num_embeddings": "${data_config.small_spatio_bin}",
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    text_font_color: EmbeddingConfig = EmbeddingConfig(
        "${text_element_embedding_flag.text_font_color}",
        "${data_config.color_bin}",
        "text_font_color",
        "nn.Embedding",
        {
            "num_embeddings": "${data_config.color_bin}",
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    text_line_count: EmbeddingConfig = EmbeddingConfig(
        "${text_element_embedding_flag.text_line_count}",
        "${data_config.max_text_line_count}",
        "text_line_count",
        "nn.Embedding",
        {
            "num_embeddings": "${data_config.max_text_line_count}",
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    text_char_count: EmbeddingConfig = EmbeddingConfig(
        "${text_element_embedding_flag.text_char_count}",
        "${data_config.max_text_char_count}",
        "text_char_count",
        "nn.Embedding",
        {
            "num_embeddings": "${data_config.max_text_char_count}",
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    text_height: EmbeddingConfig = EmbeddingConfig(
        "${text_element_embedding_flag.text_height}",
        "${data_config.large_spatio_bin}",
        "text_height",
        "nn.Embedding",
        {
            "num_embeddings": "${data_config.large_spatio_bin}",
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    text_width: EmbeddingConfig = EmbeddingConfig(
        "${text_element_embedding_flag.text_width}",
        "${data_config.large_spatio_bin}",
        "text_width",
        "nn.Embedding",
        {
            "num_embeddings": "${data_config.large_spatio_bin}",
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    text_top: EmbeddingConfig = EmbeddingConfig(
        "${text_element_embedding_flag.text_top}",
        "${data_config.large_spatio_bin}",
        "text_top",
        "nn.Embedding",
        {
            "num_embeddings": "${data_config.large_spatio_bin}",
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    text_left: EmbeddingConfig = EmbeddingConfig(
        "${text_element_embedding_flag.text_left}",
        "${data_config.large_spatio_bin}",
        "text_left",
        "nn.Embedding",
        {
            "num_embeddings": "${data_config.large_spatio_bin}",
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    text_center_y: EmbeddingConfig = EmbeddingConfig(
        "${text_element_embedding_flag.text_center_y}",
        "${data_config.large_spatio_bin}",
        "text_center_y",
        "nn.Embedding",
        {
            "num_embeddings": "${data_config.large_spatio_bin}",
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    text_center_x: EmbeddingConfig = EmbeddingConfig(
        "${text_element_embedding_flag.text_center_x}",
        "${data_config.large_spatio_bin}",
        "text_center_x",
        "nn.Embedding",
        {
            "num_embeddings": "${data_config.large_spatio_bin}",
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    text_align_type: EmbeddingConfig = EmbeddingConfig(
        "${text_element_embedding_flag.text_align_type}",
        "3",
        "text_align_type",
        "nn.Embedding",
        {
            "num_embeddings": "3",
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    text_angle: EmbeddingConfig = EmbeddingConfig(
        "${text_element_embedding_flag.text_angle}",
        "${data_config.small_spatio_bin}",
        "text_angle",
        "nn.Embedding",
        {
            "num_embeddings": "${data_config.small_spatio_bin}",
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    text_letter_spacing: EmbeddingConfig = EmbeddingConfig(
        "${text_element_embedding_flag.text_letter_spacing}",
        "${data_config.small_spatio_bin}",
        "text_letter_spacing",
        "nn.Embedding",
        {
            "num_embeddings": "${data_config.small_spatio_bin}",
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    text_line_height_scale: EmbeddingConfig = EmbeddingConfig(
        "${text_element_embedding_flag.text_line_height_scale}",
        "${data_config.small_spatio_bin}",
        "text_line_height_scale",
        "nn.Embedding",
        {
            "num_embeddings": "${data_config.small_spatio_bin}",
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    text_line_height_size: EmbeddingConfig = EmbeddingConfig(
        "${text_element_embedding_flag.text_line_height_size}",
        "${data_config.small_spatio_bin}",
        "text_line_height_size",
        "nn.Embedding",
        {
            "num_embeddings": "${data_config.small_spatio_bin}",
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )

    text_capitalize: EmbeddingConfig = EmbeddingConfig(
        "${text_element_embedding_flag.text_capitalize}",
        "2",
        "text_capitalize",
        "nn.Embedding",
        {
            "num_embeddings": "2",
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    text_emb: EmbeddingConfig = EmbeddingConfig(
        "${text_element_embedding_flag.text_emb}",
        "${model_config.clip_dim}",
        "text_emb",
        "nn.Linear",
        {
            "in_features": "${model_config.clip_dim}",
            "out_features": "${model_config.d_model}",
        },
        None,
        None,
    )
    text_local_img_emb: EmbeddingConfig = EmbeddingConfig(
        "${text_element_embedding_flag.text_local_img_emb}",
        "${model_config.clip_dim}",
        "text_local_img_emb",
        "nn.Linear",
        {
            "in_features": "${model_config.clip_dim}",
            "out_features": "${model_config.d_model}",
        },
        None,
        None,
    )


@dataclass
class CanvasContextEmbeddingAttributeConfig:
    canvas_bg_img: EmbeddingConfig = EmbeddingConfig(
        "${canvas_embedding_flag.canvas_bg_img}",
        2048,
        "canvas_bg_img",
        None,
        None,
        "build_resnet_feat_extractor",
        "get_feat",
    )
    canvas_aspect_ratio: EmbeddingConfig = EmbeddingConfig(
        "${canvas_embedding_flag.canvas_aspect_ratio}",
        "${data_config.small_spatio_bin}",
        "canvas_aspect_ratio",
        "nn.Embedding",
        {
            "num_embeddings": "${data_config.small_spatio_bin}",
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    canvas_text_num: EmbeddingConfig = EmbeddingConfig(
        "${canvas_embedding_flag.canvas_text_num}",
        50,
        "canvas_text_num",
        "nn.Embedding",
        {
            "num_embeddings": 50,
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    canvas_group: EmbeddingConfig = EmbeddingConfig(
        "${canvas_embedding_flag.canvas_group}",
        6,
        "canvas_group",
        "nn.Embedding",
        {
            "num_embeddings": 6,
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    canvas_format: EmbeddingConfig = EmbeddingConfig(
        "${canvas_embedding_flag.canvas_format}",
        67,
        "canvas_format",
        "nn.Embedding",
        {
            "num_embeddings": 67,
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    canvas_category: EmbeddingConfig = EmbeddingConfig(
        "${canvas_embedding_flag.canvas_category}",
        23,
        "canvas_category",
        "nn.Embedding",
        {
            "num_embeddings": 23,
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    canvas_height: EmbeddingConfig = EmbeddingConfig(
        "${canvas_embedding_flag.canvas_height}",
        46,
        "canvas_height",
        "nn.Embedding",
        {
            "num_embeddings": 46,
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    canvas_width: EmbeddingConfig = EmbeddingConfig(
        "${canvas_embedding_flag.canvas_width}",
        41,
        "canvas_width",
        "nn.Embedding",
        {
            "num_embeddings": 41,
            "embedding_dim": "${model_config.d_model}",
        },
        None,
        None,
    )
    canvas_bg_img_emb: EmbeddingConfig = EmbeddingConfig(
        "${canvas_embedding_flag.canvas_bg_img_emb}",
        "${model_config.clip_dim}",
        "canvas_bg_img_emb",
        "nn.Linear",
        {
            "in_features": "${model_config.clip_dim}",
            "out_features": "${model_config.d_model}",
        },
        None,
        "canvas_bg_img_emb_layer",
    )
