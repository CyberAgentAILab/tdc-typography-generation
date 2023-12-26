from dataclasses import dataclass
from typography_generation.config.attribute_config import (
    CanvasContextEmbeddingAttributeConfig,
    TextElementContextEmbeddingAttributeConfig,
    TextElementContextPredictionAttributeConfig,
)


@dataclass
class MetaInfo:
    model_name: str = "bart"
    dataset: str = "crello"
    data_dir: str = "crello"


@dataclass
class DataConfig:
    font_num: int = 288
    large_spatio_bin: int = 64
    small_spatio_bin: int = 16
    color_bin: int = 64
    max_text_char_count: int = 50
    max_text_line_count: int = 50
    font_emb_type: str = "label"
    font_emb_weight: float = 1.0
    font_emb_dim: int = 40
    font_emb_name: str = "mfc"
    order_type: str = "raster_scan_order"
    seq_length: int = 50


@dataclass
class ModelConfig:
    d_model: int = 256
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    n_head: int = 8
    dropout: float = 0.1
    bert_dim: int = 768
    clip_dim: int = 512
    mlp_dim: int = 3584
    mfc_dim: int = 3584
    bypass: bool = True
    seq_length: int = 50
    std_ratio: float = 1.0


@dataclass
class TrainConfig:
    epochs: int = 31
    save_epoch: int = 5
    batch_size: int = 32
    num_worker: int = 2
    train_only: bool = False
    show_interval: int = 100
    learning_rate: float = 0.0002
    weight_decay: float = 0.01
    optimizer: str = "adam"


@dataclass
class TestConfig:
    prediction_mode: int = 6
    autoregressive_prediction: bool = False
    autoregressive_order: bool = False
    sampling_mode: str = "topp"
    sampling_param: float = 0
    sampling_param_geometry: float = 0
    sampling_param_semantic: float = 0
    sampling_num: int = 10
    structure_preserving: bool = False


@dataclass
class TextElementEmbeddingFlag:
    text_font: bool = True
    text_font_size: bool = True
    text_font_size_raw: bool = False
    text_font_color: bool = True
    text_line_count: bool = True
    text_char_count: bool = True
    text_height: bool = False
    text_width: bool = False
    text_top: bool = False
    text_left: bool = False
    text_center_y: bool = True
    text_center_x: bool = True
    text_align_type: bool = False
    text_angle: bool = False
    text_letter_spacing: bool = False
    text_line_height_scale: bool = False
    text_line_height_size: bool = False
    text_capitalize: bool = False
    text_emb: bool = True
    text_local_img_emb: bool = True


@dataclass
class CanvasEmbeddingFlag:
    canvas_bg_img: bool = False
    canvas_bg_img_emb: bool = True
    canvas_aspect_ratio: bool = True
    canvas_text_num: bool = True
    canvas_group: bool = False
    canvas_format: bool = False
    canvas_category: bool = False
    canvas_width: bool = False
    canvas_height: bool = False


@dataclass
class TextElementPredictionTargetFlag:
    text_font: bool = True
    text_font_emb: bool = False
    text_font_size: bool = True
    text_font_size_raw: bool = False
    text_font_color: bool = True
    text_angle: bool = True
    text_letter_spacing: bool = True
    text_line_height_scale: bool = True
    text_line_height_size: bool = False
    text_capitalize: bool = True
    text_align_type: bool = True
    text_center_y: bool = False
    text_center_x: bool = False
    text_height: bool = False
    text_width: bool = False


@dataclass
class GlobalConfig:
    meta_info: MetaInfo = MetaInfo()
    model_config: ModelConfig = ModelConfig()
    data_config: DataConfig = DataConfig()
    train_config: TrainConfig = TrainConfig()
    test_config: TestConfig = TestConfig()
    text_element_embedding_flag: TextElementEmbeddingFlag = TextElementEmbeddingFlag()
    canvas_embedding_flag: CanvasEmbeddingFlag = CanvasEmbeddingFlag()
    text_element_prediction_flag: TextElementPredictionTargetFlag = (
        TextElementPredictionTargetFlag()
    )
    canvas_embedding_attribute_config: CanvasContextEmbeddingAttributeConfig = (
        CanvasContextEmbeddingAttributeConfig()
    )
    text_element_embedding_attribute_config: TextElementContextEmbeddingAttributeConfig = (
        TextElementContextEmbeddingAttributeConfig()
    )
    text_element_prediction_attribute_config: TextElementContextPredictionAttributeConfig = (
        TextElementContextPredictionAttributeConfig()
    )
