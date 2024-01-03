import argparse
import logging
import os
from typing import Any, Dict, Tuple

import logzero
import torch
from logzero import logger
from typography_generation.config.config_args_util import (
    args2add_data_inputs,
    get_global_config,
    get_global_config_input,
    get_model_config_input,
    get_prefix_lists,
    get_sampling_config,
    get_train_config_input,
)
from typography_generation.config.default import (
    get_datapreprocess_config,
    get_font_config,
)
from typography_generation.io.build_dataset import (
    build_test_dataset,
    build_train_dataset,
)
from typography_generation.model.model import create_model

from typography_generation.preprocess.map_features import map_features

from typography_generation.tools.evaluator import Evaluator

from typography_generation.tools.sampler import Sampler
from typography_generation.tools.structure_preserved_sampler import (
    StructurePreservedSampler,
)
from typography_generation.tools.train import Trainer


def get_save_dir(job_dir: str) -> str:
    save_dir = os.path.join(job_dir, "logs")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def make_logfile(job_dir: str, debug: bool = False) -> None:
    if debug is True:
        logzero.loglevel(logging.DEBUG)
    else:
        logzero.loglevel(logging.INFO)

    os.makedirs(job_dir, exist_ok=True)
    file_name = f"{job_dir}/log.log"
    logzero.logfile(file_name)


def train(args: Any) -> None:
    logger.info("training")
    data_dir = args.datadir
    global_config_input = get_global_config_input(data_dir, args)
    config = get_global_config(**global_config_input)
    gpu = args.gpu
    model_name, model_kwargs = get_model_config_input(config)

    logger.info("model creation")
    model = create_model(
        model_name,
        **model_kwargs,
    )

    logger.info(f"log file location {args.jobdir}/log.log")
    make_logfile(args.jobdir, args.debug)
    save_dir = get_save_dir(args.jobdir)
    logger.info(f"save_dir {save_dir}")
    prefix_list_object = get_prefix_lists(config)
    font_config = get_font_config(config)
    prediction_config_element = config.text_element_prediction_attribute_config

    train_kwargs = get_train_config_input(config, args.debug)
    logger.info(f"build trainer")
    dataset, dataset_val = build_train_dataset(
        data_dir,
        prefix_list_object,
        font_config,
        use_extended_dataset=args.use_extended_dataset,
        debug=args.debug,
    )

    model_trainer = Trainer(
        model,
        gpu,
        save_dir,
        dataset,
        dataset_val,
        prefix_list_object,
        prediction_config_element,
        **train_kwargs,
    )
    logger.info("start training")
    model_trainer.train_model()


def train_eval(args: Any) -> None:
    logger.info("training")
    data_dir = args.datadir
    global_config_input = get_global_config_input(data_dir, args)
    config = get_global_config(**global_config_input)
    gpu = args.gpu
    model_name, model_kwargs = get_model_config_input(config)

    logger.info("model creation")
    model = create_model(
        model_name,
        **model_kwargs,
    )

    logger.info(f"log file location {args.jobdir}/log.log")
    make_logfile(args.jobdir, args.debug)
    save_dir = get_save_dir(args.jobdir)
    logger.info(f"save_dir {save_dir}")
    prefix_list_object = get_prefix_lists(config)
    font_config = get_font_config(config)
    prediction_config_element = config.text_element_prediction_attribute_config

    train_kwargs = get_train_config_input(config, args.debug)
    logger.info(f"build trainer")

    dataset, dataset_val = build_train_dataset(
        data_dir,
        prefix_list_object,
        font_config,
        use_extended_dataset=args.use_extended_dataset,
        debug=args.debug,
    )

    model_trainer = Trainer(
        model,
        gpu,
        save_dir,
        dataset,
        dataset_val,
        prefix_list_object,
        prediction_config_element,
        **train_kwargs,
    )
    logger.info("start training")
    model_trainer.train_model()

    dataset = build_test_dataset(
        data_dir,
        prefix_list_object,
        font_config,
        use_extended_dataset=args.use_extended_dataset,
        debug=args.debug,
    )

    evaluator = Evaluator(
        model,
        gpu,
        save_dir,
        dataset,
        prefix_list_object,
        debug=args.debug,
    )
    logger.info("start evaluation")
    evaluator.eval_model()


# def _train_font_embedding(args: Any) -> None:
#     train_font_embedding(args.datadir, args.jobdir, args.gpu)


def loadweight(weight_file: Any, gpu: bool, model: Any) -> Any:
    if weight_file == "":
        pass
    else:
        if gpu is False:
            state_dict = torch.load(weight_file, map_location=torch.device("cpu"))
        else:
            state_dict = torch.load(weight_file)
        model.load_state_dict(state_dict)
    return model


def sample(args: Any) -> None:
    data_dir = args.datadir
    global_config_input = get_global_config_input(data_dir, args)
    config = get_global_config(**global_config_input)
    gpu = args.gpu
    model_name, model_kwargs = get_model_config_input(config)

    logger.info("model creation")
    model = create_model(
        model_name,
        **model_kwargs,
    )
    weight = args.weight
    model = loadweight(weight, gpu, model)

    logger.info(f"log file location {args.jobdir}/log.log")
    make_logfile(args.jobdir, args.debug)
    save_dir = get_save_dir(args.jobdir)
    logger.info(f"save_dir {save_dir}")
    prefix_list_object = get_prefix_lists(config)
    font_config = get_font_config(config)
    sampling_config = get_sampling_config(config)

    dataset = build_test_dataset(
        data_dir,
        prefix_list_object,
        font_config,
        use_extended_dataset=args.use_extended_dataset,
        debug=args.debug,
    )

    sampler = Sampler(
        model,
        gpu,
        save_dir,
        dataset,
        prefix_list_object,
        sampling_config,
        debug=args.debug,
    )
    logger.info("start sampling")
    sampler.sample()


def structure_preserved_sample(args: Any) -> None:
    data_dir = args.datadir
    global_config_input = get_global_config_input(data_dir, args)
    config = get_global_config(**global_config_input)
    gpu = args.gpu
    model_name, model_kwargs = get_model_config_input(config)

    logger.info("model creation")
    model = create_model(
        model_name,
        **model_kwargs,
    )
    weight = args.weight
    model = loadweight(weight, gpu, model)

    logger.info(f"log file location {args.jobdir}/log.log")
    make_logfile(args.jobdir, args.debug)
    save_dir = get_save_dir(args.jobdir)
    logger.info(f"save_dir {save_dir}")
    prefix_list_object = get_prefix_lists(config)
    font_config = get_font_config(config)
    sampling_config = get_sampling_config(config)

    dataset = build_test_dataset(
        data_dir,
        prefix_list_object,
        font_config,
        use_extended_dataset=args.use_extended_dataset,
        debug=args.debug,
    )

    sampler = StructurePreservedSampler(
        model,
        gpu,
        save_dir,
        dataset,
        prefix_list_object,
        sampling_config,
        debug=args.debug,
    )
    logger.info("start sampling")
    sampler.sample()


def evaluation_pattern(args: Any, prefix: str, evaluation_class: Any) -> None:
    logger.info(f"{prefix}")
    data_dir = args.datadir
    global_config_input = get_global_config_input(data_dir, args)
    config = get_global_config(**global_config_input)
    gpu = args.gpu
    model_name, model_kwargs = get_model_config_input(config)

    logger.info("model creation")
    model = create_model(
        model_name,
        **model_kwargs,
    )
    weight = args.weight
    model = loadweight(weight, gpu, model)

    logger.info(f"log file location {args.jobdir}/log.log")
    make_logfile(args.jobdir, args.debug)
    save_dir = get_save_dir(args.jobdir)
    logger.info(f"save_dir {save_dir}")
    prefix_list_object = get_prefix_lists(config)
    font_config = get_font_config(config)

    dataset = build_test_dataset(
        data_dir,
        prefix_list_object,
        font_config,
        use_extended_dataset=args.use_extended_dataset,
        debug=args.debug,
    )
    evaluator = evaluation_class(
        model,
        gpu,
        save_dir,
        dataset,
        prefix_list_object,
        dataset_division="test",
        debug=args.debug,
    )
    logger.info("start evaluation")
    evaluator.eval_model()


def evaluation(args: Any) -> None:
    evaluation_pattern(args, "evaluation", Evaluator)


def _map_features(args: Any) -> None:
    logger.info(f"map_features")
    map_features(args.datadir)


COMMANDS = {
    "train": train,
    "train_evaluation": train_eval,
    "sample": sample,
    "structure_preserved_sample": structure_preserved_sample,
    "evaluation": evaluation,
    "map_features": _map_features,
}


if __name__ == "__main__":
    logger.info("start")
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="job option")
    parser.add_argument(
        "--configname",
        type=str,
        default="bart",
        help="config option",
    )
    parser.add_argument(
        "--testconfigname",
        type=str,
        default="test_config",
        help="test config option",
    )
    parser.add_argument(
        "--modelconfigname",
        type=str,
        default="model_config",
        help="test config option",
    )
    parser.add_argument(
        "--canvasembeddingflagconfigname",
        type=str,
        default="canvas_embedding_flag_config",
        help="canvas embedding flag config option",
    )
    parser.add_argument(
        "--elementembeddingflagconfigname",
        type=str,
        default="text_element_embedding_flag_config",
        help="element embedding flag config option",
    )
    parser.add_argument(
        "--elementpredictionflagconfigname",
        type=str,
        default="text_element_prediction_flag_config",
        help="element prediction flag config option",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="data",
        help="data location",
    )
    parser.add_argument(
        "--jobdir",
        type=str,
        default=".",
        help="results location",
    )
    parser.add_argument(
        "--job-dir",
        type=str,
        default=".",
        help="dummy",
    )
    parser.add_argument(
        "--weight",
        type=str,
        default="",
        help="weight file location",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="gpu option",
    )
    parser.add_argument(
        "--use_extended_dataset",
        action="store_true",
        help="dataset option",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug option",
    )
    args = parser.parse_args()
    module = COMMANDS[args.command]
    module(args)
