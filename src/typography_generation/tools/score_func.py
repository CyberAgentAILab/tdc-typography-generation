import itertools
from typing import Dict, List, Tuple

import numpy as np
from _io import TextIOWrapper
from logzero import logger as log

from typography_generation.tools.color_func import deltaE_ciede2000, rgb2lab


class EvalDataInstance:
    def __init__(
        self,
        attribute_list: List,
    ) -> None:
        self.attribute_list = attribute_list
        self.target_list = ["pred_token", "gt_token", "pred", "gt"]
        self.reset()

    def reset(self) -> None:
        for att in self.attribute_list:
            for tar in self.target_list:
                registration_name = f"{att}_{tar}"
                setattr(self, registration_name, [])

    def rigister_att(
        self,
        text_num: int,
        prefix: str,
        target_pred_token: np.array,
        target_gt_token: np.array,
        target_pred: np.array,
        target_gt: np.array,
        start_index: int = 0,
    ) -> None:
        for i in range(start_index, text_num):
            registration_name = f"{prefix}_pred_token"
            getattr(self, registration_name).append(target_pred_token[i])
            registration_name = f"{prefix}_gt_token"
            getattr(self, registration_name).append(target_gt_token[i])
            registration_name = f"{prefix}_pred"
            getattr(self, registration_name).append(target_pred[i])
            registration_name = f"{prefix}_gt"
            getattr(self, registration_name).append(target_gt[i])


class EvalDataEntire:
    def __init__(
        self, attribute_list: List, save_dir: str, save_file_prefix: str = "score"
    ) -> None:
        self.attribute_list = attribute_list
        self.text_num = {}
        self.overlap_scores = {}
        self.data_index_list = []
        self.target_list = ["pred_token", "gt_token", "pred", "gt"]
        for att in self.attribute_list:
            for tar in self.target_list:
                registration_name = f"{att}_{tar}"
                setattr(self, registration_name, {})

        save_file = f"{save_dir}/{save_file_prefix}.txt"
        self.f = open(save_file, "w")

    def update_prediction_data(
        self,
        index: str,
        instance_obj: EvalDataInstance,
        prefix: str,
    ) -> None:
        getattr(self, f"{prefix}_pred_token")[index] = getattr(
            instance_obj, f"{prefix}_pred_token"
        )
        getattr(self, f"{prefix}_gt_token")[index] = getattr(
            instance_obj, f"{prefix}_gt_token"
        )
        getattr(self, f"{prefix}_pred")[index] = getattr(instance_obj, f"{prefix}_pred")
        getattr(self, f"{prefix}_gt")[index] = getattr(instance_obj, f"{prefix}_gt")

    def update_sampling_data(
        self,
        index: str,
        instance_obj: EvalDataInstance,
        prefix: str,
    ) -> None:
        primary_index, sub_index = index.split("_")
        if int(sub_index) > 0:
            getattr(self, prefix)[primary_index].append(getattr(instance_obj, prefix))
        else:
            getattr(self, prefix)[primary_index] = []
            getattr(self, prefix)[primary_index].append(getattr(instance_obj, prefix))

    def show_classification_score(
        self, att: str, topk: int = 10, show_topk: List = [0, 5]
    ) -> None:
        log.info(f"{att}")
        self.f.write(f"{att} \n")
        compute_score(
            getattr(self, f"{att}_pred"),
            getattr(self, f"{att}_gt"),
            topk,
            show_topk,
            self.f,
        )

    def show_abs_erros(
        self, prefix: str, blanktype: str = "", topk: int = 10, show_topk: List = [0, 5]
    ) -> None:
        log.info(f"{prefix}")
        self.f.write(f"{prefix} \n")
        compute_abs_error_score(
            getattr(self, f"{prefix}_pred"),
            getattr(self, f"{prefix}_gt"),
            self.f,
        )

    def show_font_color_scores(
        self, blanktype: str = "", topk: int = 10, show_topk: List = [0, 5]
    ) -> None:
        log.info(f"font_color")
        self.f.write(f"font_color \n")
        compute_color_score(
            getattr(self, f"text_font_color_pred"),
            getattr(self, f"text_font_color_gt"),
            self.f,
        )

    def show_structure_score(self, att: str) -> None:
        log.info(f"{att}")
        self.f.write(f"{att} \n")
        compute_bigram_score(
            getattr(self, f"text_num"),
            getattr(self, f"{att}_pred_token"),
            getattr(self, f"{att}_gt_token"),
            self.f,
        )

    def show_visual_similarity_scores(self) -> None:
        registration_name = "l2error"
        dict_average_score(registration_name, getattr(self, registration_name))
        registration_name = "psnr"
        dict_average_score(registration_name, getattr(self, registration_name))

    def show_time_score(self) -> None:
        registration_name = "time"
        dict_average_score(registration_name, getattr(self, registration_name))

    def show_diversity_scores(self, attribute_list: List) -> None:
        for att in attribute_list:
            log.info(f"{att}")
            self.f.write(f"{att} \n")
            compute_label_diversity_score(
                getattr(self, "data_index_list"),
                getattr(self, f"text_num"),
                getattr(self, f"{att}_pred_token"),
                self.f,
            )

    def show_alpha_overlap_score(self) -> None:
        compute_alpha_overlap(
            getattr(self, "overlap_scores"),
            self.f,
        )


def compute_abs_error_score(
    eval_list: dict,
    gt_list: dict,
    f: TextIOWrapper,
) -> None:
    cnt_img = 0
    l1_distance = 0.0
    r_l1_distance = 0.0
    for index in eval_list.keys():
        g = gt_list[index]
        if len(g) > 0:
            e = eval_list[index]
            d = 0.0
            rd = 0.0
            for i, (pred, gt) in enumerate(zip(e, g)):
                pred = pred[0]
                _d = abs(gt - pred)
                _rd = abs(gt - pred) / max(abs(float(gt)), 1e-5)
                d += _d
                rd += _rd
            l1_distance += d / len(g)
            r_l1_distance += rd / len(g)
            cnt_img += 1
    l1_distance /= cnt_img
    log.info(f"l1_distance {l1_distance}")
    f.write(f"l1_distance {l1_distance} \n")
    r_l1_distance /= cnt_img
    log.info(f"r_l1_distance {r_l1_distance}")
    f.write(f"r_l1_distance {r_l1_distance} \n")


def compute_color_score(
    font_color_eval_list: dict,
    font_color_gt_list: dict,
    f: TextIOWrapper,
) -> None:
    cnt_img = 0
    color_distance = 0.0
    for index in font_color_eval_list.keys():
        e = font_color_eval_list[index]
        g = font_color_gt_list[index]
        if len(g) > 0:
            d = 0.0
            for i, (pred, gt) in enumerate(zip(e, g)):
                pred = pred[0]
                lab_p = rgb2lab(np.array(pred).reshape(1, 1, 3).astype(np.float32))
                lab_g = rgb2lab(np.array(gt).reshape(1, 1, 3).astype(np.float32))
                _d = deltaE_ciede2000(lab_p, lab_g)
                d += _d[0][0]
            color_distance += d / len(g)
            cnt_img += 1
    color_distance /= cnt_img
    log.info(f"color_distance {color_distance}")
    f.write(f"color_distance {color_distance} \n")


def compute_score(
    eval_list: dict,
    gt_list: dict,
    topk: int,
    show_topk: List,
    f: TextIOWrapper,
) -> Tuple:
    cnt_elm = 0
    cnt_img = 0
    topk_acc_elm = {}
    topk_acc_img = {}
    for k in range(topk):
        topk_acc_elm[k] = 0.0
        topk_acc_img[k] = 0.0
    for index in eval_list.keys():
        e = eval_list[index]
        g = gt_list[index]
        topk_acc_tmp = {}
        for k in range(topk):
            topk_acc_tmp[k] = 0.0
        if len(g) > 0:
            cnt_gt = 0
            for i, gt in enumerate(g):
                flag = 0
                for k in range(min(topk, len(e[i]))):
                    if int(gt) == int(e[i][k]):
                        flag = 1.0
                    topk_acc_elm[k] += flag
                    topk_acc_tmp[k] += flag
                cnt_gt += 1
                cnt_elm += 1
            for k in range(topk):
                if cnt_gt > 0:
                    topk_acc_img[k] += topk_acc_tmp[k] / cnt_gt
            cnt_img += 1
    for k in range(topk):
        topk_acc_elm[k] /= cnt_elm
        topk_acc_img[k] /= cnt_img
        if k in show_topk:
            log.info(f"top{k} img_level_acc {topk_acc_img[k]}")
            f.write(f"top{k} img_level_acc {topk_acc_img[k]} \n")
    return topk_acc_elm, topk_acc_img


def dict_average_score(prefix: str, score_dict: dict) -> None:
    score_mean = 0
    cnt = 0
    for index in score_dict.keys():
        score_mean += score_dict[index]
        cnt += 1
    log.info("{} {}".format(prefix, score_mean / cnt))


def compute_unigram_label_score(
    text_num: int,
    text_target_mask: np.array,
    font_pred: np.array,
    font_gt: np.array,
    ignore_labels: List[int],
) -> float:
    cnt = 0
    correct_cnt = 0
    for i in range(text_num):
        fi = font_pred[i]
        if fi in ignore_labels or text_target_mask[i] == 0:
            continue
        else:
            cnt += 1
        for j in range(text_num):
            fj = int(font_gt[j])
            if fi == fj:
                correct_cnt += 1
                break
    if cnt > 0:
        score = float(correct_cnt) / cnt
    else:
        score = 0
    return score


def compute_bigram_label_score(
    pred: np.array,
    gt: np.array,
) -> float:
    text_num = len(gt)
    text_cmb = list(itertools.combinations(list(range(text_num)), 2))
    cnt = 0
    correct_cnt = 0
    for pi, pj in text_cmb:
        fpi = pred[pi][0]
        fpj = pred[pj][0]
        cnt += 1
        for gi, gj in text_cmb:
            fgi = int(gt[gi])
            fgj = int(gt[gj])
            if (fpi == fgi) and (fpj == fgj):
                correct_cnt += 1
                break
            elif (fpi == fgj) and (fpj == fgi):
                correct_cnt += 1
                break
    if cnt > 0:
        score = float(correct_cnt) / cnt
    else:
        score = 0
    return score


def get_binary_classification_scores(
    l11cnt: int, l00cnt: int, l10cnt: int, l01cnt: int
) -> Tuple:
    if l11cnt + l10cnt > 0:
        precision = float(l11cnt) / (l11cnt + l10cnt)
    else:
        precision = 0

    if l11cnt + l01cnt > 0:
        recall = float(l11cnt) / (l11cnt + l01cnt)
    else:
        recall = 0

    if l00cnt + l01cnt > 0:
        precision_inv = float(l00cnt) / (l00cnt + l01cnt)
    else:
        precision_inv = 0

    if l00cnt + l10cnt > 0:
        recall_inv = float(l00cnt) / (l00cnt + l10cnt)
    else:
        recall_inv = 0

    if precision + recall > 0:
        fvalue = 2 * precision * recall / (precision + recall)
    else:
        fvalue = 0

    if 2 * l11cnt + l01cnt + l10cnt == 0:
        _fvalue = np.nan
    else:
        _fvalue = 2 * l11cnt / (2 * l11cnt + l01cnt + l10cnt)

    if precision_inv + recall_inv > 0:
        fvalue_inv = 2 * precision_inv * recall_inv / (precision_inv + recall_inv)
    else:
        fvalue_inv = 0

    if 2 * l00cnt + l01cnt + l10cnt == 0:
        _fvalue_inv = np.nan
    else:
        _fvalue_inv = 2 * l00cnt / (2 * l00cnt + l01cnt + l10cnt)

    if l11cnt + l00cnt + l01cnt + l10cnt > 0:
        accuracy = float(l11cnt + l00cnt) / (l11cnt + l00cnt + l01cnt + l10cnt)
    else:
        accuracy = 0
    if l00cnt + l01cnt > 0:
        spcecificity = float(l00cnt) / (l00cnt + l01cnt)
    else:
        spcecificity = 0
    return (
        accuracy,
        spcecificity,
        precision,
        recall,
        fvalue,
        precision_inv,
        recall_inv,
        fvalue_inv,
        _fvalue,
        _fvalue_inv,
    )


def compute_bigram_structure_score(
    pred: np.array,
    gt: np.array,
) -> Tuple:
    text_num = len(gt)
    text_cmb = list(itertools.combinations(list(range(text_num)), 2))
    l11cnt = 0
    l00cnt = 0
    l10cnt = 0
    l01cnt = 0
    for pi, pj in text_cmb:
        fpi = pred[pi][0]
        fpj = pred[pj][0]
        fgi = int(gt[pi])
        fgj = int(gt[pj])
        if (fpi == fpj) and (fgi == fgj):
            l11cnt += 1
            # l00cnt += 1
        if (fpi != fpj) and (fgi != fgj):
            l00cnt += 1
            # l11cnt += 1
        if (fpi != fpj) and (fgi == fgj):
            l10cnt += 1
            # l01cnt += 1
        if (fpi == fpj) and (fgi != fgj):
            l01cnt += 1
            # l10cnt += 1
    scores = get_binary_classification_scores(l11cnt, l00cnt, l10cnt, l01cnt)
    return scores, (l11cnt, l00cnt, l10cnt, l01cnt)


def get_structure_type(
    gt: np.array,
) -> float:
    text_num = len(gt)
    text_cmb = list(itertools.combinations(list(range(text_num)), 2))
    consistency_num = 0
    contrast_num = 0
    for pi, pj in text_cmb:
        fgi = int(gt[pi])
        fgj = int(gt[pj])
        if fgi == fgj:
            consistency_num += 1
        else:
            contrast_num += 1
    if text_num <= 1:
        flag = 0  # uncount
    elif consistency_num == 0:
        flag = 1  # no consistency
    elif contrast_num == 0:
        flag = 2  # no contrast
    else:
        flag = 3  # others
    return flag


def compute_bigram_score(
    text_num_list: dict,
    pred_list: dict,
    gt_list: dict,
    f: TextIOWrapper,
) -> Tuple:
    cnt = 0
    structure_accuracy_mean = 0.0
    structure_precision_mean = 0.0
    structure_recall_mean = 0.0
    structure_fvalue_mean = 0.0
    structure_spcecificity_mean = 0.0
    structure_precision_inv_mean = 0.0
    structure_recall_inv_mean = 0.0
    structure_fvalue_inv_mean = 0.0
    label_score_mean = 0.0
    l11cnt_all = 0
    l00cnt_all = 0
    l10cnt_all = 0
    l01cnt_all = 0
    diff_case_scores = {}
    diff_case_counts = {}
    diff_case_scores[1] = 0.0  # no consistency
    diff_case_scores[2] = 0.0  # no contrast
    diff_case_scores[3] = 0.0  # others
    diff_case_counts[1] = 0
    diff_case_counts[2] = 0
    diff_case_counts[3] = 0
    structure_nanmean = []
    for index in pred_list.keys():
        text_num = text_num_list[index]
        if text_num == 0:
            continue
        pred = pred_list[index]
        gt = gt_list[index]
        flag = get_structure_type(gt)
        scores, counts = compute_bigram_structure_score(pred, gt)
        (
            structure_accuracy,
            structure_spcecificity,
            structure_precision,
            structure_recall,
            structure_fvalue,
            structure_precision_inv,
            structure_recall_inv,
            structure_fvalue_inv,
            _structure_fvalue,
            _structure_fvalue_inv,
        ) = scores
        l11cnt, l00cnt, l10cnt, l01cnt = counts
        l11cnt_all += l11cnt
        l00cnt_all += l00cnt
        l10cnt_all += l10cnt
        l01cnt_all += l01cnt
        label_score = compute_bigram_label_score(pred, gt)
        structure_accuracy_mean += structure_accuracy
        structure_spcecificity_mean += structure_spcecificity
        structure_precision_mean += structure_precision
        structure_recall_mean += structure_recall
        structure_fvalue_mean += structure_fvalue
        structure_precision_inv_mean += structure_precision_inv
        structure_recall_inv_mean += structure_recall_inv
        structure_fvalue_inv_mean += structure_fvalue_inv
        label_score_mean += label_score

        if flag == 0:
            pass
        elif flag == 1:  # no consistency
            diff_case_scores[1] += structure_fvalue_inv
            diff_case_counts[1] += 1
        elif flag == 2:  # no contrast
            diff_case_scores[2] += structure_fvalue
            diff_case_counts[2] += 1
        elif flag == 3:  # others
            diff_case_scores[3] += (structure_fvalue + structure_fvalue_inv) / 2.0
            diff_case_counts[3] += 1
        structure_nanmean.append(_structure_fvalue)
        structure_nanmean.append(_structure_fvalue_inv)

        cnt += 1
    structure_accuracy_mean /= cnt
    structure_spcecificity_mean /= cnt
    structure_precision_mean /= cnt
    structure_recall_mean /= cnt
    structure_fvalue_mean /= cnt
    structure_precision_inv_mean /= cnt
    structure_recall_inv_mean /= cnt
    structure_fvalue_inv_mean /= cnt
    label_score_mean /= cnt
    log.info("structure_accuracy {:.3f}".format(structure_accuracy_mean))
    f.write("structure_accuracy {:.3f} \n".format(structure_accuracy_mean))

    # log.info("label_score {:.3f}".format(label_score_mean))
    # f.write("label_score {:.3f} \n".format(label_score_mean))
    log.info("structure nanmean {:.3f}".format(np.nanmean(structure_nanmean)))
    f.write("structure nanmean {:.3f} \n".format(np.nanmean(structure_nanmean)))
    for i in range(1, 4):
        if diff_case_counts[i] > 0:
            log.info(
                "structure_case_score{} count:{} {:.3f}".format(
                    i, diff_case_counts[i], diff_case_scores[i] / diff_case_counts[i]
                )
            )
            f.write(
                "structure_case_score{} count:{} {:.3f} \n".format(
                    i, diff_case_counts[i], diff_case_scores[i] / diff_case_counts[i]
                )
            )
        else:
            log.info("structure_case_score{} count:{} -".format(i, diff_case_counts[i]))
            f.write(
                "structure_case_score{} count:{} - \n".format(i, diff_case_counts[i])
            )

    scores = get_binary_classification_scores(
        l11cnt_all, l00cnt_all, l10cnt_all, l01cnt_all
    )
    (
        structure_accuracy,
        structure_spcecificity,
        structure_precision,
        structure_recall,
        structure_fvalue,
        structure_precision_inv,
        structure_recall_inv,
        structure_fvalue_inv,
        _structure_fvalue,
        _structure_fvalue_inv,
    ) = scores
    return structure_accuracy_mean, label_score_mean


def compute_label_diversity_score(
    data_index_list: List,
    text_num_list: Dict,
    pred_list: Dict,
    f: TextIOWrapper,
    sampling_num: int = 10,
) -> None:
    def compute_label_diversity(pred_labels: List, text_num: int) -> float:
        unique_num_rate_avg = 0.0
        for k in range(text_num):
            labels = []
            for j in range(len(pred_labels)):
                l = int(pred_labels[j][k][0])
                labels.append(l)
            unique_num_rate = len(set(labels)) / float(len(pred_labels))
            unique_num_rate_avg += unique_num_rate
        unique_num_rate_avg /= text_num
        return unique_num_rate_avg

    label_diversity_avg = 0.0
    cnt = 0
    for index in data_index_list:
        text_num = text_num_list[f"{index}_0"]
        pred_lists = []
        for n in range(sampling_num):
            preds = pred_list[f"{index}_{n}"]
            pred_lists.append(preds)
        if text_num > 0:
            label_diversity = compute_label_diversity(pred_lists, text_num)
            label_diversity_avg += label_diversity
            cnt += 1
    label_diversity_avg /= cnt
    log.info("diversity score {:.1f}".format(label_diversity_avg * 100))
    f.write("diversity score {:.1f}\n".format(label_diversity_avg * 100))


def compute_alpha_overlap(
    overlap_scores: Dict,
    f: TextIOWrapper,
) -> None:
    overlap_score_all = 0
    cnt_all = 0
    data_index_list = list(overlap_scores.keys())
    for index in data_index_list:
        overlap_score = overlap_scores[f"{index}"]
        if overlap_score is not None:
            overlap_score_all += overlap_score
            cnt_all += 1
    if cnt_all > 0:
        overlap_score_all = overlap_score_all / cnt_all

        log.info("alpha overlap score {:.2f}".format(overlap_score_all))
        f.write("alpha overlap score {:.2f}\n".format(overlap_score_all))


def _compute_alpha_overlap(
    alpha_map_list: List,
) -> None:
    overlap_score = 0
    cnt = 0
    for i in range(len(alpha_map_list)):
        alpha_i = alpha_map_list[i]
        for j in range(len(alpha_map_list)):
            if i == j:
                continue
            else:
                alpha_j = alpha_map_list[j]
                overlap = np.sum(alpha_i * alpha_j)
                if np.sum(alpha_i) > 0:
                    recall = overlap / np.sum(alpha_i)
                    overlap_score += recall
                    cnt += 1
    if cnt > 0:
        overlap_score = overlap_score / cnt
        return overlap_score
    else:
        return None
