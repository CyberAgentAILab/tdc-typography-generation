from typing import Dict, List

from logzero import logger

############################################################
#  Prediction Recoder
############################################################


class PredictionRecoder:
    def __init__(self, prefix_list_target: List, topk: int = 5):
        super(PredictionRecoder, self).__init__()
        self.prefix_list_target = prefix_list_target
        self.topk = topk
        self.show_topk = list(range(self.topk))
        self.register(prefix_list_target)
        self.epoch = 0

    def register(self, prefix_list_target: List) -> None:
        self.all_target_list = []
        self.cl_target_list = []
        self.loss_target_list = []
        for prefix in prefix_list_target:
            self.all_target_list.append(f"{prefix}_cl")
            self.all_target_list.append(f"{prefix}_loss")
            self.cl_target_list.append(f"{prefix}_cl")
            self.loss_target_list.append(f"{prefix}_loss")
        self.regist_cl_recorder_set(self.cl_target_list)
        self.regist_recorder_set(self.loss_target_list)

    def regist_recorder_set(self, target_list: List) -> None:
        for tar in target_list:
            self.regist_record_target(tar)

    def regist_cl_recorder_set(self, target_list: List) -> None:
        for tar in target_list:
            self.regist_cl_recorder(tar)

    def regist_cl_recorder(self, registration_name: str) -> None:
        self.regist_record_target(f"{registration_name}_pred")
        self.regist_record_target(f"{registration_name}_gt")

    def regist_record_target(self, registration_name: str) -> None:
        setattr(self, registration_name, [])
        setattr(self, f"{registration_name}_history", [])
        for k in range(len(self.show_topk)):
            setattr(self, f"{registration_name}_history_{self.show_topk[k]}", [])

    def __call__(self, recoder_items: Dict) -> None:
        for name, (pred, gt, loss) in recoder_items.items():
            clname = f"{name}_cl"
            lossname = f"{name}_loss"
            if clname in self.cl_target_list:
                for p, g in zip(pred, gt):
                    getattr(self, f"{clname}_pred").append(p)
                    getattr(self, f"{clname}_gt").append(g)
            if lossname in self.loss_target_list:
                getattr(self, lossname).append(loss)

    def reset(self) -> None:
        for name in self.all_target_list:
            if name in self.cl_target_list:
                setattr(self, f"{name}_pred", [])
                setattr(self, f"{name}_gt", [])
            if name in self.loss_target_list:
                setattr(self, name, [])

    def store_score(self) -> None:
        for name in self.all_target_list:
            if name in self.cl_target_list:
                for k in range(len(self.show_topk)):
                    getattr(self, f"{name}_pred_history_{self.show_topk[k]}").append(
                        getattr(self, f"{name}{self.show_topk[k]}_acc")
                    )
            if name in (self.loss_target_list):
                getattr(self, f"{name}_history").append(getattr(self, f"{name}_mean"))

    def compute_score(self) -> None:
        for name in self.all_target_list:
            if name in self.cl_target_list:
                topkacc = {}
                for k in range(self.topk):
                    topkacc[k] = 0
                for p, g in zip(
                    getattr(self, f"{name}_pred"), getattr(self, f"{name}_gt")
                ):
                    flag = 0
                    for k in range(min(self.topk, len(p))):
                        if p[k] == g:
                            flag = 1
                        topkacc[k] += flag
                for k in range(min(self.topk, len(topkacc))):
                    setattr(
                        self,
                        f"{name}{k}_acc",
                        topkacc[k] / max(len(getattr(self, f"{name}_pred")), 1),
                    )
            if name in (self.loss_target_list):
                mean_loss = 0
                for l in getattr(self, f"{name}"):
                    mean_loss += l
                setattr(
                    self,
                    f"{name}_mean",
                    mean_loss / max(len(getattr(self, f"{name}")), 1),
                )
        score_dict = self.show_scores()
        return score_dict

    def show_scores(self) -> None:
        score_dict = {}
        for name in self.all_target_list:
            if name in self.cl_target_list:
                for k in range(len(self.show_topk)):
                    logger.info(
                        f"{name}{self.show_topk[k]}_acc:{getattr(self, f'{name}{self.show_topk[k]}_acc')}"
                    )
                score_dict[name] = getattr(self, f"{name}{self.show_topk[0]}_acc")
            if name in self.loss_target_list:
                logger.info(f"{name}_mean:{getattr(self, f'{name}_mean')}")
                score_dict[name] = getattr(self, f"{name}_mean")
        return score_dict

    def show_history_scores(self) -> None:
        for i in range(self.epoch):
            logger.info(f"epoch {i}")
            for name in self.all_target_list:
                if name in self.cl_target_list:
                    for k in range(len(self.show_topk)):
                        logger.info(
                            f"{name}{self.show_topk[k]}acc:{getattr(self, f'{name}_pred_history_{self.show_topk[k]}')[i]}"
                        )
                if name in self.loss_target_list:
                    logger.info(f"{name}_mean:{getattr(self, f'{name}_history')[i]}")

    def step_epoch(self) -> None:
        self.compute_score()
        self.store_score()
        self.update_epoch()

    def update_epoch(self) -> None:
        self.epoch += 1
