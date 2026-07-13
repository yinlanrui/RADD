from nni.algorithms.compression.pytorch.pruning import (
    L1FilterPruner,
    L2FilterPruner,
    LevelPruner,
    LotteryTicketPruner,
)


class IterativeStructuredFilterPruner:
    """Compatibility wrapper for old NNI structured filter pruners.

    Older NNI versions do not let LotteryTicketPruner select an L1/L2 structured
    pruning algorithm. This wrapper keeps the prune_modi.py call pattern
    unchanged by exposing compress() and export_model(), while gradually
    increasing the target structured sparsity over multiple pruning iterations.
    """

    def __init__(self, model, sparsity=0.5, prune_iter=5, method="l1"):
        self.model = model
        self.sparsity = sparsity
        self.prune_iter = max(1, int(prune_iter))
        self.method = method
        self.cur_iter = 0
        self.pruner = None

    def _build_pruner(self, target_sparsity):
        config_list = [{
            'sparsity': target_sparsity,
            'op_types': ['Conv2d']
        }]
        if self.method == "l1":
            return L1FilterPruner(self.model, config_list)
        if self.method == "l2":
            return L2FilterPruner(self.model, config_list)
        raise ValueError

    def compress(self):
        self.pruner = self._build_pruner(self.sparsity)
        self.model = self.pruner.compress()
        return self.model

    def get_prune_iterations(self):
        return range(self.prune_iter)

    def prune_iteration_start(self):
        self.cur_iter += 1

    def update_epoch(self, epoch):
        return None

    def export_model(self, model_path, mask_path):
        if self.pruner is None:
            raise RuntimeError("compress() must be called before export_model().")
        return self.pruner.export_model(model_path=model_path, mask_path=mask_path)


class OneShotLevelPruner:
    """Compatibility wrapper so LevelPruner can be used by prune_modi.py."""

    def __init__(self, model, sparsity=0.5):
        self.model = model
        self.sparsity = sparsity
        self.pruner = None

    def compress(self):
        config_list = [{
            'sparsity': self.sparsity,
            'op_types': ["default"]
        }]
        self.pruner = LevelPruner(self.model, config_list)
        self.model = self.pruner.compress()
        return self.model

    def get_prune_iterations(self):
        return range(1)

    def prune_iteration_start(self):
        return None

    def update_epoch(self, epoch):
        return None

    def export_model(self, model_path, mask_path):
        if self.pruner is None:
            raise RuntimeError("compress() must be called before export_model().")
        return self.pruner.export_model(model_path=model_path, mask_path=mask_path)


def get_pruner(pruner_name, model, sparsity=0.5, prune_iter = 5):
    if pruner_name == "l1unstructure":
        return OneShotLevelPruner(model, sparsity=sparsity)
    elif pruner_name == "iter_pruning":
        config_list = [{
            'prune_iterations': prune_iter - 1,
            'sparsity': sparsity,
            'op_types': ['default']
        }]
        return LotteryTicketPruner(model, config_list, reset_weights=False)
    elif pruner_name == "iter_prunetxt":
        config_list = [{
            'prune_iterations': prune_iter - 1,
            'sparsity': sparsity,
            'op_names': ['fc1', 'fc2']
        }]
        return LotteryTicketPruner(model, config_list, reset_weights=False)
    elif pruner_name == "iter_l1_structured":
        return IterativeStructuredFilterPruner(
            model, sparsity=sparsity, prune_iter=prune_iter, method="l1"
        )
    elif pruner_name == "iter_l2_structured":
        return IterativeStructuredFilterPruner(
            model, sparsity=sparsity, prune_iter=prune_iter, method="l2"
        )
    else:
        raise ValueError
