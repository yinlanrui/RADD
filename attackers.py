import torch
import torch.nn.functional as F
import numpy as np
from attacker_threshold import ThresholdAttacker
from base_model import BaseModel
from torch.utils.data import DataLoader, TensorDataset
from utils import seed_worker



class MiaAttack:
    def __init__(self, victim_model, victim_pruned_model, victim_train_loader, victim_test_loader,
                 shadow_model_list, shadow_pruned_model_list, shadow_train_loader_list, shadow_test_loader_list,
                 num_cls=10, batch_size=128,  device="cuda",
                 lr=0.001, optimizer="sgd", epochs=100, weight_decay=5e-4,
                 # lr=0.001, optimizer="adam", epochs=100, weight_decay=5e-4,
                 attack_original=False
                 ):
        self.victim_model = victim_model
        self.victim_pruned_model = victim_pruned_model
        self.victim_train_loader = victim_train_loader
        self.victim_test_loader = victim_test_loader
        self.shadow_model_list = shadow_model_list
        self.shadow_pruned_model_list = shadow_pruned_model_list
        self.shadow_train_loader_list = shadow_train_loader_list
        self.shadow_test_loader_list = shadow_test_loader_list
        self.num_cls = num_cls
        self.device = device
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.attack_original = attack_original
        self._prepare()

    def _prepare(self):
        attack_in_predicts_list, attack_in_targets_list, attack_in_sens_list = [], [], []
        attack_out_predicts_list, attack_out_targets_list, attack_out_sens_list = [], [], []
        for shadow_model, shadow_pruned_model, shadow_train_loader, shadow_test_loader in zip(
                self.shadow_model_list, self.shadow_pruned_model_list, self.shadow_train_loader_list,
                self.shadow_test_loader_list):

            if self.attack_original:
                attack_in_predicts, attack_in_targets, attack_in_sens = \
                    shadow_model.predict_target_sensitivity(shadow_train_loader)
                attack_out_predicts, attack_out_targets, attack_out_sens = \
                    shadow_model.predict_target_sensitivity(shadow_test_loader)
            else:
                attack_in_predicts, attack_in_targets, attack_in_sens = \
                    shadow_pruned_model.predict_target_sensitivity(shadow_train_loader)
                attack_out_predicts, attack_out_targets, attack_out_sens = \
                    shadow_pruned_model.predict_target_sensitivity(shadow_test_loader)

            attack_in_predicts_list.append(attack_in_predicts)
            attack_in_targets_list.append(attack_in_targets)
            attack_in_sens_list.append(attack_in_sens)
            attack_out_predicts_list.append(attack_out_predicts)
            attack_out_targets_list.append(attack_out_targets)
            attack_out_sens_list.append(attack_out_sens)

        self.attack_in_predicts = torch.cat(attack_in_predicts_list, dim=0)
        self.attack_in_targets = torch.cat(attack_in_targets_list, dim=0)
        self.attack_in_sens = torch.cat(attack_in_sens_list, dim=0)
        self.attack_out_predicts = torch.cat(attack_out_predicts_list, dim=0)
        self.attack_out_targets = torch.cat(attack_out_targets_list, dim=0)
        self.attack_out_sens = torch.cat(attack_out_sens_list, dim=0)

        if self.attack_original:
            self.victim_in_predicts, self.victim_in_targets, self.victim_in_sens = \
                self.victim_model.predict_target_sensitivity(self.victim_train_loader)
            self.victim_out_predicts, self.victim_out_targets, self.victim_out_sens = \
                self.victim_model.predict_target_sensitivity(self.victim_test_loader)
        else:
            self.victim_in_predicts, self.victim_in_targets, self.victim_in_sens = \
                self.victim_pruned_model.predict_target_sensitivity(self.victim_train_loader)
            self.victim_out_predicts, self.victim_out_targets, self.victim_out_sens = \
                self.victim_pruned_model.predict_target_sensitivity(self.victim_test_loader)

    def nn_attack(self, mia_type="nn_sens_cls", model_name="mia_fc"):
        attack_predicts = torch.cat([self.attack_in_predicts, self.attack_out_predicts], dim=0)
        attack_sens = torch.cat([self.attack_in_sens, self.attack_out_sens], dim=0)
        attack_targets = torch.cat([self.attack_in_targets, self.attack_out_targets], dim=0)
        attack_targets = F.one_hot(attack_targets, num_classes=self.num_cls).float()
        attack_labels = torch.cat([torch.ones(self.attack_in_targets.size(0)),
                                   torch.zeros(self.attack_out_targets.size(0))], dim=0).long()

        victim_predicts = torch.cat([self.victim_in_predicts, self.victim_out_predicts], dim=0)
        victim_sens = torch.cat([self.victim_in_sens, self.victim_out_sens], dim=0)
        victim_targets = torch.cat([self.victim_in_targets, self.victim_out_targets], dim=0)
        victim_targets = F.one_hot(victim_targets, num_classes=self.num_cls).float()
        victim_labels = torch.cat([torch.ones(self.victim_in_targets.size(0)),
                                   torch.zeros(self.victim_out_targets.size(0))], dim=0).long()

        if mia_type == "nn_cls":
            new_attack_data = torch.cat([attack_predicts, attack_targets], dim=1)
            new_victim_data = torch.cat([victim_predicts, victim_targets], dim=1)
        elif mia_type == "nn_top3":
            new_attack_data, _ = torch.topk(attack_predicts, k=3, dim=-1)
            new_victim_data, _ = torch.topk(victim_predicts, k=3, dim=-1)
        elif mia_type == "nn_sens_cls":
            new_attack_data = torch.cat([attack_predicts, attack_sens, attack_targets], dim=1)
            new_victim_data = torch.cat([victim_predicts, victim_sens, victim_targets], dim=1)
        else:
            new_attack_data = attack_predicts
            new_victim_data = victim_predicts

        attack_train_dataset = TensorDataset(new_attack_data, attack_labels)
        attack_train_dataloader = DataLoader(
            attack_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
            worker_init_fn=seed_worker)
        attack_test_dataset = TensorDataset(new_victim_data, victim_labels)
        attack_test_dataloader = DataLoader(
            attack_test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
            worker_init_fn=seed_worker)

        attack_model = BaseModel(
            model_name, device=self.device, num_cls=new_victim_data.size(1), optimizer=self.optimizer, lr=self.lr,
            weight_decay=self.weight_decay, epochs=self.epochs)

        for epoch in range(self.epochs):
            train_acc, train_loss = attack_model.train(attack_train_dataloader)
            test_acc, test_loss = attack_model.test(attack_test_dataloader)
        return test_acc

    def threshold_attack(self):
        victim_in_predicts = self.victim_in_predicts.numpy()
        victim_out_predicts = self.victim_out_predicts.numpy()

        attack_in_predicts = self.attack_in_predicts.numpy()
        attack_out_predicts = self.attack_out_predicts.numpy()
        attacker = ThresholdAttacker((attack_in_predicts, self.attack_in_targets.numpy()),
                                 (attack_out_predicts, self.attack_out_targets.numpy()),
                                 (victim_in_predicts, self.victim_in_targets.numpy()),
                                 (victim_out_predicts, self.victim_out_targets.numpy()),
                                 self.num_cls)
        confidence, entropy, modified_entropy = attacker._mem_inf_benchmarks()
        top1_conf, _, _ = attacker._mem_inf_benchmarks_non_cls()
        return confidence * 100., entropy * 100., modified_entropy * 100., \
               top1_conf * 100.

    def lira_attack(self, class_conditional=True, min_class_samples=20, eps=1e-6):
        """Fast offline LiRA-style attack on final model outputs.

        This implementation reuses the existing adaptive shadow-pruned models.
        It fits Gaussian distributions to shadow member/non-member true-label
        confidence logits, then scores victim samples by a likelihood ratio.
        """
        shadow_in_features = self._lira_features(self.attack_in_predicts, self.attack_in_targets, eps)
        shadow_out_features = self._lira_features(self.attack_out_predicts, self.attack_out_targets, eps)
        victim_in_features = self._lira_features(self.victim_in_predicts, self.victim_in_targets, eps)
        victim_out_features = self._lira_features(self.victim_out_predicts, self.victim_out_targets, eps)

        shadow_in_labels = self.attack_in_targets.numpy()
        shadow_out_labels = self.attack_out_targets.numpy()
        victim_in_labels = self.victim_in_targets.numpy()
        victim_out_labels = self.victim_out_targets.numpy()

        global_params = self._fit_lira_params(shadow_in_features, shadow_out_features, eps)
        class_params = {}
        if class_conditional:
            for cls in range(self.num_cls):
                in_mask = shadow_in_labels == cls
                out_mask = shadow_out_labels == cls
                if in_mask.sum() >= min_class_samples and out_mask.sum() >= min_class_samples:
                    class_params[cls] = self._fit_lira_params(
                        shadow_in_features[in_mask], shadow_out_features[out_mask], eps)

        victim_in_scores = self._score_lira_samples(
            victim_in_features, victim_in_labels, class_params, global_params)
        victim_out_scores = self._score_lira_samples(
            victim_out_features, victim_out_labels, class_params, global_params)
        shadow_in_scores = self._score_lira_samples(
            shadow_in_features, shadow_in_labels, class_params, global_params)
        shadow_out_scores = self._score_lira_samples(
            shadow_out_features, shadow_out_labels, class_params, global_params)

        threshold = self._best_lira_threshold(shadow_in_scores, shadow_out_scores)
        attack_acc = 0.5 * (
            np.mean(victim_in_scores >= threshold) + np.mean(victim_out_scores < threshold))
        auc = self._binary_auc(
            np.concatenate([victim_in_scores, victim_out_scores]),
            np.concatenate([np.ones_like(victim_in_scores), np.zeros_like(victim_out_scores)]))

        return {
            "acc": attack_acc * 100.0,
            "auc": auc * 100.0,
            "tpr_at_1_fpr": self._tpr_at_fpr(victim_in_scores, victim_out_scores, 0.01) * 100.0,
            "tpr_at_0_5_fpr": self._tpr_at_fpr(victim_in_scores, victim_out_scores, 0.005) * 100.0,
        }

    @staticmethod
    def _lira_features(predicts, targets, eps):
        predicts = predicts.numpy()
        targets = targets.numpy()
        conf = predicts[np.arange(targets.shape[0]), targets]
        conf = np.clip(conf, eps, 1.0 - eps)
        return np.log(conf) - np.log1p(-conf)

    @staticmethod
    def _fit_lira_params(in_features, out_features, eps):
        return {
            "in_mu": float(np.mean(in_features)),
            "in_std": float(max(np.std(in_features), eps)),
            "out_mu": float(np.mean(out_features)),
            "out_std": float(max(np.std(out_features), eps)),
        }

    @staticmethod
    def _normal_logpdf(values, mu, std):
        return -0.5 * np.square((values - mu) / std) - np.log(std) - 0.5 * np.log(2.0 * np.pi)

    @classmethod
    def _score_lira_samples(cls, features, labels, class_params, global_params):
        scores = np.empty_like(features, dtype=np.float64)
        for idx, (feature, label) in enumerate(zip(features, labels)):
            params = class_params.get(int(label), global_params)
            log_in = cls._normal_logpdf(feature, params["in_mu"], params["in_std"])
            log_out = cls._normal_logpdf(feature, params["out_mu"], params["out_std"])
            scores[idx] = log_in - log_out
        return scores

    @staticmethod
    def _best_lira_threshold(in_scores, out_scores):
        values = np.unique(np.concatenate([in_scores, out_scores]))
        if values.size == 0:
            return 0.0
        best_threshold = values[0]
        best_acc = -1.0
        for threshold in values:
            acc = 0.5 * (np.mean(in_scores >= threshold) + np.mean(out_scores < threshold))
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        return best_threshold

    @staticmethod
    def _tpr_at_fpr(in_scores, out_scores, fpr):
        if len(in_scores) == 0 or len(out_scores) == 0:
            return 0.0
        threshold = np.quantile(out_scores, 1.0 - fpr)
        return float(np.mean(in_scores >= threshold))

    @staticmethod
    def _binary_auc(scores, labels):
        order = np.argsort(scores)
        sorted_scores = scores[order]
        sorted_ranks = np.empty(len(scores), dtype=np.float64)
        start = 0
        while start < len(scores):
            end = start + 1
            while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
                end += 1
            sorted_ranks[start:end] = 0.5 * (start + 1 + end)
            start = end
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[order] = sorted_ranks
        pos = labels == 1
        n_pos = np.sum(pos)
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.0
        return float((np.sum(ranks[pos]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
