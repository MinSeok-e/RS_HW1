import math

from recbole.model.general_recommender.bpr import BPR
from recbole.model.loss import EmbLoss


class RegularizedBPR(BPR):
    """BPR with sampled-embedding squared L2 regularization.

    loss = mean(BPR loss)
           + reg_lambda / (2 * batch_size)
             * sum(||user||^2 + ||positive||^2 + ||negative||^2)

    Repeated IDs contribute once per occurrence in the batch.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.reg_lambda = float(config["reg_lambda"])
        if not math.isfinite(self.reg_lambda) or self.reg_lambda < 0:
            raise ValueError("reg_lambda must be finite and nonnegative.")

        if float(config["weight_decay"]) != 0:
            raise ValueError("Set weight_decay=0 to avoid double regularization.")

        self.embedding_reg_loss = EmbLoss(norm=2)

    def calculate_loss(self, interaction):
        ranking_loss = super().calculate_loss(interaction)

        user_e = self.get_user_embedding(interaction[self.USER_ID])
        pos_e = self.get_item_embedding(interaction[self.ITEM_ID])
        neg_e = self.get_item_embedding(interaction[self.NEG_ITEM_ID])

        reg_loss = self.embedding_reg_loss(
            user_e,
            pos_e,
            neg_e,
            require_pow=True,
        ).squeeze()

        return ranking_loss + self.reg_lambda * reg_loss
