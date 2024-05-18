import hydra
import numpy as np
from scipy.stats import rankdata, norm, mannwhitneyu
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class MannWhitneyUTest:
    def __init__(self, group1: np.array, group2: np.array) -> None:
        self.group1 = group1
        self.group2 = group2

    def run(self):
        n1 = len(self.group1)
        n2 = len(self.group2)

        # calculate rank of group1 and group2
        rank1, rank2 = self._calculate_ranks(n1, n2)
        u = self._calculate_u(n1, n2, rank1, rank2)

        # calculate z-score
        mean_u = n1 * n2 / 2
        std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        u_corrected = u - 0.5 if u < mean_u else u + 0.5
        z_score = self._calculate_z_score(u_corrected, mean_u, std_u)

        # calculate p-value
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        return u_corrected, p_value

    def __call__(self):
        self.run()

    def _calculate_ranks(self, n1: np.array, n2: np.array) -> Tuple[int, int]:
        merged_group = np.concatenate([self.group1, self.group2])
        rank = rankdata(merged_group)
        rank1 = np.sum(rank[:n1], axis=None)
        rank2 = np.sum(rank[n1:], axis=None)
        return rank1, rank2

    def _calculate_u(self, n1: np.array, n2: np.array, rank1: int, rank2: int) -> float:
        u1 = n1 * n2 + (n1 * (n1 + 1) / 2) - rank1
        u2 = n1 * n2 + (n2 * (n2 + 1) / 2) - rank2
        return min(u1, u2)

    def _calculate_z_score(self, u: float, mean_u: float, std_u: float) -> float:
        return (u - mean_u) / std_u


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    group1 = np.random.randint(10, size=cfg.G1.n_sample)
    group2 = np.random.randint(10, size=cfg.G2.n_sample)

    mwu_user_defined = MannWhitneyUTest(group1, group2)
    u, p_value = mwu_user_defined.run()
    logger.info(f"the result of user-defined Mann-Whitney U test")
    logger.info(f"u: {u}, p_value: {p_value}")

    u, p_value = mannwhitneyu(group1, group2)
    logger.info(f"the result of scipy Mann-Whitney U test")
    logger.info(f"u: {u}, p_value: {p_value}")


if __name__ == "__main__":
    main()
