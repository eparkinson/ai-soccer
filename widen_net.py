"""
Widen a PPOBrain's networks without changing what they do (Net2WiderNet).

Every hidden unit is duplicated and its outgoing weights are split between the two
copies, so the widened network computes exactly the same function. The split is
uneven (--split, e.g. 0.55 / 0.45) so the copies receive different gradients and can
learn different things from the first update on.

    poetry run python widen_net.py runs/league2/PPO-big/latest.npz start-PPO-bigger.npz --width 512
"""

import argparse

import numpy as np

from aisoccer.brains.PPOBrain import PPOBrain


def widen(params, width, split):
    """params: [W1, b1, W2, b2, ..., Wout, bout] of a tanh MLP; every hidden layer -> width."""
    params = [np.array(p, dtype=float) for p in params]
    for layer in range(len(params) // 2 - 1):
        w_in, b, w_out = params[2 * layer], params[2 * layer + 1], params[2 * layer + 2]
        n = len(b)
        copies = np.arange(width) % n  # new unit j copies old unit j mod n
        assert width % n == 0, "width must be a multiple of the old width"
        per_unit = width // n
        share = np.where(np.arange(width) < n, split, (1 - split) / (per_unit - 1)) if per_unit > 1 else 1.0
        params[2 * layer] = w_in[:, copies]
        params[2 * layer + 1] = b[copies]
        params[2 * layer + 2] = w_out[copies, :] * np.asarray(share)[:, None]
    return params


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="PPOBrain .npz, or a train_ppo latest.npz")
    parser.add_argument("out")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--split", type=float, default=0.55)
    args = parser.parse_args()

    data = np.load(args.source, allow_pickle=True)
    weights = data["state"].item() if "state" in data.files else PPOBrain.load_weights(args.source)
    wide = {
        "policy": widen(weights["policy"], args.width, args.split),
        "value": widen(weights["value"], args.width, args.split),
        "log_std": np.array(weights["log_std"], dtype=float),
    }
    PPOBrain.save_weights(args.out, wide)

    # Check: the widened networks give the same outputs.
    from aisoccer.ppo import MLP

    rng = np.random.default_rng(0)
    x = rng.normal(size=(1000, np.asarray(weights["policy"][0]).shape[0]))
    for net in ("policy", "value"):
        old, new = MLP([1, 1]), MLP([1, 1])
        old.params = [np.asarray(p, dtype=float) for p in weights[net]]
        new.params = wide[net]
        print(f"{net}: {[p.shape for p in new.params[::2]]}, max output difference "
              f"{np.max(np.abs(old(x) - new(x))):.2e}")


if __name__ == "__main__":
    main()
