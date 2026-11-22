from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N, STEPS = 40, 20
FIG_DIR = Path(__file__).resolve().parent
FIG_DIR.mkdir(parents=True, exist_ok=True)

left = lambda x: np.roll(x, 1)    # x_{i-1}
right = lambda x: np.roll(x, -1)  # x_{i+1}
true_step = right                  # backward motion: x_i(t+1) = x_{i+1}(t)

# ---------- (a)(i) least squares for a, b on the isolated slowdown ----------
x0 = np.zeros(N); x0[20] = 1.0
y = true_step(x0)                                  # 1 at sensor 19
F = np.stack([x0, left(x0) + right(x0)], axis=1)   # features for a, b at every sensor
(a, b), *_ = np.linalg.lstsq(F, y, rcond=None)
sse = np.sum((F @ [a, b] - y) ** 2)
print(f"(a)(i)  a = {a:.4f}, b = {b:.4f}, SSE = {sse:.4f}")
pred = F @ [a, b]
print("        predictions at 19, 20, 21:", pred[19:22], " targets:", y[19:22])

# ---------- (a)(ii) roll out the symmetric layer vs the true motion ----------
i = np.arange(N)
bump = np.exp(-((i - 20) ** 2) / 8)
pred_seq, true_seq = [bump], [bump]
for _ in range(STEPS):
    p = pred_seq[-1]
    pred_seq.append(a * p + b * (left(p) + right(p)))
    true_seq.append(true_step(true_seq[-1]))
pred_seq, true_seq = np.array(pred_seq), np.array(true_seq)

def centre(seq):  # circular centre of mass, for reporting
    ang = 2 * np.pi * i / N
    return (np.angle((seq * np.exp(1j * ang)).sum(1)) % (2 * np.pi)) * N / (2 * np.pi)

for name, seq in [("symmetric layer", pred_seq), ("true motion", true_seq)]:
    print(f"(a)(ii) {name:16s}: centre t=0 {centre(seq)[0]:5.2f} -> t=20 {centre(seq)[-1]:5.2f}, "
          f"peak {seq[0].max():.3f} -> {seq[-1].max():.3f}, total {seq[0].sum():.3f} -> {seq[-1].sum():.3f}")

vmax = max(pred_seq.max(), true_seq.max())
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, constrained_layout=True)
for ax, seq, title in [(axes[0], pred_seq, rf"Symmetric layer ($a={a:.2f}$, $b={b:.2f}$)"),
                       (axes[1], true_seq, r"True motion $x_i(t+1)=x_{i+1}(t)$")]:
    im = ax.imshow(seq, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=vmax,
                   extent=[-0.5, N - 0.5, -0.5, STEPS + 0.5])
    ax.set_title(title); ax.set_xlabel("sensor index $i$")
    ax.set_yticks(range(0, STEPS + 1, 5)); ax.set_xticks(range(0, N, 5))
axes[0].set_ylabel("time step $t$")
fig.colorbar(im, ax=axes, label="slowdown $x_i(t)$", shrink=0.9)
fig.savefig(FIG_DIR / "q2_heatmaps.pdf"); fig.savefig(FIG_DIR / "q2_heatmaps.png", dpi=150)

# ---------- (b) direction-aware weights / 1D convolution ----------
rng = np.random.default_rng(0)
xr = rng.standard_normal(N)
a_b, wm, wp = 0.0, 0.0, 1.0
assert np.allclose(a_b * xr + wm * left(xr) + wp * right(xr), true_step(xr))
kernel = np.array([wm, a_b, wp])                      # multiplies [x_{i-1}, x_i, x_{i+1}]
padded = np.concatenate([xr[-1:], xr, xr[:1]])        # circular padding of 1
conv = np.array([kernel @ padded[k:k + 3] for k in range(N)])
assert np.allclose(conv, true_step(xr))
print(f"(b)     a=0, w-=0, w+=1 exact on random input; kernel {kernel} with circular padding OK")

# ---------- (c) equal neighbour weights + history ----------
ac, bc, cc = 0.0, 1.0, -1.0
layer_c = lambda xt, xprev: ac * xt + bc * (left(xt) + right(xt)) + cc * xprev
for direction, step in [("backward", right), ("forward", left)]:
    x_prev = rng.standard_normal(N)
    x_t = step(x_prev)
    ok = np.allclose(layer_c(x_t, x_prev), step(x_t))
    # also roll out 20 steps from the bump
    xp, xt = bump, step(bump)
    for _ in range(STEPS):
        xp, xt = xt, layer_c(xt, xp)
    ok_roll = np.allclose(xt, np.roll(bump, -(STEPS + 1) if direction == "backward" else STEPS + 1))
    print(f"(c)     {direction:8s} motion: a=0, b=1, c=-1 one-step exact={ok}, 20-step rollout exact={ok_roll}")
