# %%
# ============================================================
# 2-state HMM（含 O1–H–O2 角度）分段筆記本
# ------------------------------------------------------------
# 使用步驟：
#   1. 先按上方「📂 上傳檔案」選擇 ARC 軌跡檔（.arc）
#   2. 由上而下逐段執行（Shift+Enter），或按「全部執行」
# 各段內容：
#   [1] 參數設定  [2] 讀取 ARC  [3] 計算每個 H 的 r1、r2、角度
#   [4] 7:3 切分訓練/測試集  [5] HMM 函式  [6] 訓練
#   [7] 查看模型參數  [8] 解碼與作圖  [9] 轉移矩陣熱圖
#   [10] 質子移轉事件判定（L→S→L）  [11] 測試集共享期盒型圖
#   [12] 各事件鄰近氧原子距離演化
#   ── 第二層 2-state HMM（移轉 / 未移轉）──
#   [13] 建立 δ(t) 序列  [14] 訓練第二層 HMM  [15] 判定與比較
#   [16] 第二層作圖  [17] 各事件 δ(t) 與解碼狀態
# ============================================================
import os, glob, re, time, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)   # 隱藏 pandas 的 pyarrow 提示
import numpy as np
import pandas as pd

ARC_PATH = None          # None = 自動抓取已上傳的 .arc 檔；也可填檔名
SEED = 42                # 切分與初始化用的亂數種子
TRAIN_RATIO = 0.7        # 訓練集比例（7:3）
RESTARTS = 3             # Baum–Welch 重新起始次數
MAX_ITER = 200           # 每次起始最多迭代次數
LOG_GAP_OFFSET_A = 0.001 # log(Δr + 0.001 Å)，避免 Δr = 0 時取 log 發散
FEATURES = ["r1_A", "r2_A", "gap_A", "log_gap", "O1_H_O2_angle_deg"]

if ARC_PATH is None:
    found = sorted(glob.glob("*.arc") + glob.glob("*.ARC"))
    if not found:
        raise FileNotFoundError("找不到 .arc 檔，請先按上方「📂 上傳檔案」上傳 ARC 軌跡檔。")
    ARC_PATH = found[0]
print("ARC 檔：", ARC_PATH, f"({os.path.getsize(ARC_PATH) / 1e6:.1f} MB)")

# %%
# [2] 讀取 ARC：每個 !DATE 區塊為一個 frame，保留原子編號、元素、座標與 PBC 晶胞
def cell_matrix(a, b, c, alpha, beta, gamma):
    al, be, ga = np.deg2rad([alpha, beta, gamma])
    ax = [a, 0.0, 0.0]
    bx = [b * np.cos(ga), b * np.sin(ga), 0.0]
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.array([ax, bx, [cx, cy, cz]])

def read_arc(path):
    frames = []          # 每個 frame：dict(cell, ids, elements, xyz)
    cell, ids, elems, xyz = None, [], [], []
    def flush():
        if ids:
            order = np.argsort(ids)
            frames.append({
                "cell": cell,
                "ids": np.asarray(ids)[order],
                "elements": np.asarray(elems)[order],
                "xyz": np.asarray(xyz, dtype=float)[order],
            })
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("!DATE"):
                flush()
                cell, ids, elems, xyz = None, [], [], []
                continue
            if not s or s.startswith("!") or s.lower() == "end":
                continue
            t = s.split()
            if t[0].upper() == "PBC" and len(t) >= 7:
                cell = cell_matrix(*map(float, t[1:7]))
                continue
            m = re.match(r"^([A-Za-z]+)(\d+)$", t[0])
            if m and len(t) >= 4:
                ids.append(int(m.group(2)))
                elem = t[-2] if len(t) >= 8 else m.group(1)
                elems.append(elem[:1].upper() + elem[1:].lower())
                xyz.append([float(t[1]), float(t[2]), float(t[3])])
    flush()
    return frames

t0 = time.time()
frames = read_arc(ARC_PATH)
f0 = frames[0]
print(f"讀到 {len(frames)} 個 frame（{time.time() - t0:.1f} s）")
print("第一個 frame 原子組成：", pd.Series(f0["elements"]).value_counts().to_dict())
print("PBC 晶胞矩陣 (Å)：\n", np.round(f0["cell"], 4))

# %%
# [3] 計算每個 H 的 r1、r2（最近、第二近的 O，最小映像距離）與 O1–H–O2 角度（H 為頂點）
def h_geometry(frames):
    rows = []
    for k, fr in enumerate(frames, start=1):
        ids, el, xyz = fr["ids"], fr["elements"], fr["xyz"]
        h_mask, o_mask = el == "H", el == "O"
        H, O = xyz[h_mask], xyz[o_mask]
        h_ids, o_ids = ids[h_mask], ids[o_mask]
        M = fr["cell"]; Minv = np.linalg.inv(M)
        d = O[None, :, :] - H[:, None, :]                 # (nH, nO, 3) H→O 向量
        f = d @ Minv
        d = (f - np.round(f)) @ M                         # 最小映像
        r = np.linalg.norm(d, axis=2)
        order = np.argsort(r, axis=1, kind="stable")[:, :2]   # 同距離時取編號小的 O
        i = np.arange(len(H))
        v1, v2 = d[i, order[:, 0]], d[i, order[:, 1]]
        r1, r2 = r[i, order[:, 0]], r[i, order[:, 1]]
        cosang = np.clip(np.sum(v1 * v2, axis=1) / np.maximum(r1 * r2, 1e-8), -1, 1)
        rows.append(pd.DataFrame({
            "H_ID": h_ids, "Frame": k,
            "O1_ID": o_ids[order[:, 0]], "O2_ID": o_ids[order[:, 1]],
            "r1_A": r1, "r2_A": r2,
            "O1_H_O2_angle_deg": np.degrees(np.arccos(cosang)),
        }))
    data = pd.concat(rows, ignore_index=True)
    data["gap_A"] = (data["r2_A"] - data["r1_A"]).clip(lower=0.0)          # Δr = r2 − r1
    data["log_gap"] = np.log(data["gap_A"] + LOG_GAP_OFFSET_A)             # log(Δr + 0.001)
    return data.sort_values(["H_ID", "Frame"]).reset_index(drop=True)

t0 = time.time()
data = h_geometry(frames)
print(f"共 {data['H_ID'].nunique()} 個 H × {data['Frame'].nunique()} frames = {len(data)} 筆（{time.time() - t0:.1f} s）")
print(data[FEATURES].describe().round(3).to_string())
data.head(10)

# %%
# [4] 以 H 原子（整條軌跡）為單位，依 7:3 隨機切成訓練集 / 測試集，並用訓練集做標準化
h_ids = sorted(data["H_ID"].unique().tolist())
rng = np.random.default_rng(SEED)
perm = rng.permutation(h_ids)
n_train = int(round(TRAIN_RATIO * len(h_ids)))
train_ids = sorted(int(v) for v in perm[:n_train])
test_ids = sorted(int(v) for v in perm[n_train:])

def arrange(ids):
    return np.stack([data[data["H_ID"] == h].sort_values("Frame")[FEATURES].to_numpy(float) for h in ids])

train_raw, test_raw = arrange(train_ids), arrange(test_ids)
feat_mean = train_raw.reshape(-1, len(FEATURES)).mean(axis=0)
feat_std = train_raw.reshape(-1, len(FEATURES)).std(axis=0)
train_x = (train_raw - feat_mean) / feat_std
test_x = (test_raw - feat_mean) / feat_std

print(f"訓練集：{len(train_ids)} 個 H，測試集：{len(test_ids)} 個 H（seed = {SEED}）")
print("訓練集陣列形狀 (H 數, frame 數, 特徵數)：", train_x.shape)
print("測試集 H_ID：", test_ids)
pd.DataFrame({"mean": feat_mean, "std": feat_std}, index=FEATURES).round(4)

# %%
# [5] 2-state 對角高斯 HMM：forward–backward、Viterbi 與 Baum–Welch
EPS = 1e-8

def log_emission(x, means, variances):
    return -0.5 * np.sum(np.log(2 * np.pi * variances)[None, None]
                         + (x[:, :, None, :] - means[None, None]) ** 2 / variances[None, None], axis=3)

def logsumexp(v, axis):
    m = np.max(v, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.sum(np.exp(v - m), axis=axis, keepdims=True)), axis=axis)

def forward_backward(x, start, trans, means, variances, posteriors=True):
    ls, lt = np.log(np.clip(start, EPS, None)), np.log(np.clip(trans, EPS, None))
    e = log_emission(x, means, variances)
    N, T, S = e.shape
    a = np.empty_like(e)
    a[:, 0] = ls + e[:, 0]
    for t in range(1, T):
        a[:, t] = e[:, t] + logsumexp(a[:, t - 1, :, None] + lt[None], axis=1)
    ll = logsumexp(a[:, -1], axis=1)
    if not posteriors:
        return ll
    b = np.zeros_like(e)
    for t in range(T - 2, -1, -1):
        b[:, t] = logsumexp(lt[None] + e[:, t + 1, None, :] + b[:, t + 1, None, :], axis=2)
    gamma = np.exp(a + b - ll[:, None, None])
    xi = np.exp(a[:, :-1, :, None] + lt[None, None] + e[:, 1:, None, :] + b[:, 1:, None, :]
                - ll[:, None, None, None])
    return ll, gamma, xi

def viterbi(x, start, trans, means, variances):
    ls, lt = np.log(np.clip(start, EPS, None)), np.log(np.clip(trans, EPS, None))
    e = log_emission(x, means, variances)
    N, T, S = e.shape
    score = ls + e[:, 0]
    back = np.zeros((N, T, S), dtype=np.int8)
    for t in range(1, T):
        ch = score[:, :, None] + lt[None]
        back[:, t] = ch.argmax(axis=1)
        score = e[:, t] + ch.max(axis=1)
    path = np.empty((N, T), dtype=int)
    path[:, -1] = score.argmax(axis=1)
    for t in range(T - 1, 0, -1):
        path[:, t - 1] = back[np.arange(N), t, path[:, t]]
    return path

def fit_hmm(x, seed=42, restarts=3, max_iter=200, verbose=True):
    N, T, F = x.shape
    fits = []
    for r in range(restarts):
        rng = np.random.default_rng(seed + r)
        q25, q75 = np.quantile(x[:, :, 2].ravel(), [0.25, 0.75])   # 以 gap 特徵初始化兩個狀態
        means = np.vstack([np.full(F, q25), np.full(F, q75)]) + rng.normal(0, 0.05, (2, F))
        variances = np.tile(np.var(x.reshape(-1, F), axis=0) + 0.05, (2, 1))
        start = np.array([0.5, 0.5])
        trans = np.array([[0.985, 0.015], [0.015, 0.985]])
        prev, history = -np.inf, []
        for it in range(1, max_iter + 1):
            ll, gamma, xi = forward_backward(x, start, trans, means, variances)
            total = float(ll.sum()); history.append(total)
            w = gamma.sum(axis=(0, 1)) + EPS
            start = gamma[:, 0].mean(axis=0)
            trans = xi.sum(axis=(0, 1)) + 1e-3
            trans /= trans.sum(axis=1, keepdims=True)
            means = np.einsum("nts,ntf->sf", gamma, x) / w[:, None]
            variances = np.einsum("nts,ntsf->sf", gamma, (x[:, :, None, :] - means[None, None]) ** 2) / w[:, None]
            variances = np.maximum(variances, 1e-4)
            if it > 3 and abs(total - prev) < 1e-5 * (1 + abs(prev)):
                break
            prev = total
        final = float(forward_backward(x, start, trans, means, variances, posteriors=False).sum())
        if verbose:
            print(f"  restart {r}（seed {seed + r}）：{it} 次迭代，log-likelihood = {final:.2f}")
        fits.append(dict(restart=r, start=start, trans=trans, means=means,
                         variances=variances, history=history, ll=final))
    return max(fits, key=lambda f: f["ll"]), fits

print("HMM 函式已定義")

# %%
# [6] 以訓練集訓練 2-state HMM（Baum–Welch，多次起始取最佳）
t0 = time.time()
print(f"訓練中：{train_x.shape[0]} 條軌跡 × {train_x.shape[1]} frames ...")
best, fits = fit_hmm(train_x, seed=SEED, restarts=RESTARTS, max_iter=MAX_ITER)
start, trans, means_z, vars_z = best["start"], best["trans"], best["means"], best["variances"]

# 換回原始單位；log_gap 平均較小的狀態 = Shared-like（H 介於兩個 O 之間）
means_orig = means_z * feat_std + feat_mean
sd_orig = np.sqrt(vars_z) * feat_std
shared = int(np.argmin(means_orig[:, FEATURES.index("log_gap")]))
localized = 1 - shared
state_names = {localized: "Localized-like", shared: "Shared-like"}

train_ll = forward_backward(train_x, start, trans, means_z, vars_z, posteriors=False)
test_ll = forward_backward(test_x, start, trans, means_z, vars_z, posteriors=False)
print(f"完成，選用 restart {best['restart']}，耗時 {time.time() - t0:.1f} s")

# %%
# [7] 查看訓練完的模型參數
names = [state_names[0], state_names[1]]
print("狀態對應：", state_names, "\n")

print("初始機率 π：")
print(pd.Series(start, index=names).round(4).to_string(), "\n")

print("轉移矩陣 A（列 = 目前狀態，欄 = 下一個狀態）：")
print(pd.DataFrame(trans, index=names, columns=names).round(5).to_string(), "\n")

print("平均停留時間（1 frame = 1 fs）：")
for s in (0, 1):
    print(f"  {state_names[s]:15s} 1/(1 − A_ii) = {1 / (1 - trans[s, s]):.1f} fs")

print("\n發射分布平均值（原始單位）：")
print(pd.DataFrame(means_orig, index=names, columns=FEATURES).round(4).to_string())
print("\n發射分布標準差（原始單位）：")
print(pd.DataFrame(sd_orig, index=names, columns=FEATURES).round(4).to_string())

n_par = 1 + 2 + 2 * len(FEATURES) * 2          # π(1) + A(2) + 平均(2F) + 變異數(2F)
n_obs = train_x.shape[0] * train_x.shape[1]
train_pf = train_ll.sum() / n_obs
test_pf = test_ll.sum() / (test_x.shape[0] * test_x.shape[1])
print("\n模型評估：")
print(f"  訓練集 log-likelihood / frame：{train_pf:.4f}")
print(f"  測試集 log-likelihood / frame：{test_pf:.4f}")
print(f"  差值（測試 − 訓練）        ：{test_pf - train_pf:+.4f}")
print(f"  參數數量                  ：{n_par}")
print(f"  訓練集 BIC                ：{-2 * train_ll.sum() + n_par * np.log(n_obs):.1f}")

# %%
# [8] 解碼（Viterbi）並畫圖：各狀態的 r1、r2、Δr、角度分布，以及訓練收斂曲線
import matplotlib.pyplot as plt

all_ids = train_ids + test_ids
path = viterbi(np.concatenate([train_x, test_x]), start, trans, means_z, vars_z)
decoded = pd.concat([
    data[data["H_ID"] == h].sort_values("Frame").assign(
        split="train" if h in train_ids else "test",
        state=[state_names[s] for s in path[i]])
    for i, h in enumerate(all_ids)], ignore_index=True)

print("各狀態 frame 數與平均幾何：")
print(decoded.groupby(["split", "state"])[["r1_A", "r2_A", "gap_A", "O1_H_O2_angle_deg"]]
      .agg(["count", "mean"]).round(3).to_string())

colors = {"Localized-like": "#3569A8", "Shared-like": "#D46A4C"}
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
for ax, (col, label) in zip(axes.ravel(), [
        ("r1_A", "r1 (Å)"), ("r2_A", "r2 (Å)"), ("gap_A", "Δr = r2 − r1 (Å)"),
        ("log_gap", "log(Δr + 0.001)"), ("O1_H_O2_angle_deg", "O1–H–O2 angle (deg)")]):
    for name in ["Localized-like", "Shared-like"]:
        ax.hist(decoded.loc[decoded["state"] == name, col], bins=70, density=True,
                alpha=0.55, color=colors[name], label=name)
    ax.set_xlabel(label); ax.set_ylabel("Density")
axes[0, 0].legend(frameon=False)
ax = axes[1, 2]
for f in fits:
    ax.plot(f["history"], label=f"restart {f['restart']}")
ax.set_xlabel("Iteration"); ax.set_ylabel("Train log-likelihood"); ax.legend(frameon=False)
fig.tight_layout()

# %%
# [9] 轉移機率矩陣熱圖（報告圖 6）
order = [shared, localized]                     # 列/欄順序：Shared-like, Localized-like
A = trans[np.ix_(order, order)]
labels = [state_names[s] for s in order]
fig, ax = plt.subplots(figsize=(5.2, 4.2))
im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{A[i, j]:.4f}", ha="center", va="center",
                color="white" if A[i, j] > 0.5 else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
ax.set_title("Transition matrix")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
print(pd.DataFrame(A, index=labels, columns=labels).round(4).to_string())

# %%
# [10] 質子移轉事件判定
#   1. 以 Viterbi 解碼的狀態序列，找出完整的 L → S → L 片段（Localized → Shared → Localized）
#   2. 記錄進入 S 前最後一個 L frame 的最近氧 O_A，以及回到 L 第一個 frame 的最近氧 O_B
#   3. 若 O_A ≠ O_B，判定為質子移轉候選事件；O_A = O_B 則視為震盪後回到原氧（不計）
def lsl_episodes(hdf):
    st = (hdf["state"].to_numpy() == "Shared-like")          # True = S, False = L
    ox, fr = hdf["O1_ID"].to_numpy(), hdf["Frame"].to_numpy()
    rows = []
    for b in np.where(~st[:-1] & st[1:])[0] + 1:              # L→S 的第一個 S frame
        back = np.where(~st[b:])[0]
        if not len(back):                                     # 軌跡結束前沒有回到 L，不算完整片段
            continue
        e = b + int(back[0])                                  # 回到 L 的第一個 frame
        rows.append({
            "H_ID": int(hdf["H_ID"].iloc[0]), "split": hdf["split"].iloc[0],
            "shared_start_frame": int(fr[b]), "shared_end_frame": int(fr[e - 1]),
            "shared_duration_fs": float(e - b),               # 1 frame = 1 fs
            "O_A": int(ox[b - 1]), "O_B": int(ox[e]),
        })
    return rows

episodes = pd.DataFrame([r for _, g in decoded.groupby("H_ID", sort=True)
                         for r in lsl_episodes(g.sort_values("Frame"))])
episodes["PT_candidate"] = episodes["O_A"] != episodes["O_B"]
pt = episodes[episodes["PT_candidate"]].reset_index(drop=True)
pt.insert(0, "PT_id", np.arange(1, len(pt) + 1))

print("完整 L→S→L 片段數：")
print(episodes.groupby("split")["PT_candidate"].agg(L_S_L片段="count", 質子移轉候選="sum").to_string())
test_pt = pt[pt["split"] == "test"].reset_index(drop=True)
print(f"\n測試集質子移轉候選事件（{len(test_pt)} 筆）：")
print(test_pt[["H_ID", "shared_start_frame", "shared_end_frame", "shared_duration_fs", "O_A", "O_B"]].to_string(index=False))

# %%
# [11] 測試集質子移轉候選事件：中間 Shared-like 持續時間盒型圖（報告圖 7）
fig, ax = plt.subplots(figsize=(8.2, 4.8))
if len(test_pt):
    ax.boxplot(test_pt["shared_duration_fs"], showfliers=True)
    xs = np.linspace(0.90, 1.10, len(test_pt))
    cmap = plt.get_cmap("tab10")
    for k, (h, g) in enumerate(test_pt.groupby("H_ID", sort=True)):
        rng_txt = "; ".join(f"{a}-{b} fs" for a, b in zip(g["shared_start_frame"], g["shared_end_frame"]))
        ax.scatter(xs[g.index], g["shared_duration_fs"], s=55, color=cmap(k % 10),
                   edgecolor="black", linewidth=0.5, zorder=3, label=f"H{h} (S: {rng_txt})")
    ax.legend(title="Hydrogen atom", frameon=False, fontsize=8, title_fontsize=9,
              loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_xticks([1]); ax.set_xticklabels(["Test set"])
    print(test_pt["shared_duration_fs"].describe().round(1).to_string())
else:
    ax.text(0.5, 0.5, "No test-set PT candidates", ha="center", va="center")
ax.set_ylabel("Middle Shared-like duration in L-S-L (fs)")
ax.set_title("Test-set PT-containing L-S-L episodes")
fig.tight_layout()

# %%
# [12] 各測試集事件：H 與最近兩顆氧原子的距離隨時間演化，依氧原子編號上色（報告圖 8）
hs = sorted(test_pt["H_ID"].unique())
if hs:
    ncol = 2
    nrow = (len(hs) + 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.6 * nrow), squeeze=False)
    for ax, h in zip(axes.ravel(), hs):
        g = data[data["H_ID"] == h].sort_values("Frame")
        pts = pd.concat([g[["Frame", "O1_ID", "r1_A"]].set_axis(["Frame", "O", "r"], axis=1),
                         g[["Frame", "O2_ID", "r2_A"]].set_axis(["Frame", "O", "r"], axis=1)])
        for k, (o, q) in enumerate(pts.groupby("O", sort=False)):
            ax.scatter(q["Frame"], q["r"], s=4, color=plt.get_cmap("tab20")(k % 20), label=f"O{o}")
        for _, ev in test_pt[test_pt["H_ID"] == h].iterrows():
            ax.axvspan(ev["shared_start_frame"], ev["shared_end_frame"], color="#D46A4C", alpha=0.12)
        ax.set_title(f"H{h}: nearest 2 O atoms by frame")
        ax.set_xlabel("Frame (fs)"); ax.set_ylabel("Distance (Å)")
        ax.legend(title="Nearest O atom", fontsize=7, markerscale=3, ncol=2, frameon=False)
    for ax in axes.ravel()[len(hs):]:
        ax.axis("off")
    fig.tight_layout()
    print("淡紅色區塊 = 該事件中間的 Shared-like 區段")

# %%
# ============================================================
# 第二層 2-state HMM：判斷每個 L→S→L 片段「有沒有移轉」
# ------------------------------------------------------------
# 想法：第一層 HMM 只告訴我們質子處於 Localized 或 Shared，
#       第二層則針對每個 L→S→L 片段，固定一對氧原子：
#         O_A = 進入 S 前的最近氧（原本的供體）
#         O_B = S 期間最常出現的另一顆氧（可能的受體）
#       觀測值為質子座標 δ(t) = r(H, O_A) − r(H, O_B)
#         δ < 0：質子在 O_A 側；δ > 0：質子在 O_B 側
#       以 2-state HMM 學習「O_A 側（未移轉）」與「O_B 側（已移轉）」。
#       兩個狀態設為對稱（平均值 −m / +m、共用變異數），因為交換 O_A 與 O_B 時 δ 只是變號；
#       δ ≈ 0 的共享構型兩側都可能，由轉移機率（時間連續性）決定，因此短暫震盪（rattling）不會被判成移轉。
# [13] 建立每個片段的 δ(t) 序列
# ============================================================
PRE_FS, POST_FS = 20, 20        # 片段前後各多取的 fs 數（前段 L、後段 L）
STABLE_FS = 10                  # 判定移轉：回到 L 後需在 O_B 側連續停留的 fs 數

frame_index = [{int(a): k for k, a in enumerate(fr["ids"])} for fr in frames]

cell_inv = [np.linalg.inv(fr["cell"]) for fr in frames]

def pair_distances(frame_numbers, a, b):
    """H(a) 到 O(b) 在多個 frame 的最小映像距離"""
    out = np.empty(len(frame_numbers))
    for k, f in enumerate(frame_numbers):
        fr, idx = frames[f - 1], frame_index[f - 1]
        d = fr["xyz"][idx[b]] - fr["xyz"][idx[a]]
        q = d @ cell_inv[f - 1]
        out[k] = np.linalg.norm((q - np.round(q)) @ fr["cell"])
    return out

seqs = []
for ep_id, ep in episodes.reset_index(drop=True).iterrows():
    h = int(ep["H_ID"]); s0, s1 = int(ep["shared_start_frame"]), int(ep["shared_end_frame"])
    g = data[data["H_ID"] == h].set_index("Frame")
    during = pd.concat([g.loc[s0:s1, "O1_ID"], g.loc[s0:s1, "O2_ID"]])
    others = during[during != ep["O_A"]]
    if others.empty:
        continue
    oA, oB = int(ep["O_A"]), int(others.value_counts().idxmax())
    f_lo, f_hi = max(1, s0 - PRE_FS), min(len(frames), s1 + 1 + POST_FS)
    fr_range = np.arange(f_lo, f_hi + 1)
    delta = pair_distances(fr_range, h, oA) - pair_distances(fr_range, h, oB)
    seqs.append(dict(ep_id=ep_id, H_ID=h, split=ep["split"], O_A=oA, O_B=oB,
                     s0=s0, s1=s1, frames=fr_range, delta=delta,
                     layer1_PT=bool(ep["PT_candidate"]), dur=float(ep["shared_duration_fs"])))

print(f"共 {len(seqs)} 個 L→S→L 片段（訓練 {sum(s['split'] == 'train' for s in seqs)}、"
      f"測試 {sum(s['split'] == 'test' for s in seqs)}）")
print(f"每段序列長度：{min(len(s['delta']) for s in seqs)}–{max(len(s['delta']) for s in seqs)} frames")

# %%
# [14] 第二層 HMM：可處理不同長度序列的 1 維高斯 HMM（用遮罩補齊長度），以訓練集片段訓練
def pad(seq_list):
    T = max(len(s) for s in seq_list)
    X = np.zeros((len(seq_list), T)); Mk = np.zeros((len(seq_list), T), bool)
    for i, s in enumerate(seq_list):
        X[i, :len(s)] = s; Mk[i, :len(s)] = True
    return X, Mk

def fb_masked(X, Mk, start, A, mu, var):
    e = -0.5 * (np.log(2 * np.pi * var)[None, None] + (X[:, :, None] - mu) ** 2 / var)
    e = np.where(Mk[:, :, None], e, 0.0)            # 補齊的 frame 不提供觀測資訊
    lA, N, T = np.log(A), X.shape[0], X.shape[1]
    a = np.empty((N, T, 2)); a[:, 0] = np.log(start) + e[:, 0]
    for t in range(1, T):
        a[:, t] = e[:, t] + logsumexp(a[:, t - 1, :, None] + lA[None], axis=1)
    ll = logsumexp(a[:, -1], axis=1)
    b = np.zeros((N, T, 2))
    for t in range(T - 2, -1, -1):
        b[:, t] = logsumexp(lA[None] + e[:, t + 1, None, :] + b[:, t + 1, None, :], axis=2)
    gamma = np.exp(a + b - ll[:, None, None]) * Mk[:, :, None]
    xi = np.exp(a[:, :-1, :, None] + lA[None, None] + e[:, 1:, None, :] + b[:, 1:, None, :]
                - ll[:, None, None, None]) * Mk[:, 1:, None, None]
    return ll, gamma, xi, e

def viterbi_masked(X, Mk, start, A, mu, var):
    e = -0.5 * (np.log(2 * np.pi * var)[None, None] + (X[:, :, None] - mu) ** 2 / var)
    e = np.where(Mk[:, :, None], e, 0.0)
    lA, N, T = np.log(A), X.shape[0], X.shape[1]
    score = np.log(start) + e[:, 0]; back = np.zeros((N, T, 2), np.int8)
    for t in range(1, T):
        ch = score[:, :, None] + lA[None]
        back[:, t] = ch.argmax(axis=1); score = e[:, t] + ch.max(axis=1)
    path = np.empty((N, T), int); path[:, -1] = score.argmax(axis=1)
    for t in range(T - 1, 0, -1):
        path[:, t - 1] = back[np.arange(N), t, path[:, t]]
    return path

train_seqs = [s for s in seqs if s["split"] == "train"]
Xtr, Mtr = pad([s["delta"] for s in train_seqs])
start2 = np.array([0.5, 0.5]); A2 = np.array([[0.99, 0.01], [0.01, 0.99]])
mu2 = np.array([-0.5, 0.5]); var2 = np.array([0.1, 0.1])
prev = -np.inf
t0 = time.time()
for it in range(1, MAX_ITER + 1):
    ll, gamma, xi, _ = fb_masked(Xtr, Mtr, start2, A2, mu2, var2)
    total = float(ll.sum())
    w = gamma.sum(axis=(0, 1)) + EPS
    start2 = gamma[:, 0].mean(axis=0) + 1e-6; start2 /= start2.sum()
    A2 = xi.sum(axis=(0, 1)) + 1e-3; A2 /= A2.sum(axis=1, keepdims=True)
    # 對稱限制：O_A 側與 O_B 側互為鏡像（μ_A = −m、μ_B = +m，共用變異數）
    m = float((gamma[:, :, 1] * Xtr - gamma[:, :, 0] * Xtr).sum() / w.sum())
    mu2 = np.array([-m, m])
    var2 = np.full(2, max(float((gamma * (Xtr[:, :, None] - mu2) ** 2).sum() / w.sum()), 1e-4))
    if it > 3 and abs(total - prev) < 1e-6 * (1 + abs(prev)):
        break
    prev = total
side_A = int(np.argmin(mu2)); side_B = 1 - side_A
names2 = {side_A: "O_A side (not transferred)", side_B: "O_B side (transferred)"}
print(f"對稱高斯：m = {m:.4f} Å")
print(f"第二層 HMM 訓練完成：{it} 次迭代，{time.time() - t0:.1f} s，log-likelihood = {total:.1f}\n")
o2 = [side_A, side_B]; lab2 = ["O_A side", "O_B side"]
print("初始機率：", dict(zip(lab2, start2[o2].round(4))))
print("轉移矩陣（列 = 目前，欄 = 下一個）：")
print(pd.DataFrame(A2[np.ix_(o2, o2)], index=lab2, columns=lab2).round(4).to_string())
print("\n發射分布（δ = r(H,O_A) − r(H,O_B)，Å）：")
print(pd.DataFrame({"mean_δ": mu2[o2], "std_δ": np.sqrt(var2[o2]),
                    "平均停留時間_fs": 1 / (1 - np.diag(A2)[o2])}, index=lab2).round(4).to_string())

# %%
# [15] 用第二層 HMM 解碼所有片段，判定是否移轉，並與第一層規則（O_A ≠ O_B）比較
#   第二層判定為「移轉」：回到 L 的那個 frame 起，Viterbi 狀態在 O_B 側連續停留 ≥ STABLE_FS
X_all, M_all = pad([s["delta"] for s in seqs])
path2 = viterbi_masked(X_all, M_all, start2, A2, mu2, var2)
_, post2, _, _ = fb_masked(X_all, M_all, start2, A2, mu2, var2)     # 每個 frame 在 O_B 側的後驗機率
rows = []
for i, s in enumerate(seqs):
    p = path2[i, :len(s["delta"])]
    ret = int(np.where(s["frames"] == s["s1"] + 1)[0][0]) if (s["s1"] + 1) in s["frames"] else len(p) - 1
    after = p[ret:ret + STABLE_FS]
    started_A = p[0] == side_A
    transferred = bool(started_A and len(after) == STABLE_FS and np.all(after == side_B))
    s["path"] = p
    sw = np.where((p[1:] == side_B) & (p[:-1] == side_A))[0] + 1     # A→B 切換點
    sw = sw[sw <= ret]
    transfer_frame = int(s["frames"][sw[-1]]) if (transferred and len(sw)) else None   # 最後一次（確定）越過的時間
    p_transfer = float(post2[i, ret:ret + STABLE_FS, side_B].mean()) if len(after) else np.nan
    rows.append(dict(H_ID=s["H_ID"], split=s["split"], shared_start_frame=s["s0"], shared_end_frame=s["s1"],
                     shared_duration_fs=s["dur"], O_A=s["O_A"], O_B=s["O_B"],
                     layer1_PT=s["layer1_PT"], layer2_PT=transferred, P_transfer=round(p_transfer, 4),
                     transfer_frame=transfer_frame,
                     n_side_switches=int(np.sum(p[1:] != p[:-1]))))
judge = pd.DataFrame(rows)
judge["transfer_frame"] = judge["transfer_frame"].astype("Int64")
for sp in ["train", "test"]:
    j = judge[judge["split"] == sp]
    print(f"【{sp}】第一層規則 vs 第二層 HMM（列 = 第一層，欄 = 第二層）")
    print(pd.crosstab(j["layer1_PT"].map({True: "移轉", False: "未移轉"}),
                      j["layer2_PT"].map({True: "移轉", False: "未移轉"})).to_string(), "\n")
test_l2 = judge[(judge["split"] == "test") & (judge["layer1_PT"] | judge["layer2_PT"])].reset_index(drop=True)
print("測試集中任一層判定為移轉的片段：")
print(test_l2[["H_ID", "shared_start_frame", "shared_end_frame", "shared_duration_fs",
               "O_A", "O_B", "layer1_PT", "layer2_PT", "P_transfer", "transfer_frame", "n_side_switches"]].to_string(index=False))
print("\nP_transfer = 回到 L 後 STABLE_FS 個 frame 在 O_B 側的平均後驗機率；n_side_switches = 片段內 O_A/O_B 側切換次數；transfer_frame = 質子最後一次越到 O_B 側的時間（第二層才有的資訊）")
print("所有片段 P_transfer 分布：")
print(pd.cut(judge["P_transfer"], [-0.01, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]).value_counts().sort_index().to_string())

# %%
# [16] 第二層 HMM 作圖：(a) δ 分布與兩個高斯 (b) 第二層轉移矩陣 (c) 測試集第二層判定移轉事件的共享期盒型圖
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.2, 1, 1.3]})
ax = axes[0]
allx = np.concatenate([s["delta"] for s in train_seqs])
ax.hist(allx, bins=120, density=True, color="#bbbbbb", alpha=0.7, label="train δ")
xx = np.linspace(allx.min(), allx.max(), 400)
for st, c in [(side_A, "#3569A8"), (side_B, "#D46A4C")]:
    wgt = gamma.sum(axis=(0, 1))[st] / gamma.sum()
    ax.plot(xx, wgt * np.exp(-0.5 * (xx - mu2[st]) ** 2 / var2[st]) / np.sqrt(2 * np.pi * var2[st]),
            color=c, lw=2, label=names2[st])
ax.set_xlabel("δ = r(H,O_A) − r(H,O_B) (Å)"); ax.set_ylabel("Density"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Layer-2 emission")

ax = axes[1]
B = A2[np.ix_(o2, o2)]
im = ax.imshow(B, cmap="Blues", vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{B[i, j]:.4f}", ha="center", va="center", color="white" if B[i, j] > 0.5 else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(lab2); ax.set_yticks([0, 1]); ax.set_yticklabels(lab2)
ax.set_title("Layer-2 transition matrix"); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = axes[2]
tp2 = judge[(judge["split"] == "test") & judge["layer2_PT"]].reset_index(drop=True)
if len(tp2):
    ax.boxplot(tp2["shared_duration_fs"], showfliers=True)
    xs = np.linspace(0.90, 1.10, len(tp2)); cmap = plt.get_cmap("tab10")
    for k, (h, g) in enumerate(tp2.groupby("H_ID", sort=True)):
        txt = "; ".join(f"{a}-{b} fs" for a, b in zip(g["shared_start_frame"], g["shared_end_frame"]))
        ax.scatter(xs[g.index], g["shared_duration_fs"], s=55, color=cmap(k % 10), edgecolor="black",
                   linewidth=0.5, zorder=3, label=f"H{h} (S: {txt})")
    ax.legend(title="Hydrogen atom", frameon=False, fontsize=7, title_fontsize=8,
              loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_xticks([1]); ax.set_xticklabels(["Test set"])
else:
    ax.text(0.5, 0.5, "No layer-2 transfers in test set", ha="center", va="center")
ax.set_ylabel("Middle Shared-like duration (fs)"); ax.set_title("Test-set layer-2 PT episodes")
fig.tight_layout()

# %%
# [17] 測試集各片段的 δ(t) 與第二層解碼狀態（藍 = O_A 側，紅 = O_B 側；灰底 = 第一層 Shared-like 區段；紅虛線 = 移轉時間）
show = [s for s in seqs if s["split"] == "test" and (s["layer1_PT"] or
        judge.loc[(judge["H_ID"] == s["H_ID"]) & (judge["shared_start_frame"] == s["s0"]), "layer2_PT"].any())]
if show:
    ncol = 2; nrow = (len(show) + 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.0 * nrow), squeeze=False)
    for ax, s in zip(axes.ravel(), show):
        ax.axvspan(s["s0"], s["s1"], color="#999999", alpha=0.15)
        col = np.where(s["path"] == side_B, "#D46A4C", "#3569A8")
        ax.scatter(s["frames"], s["delta"], c=col, s=5)
        ax.axhline(0, color="black", lw=0.6, ls=":")
        tf = judge.loc[(judge["H_ID"] == s["H_ID"]) & (judge["shared_start_frame"] == s["s0"]), "transfer_frame"].iloc[0]
        if pd.notna(tf):
            ax.axvline(tf, color="#D46A4C", lw=1, ls="--")
        l2 = judge.loc[(judge["H_ID"] == s["H_ID"]) & (judge["shared_start_frame"] == s["s0"]), "layer2_PT"].iloc[0]
        ax.set_title(f"H{s['H_ID']}  O{s['O_A']}->O{s['O_B']}  layer-1: {'PT' if s['layer1_PT'] else 'no PT'}"
                     f"  layer-2: {'PT' if l2 else 'no PT'}", fontsize=9)
        ax.set_xlabel("Frame (fs)"); ax.set_ylabel("δ (Å)")
    for ax in axes.ravel()[len(show):]:
        ax.axis("off")
    fig.tight_layout()
