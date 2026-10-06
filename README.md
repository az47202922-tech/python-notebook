# 2-state HMM 質子移轉辨識

利用隱馬可夫模型（HMM）辨識化學機械平坦化（CMP）研磨液 DFT-MD 軌跡中的質子移轉事件。
觀測特徵：r1、log(Δr + 0.001 Å)、O1–H–O2 角度；以 H 原子為單位 7:3 切分訓練 / 測試集。

## 執行方式

| 方式 | 連結 |
|---|---|
| Google Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/az47202922-tech/python-notebook/blob/main/hmm_2state.ipynb) |
| 網頁版（瀏覽器內執行） | https://az47202922-tech.github.io/python-notebook/?nb=hmm_2state |

兩種方式都需要自行上傳 ARC 軌跡檔（`.arc`），檔案不包含在此 repo 中。

## 檔案

- `hmm_2state.ipynb`：Colab / Jupyter 筆記本
- `hmm_2state.py`：同內容的 Python 腳本（以 `# %%` 分段，網頁版使用）
- `index.html`：網頁版分段執行環境（Pyodide）
