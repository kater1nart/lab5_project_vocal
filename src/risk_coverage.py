"""Риск-покрытие итогового контроллера: доля автоматически размеченных треков vs доля ошибок среди них.

Используются вероятности модальных экспертов без утечки (ЛР3, data/stream.parquet): val — для выбора порога под SLA,
test (MTAT) и Jamendo 2010-2017 — для проверки. Правило: среднее вероятностей трёх модальностей, автометка при max ≥ τ.
Дополнительно — правило согласия: автометка только при единогласии трёх экспертов.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STREAM = ROOT.parents[1] / "3" / "lab3_marl_vocal" / "data" / "stream.parquet"
plt.rcParams.update({"font.family": "Arial", "font.size": 11})


def probs(d):
    P = np.stack([d[[f"p_{m}_{k}" for k in range(4)]].values for m in ["text", "ts", "img"]], 1)
    return P, d.y.values


def curve(P, y):
    avg = P.mean(1)
    conf, pred = avg.max(1), avg.argmax(1)
    taus = np.linspace(0.25, 0.95, 141)
    rows = []
    for t in taus:
        m = conf >= t
        rows.append({"tau": t, "coverage": m.mean(), "auto_error": 1 - (pred[m] == y[m]).mean() if m.any() else np.nan})
    return pd.DataFrame(rows)


def main():
    st = pd.read_parquet(STREAM)
    sets = {"val": st[st.split == "val"], "MTAT test": st[st.split == "test"],
            "Jamendo 2010–2017": st[(st.split == "sdd") & (st.release_year >= 2010)]}
    C = {k: curve(*probs(v)) for k, v in sets.items()}
    out = []
    for sla in [0.20, 0.10, 0.05]:
        cv = C["val"].dropna()
        ok = cv[cv.auto_error <= sla]
        tau = ok.tau.min() if len(ok) else np.nan  # минимальный порог, при котором SLA выполняется на val
        r = {"SLA (ошибка автометок ≤)": sla, "τ (по val)": tau}
        for k in ["MTAT test", "Jamendo 2010–2017"]:
            c = C[k].iloc[(C[k].tau - tau).abs().argmin()] if np.isfinite(tau) else None
            r[f"{k}: покрытие"] = c.coverage if c is not None else np.nan
            r[f"{k}: ошибка"] = c.auto_error if c is not None else np.nan
        out.append(r)
    for k in ["MTAT test", "Jamendo 2010–2017"]:  # рабочий порог ЛР3 и правило единогласия
        c = C[k].iloc[(C[k].tau - 0.425).abs().argmin()]
        P, y = probs(sets[k])
        am = P.argmax(-1)
        un = (am[:, 0] == am[:, 1]) & (am[:, 1] == am[:, 2])
        out.append({"SLA (ошибка автометок ≤)": f"τ = 0,425 (ЛР3), {k}", "τ (по val)": 0.425,
                    f"{k}: покрытие": c.coverage, f"{k}: ошибка": c.auto_error})
        out.append({"SLA (ошибка автометок ≤)": f"единогласие трёх экспертов, {k}", "τ (по val)": np.nan,
                    f"{k}: покрытие": un.mean(), f"{k}: ошибка": 1 - (am[un, 0] == y[un]).mean()})
    t = pd.DataFrame(out)
    t.to_csv(ROOT / "results" / "risk_coverage.csv", index=False)
    print(t.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for k, c in C.items():
        ax.plot(c.coverage, c.auto_error, label=k)
    ax.axhline(0.10, color="black", ls="--", lw=0.8)
    ax.text(0.02, 0.105, "SLA 10%", fontsize=9)
    ax.set_xlabel("доля треков, размеченных автоматически (покрытие)")
    ax.set_ylabel("доля ошибок среди автометок")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(ROOT / "report" / "figs" / "risk_coverage.png", dpi=150)


if __name__ == "__main__":
    main()
