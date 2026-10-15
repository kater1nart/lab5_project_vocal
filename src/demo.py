"""Демонстрационный сценарий защиты: поток из 8 треков через итоговую систему.

MTAT test (исполнители не встречались в обучении) — 6 треков, Jamendo (другой каталог) — 2 трека.
Бюджет модерации в демо — 3 проверки, чтобы показать работу контроллера риска.
Выход: results/demo_log.jsonl (журнал решений), results/demo_table.md.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import LAB2, ROOT, STUDY, VocalLabeler  # noqa: E402

LAB1 = STUDY / "1" / "lab1_vocal_llm"


def pick_tracks():
    df = pd.read_csv(LAB1 / "data" / "processed" / "dataset_final.csv")
    te = df[df.split == "test"]
    ids = [2845, 1726, 6267, 28690, 2069, 47766]  # примеры из анализа ошибок ЛР1/ЛР2: разные классы, согласие и конфликт модальностей
    rows = []
    for cid in ids:
        r = te[te.clip_id == cid].iloc[0]
        rows.append(dict(track_id=f"MTAT-{cid}", artist=r.artist, album=r.album, title=r.title, true=r.label,
                         path=str(LAB1 / "data" / "raw" / "mp3" / r.mp3_path), offset=0.0))
    import pyarrow.parquet as pq
    sdd = pd.read_csv(LAB2 / "data" / "processed" / "sdd_labeled.csv")
    want = sdd[sdd.release_year >= 2010].groupby("label").head(1)
    want = want[want.label.isin(["female", "instrumental"])]
    for f in sorted((LAB2 / "data" / "raw" / "sdd").glob("train-*.parquet")):
        t = pq.read_table(f, columns=["track_id", "path"]).to_pandas()
        for r in want.itertuples():
            m = t[t.track_id == r.track_id]
            if len(m) and not any(x["track_id"] == f"JAM-{r.track_id}" for x in rows):
                p = f"/tmp/jam_{r.track_id}.mp3"
                Path(p).write_bytes(m.iloc[0].path["bytes"])
                rows.append(dict(track_id=f"JAM-{r.track_id}", artist=r.ARTIST_NAME, album=r.ALBUM_NAME, title=r.TRACK_NAME,
                                 true=r.label, path=p, offset=30.0))
    return rows


def main():
    tracks = pick_tracks()
    sys_ = VocalLabeler(budget=3)
    out = []
    for t in tracks:
        rec = sys_.decide(t["track_id"], t["artist"], t["album"], t["title"], t["path"], t["offset"])
        rec["true_label"] = t["true"]
        out.append(rec)
        print(json.dumps(rec, ensure_ascii=False))
    with open(ROOT / "results" / "demo_log.jsonl", "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tab = pd.DataFrame([{"Трек": r["track"][:45], "Истина": r["true_label"], "Голоса: текст / ряд / изображение":
                         " / ".join(v.split(" ")[0] for v in r["votes"].values()), "Согласие": r["agreement"],
                         "Метка системы": r["label"], "Уверенность": r["confidence"], "Решение": r["action"],
                         "Бюджет": r["budget_left"], "Время, с": r["latency_s"]} for r in out])
    (ROOT / "results" / "demo_table.md").write_text(tab.to_markdown(index=False))
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
