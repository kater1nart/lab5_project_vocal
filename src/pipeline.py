"""Итоговая система разметки каталога по типу фронт-вокала (сквозной пайплайн ВКР).

Модули:
 1. Приём и валидация трека: метаданные (исполнитель, альбом, название) + аудио; проверка длительности, тишины, частоты.
 2. LLM-модуль (ЛР1): исключён из контура решений — VLM по спектрограмме на уровне угадывания (Macro-F1 ≤ 0,24).
 3. Мультимодальный модуль (ЛР2): три модальных эксперта — текст (TF-IDF + LogReg), временной ряд звука (120 статистик
    + LogReg), спектрограмма (CLIP ViT-B/32 + LogReg); слияние — среднее вероятностей.
 4. Модуль маршрутизации (ЛР3): централизованное правило «равные веса + порог τ» (лучше 8 MARL-алгоритмов):
    автометка, если max средней вероятности ≥ τ и бюджет модерации не исчерпан, иначе — модератор.
 5. Контроллер риска: бюджет модерации, журналирование решения (JSON Lines), объяснение для модератора
    (голоса модальностей, согласие, уверенность, причина решения), предупреждения о сдвиге (низкое согласие / тишина).
Обучение: модели модальностей обучаются на MTAT train (как в ЛР2), τ = 0,425 подобран на val в ЛР3.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT.parents[1]
LAB2 = STUDY / "2" / "lab2_multimodal_vocal"
sys.path.insert(0, str(LAB2 / "src"))
CLASSES = ["male", "female", "group", "instrumental"]
RU = {"male": "мужской вокал", "female": "женский вокал", "group": "группа / хор", "instrumental": "инструментал"}
TAU = 0.425  # ЛР3, подобран на val


# ------------------------------------------------------------------ обучение и сохранение экспертов
def train_and_save():
    from sklearn.linear_model import LogisticRegression
    import experiment as e2  # модуль ЛР2 (путь добавлен в sys.path), нужен для сериализации энкодеров
    df, seq, emb, _ = e2.load("mtat")
    tr = (df.split == "train").values
    enc = e2.Encoders().fit(df[tr], seq[tr], emb[tr])
    F = enc.transform(df[tr], seq[tr], emb[tr])
    y = df.y.values[tr]
    experts = {k: LogisticRegression(C=10, max_iter=4000, class_weight="balanced").fit(F[key], y)
               for k, key in [("text", "text"), ("ts", "ts"), ("img", "img")]}
    joblib.dump({"enc": enc, "experts": experts, "tau": TAU, "classes": CLASSES}, ROOT / "models" / "vocal_pipeline.joblib")
    print("saved models/vocal_pipeline.joblib")


# ------------------------------------------------------------------ инференс
class VocalLabeler:
    def __init__(self, budget=15):
        import importlib.util
        import open_clip
        import torch
        b = joblib.load(ROOT / "models" / "vocal_pipeline.joblib")
        self.enc, self.experts, self.tau = b["enc"], b["experts"], b["tau"]
        spec = importlib.util.spec_from_file_location("lab2_features", LAB2 / "src" / "features.py")
        self.fx = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.fx)
        self.torch = torch
        self.clip, _, self.prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        self.clip.eval()
        self.budget, self.budget0 = budget, budget
        self.log = []

    def validate(self, y, sr):
        issues = []
        if len(y) < sr * 5:
            issues.append("аудио короче 5 с")
        if np.sqrt(np.mean(y ** 2)) < 1e-3:
            issues.append("почти тишина")
        return issues

    def experts_proba(self, artist, album, title, audio_path, offset=0.0):
        import io
        import librosa
        import pandas as pd
        from PIL import Image
        t0 = time.perf_counter()
        y, _ = librosa.load(audio_path, sr=16000, mono=True, offset=offset, duration=10.0)  # гармонизация (ЛР2, S4)
        issues = self.validate(y, 16000)
        y = librosa.resample(y, orig_sr=16000, target_sr=self.fx.SR)
        seq = self.fx.timeseries(y)[None]
        png = "/tmp/_vocal_demo_spec.png"
        self.fx.render_spec(y, png)
        with self.torch.no_grad():
            e = self.clip.encode_image(self.prep(Image.open(png).convert("RGB"))[None])
            e = (e / e.norm(dim=-1, keepdim=True)).numpy()
        df = pd.DataFrame({"text": [f"{artist} | {album} | {title}"]})
        F = self.enc.transform(df, seq, e)
        P = {k: m.predict_proba(F[k])[0] for k, m in self.experts.items()}
        return P, issues, time.perf_counter() - t0

    def decide(self, track_id, artist, album, title, audio_path, offset=0.0):
        P, issues, lat = self.experts_proba(artist, album, title, audio_path, offset)
        avg = np.mean([P[k] for k in P], 0)
        votes = {k: CLASSES[int(np.argmax(v))] for k, v in P.items()}
        n_agree = max(list(votes.values()).count(c) for c in CLASSES)
        label, conf = CLASSES[int(avg.argmax())], float(avg.max())
        if issues:
            action, reason = "модератор", "валидация аудио: " + "; ".join(issues)
        elif conf >= self.tau:
            action, reason = "автометка", f"уверенность {conf:.2f} ≥ порога {self.tau}"
        else:
            action, reason = "модератор", f"уверенность {conf:.2f} < порога {self.tau}"
        if action == "модератор":
            if self.budget > 0:
                self.budget -= 1
            else:
                action, reason = "автометка (бюджет исчерпан)", reason + "; бюджет модерации исчерпан — метка ставится с флагом риска"
        warn = []
        if n_agree == 1:
            warn.append("модальности не согласны между собой — повышенный риск ошибки")
        rec = {"track_id": track_id, "track": f"{artist} — {title}", "label": label, "label_ru": RU[label],
               "confidence": round(conf, 3), "action": action, "reason": reason,
               "votes": {k: f"{v} ({P[k].max():.2f})" for k, v in votes.items()}, "agreement": f"{n_agree} из 3",
               "warnings": warn, "budget_left": self.budget, "latency_s": round(lat, 3),
               "probs": {c: round(float(p), 3) for c, p in zip(CLASSES, avg)}}
        self.log.append(rec)
        return rec


if __name__ == "__main__":
    train_and_save()
