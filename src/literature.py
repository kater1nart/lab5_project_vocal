"""Матрица литературы проекта: объединение и дедупликация обзоров ЛР1-4 + источники ЛР5 (39 первичных источников)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = "https://arxiv.org/abs/"
L = [
    # блок, источник, ссылка, ЛР, вклад, данные/задача, метрики, использование в проекте
    ("Данные", "Law E. et al. Evaluation of algorithms using games: the case of music tagging. ISMIR 2009", "https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset", "1-4", "датасет MagnaTagATune, краудсорсинговые теги", "25,9 тыс. клипов", "—", "основной набор; метки типа вокала из тегов"),
    ("Данные", "Bogdanov D. et al. The MTG-Jamendo Dataset for Automatic Music Tagging. ICML 2019 Workshop ML4MD", "https://github.com/MTG/mtg-jamendo-dataset", "2,3", "открытые метаданные и даты релиза Jamendo", "55 тыс. треков", "ROC-AUC", "текстовая модальность и даты для отложенного периода"),
    ("Данные", "Manco I. et al. The Song Describer Dataset. NeurIPS 2023 Workshop ML for Audio", A + "2311.10057", "2,3", "описания треков слушателями", "1,1 тыс. описаний / 706 треков", "—", "авторский набор для переноса (метки из описаний)"),
    ("Данные", "Won M. et al. Evaluation of CNN-based Automatic Music Tagging Models. SMC 2020", A + "2006.00751", "1", "единый протокол сравнения теггеров", "MTAT, MSD, Jamendo", "ROC/PR-AUC", "принцип единого протокола; supervised baseline"),
    ("Мультимодальность", "Oramas S. et al. Multi-label Music Genre Classification from Audio, Text, and Images. ISMIR 2017", A + "1707.04916", "2", "слияние аудио, текста и обложек", "MuMu, 31 тыс. альбомов", "AUC", "ближайший музыкальный аналог тройного слияния"),
    ("LLM / VLM", "Dixit S., Heller L., Donahue C. Vision Language Models Are Few-Shot Audio Spectrogram Classifiers. 2024", A + "2411.12058", "1,2", "VLM классифицируют звук по спектрограмме", "ESC-10", "accuracy", "обоснование модальности «спектрограмма»; в ЛР1 подход не сработал"),
    ("LLM / VLM", "Gardner J. et al. LLark: A Multimodal Instruction-Following Language Model for Music. ICML 2024", A + "2310.07160", "1", "музыкальный энкодер + LLM, instruction tuning", "MTAT, MusicCaps и др.", "accuracy, human eval", "LLM для музыки требуют дообучения"),
    ("LLM / VLM", "Weck B., Manco I. et al. MuChoMusic. ISMIR 2024", A + "2408.01337", "1", "бенчмарк понимания музыки аудио-LLM", "1 187 вопросов", "accuracy", "модели опираются на текст, а не звук — подтверждено в ЛР1"),
    ("LLM / VLM", "Radford A. et al. Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML 2021", A + "2103.00020", "2", "контрастивное пространство изображение—текст", "400 млн пар", "zero-shot acc", "энкодер спектрограммы; zero-shot baseline (неудачный)"),
    ("LLM / VLM", "Liu H. et al. Visual Instruction Tuning (LLaVA). NeurIPS 2023", A + "2304.08485", "1,2", "instruction tuning VLM", "158 тыс. инструкций", "GPT-оценка", "контекст VLM из ЛР1"),
    ("LLM / VLM", "Bai S. et al. Qwen2.5-VL Technical Report. 2025", A + "2502.13923", "1", "открытая VLM с многими изображениями", "—", "бенчмарки VQA", "модели сравнения в ЛР1"),
    ("Мультимодальность", "Baltrušaitis T., Ahuja C., Morency L.-P. Multimodal Machine Learning: A Survey and Taxonomy. TPAMI 2019", A + "1705.09406", "2", "таксономия early/late/hybrid fusion", "обзор", "—", "классификация схем слияния"),
    ("Мультимодальность", "Wang W., Tran D., Feiszli M. What Makes Training Multi-Modal Classification Networks Hard? CVPR 2020", A + "1905.12681", "2,4", "мультимодальная сеть может проиграть одномодальной", "Kinetics", "accuracy", "объясняет провал Gated-MLP и доминирование модальности"),
    ("Мультимодальность", "Ma M. et al. Are Multimodal Transformers Robust to Missing Modality? CVPR 2022", A + "2204.05454", "2", "деградация при пропуске модальности", "MM-IMDb и др.", "F1", "эксперимент с пропуском модальности, modality dropout"),
    ("Мультимодальность", "Elizalde B. et al. CLAP: Learning Audio Concepts From Natural Language Supervision. ICASSP 2023", A + "2206.04769", "2", "аудио—текст контрастивное пространство", "128 тыс. пар", "zero-shot acc", "направление развития аудио-энкодера"),
    ("Мультимодальность", "Huang Q. et al. MuLan: A Joint Embedding of Music Audio and Natural Language. ISMIR 2022", A + "2208.12415", "2", "совместное пространство музыки и текста", "44 млн треков", "AUC, mAP", "предел подхода: закрытые данные индустриального масштаба"),
    ("Мультимодальность", "Delbouys R. et al. Music Mood Detection Based on Audio and Lyrics with Deep Neural Net. ISMIR 2018", A + "1809.07276", "2", "mid-level fusion аудио и лирики", "18 тыс. треков", "R²", "выигрыш fusion зависит от задачи"),
    ("MARL", "Lowe R. et al. Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG). NeurIPS 2017", A + "1706.02275", "3", "централизованный критик, децентрализованные акторы", "MPE", "reward", "алгоритм сравнения (дискретный вариант)"),
    ("MARL", "Foerster J. et al. Counterfactual Multi-Agent Policy Gradients (COMA). AAAI 2018", A + "1705.08926", "3", "контрфактический credit assignment", "StarCraft", "win rate", "алгоритм сравнения; выродился в ЛР3"),
    ("MARL", "Sunehag P. et al. Value-Decomposition Networks. AAMAS 2018", A + "1706.05296", "3", "Q_tot = ΣQ_i", "2D-игры", "reward", "алгоритм сравнения"),
    ("MARL", "Rashid T. et al. QMIX. ICML 2018", A + "1803.11485", "3", "монотонное смешивание Q", "SMAC", "win rate", "второй лучший MARL в ЛР3"),
    ("MARL", "Yu C. et al. The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO). NeurIPS 2022", A + "2103.01955", "3", "PPO с централизованным критиком", "MPE, SMAC, Hanabi", "win rate, reward", "лучший MARL в ЛР3"),
    ("MARL", "Papoudakis G. et al. Benchmarking Multi-Agent Deep RL Algorithms in Cooperative Tasks. NeurIPS 2021 D&B", A + "2006.07869", "3", "единое сравнение 8 MARL-алгоритмов", "SMAC, LBF, RWARE", "return", "набор из 8 алгоритмов и протокол"),
    ("MARL (РФ)", "Skrynnik A. et al. (AIRI) POGEMA: A Benchmark Platform for Cooperative Multi-Agent Pathfinding. 2024", A + "2407.14931", "3", "открытая платформа и протокол кооперативного MARL", "POGEMA", "success rate", "российский пример воспроизводимого протокола"),
    ("Человек в контуре", "Madras D., Pitassi T., Zemel R. Predict Responsibly: Learning to Defer. NeurIPS 2018", A + "1711.06664", "3,5", "модель учится передавать решение эксперту", "классификация", "accuracy, fairness", "роль контроллера «автометка / модератор»"),
    ("Человек в контуре", "Geifman Y., El-Yaniv R. Selective Classification for Deep Neural Networks. NeurIPS 2017", A + "1705.08500", "5", "риск-покрытие, порог по уверенности под заданный риск", "CIFAR, ImageNet", "risk–coverage", "подбор порога под SLA ошибок (ЛР5)"),
    ("Безопасный RL", "Achiam J. et al. Constrained Policy Optimization. ICML 2017", A + "1705.10528", "3", "RL с ограничениями на издержки", "MuJoCo", "return, cost", "SLA ошибок как ограничение (не встроено — ограничение)"),
    ("Безопасный RL", "Gu S. et al. Multi-Agent Constrained Policy Optimisation. 2021", A + "2110.02793", "3", "MARL с ограничениями безопасности", "Safe MAMuJoCo", "reward, cost", "учёт нарушений ограничений"),
    ("Оптимизация", "Kingma D., Ba J. Adam: A Method for Stochastic Optimization. ICLR 2015", A + "1412.6980", "4", "моменты градиента", "CV, NLP", "loss", "лучший оптимизатор для мультимодальной модели"),
    ("Оптимизация", "Loshchilov I., Hutter F. Decoupled Weight Decay Regularization (AdamW). ICLR 2019", A + "1711.05101", "2,4", "раздельный weight decay", "CIFAR, ImageNet", "test error", "исходный оптимизатор ЛР2; sensitivity по wd"),
    ("Оптимизация", "Wilson A. et al. The Marginal Value of Adaptive Gradient Methods in Machine Learning. NeurIPS 2017", A + "1705.08292", "4", "адаптивность ≠ обобщение", "CIFAR, PTB", "test error", "разделение сходимости и обобщения"),
    ("Оптимизация", "Smith S., Le Q. A Bayesian Perspective on Generalization and Stochastic Gradient Descent. ICLR 2018", A + "1710.06451", "4", "шум SGD ∝ lr / batch", "MNIST, CIFAR", "test acc", "чувствительность к batch size"),
    ("Оптимизация (РФ)", "Gorbunov E., Danilova M., Gasnikov A. Stochastic Optimization with Heavy-Tailed Noise via Accelerated Gradient Clipping. NeurIPS 2020", A + "2005.10785", "4", "высоковероятностные оценки при тяжёлых хвостах", "выпуклые задачи, BERT", "high-prob. bounds", "анализ хвостов распределения метрик по seed"),
    ("Оптимизация (РФ)", "Beznosikov A. et al. First Order Methods with Markovian Noise. NeurIPS 2023", A + "2305.15938", "4", "сходимость при марковском шуме", "теория", "сходимость", "сценарий Markov-switching; зависимые объекты"),
    ("Воспроизводимость", "Bouthillier X. et al. Accounting for Variance in Machine Learning Benchmarks. MLSys 2021", A + "2103.03098", "4", "дисперсия источников случайности", "CV, NLP", "std, P(A>B)", "разложение дисперсии, 5–10 seed"),
    ("Калибровка", "Guo C. et al. On Calibration of Modern Neural Networks. ICML 2017", A + "1706.04599", "1-5", "ECE, переуверенность, температурное масштабирование", "CIFAR, ImageNet", "ECE", "метрика калибровки во всех ЛР; перекалибровка при сдвиге"),
    ("Ответственный ИИ", "Gebru T. et al. Datasheets for Datasets. CACM 2021", A + "1803.09010", "5", "стандарт документирования данных", "—", "—", "data card проекта"),
    ("Ответственный ИИ", "Mitchell M. et al. Model Cards for Model Reporting. FAT* 2019", A + "1810.03993", "5", "стандарт карточки модели", "—", "—", "карточка итоговой системы"),
    ("Ответственный ИИ", "NIST. Artificial Intelligence Risk Management Framework (AI RMF 1.0). NIST AI 100-1, 2023", "https://doi.org/10.6028/NIST.AI.100-1", "5", "управление рисками ИИ: govern, map, measure, manage", "—", "—", "карта рисков и роль человека"),
]


def main():
    df = pd.DataFrame(L, columns=["Блок", "Источник", "Ссылка", "ЛР", "Метод / вклад", "Данные / задача", "Метрики", "Использование в проекте"])
    df.insert(0, "№", range(1, len(df) + 1))
    assert df.Источник.is_unique and df.Ссылка.is_unique, "дубликаты"
    df.to_csv(ROOT / "results" / "literature_matrix.csv", index=False)
    df.to_excel(ROOT / "results" / "Матрица_литературы_ЛР5.xlsx", index=False)
    md = ["| № | Источник | Блок | ЛР | Метод / вклад | Использование в проекте |", "|---|---|---|---|---|---|"]
    for r in df.itertuples():
        md.append(f"| {r[1]} | {r.Источник}. [ссылка]({r.Ссылка}) | {r.Блок} | {r.ЛР} | {r[6]} | {r[9]} |")
    (ROOT / "results" / "literature_matrix.md").write_text("\n".join(md))
    print(len(df), df.Блок.value_counts().to_dict())


if __name__ == "__main__":
    main()
