# Close-proximity high-speed Ball (football) detection system model for Indoor training scenarios

## Abstract
> **Совет:** Пишется в самом конце. Должен содержать 4 предложения: 1. Проблема (motion blur, latency in indoor sports, неприменимость open-source данных). 2. Наше решение (Custom Dataset + Sport-YOLO v2 + Hardware-aware pipeline). 3. Методология (Dynamic calibration, TensorRT, удаление макро-слоев, NWD Loss). 4. Главный результат (достигнутый FPS, миллисекунды задержки, NWD-mAP, точность в 3D).

## 1. Introduction
* Context: The necessity of accurate 3D ball tracking in indoor football training scenarios (Footbonaut arena).
* The Problem: High-speed movement in close proximity causes severe motion blur ("ghost balls"). Standard Horizontal Bounding Boxes (HBB), traditional CV models, and models trained on broadcast-camera datasets fail or introduce unacceptable latency.
* Proposed Solution: An end-to-end hardware-aware system (Sport-YOLO) with a custom empirically filtered dataset.

## 2. Related Work
* Evolution of YOLO architectures (up to YOLO26 NMS-free design) and transformers like RT-DETR/RF-DETR.
* Existing multi-camera tracking systems and their limitations in budget and latency.
* Limitations of open-source sports datasets for close-proximity indoor tracking.

## 3. System Architecture and Hardware
> **Совет:** Вставьте сюда диаграмму `Preprocess -> Inference -> Algorithms -> Postprocess`.
* **Edge Computing:** System designed for NVIDIA Jetson and Tang nano 20k implementation per camera.
* **Threading Pipeline:** Asynchronous buffer design for zero-bottleneck execution. Semaphore synchronization.
* **Data Flow:** JSON output metrics delivered to a local FastAPI server for metadata exchange and User monitoring (UI).

## 4. Methodology
### 4.1. Custom Dataset Engineering & Justification
* **Broadcast vs. Close-Proximity:** Why massive open-source sports datasets fail in Footbonaut environments. These datasets predominantly feature broadcast/stadium views with distant objects, whereas our scenario involves high-speed, close-proximity indoor trajectories that cause severe motion blur.
* **Dataset Cleaning:** Targeted annotation and manual filtering of edge cases (e.g., overlapping with player limbs) to eliminate "ghost ball" artifacts that open-source models usually struggle with.
* **Hyperparameter Optimization:** Resolution, Batch Size, Learning Rate, and aggressive motion blur augmentation.

### 4.2. Sport-YOLO v2 (Vision Engine)
Архитектура Sport-YOLO v2 представляет собой глубоко модифицированную, строго пространственную (spatial-only) модель, оптимизированную для высокоскоростной детекции микро-объектов. Мы полностью отказались от макро-слоев (P4/P5), сместив фокус на высокоразрешающие карты признаков P2 (stride-4) и P3 (stride-8).

**4.2.1. Hybrid Geometric Stem (Invariant + Empirical Feature Extraction)**

Для повышения устойчивости к освещению и размытию, мы предлагаем **гибридный Geometric Stem** — двухветвевую архитектуру, объединяющую фиксированные математические фильтры с эмпирическими ядрами, извлеченными из обученной модели YOLO26.

**Branch A: Mathematical Filter Bank (Depthwise).** Входной тензор $\mathbf{X} \in \mathbb{R}^{B \times 9 \times H \times W}$ обрабатывается через фиксированные банки фильтров (depthwise convolution, $1 \times 7 \times 7$ на каждый канал). Мы применяем фильтры Габора ($G_{\theta}$) для выделения направленных текстур при $\theta \in \{0, \frac{\pi}{4}, \frac{\pi}{2}, \frac{3\pi}{4}\}$:
$$G(x,y; \theta) = \exp\left(-\frac{x_r^2 + (\gamma y_r)^2}{2\sigma^2}\right) \cos\left(2\pi \frac{x_r}{\lambda}\right)$$
Для изотропной детекции пятен (blob detection) применяются фильтры Лапласиана Гауссиана (LoG) при $\sigma \in \{1.5, 2.5\}$:
$$L(x, y; \sigma) = - \frac{1}{\pi\sigma^4} \left( 1 - \frac{x^2+y^2}{2\sigma^2} \right) \exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)$$
Эти 6 фильтров, применённые к каждому из 9 каналов, дают $C_{\text{gabor}} = 54$ выходных каналов. Веса **никогда не обновляются** (`requires_grad=False`).

**Branch B: Empirical YOLO Kernels (Standard Conv).** Параллельная ветвь использует стандартную (не depthwise) свертку `Conv2d(9, N, 3)` для сохранения кросс-канальных цветовых корреляций. Ключевое наблюдение: depthwise фильтры ($1 \times k \times k$) не способны захватить паттерны, зависящие от RGB-взаимодействий (например, "белый мяч на зелёном поле"). Стандартная свертка с весами $3 \times k \times k$ сохраняет эти кросс-канальные зависимости.

Ядра извлекаются из обученной модели YOLO26s conv0 ($32 \times 3 \times 3 \times 3$) с помощью standalone-инструмента `extract_kernels.py`, который вычисляет для каждого фильтра:
1. **L2-норму** $\|\mathbf{W}_i\|_2$ — для удаления "мертвых" фильтров;
2. **Радиальную симметрию** — степень изотропности паттерна;
3. **Энтропию** — информативность пространственного распределения весов;
4. **Gabor-similarity** $r_i$ — максимальную корреляцию с банком стандартных фильтров Габора.

Полученные ядра нормализуются по амплитуде для совместимости с масштабом Gabor/LoG фильтров и реплицируются $3 \times 3 \rightarrow 9$ каналов для 3-frame RGB стека.

**Выбор $N=8$ из 32 фильтров YOLO26 conv0 обоснован тремя факторами:**

*(i) Порог информационной избыточности.* Из 32 фильтров conv0 обученной YOLO26s: 3 оказались "мертвыми" ($\|\mathbf{W}\|_2 < 0.5$, веса обнулились в ходе тренировки), 21 — Gabor-избыточными ($r_i \geq 0.65$, паттерны дублируют существующий банк математических фильтров). Оставшиеся 11 уникальных фильтров имеют $r_i \in [0.28, 0.70]$. Из них отсекаются 3 пограничных ($r_i > 0.65$): F5 ($r=0.696$), F7 ($r=0.663$ — высокая коллинеарность с F30), F22 ($r=0.688$). Финальные 8 фильтров ($r_i \in [0.28, 0.65]$) **уверенно** отличаются от математического банка и покрывают все основные семантические категории: радиальный градиент кривизны мяча (F4, F26), направленный motion blur (F10, F30), текстурные паттерны (F12, F16), а также два максимально уникальных ядра (F17: $r=0.28$, F10: $r=0.37$).

*(ii) Эффективность в режиме малого числа параметров.* Для nano-модели (549K параметров), $1 \times 1$ mix-слой проецирует $(54 + N) \rightarrow 16$ каналов. Каждый дополнительный эмпирический канал добавляет 16 весов к mix-слою. При $N=8$ добавляется 128 дополнительных mix-весов (суммарно +776 параметров, 0.14%). При $N=11$ — 176 весов. Маргинальный информационный выигрыш от трёх пограничных фильтров (все $r \approx 0.7$) не оправдывает даже этих 48 дополнительных параметров, особенно при ортогональной инициализации mix-слоя.

*(iii) Аппаратное выравнивание.* GPU наиболее эффективно обрабатывают тензоры, когда измерение каналов кратно 8. Существующий depthwise выход — 54 канала. Добавление 8 даёт 62-канальный промежуточный тензор, который $1 \times 1$ mix немедленно проецирует в $C_0 = 16$ (кратно 16 для Tensor Cores).

**Комбинирование и обучение.** Выходы обеих ветвей конкатенируются:
$$\mathbf{X}_{\text{hybrid}} = [\text{Gabor}(\mathbf{X}) \parallel \text{LoG}(\mathbf{X}) \parallel \text{Empirical}(\mathbf{X})]$$
и смешиваются через $1 \times 1$ параметрическую свертку: $\mathbf{X}_{mix} = \text{SiLU}(\text{BN}(\mathbf{W}_{1\times 1} * \mathbf{X}_{\text{hybrid}}))$. Эмпирические ядра **замораживаются** на первой фазе обучения (Phase 1, `freeze_epochs`), затем размораживаются с дифференциальным learning rate во второй фазе. Mix-слой подвергается ортогональной инициализации через SVD, обеспечивая разнообразие комбинаций фильтров обеих ветвей.

**Ablation Plan.** Влияние выбора $N$ на NWD-mAP будет валидировано через 3-строчную аблацию: $N=0$ (baseline без эмпирической ветви), $N=8$ (консервативный порог), $N=11$ (все уникальные). Если $\text{mAP}(N=8) \approx \text{mAP}(N=11)$, порог $r < 0.65$ признаётся оптимальным; если $N=11$ значительно выше, порог корректируется.

**4.2.2. Blur Resilience via Deformable Convolutions (DCNv4)**
Движение мяча на высоких скоростях вблизи камеры создает сильный "motion blur", деформируя форму объекта до вытянутого эллипса. В Sport-YOLO v2 мы заменили стандартные $3 \times 3$ свертки в Residual Bottleneck на PyTorch-native Deformable Convolutions (DCNv4).
Для стандартной сетки $\mathcal{R}$, выходная точка $p_0$ вычисляется с учетом динамически обучаемого поля смещений $\Delta p_n$:
$$y(p_0) = \sum_{p_n \in \mathcal{R}} w(p_n) \cdot x(p_0 + p_n + \Delta p_n)$$
Это позволяет рецептивному полю сети динамически "обволакивать" размытый контур мяча.

**4.2.3. Normalized Wasserstein Distance (NWD) Loss**
Для микро-объектов сдвиг всего на 1-2 пикселя приводит к коллапсу значения метрики IoU. Sport-YOLO v2 моделирует Bounding Boxes как 2D-распределения Гаусса.
Дистанция Вассерштейна второго порядка ($W_2$) между предсказанным ($\mu_p, w_p, h_p$) и целевым ($\mu_t, w_t, h_t$) распределениями определяется как:
$$W_2^2 = \|\mu_p - \mu_t\|_2^2 + \frac{(w_p - w_t)^2 + (h_p - h_t)^2}{4}$$
Финальная функция потерь NWD нормализуется через экспоненциальное отображение:
$$\mathcal{L}_{NWD} = 1 - \exp\left(-\frac{\sqrt{W_2^2}}{C}\right)$$
Где $C = 12.8$ — константа нормализации. Эта функция обеспечивает стабильные градиенты даже при отсутствии перекрытия боксов.

**4.2.4. Optimization via MuSGD and Stiefel Manifold**
Веса сверточных слоев инициализируются на многообразии Штифеля с условием ортогональности $\mathbf{W} \mathbf{W}^T = \mathbf{I}$. Гибридный оптимизатор MuSGD ортогонализирует 2D градиенты $\mathbf{G}_{2D}$ через 5-шаговые итерации Ньютона-Шульца, что обеспечивает строгую ротационную эквивариантность.

### 4.3. Adaptive Tracking & Resource Management
Для обеспечения работы системы в реальном времени (100+ FPS) на edge-устройствах с ограниченным вычислительным бюджетом, мы разработали асинхронный пайплайн с динамическим управлением ресурсами.

**4.3.1. 3D Kinematic Tracking and Velocity Estimation (Tracker3D)**
Модуль `Tracker3D` вычисляет истинную метрическую скорость мяча в физическом пространстве. Имея историю 3D-координат с временными метками $\Delta t$, сырой вектор скорости вычисляется через евклидово расстояние:
$$v_{raw} = \frac{\sqrt{(X_t - X_{t-1})^2 + (Y_t - Y_{t-1})^2 + (Z_t - Z_{t-1})^2}}{\Delta t}$$
Поскольку детекции на высоких скоростях подвержены микро-шумам (jitter), мы применяем фильтрацию порога (`jitter_thresh`) и экспоненциальное скользящее среднее (EMA) для сглаживания вектора скорости:
$$V_t = \alpha_{vel} \cdot v_{raw} + (1 - \alpha_{vel}) \cdot V_{t-1}$$
Параллельно, дискретный Фильтр Калмана оценивает линейную скорость и ускорение, предсказывая положение мяча на следующем кадре, что эффективно компенсирует кратковременные перекрытия (occlusions).

**4.3.2. Velocity-Based Adaptive Frame Skipping**
Традиционные системы запускают тяжелый инференс нейросети на каждом кадре. В нашей системе частота детекции динамически адаптируется на основе вычисленного вектора скорости. Если мяч движется с высокой скоростью по предсказуемой баллистической траектории, система полностью пропускает вызов Sport-YOLO (экономя циклы GPU) и использует координаты, предсказанные Фильтром Калмана. Инференс возобновляется при замедлении объекта или смене направления.

**4.3.3. Dynamic Region of Interest (ROI)**
Основываясь на предсказаниях Калмана и эмпирически откалиброванном размере мяча (зависящем от глубины $Z$), система вырезает динамический Region of Interest (ROI). Это снижает размерность входного тензора для Sport-YOLO, радикально сокращая количество GFLOPs на кадр.

**4.3.4. Budget-Efficient Camera Switching Algorithm**
В мульти-камерной конфигурации ("Garage" 4-Camera System) наш алгоритм адаптивного переключения оценивает пространственное положение мяча и историческую точность детекций в каждой зоне, чтобы динамически активировать только оптимальную пару (или тройку) камер для триангуляции. Неактивные узлы переходят в режим ожидания, снижая общее энергопотребление системы и избегая узких мест в сети.

## 5. Spatial 3D Calibration and Depth Estimation
Для обеспечения высокоточного трекинга мяча в физических координатах (метрическая система) в условиях арены Footbonaut, наша система использует многокамерную геометрическую модель. Процесс перевода 2D-координат пикселей в единое 3D-пространство состоит из нескольких математически строгих этапов.

### 5.1. Camera Projection Model and Extrinsic Calibration
Для каждой камеры $i$ преобразование 3D-точки физического мира $\mathbf{X} = [X, Y, Z, 1]^T$ в 2D-координаты изображения $\mathbf{x}_i = [u_i, v_i, 1]^T$ описывается моделью пинхол-камеры:
$$s \mathbf{x}_i = \mathbf{K}_i [\mathbf{R}_i | \mathbf{t}_i] \mathbf{X} = \mathbf{P}_i \mathbf{X}$$
Первичная оценка вектора вращения и трансляции вычисляется с использованием алгоритма **Perspective-n-Point (PnP)** в комбинации с **RANSAC** для фильтрации выбросов от статических маркеров.

### 5.2. N-View DLT Triangulation
Для каждой камеры $i$ с матрицей проекции $\mathbf{P}_i \in \mathbb{R}^{3 \times 4}$, векторное произведение $\mathbf{x}_i \times (\mathbf{P}_i \mathbf{X}) = \mathbf{0}$ дает два линейно независимых уравнения:
$$u_i (\mathbf{p}_{i,3}^T \mathbf{X}) - (\mathbf{p}_{i,1}^T \mathbf{X}) = 0$$
$$v_i (\mathbf{p}_{i,3}^T \mathbf{X}) - (\mathbf{p}_{i,2}^T \mathbf{X}) = 0$$
Оптимальное решение для $\mathbf{X}$ находится через Сингулярное Разложение (SVD) матрицы $\mathbf{A}$, где решением является собственный вектор, соответствующий наименьшему сингулярному значению.

### 5.3. Global Bundle Adjustment and Optimization
Для минимизации накопленной ошибки мы применяем **Global Bundle Adjustment (GBA)**. Задача заключается в минимизации общей ошибки репроекции (reprojection error) для всех наблюдаемых точек $j$ на всех камерах $i$:
$$\min_{\mathbf{P}_i, \mathbf{X}_j} \sum_{i=1}^N \sum_{j=1}^M v_{ij} \| \mathbf{x}_{ij} - \pi(\mathbf{P}_i, \mathbf{X}_j) \|^2$$
Где $v_{ij} = 1$, если точка $j$ видна на камере $i$ (иначе $0$), а $\pi$ — функция проекции. Эта нелинейная задача наименьших квадратов решается с использованием оптимизатора **Levenberg-Marquardt**.

### 5.4. Dynamic Anchoring and Affine Alignment
*Novelty:* В сценариях высокой динамики статической калибровки недостаточно. Мы используем сам высокоскоростной мяч в качестве **Динамической Виртуальной Калибровочной Мишени (Dynamic Virtual Calibration Target)**. Чтобы идеально выровнять полученное 3D-облако точек с физической базой арены (7.07m baseline), применяется SVD-основанный **алгоритм Кабша (Kabsch algorithm)** и аффинная трансформация.
Для набора предсказанных центроидов $\mathbf{X}_{pred}$ и истинных физических координат $\mathbf{X}_{true}$, мы находим оптимальную матрицу вращения $\mathbf{R}_{opt}$ и вектор смещения $\mathbf{t}_{opt}$:
$$\min_{\mathbf{R}, \mathbf{t}} \sum_{k} \| (\mathbf{R} \mathbf{X}_{pred, k} + \mathbf{t}) - \mathbf{X}_{true, k} \|^2$$
Это обеспечивает финальную подгонку метрического масштаба с точностью RMSE < 0.1м.

## 6. Experiments and Results
### 6.1. Dataset Evaluation (Ablation Study 1)
* Empirical comparison of the current industry baseline model (**YOLOv11s**) trained on prominent open-source datasets (ISSIA-CNR, SoccerNet-Tracking soccernet v3 h250, DeepSportLab, DFL Bundesliga Data Shootout, NPSPT (Non-Professional Soccer Player Tracking)) vs. Our Custom Dataset.
* "Для оценки обобщающей способности моделей (Baseline Evaluation) мы использовали SoccerNet_v3_H250 — стандартизированное подмножество SoccerNet-v3, оптимизированное для задач детекции в формате YOLO. Это позволило провести честное сравнение современных архитектур (YOLOv11, v12, v26) на репрезентативном объеме данных спортивного вещания перед их тестированием в специфических условиях indoor-арены". 
* **Justification of Custom Data:** Proving that modern models trained on broadcast-view datasets suffer massive performance drops in close-proximity scenarios due to motion blur and scale variance, highlighting the necessity of our custom dataset engineering.

#### 6.1.1. Training Setup and Hyperparameter Justification

To ensure a methodologically rigorous and fair comparison between the broadcast-view baseline dataset (SoccerNet-v3 H250) and our custom close-proximity dataset, we established a strictly controlled training environment designed to maximize the baseline model's theoretical generalization capabilities. Training was conducted over 100 epochs with an early stopping mechanism (`patience=20`), a batch size of 16, and 4 data loader threads (`workers=4`) on a dedicated GPU accelerator (`device=1`). To prevent the sub-pixel degradation of micro-objects (the football) when applying spatial augmentations, the input resolution was deliberately increased to `imgsz=960`.

To eliminate optimization bias—which could arise from discrepancies in dataset volumes if built-in heuristics (such as `optimizer='auto'`) were used—we strictly fixed the optimizer to AdamW with an initial learning rate of 0.001 and cosine annealing (`cos_lr=True`). AdamW provides superior weight regularization, which is essential when utilizing aggressive spatial distortions.

To bridge the domain gap, we implemented an intensive augmentation pipeline. Employing Mosaic augmentation with a probability of 1.0 guaranteed scale invariance and forced the network to learn the local geometry of the object rather than the global pitch context. Additional robustness to occlusions (e.g., the ball overlapping with player equipment) and scale variance was ensured by configuring `mixup=0.1`, `scale=0.5`, and `fliplr=0.5`. 

Finally, to provide the broadcast dataset with a theoretical chance to adapt to the high-speed conditions of the Footbonaut arena, we introduced an offline pre-processing stage. Using the Albumentations library, a physically accurate directed Motion Blur filter (kernel size ranging from 7 to 15 pixels, probability 0.5) was applied to the training set. This effectively simulated camera shutter-speed artifacts and the elliptical deformation ("ghost balls") characteristic of our close-proximity scenario. Such strict hyperparameter fixation guarantees that any degradation in the baseline model's accuracy on our test set is exclusively attributable to an insurmountable physical domain shift, rather than insufficient hyperparameter tuning or architectural limitations.

### 6.2. Baseline Model Selection (Ablation Study 2)
* **Table 1: Architecture Comparison:** Evaluation of state-of-the-art architectures (RT-DETR, RF-DETR, YOLOv11s, YOLOv12s, YOLOv26s) trained and tested purely on our custom dataset.
* **Metrics:** mAP@50, mAP@50-95, Recall, Params (M), GFLOPs, Latency PyTorch (ms).
* **Justification:** While YOLO11s/12s show strong results, YOLOv26s is selected as the foundation architecture for Sport-YOLO due to its NMS-free (One-to-One) design, providing stable and fast inference without post-processing bottlenecks.

### 6.3. Sport-YOLO Hardware-Aware Performance
* **Table 2: Sport-YOLO Superiority:** Direct comparison between Baseline YOLO26s and Sport-YOLO v2 on target edge hardware (NVIDIA Jetson).
* **Metrics:** Standard mAP50, NWD-mAP, Params (M), GFLOPs, TensorRT FP16 Latency (ms).
* **Hardware Efficiency:** Demonstrating a massive reduction in Params (~11M to ~6.4M) and GFLOPs by dropping redundant macro-layers (P4/P5), pushing TensorRT FP16 latency under 1 ms.
* **Accuracy Boost:** Highlighting the significant increase in NWD-mAP, proving that DCNv4 and NWD Loss perfectly handle motion blur and micro-object pixel shifts.
* **Graph:** Compare sportYOLO vs YOLO26 with different sizes (n,s,m,l,x). You may extract a graph from official Ultralytics blogs (coco map vs latency graphs) and insert on top of that sportyolo metrics.

### 6.4. Key Visualizations (Ablation & Trade-offs)
* **Scatter Plot (mAP vs. Latency):** Pareto Front visualization showing Sport-YOLO in the top-left corner (fastest and most accurate) compared to standard models.
* **Ablation Bar Chart:** Step-by-step performance contribution of Sport-YOLO components: Base YOLO26s -> + P2 Head -> + DCNv4 -> + NWD Loss -> + Hybrid Stem ($N=0$ vs $N=8$ vs $N=11$).
* **Visual Comparison ("Ghost Balls"):** Grid of high-speed blurred ball captures, comparing false positives/misses of the baseline model against the accurate elliptical bounding boxes of Sport-YOLO.

### 6.5. System-Level 3D Tracking Evaluation
* End-to-End latency of the entire asynchronous pipeline.
* 3D localization Accuracy (RMSE < 0.1m).
* Effectiveness of the adaptive camera switch and frame skipping in maintaining high 100+ FPS system-wide.

## 7. Conclusion
* Summary of the empirical findings (data importance) and Sport-YOLO system capabilities.
* Future work (e.g., kinematic target prediction, expanding to multi-ball scenarios or other sports).
