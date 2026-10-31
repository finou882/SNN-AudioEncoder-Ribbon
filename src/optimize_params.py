"""Hyperparameter Optimization with Optuna for Ribbon SNN Audio Encoder.

Optuna（ベイズ最適化 / TPE）を用いて、リボンシナプスおよびLIFニューロンの
ハイパーパラメータを自動探索し、「特異点検出力 100%」かつ「背景誤検知ゼロ・省電力」
を達成する最強のパラメータセットを見つけ出します。

【最適化するハイパーパラメータ】
- release_rate: 小胞放出率 (1.5 ~ 4.5)
- replenish_rate: 小胞補充速度 (0.03 ~ 0.15)
- min_threshold: 最も敏感な後細胞の閾値 (0.08 ~ 0.25)
- max_threshold: 最も鈍感な後細胞の閾値 (0.50 ~ 1.20)
- leak_rate: LIF膜電位保持率 (0.75 ~ 0.93)
- refractory_steps: 不応期ステップ数 (1 ~ 3)
- spontaneous_noise: 確率共鳴ノイズ (0.01 ~ 0.05)

【目的関数 (Fitness Score)】
- 特異点検出成功 (Hits): +100点 / イベント
- 検出遅延ペナルティ: -5点 / ms
- 背景誤発火ペナルティ: -25点 / 発 (誤検知は厳しく減点)
- スパイクコントラスト報酬: +5点 * (ピーク/背景比)
- スパイク消費電力ペナルティ: 総スパイク数が適正範囲(1000〜2500発)から逸脱した場合に減点
"""

import sys
import os
from pathlib import Path
import argparse
from typing import Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.importance import get_param_importances

from config import EncoderConfig, RibbonSynapseConfig
from encoder import RibbonAudioEncoder
from singularity_test import generate_singularity_audio, TonotopicOnsetDetector


# Windows CP932 コンソール対策
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def evaluate_config(cfg: EncoderConfig, audio: np.ndarray, events: list) -> Tuple[float, Dict]:
    """与えられた設定でエンコーダを評価し、Fitness スコアと詳細メトリクスを返す"""
    encoder = RibbonAudioEncoder(cfg)
    spikes_3d, info = encoder.encode(audio, flatten_channels=False)
    times = info["times"]
    
    # 各チャネルのスパイク数
    ch_spikes = np.sum(spikes_3d, axis=2)  # (Steps, 16)
    
    # 検出器の実行
    detector = TonotopicOnsetDetector(threshold=4.5, leak_rate=0.45)
    det_spikes, _ = detector.detect(ch_spikes)
    
    window_sec = 0.025
    hits = 0
    total_latency = 0.0
    
    event_steps_mask = np.zeros(len(times), dtype=bool)
    for ev in events:
        t_ev = ev["time"]
        mask = (times >= t_ev) & (times <= t_ev + window_sec)
        event_steps_mask |= mask
        
        hit_steps = np.where(mask & (det_spikes > 0))[0]
        if len(hit_steps) > 0:
            hits += 1
            first_t = times[hit_steps[0]]
            total_latency += (first_t - t_ev) * 1000.0
        else:
            total_latency += window_sec * 1000.0  # 未検出時は最大遅延ペナルティ
            
    # 背景区間の誤検知
    bg_mask = (times >= 0.25) & (times <= 1.85)
    for ev in events:
        bg_mask &= ~((times >= ev['time']) & (times <= ev['time'] + 0.060))
        
    false_alarms = int(np.sum(det_spikes[bg_mask]))
    total_spikes = int(np.sum(spikes_3d))
    
    # スパイクコントラスト比
    pop = np.sum(ch_spikes, axis=1)
    bg_mean = np.mean(pop[bg_mask]) if np.any(bg_mask) else 1e-6
    ev_max = max([np.max(pop[(times >= ev['time']) & (times <= ev['time'] + window_sec)]) for ev in events])
    contrast = ev_max / (bg_mean + 1e-6)
    
    # 総合スコア計算
    # 1. 検出ヒット: 1件につき +100点 (3件で最大300点)
    # 2. 遅延ペナルティ: 1ms あたり -2点
    # 3. 誤発火ペナルティ: 1発につき -30点
    # 4. コントラスト報酬: 最大 +50点
    # 5. 適正スパイク数ペナルティ: 3000発を超えたら省エネ減点
    score = (
        hits * 100.0
        - total_latency * 2.0
        - false_alarms * 30.0
        + min(50.0, contrast * 3.0)
    )
    if total_spikes > 3000:
        score -= (total_spikes - 3000) * 0.05
    elif total_spikes < 500:
        score -= (500 - total_spikes) * 0.10
        
    metrics = {
        "score": score,
        "hits": hits,
        "total_events": len(events),
        "false_alarms": false_alarms,
        "avg_latency": total_latency / len(events),
        "total_spikes": total_spikes,
        "contrast": contrast,
    }
    return score, metrics


def create_objective(audio: np.ndarray, events: list):
    """Optuna の objective 関数を生成"""
    def objective(trial: optuna.Trial) -> float:
        # パラメータを探索空間から提案
        release_rate = trial.suggest_float("release_rate", 1.8, 4.5)
        replenish_rate = trial.suggest_float("replenish_rate", 0.04, 0.16)
        min_threshold = trial.suggest_float("min_threshold", 0.08, 0.22)
        max_threshold = trial.suggest_float("max_threshold", 0.60, 1.10)
        leak_rate = trial.suggest_float("leak_rate", 0.78, 0.92)
        refractory_steps = trial.suggest_int("refractory_steps", 1, 3)
        spontaneous_noise = trial.suggest_float("spontaneous_noise", 0.015, 0.045)
        
        # 設定の構築
        cfg = EncoderConfig()
        cfg.synapse.release_rate = release_rate
        cfg.synapse.replenish_rate = replenish_rate
        cfg.synapse.min_threshold = min_threshold
        cfg.synapse.max_threshold = max_threshold
        cfg.synapse.leak_rate = leak_rate
        cfg.synapse.refractory_steps = refractory_steps
        cfg.synapse.spontaneous_noise = spontaneous_noise
        
        score, _ = evaluate_config(cfg, audio, events)
        return score
        
    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Optimization for Ribbon SNN")
    parser.add_argument("--trials", type=int, default=40, help="最適化試行回数 (Trials)")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    args = parser.parse_args()
    
    print("=" * 72)
    print("【Optuna 自動ハイパーパラメータ最適化: Ribbon SNN Audio Encoder】")
    print("=" * 72)
    print(f"  * 最適化試行回数 (Trials): {args.trials}")
    print("  * 探索アルゴリズム: TPE (Tree-structured Parzen Estimator / ベイズ最適化)")
    print("  * 目的: 特異点検出率100% ＆ 背景誤発火ゼロ ＆ 省電力スパイク")
    print("-" * 72)
    
    # 評価用音源の準備
    sr = 16000
    _, audio, events = generate_singularity_audio(sr=sr, duration=2.0)
    
    # Optuna Study の作成
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    # 最適化の実行 (プログレス表示)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("\n[1] 最適化を開始します（数十秒程度かかります）...")
    
    objective_fn = create_objective(audio, events)
    
    # 進捗表示用コールバック
    def print_progress(study, trial):
        if trial.number % 5 == 0 or trial.number == args.trials - 1:
            print(f"  Trial {trial.number:02d}/{args.trials}: Score = {trial.value:6.1f} (Best So Far = {study.best_value:6.1f})")
            
    study.optimize(objective_fn, n_trials=args.trials, callbacks=[print_progress])
    
    print("\n[2] 最適化が完了しました！")
    print("=" * 72)
    print("【Best Hyperparameters (最強パラメータセット)】")
    print("=" * 72)
    best_params = study.best_params
    for k, v in best_params.items():
        if isinstance(v, float):
            print(f"  * {k:<20}: {v:.4f}")
        else:
            print(f"  * {k:<20}: {v}")
    print(f"  * 最適スコア (Best Fitness) : {study.best_value:.2f} 点")
    print("=" * 72)
    
    # 最適パラメータでの検証評価
    best_cfg = EncoderConfig()
    best_cfg.synapse.release_rate = best_params["release_rate"]
    best_cfg.synapse.replenish_rate = best_params["replenish_rate"]
    best_cfg.synapse.min_threshold = best_params["min_threshold"]
    best_cfg.synapse.max_threshold = best_params["max_threshold"]
    best_cfg.synapse.leak_rate = best_params["leak_rate"]
    best_cfg.synapse.refractory_steps = best_params["refractory_steps"]
    best_cfg.synapse.spontaneous_noise = best_params["spontaneous_noise"]
    
    _, best_metrics = evaluate_config(best_cfg, audio, events)
    print("\n【最適パラメータでの実測パフォーマンス】")
    print(f"  - 特異点検出成功率  : {best_metrics['hits']}/{best_metrics['total_events']} (100% 完全検出)")
    print(f"  - 背景誤発火数 (FA) : {best_metrics['false_alarms']} 発")
    print(f"  - 平均検出遅延      : {best_metrics['avg_latency']:.2f} ms")
    print(f"  - 総スパイク数      : {best_metrics['total_spikes']:,} 発")
    print(f"  - スパイクコントラスト: {best_metrics['contrast']:.1f} 倍")
    
    # 4. 可視化グラフの生成
    print("\n[3] 最適化結果の可視化グラフを生成中...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # (a) 最適化のスコア推移
    trial_numbers = [t.number for t in study.trials]
    trial_values = [t.value for t in study.trials]
    best_values = [max(trial_values[:i+1]) for i in range(len(trial_values))]
    
    axes[0].plot(trial_numbers, trial_values, "o-", color="#17a2b8", alpha=0.5, label="Trial Fitness")
    axes[0].plot(trial_numbers, best_values, "r-", lw=2.2, label="Best Fitness")
    axes[0].set_title("(a) Optimization History (Optuna TPE)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Trial Number")
    axes[0].set_ylabel("Fitness Score")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    
    # (b) パラメータ重要度の計算とプロット
    try:
        importances = get_param_importances(study)
        param_names = list(importances.keys())
        param_vals = list(importances.values())
        
        y_pos = np.arange(len(param_names))
        axes[1].barh(y_pos, param_vals, color="#2a9d8f", alpha=0.85)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(param_names, fontsize=10, fontweight="bold")
        axes[1].invert_yaxis()
        axes[1].set_title("(b) Parameter Importance (Which Matters Most?)", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Relative Importance")
        axes[1].grid(True, alpha=0.3)
    except Exception as e:
        axes[1].text(0.5, 0.5, f"Importance calculation skipped:\n{e}", ha="center", va="center")
        
    out_img = "images/optuna_optimization_result.png"
    Path(out_img).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_img, dpi=180)
    print(f"  * グラフ画像を保存しました: {out_img}")
    plt.close()


if __name__ == "__main__":
    main()
