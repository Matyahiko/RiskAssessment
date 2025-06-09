# Workflow Guide

このドキュメントでは `RiskAssessment` プロジェクトの基本的な処理フローを説明します。
データの前処理からモデルトレーニング、評価までの手順をまとめています。

## 1. データ準備
`risk_pipeline/utils.py` で指定されている `DATA_DIR` に入力 CSV を配置します。
各行には 1536 次元の `embedding` 列が含まれている必要があります。

## 2. 産業別クラスタリング
```
python cluster_industry.py
```
各産業ごとにUMAP + HDBSCANを用いたクラスタリングを実施し、結果は `ART_DIR` 以下に保存されます。

## 3. 可視化
```
python visualize_industry.py
```
クラスタリング結果を散布図やリスクラベルの分布として出力します。

## 4. LightGBM ハイパーパラメータ探索
```
python tune_lightgbm.py
```
クラスタをクラスラベルとして各産業のモデルのハイパーパラメータをOptunaで探索します。

## 5. モデル学習
```
python train_final.py
```
探索したパラメータを用いて最終的な LightGBM モデルを学習し、モデルファイルを保存します。

## 6. 評価
```
python evaluate.py
```
交差検証による Accuracy/Macro F1 スコアを計算し、混同行列やレポートを出力します。

## 7. SHAP 可視化
```
python shap_industry_visuals.py
```
モデル解釈のために特徴量ごとの SHAP 値を可視化し、ヒートマップや要約グラフを生成します。

## 8. 拡張に向けて
- `risk_pipeline/utils.py` の定数を変更することでデータ保存先を変更できます。
- 追加の産業コードや特徴量を扱う際は `TARGET_CODES` や `FIN_COLS` を編集してください。
- その他のツールや前処理スクリプトを追加する場合は `docs/` にドキュメントを残すようにしてください。

