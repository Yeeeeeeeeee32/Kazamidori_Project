# Kazamidori

モデルロケットの飛行軌道・着地点を予測し、風・推力の不確実性を含むモンテカルロ計算と発射角探索を行うデスクトップアプリです。
画面はPySide6／Matplotlib、飛行計算はRocketPyを使用します。

## 起動とテスト

プロジェクト直下で、起動に使用するPythonと同じ環境へ依存関係をインストールします。

```powershell
python -m pip install -r requirements.txt -c constraints-tested.txt
python main_qt.py
```

[constraints-tested.txt](constraints-tested.txt)はPython 3.10で確認した直接依存の組合せです。全OS・全推移的依存を固定したロックファイルではありません。

```powershell
python -m pip install -r requirements-dev.txt -c constraints-tested.txt
python -B -X utf8 -m pytest -q --disable-warnings
```

テストは回帰・数式・数値計算・ファイル入出力・オフスクリーンのQt画面を扱います。実飛行との精度比較、Koinoboriの実機校正とは区別します。

## 基本操作

1. Load Airframeで機体の`.ork`／`.ork.gz`を手動選択します。取得できない値はManual Configで補完します。
2. Motor Specificationsでモータ設定と推力CSV、回収系でパラシュート設定を指定します。機体名から特定のファイルを自動選択しません。
3. 地点・発射条件・風・モード・MC条件を確認します。予報は事前取得したCSVも使用できます。
4. Run（F5）で実行します。自由モードは指定角度で計算、それ以外は発射角探索とMC検証を行います。Stop（Esc）は協調取消を要求します。
5. 軌道・着地分布・判定理由を確認し、必要に応じて結果JSON／CSV／PNGを保存します。

現在の地表テレメトリは練習用です。**Koinoboriは製作中**で、実機通信は未接続です。
GOはモデル・入力条件・有限回の試行に対する判定で、実世界の成功確率の保証ではありません。

## 文書と構成

- [開発計画](ROADMAP.md)：優先順位、次工程、保留している研究。
- [技術メモ](技術メモ/README.md)：現行仕様・制約・設計案の分野別索引。
- `main_qt.py`：起動と状態・画面・Controllerの構築。
- `core/`：機体、飛行、風、統計、最適化の計算。
- `ui_qt/`：状態、画面、操作の調停、計算ワーカー。
- `utils/`：機体ファイル、設定、結果、予報、地図、座標変換。
- `tests/`：自動テストと入力fixture。`assets/`：地図等の資産・検証用データ。

文書は最新版を上書き更新し、変更履歴・日付別監査報告を併記しません。現行仕様と未実装案は各文書内で区別します。
