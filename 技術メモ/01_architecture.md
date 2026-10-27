# アーキテクチャ

更新日：2026-10-27。状態：現行実装。

## 責務と処理経路

| 領域 | 担当 |
| --- | --- |
| `main_qt.py` | AppState、AppWindow、SimControllerを構築し起動 |
| `ui_qt/app_state.py` | 入力、結果、風履歴、監視基準を保持しSignalで通知 |
| `ui_qt/app_window.py`、`map_view.py` | Widgets／Matplotlibで表示・編集。共通AppStateを参照 |
| `ui_qt/sim_controller.py` | 読込・検証・実行・保存・判定・UI更新を調停 |
| `ui_qt/workers.py` | QThread内で飛行計算、探索、MC、統計、結果包装 |
| `core/simulation.py` | RocketPy環境、モータ、機体、開傘・着地イベント |
| `core/optimization.py`、`annealing.py` | 焼きなまし探索と候補の独立MC検証 |
| `core/monte_carlo.py`、`wind_model.py` | 風摂動、統計領域、風変換・監視補助 |
| `core/montblanc.py` | 着地直前の水平速度・変動幅・評価時間の判定 |
| `utils/` | 機体、設定、結果、気象CSV、地図、磁気偏角、座標変換 |

入力→AppState→Controllerの実行スナップショット→Worker→core→結果通知→AppState→描画という経路です。
計算結果は入力版・run_idと対応させ、古い完了結果を採用しません。
計算中は入力をロックし、プログラム経由の変更も版で検出します。風の更新は計算中も継続します。

## 単位・座標・時刻

- 内部はSI：m、kg、s、N、Pa。温度は渡し先の仕様に応じて℃／Kを明示する。
- 軌道と着地点は発射点を原点としたENU。X＝East、Y＝North、Z＝Up。
- 機体の軸方向位置はノーズ先端基準。モータ座標と混同しない。
- 地理座標との変換は`utils/geo_math.py`を共用する。緯度・経度はRocketPyの環境条件にも渡す。
- 風向は北0度・時計回りのFROM。楕円描画角は東基準・反時計回り。
- 表示はJST、API・保存日時はUTC。経過時間の評価には単調増加時計を使う。

## 非同期処理と結果の有効性

GUIの更新はQt Signal経由で行います。CPUを使うMCは現状QThread内の逐次処理です。
Stop／Closeは協調取消であり、積分・HTTP処理の途中を強制終了しません。Closeは稼働中のWorkerの終了を非同期に待ちます。
入力変更で旧GOを解除します。結果を書き出す条件は現在の入力欄ではなく実行時スナップショットから取得します。
表示レイヤーやカメラの変更は、保持した結果の再描画であり新規の飛行計算ではありません。

## 開発時の確認

対応する数式・物理・データ契約をテストと関連付けます。解析ケース、実Flightのイベント、異常入力、保存失敗、取消・終了を確認します。
実装を同じ式でなぞる試験だけに頼らず、独立した期待値や保存則を使います。
導入依存の確認版は[constraints-tested.txt](../constraints-tested.txt)、試験実行方法は[README](../README.md)を参照。
