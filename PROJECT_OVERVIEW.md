# Kazamidori Project — 現在の機能とファイル構成

確認日：2026-10-12。作業フォルダー内の実装を静的に確認した整理資料です。未コミットの変更を含む現在のコードを対象とし、構想資料の「完了」表示より実際の呼び出し経路を優先しています。GUI起動・飛行計算の実行・精度検証は今回行っていません。

追記：2026-10-13、タスク3-6の結果メニュー・表示切替を実装し、Qtの画面部品を使う自動テストとファイル入出力テストで確認しました。飛行計算の精度検証は次のタスク4-0で扱います。

## 1. プログラムの全体像

モデルロケットの機体・モーター・パラシュート・風・発射条件を入力し、飛行軌道と着地点を予測するデスクトップアプリです。風と推力の不確実性を加えたモンテカルロ計算で着地点の分布を求め、競技モードに応じた発射仰角・方位角の探索も行います。

- 画面：PySide6、グラフ：Matplotlib。
- 飛行計算：RocketPyによる6自由度計算、数値処理：NumPy／SciPy。
- 基本単位：m・kg・s。軌道は発射点を原点とした ENU 座標（X＝東、Y＝北、Z＝上）。
- 緯度経度は地図への変換に使用します。加えて、現在の `simulation.py` は RocketPy の環境生成にも緯度経度を渡しています。
- 標準起動：プロジェクト直下で `python main_qt.py`。
- 初回準備：起動に使うPythonで `python -m pip install -r requirements.txt` を実行します。

## 2. 現在の機能

| 分類 | 実装されている内容 | 主な担当 |
|---|---|---|
| 機体設定 | 平坦な `Airframe.json` を読み込み、質量、重心、寸法、ノーズ、フィン、モーター位置・乾燥質量、放出遅延の12項目を入力欄へ反映 | `app_window.py`、`sim_controller.py`、`data_loader.py` |
| 機体ファイル解析の補助関数 | RockSim `.rkt` の解析・重心・慣性計算関数は残っていますが、現行GUIの読込は `Airframe.json` に切替済み | `data_loader.py`、`geometry_math.py` |
| モーター設定 | 推力履歴CSVを読み込み、RocketPy用モーターモデルを構築 | `data_loader.py`、`simulation.py` |
| 回収系 | パラシュートのCd・面積・開傘遅延、燃焼終了後の放出遅延を設定 | `app_state.py`、`simulation.py` |
| 空力設定 | 推進中／燃焼終了後のCd、Mach–Cd曲線CSVの読込・プレビュー | `app_window.py`、`simulation.py` |
| 風モデル | 高度別風速・風向の編集、東西／南北成分への変換、補間。地上1点のみの場合はべき乗則で上空を推定 | `wind_model.py` |
| 単一飛行 | 基準条件の軌道、最高高度、着地点、飛行時間、燃焼終了・放出などの情報を計算 | `simulation.py` |
| モンテカルロ | 風・推力・突風のばらつきを与えて繰り返し計算し、着地散布、CEP50、指定確率の半径・楕円、密度分布、平均値等を算出 | `workers.py`、`monte_carlo.py` |
| 発射角度探索 | 仰角・方位角のグリッド探索、上位候補のモンテカルロ評価、採用角度での再計算 | `optimization.py`、`OptimizationWorker` |
| 可視化 | 3D軌道、時間位置マーカー、2D着地点地図、散布点、確率楕円、KDE密度表示、風の履歴 | `app_window.py`、`map_view.py` |
| オフライン地図 | OSMタイル取得・結合・保存、磁気偏角計算、地図のパン／ズーム、Ctrl＋クリックによる発射位置入力 | `map_downloader.py`、`map_view.py` |
| 実行制御 | 入力チェック、バックグラウンド計算、進捗表示、中止要求、基準軌道を先に表示、完了通知 | `sim_controller.py`、`workers.py` |
| 判定・監視 | 基準着地点の範囲確認、MC後のGO／NO-GO表示、風の基準値を固定した許容範囲監視 | `sim_controller.py`、`app_state.py` |
| 設定保存 | Cd・Cd曲線・比推力・推進薬密度をJSONに保存／復元 | `session_manager.py` |
| 結果保存・復元 | 3D／地図のPNG、軌道・MC着地点CSV、復元用JSON。JSONから結果パネル・描画・表示用の発射条件を復元 | `result_io.py`、`app_window.py`、`sim_controller.py` |
| 表示メニュー | KDE、CEP楕円・確率円、MC点群、燃焼終了点、最高到達点を切替。Simulationから詳細設定・実行・停止 | `app_window.py`、`map_view.py`、`sim_controller.py` |

### 競技モード

| モード | 現在の評価関数の概要 |
|---|---|
| 定点滞空 | 目標からの距離が小さく、飛行時間が長いほど高評価。`(許容半径 − 目標からの距離) + 飛行時間` |
| 高度 | 最高高度を評価 |
| 有翼 | 放出時刻から着地までの時間を評価 |
| 自由 | 目標半径による制限を外し、飛行時間を評価 |

有翼モードには評価関数と計算精度の分岐がありますが、専用の滑空・操舵モデルとして分離された実装は確認できません。

### 現在の計算上の特徴

- 通常の飛行計算は、先行計算で放出高度を求め、パラシュート付きの飛行を再計算する2段階構成です。
- 通常RUN後のMCには、基準飛行の放出高度を再利用する高速経路 `simulate_once_mc()` があります。試行ごとの放出高度を改めて求める方法とは異なる近似です。
- 重い処理は `QThread` で実行します。現在の通常MCはワーカースレッド内の逐次ループで、角度のグリッド探索も逐次処理です。
- 起動時の既定設定は、風の不確実性20%、推力の不確実性5%、着地確率90%、MC試行50回です。これは初期設定値であり、精度の保証を表すものではありません。

## 3. 処理の流れ

```text
main_qt.py
  ├─ AppState        入力値・結果・表示状態を共有
  ├─ AppWindow       設定画面、3D軌道、風表示、MapView
  └─ SimController   入力確認、処理開始、結果の反映
         │
         ├─ 通常RUN → SimulationWorker
         │               ├─ 風プロファイル生成
         │               ├─ 基準条件の飛行計算 → 軌道を先に表示
         │               └─ MC計算 → 着地分布・統計を表示
         │
         └─ 角度探索 → OptimizationWorker
                         ├─ optimize_launch_angle()
                         ├─ 採用した仰角・方位角を反映
                         └─ 基準飛行＋MC計算 → 結果表示

Workerの結果 → Qt Signal → SimController → AppState → 各グラフ・地図
```

入力ファイルの解釈は `utils/data_loader.py`、物理計算は `core/`、画面と計算の接続は `ui_qt/` が担当しています。

## 4. ファイル構成

```text
Kazamidori_Project/
├─ main_qt.py                    標準の起動ファイル
├─ requirements.txt             アプリと地図取得に必要なPythonライブラリ
├─ PROJECT_OVERVIEW.md           本資料
├─ core/                        飛行・風・統計・最適化
│  ├─ simulation.py             RocketPyによる飛行計算、モーター構築
│  ├─ wind_model.py             風向変換、補間、上空風推定
│  ├─ monte_carlo.py            着地分布、CEP、確率楕円、KDE、風許容判定
│  ├─ optimization.py           競技スコア、角度探索、別系統のPhase1関数
│  ├─ geometry_math.py          部品質量・重心・慣性モーメント計算
│  ├─ constants.py              物理定数・共通定数
│  ├─ mc_worker.py              プロセス並列用の補助関数（現行GUI経路では未使用）
│  └─ koinobori_api.py           気象センサー接続のダミーAPI
├─ ui_qt/                       GUIと実行制御
│  ├─ app_window.py             メイン画面、設定ダイアログ、軌道・風の描画
│  ├─ app_state.py              共有状態と変更通知、風履歴、許容範囲監視
│  ├─ sim_controller.py         UI入力とWorkerの接続、結果・判定の反映
│  ├─ workers.py                飛行計算・角度探索・地図取得のQThread
│  └─ map_view.py               地図と着地分布の描画・操作
├─ utils/                       入出力・地理処理
│  ├─ data_loader.py            Airframe・回収系JSON／各種CSV読込、旧RockSim解析
│  ├─ geo_math.py               ENUと緯度経度の近似変換、円・楕円の座標
│  ├─ map_downloader.py         地図タイル取得、画像・メタデータ出力
│  ├─ result_io.py              結果JSONの検証・保存・復元、軌道・MCのCSV保存
│  └─ session_manager.py        一部設定のJSON保存／復元
├─ assets/
│  └─ logo.png                  ロゴ画像
├─ tests/
│  ├─ core/test_optimization.py 風のべき指数計算の単体テスト（5ケース）
│  ├─ ui_qt/test_airframe_import.py  Airframe読込・編集・保存・風更新のテスト（16ケース）
│  ├─ ui_qt/test_offline_map.py  オフライン地図の表示・座標操作・取得（17ケース）
│  ├─ ui_qt/test_location_service.py  現在地取得・失敗・再試行（6ケース）
│  ├─ ui_qt/test_result_menu.py 結果メニュー・表示切替・Worker出力互換性（9ケース）
│  ├─ utils/                   地図保存（6ケース）、結果入出力（6ケース）
│  └─ fixtures/result_archive.json  結果メニュー確認用の架空データ
├─ test_physics_core.py          RocketPyの単独確認スクリプト
├─ archive/                     過去の監査報告・計画と確認用スクリプト
├─ Koinobori/
│  ├─ Koinobori_PCB/            KiCad回路図・基板・プロジェクト
│  └─ Roadmap.txt               自作風向風速計の開発計画
├─ .Jules/                      アーキテクチャ規約・監査手順・開発記録
├─ 技術メモ/                    実装状況・前提を照合した検討資料（README.mdに索引）
├─ project_ui_requirements.md   UI要求仕様
├─ Concept.txt / Roadmap.txt    全体構想・開発計画
├─ memo.txt                     作業メモ
├─ Rocket.json                 機体・回収系の統合設定例
├─ Airframe.json               機体単独設定例
├─ Parachute.json              回収系設定例
├─ Estes_B4.csv / B6 / G40      モーター推力データ（各ファイル名はEstes_*.csv）
├─ test.rkt                    RockSim形式の機体データ
├─ 定点本番機体再定義ver.ork    OpenRocketネイティブ形式の機体データ
└─ .gitignore                  ローカル入力・生成物のGit除外設定
```

各Pythonパッケージには `__init__.py` もあります。取得済みの地図は `assets/offline_map/` に保存されます。実行時には `__pycache__/` や `mc_diagnostics.log` が再生成される場合があります。

技術メモは[索引](技術メモ/README.md)と[検証結果](技術メモ/検証結果.md)から参照できます。実装済みの機能、将来案、検証保留の数値を区別しています（確認日：2026-10-24）。

### オフライン地図の使い方

- `Get Current Location` はQt Positioningを通じてOSの位置情報サービスへ一度だけ測位を要求します。WindowsではWinRTの位置情報を使います。成功すると緯度・経度と報告された水平精度を表示し、その地点の地図を読み込み／取得します。初期値やIPアドレスの推定値を現在地として扱いません。
- 現在地を取得できない場合は、Windowsの「設定 → プライバシーとセキュリティ → 位置情報」で位置情報サービスとデスクトップアプリのアクセスを確認してください。取得失敗時は入力座標と保存済み地図を保持します。タイムアウト後は再試行できます。
- 起動時の緯度・経度は設定済み座標、または保存済み地図の中心です。「現在地は未取得」と明示します。設定・保存済み地図がなければ座標欄は空欄になります。
- 通信できる環境で発射地点の緯度・経度を入力し、`Download Offline Map` を押すと周囲約1 kmの地図を保存します。同じ中心座標の正常な地図があれば、通信せずに再表示します。
- 起動時は保存済み地図を読み込みます。保存範囲内での発射位置変更、パン、ズームはオフラインで使えます。座標の編集だけではダウンロードしません。範囲外へ移動すると、別の場所の地図を原点に重ねず、範囲外であることを表示します。
- 地図上で **Ctrl＋左クリック**すると、その地点の緯度・経度がLaunch Settingsの入力欄と発射地点表示に反映されます。Ctrlを押したままドラッグして離した地点を選ぶこともできます。通常のドラッグはパン、ホイールはズームです。画像がない場所・地図の外では座標選択しません。選択だけでダウンロードや飛行計算は始まりません。
- 全タイルの取得と画像保存が成功してから地図を切り替えます。通信失敗・一部タイルの欠落・保存失敗時は、以前の地図を保持します。
- `map_meta.json` の `image_file` が現在の地図画像を指します。旧形式の `background.png` も読み込めます。地図は1エリア分で、別エリアの取得成功時に更新されます。
- 地図の回帰テストは `tests/utils/test_offline_map.py` と `tests/ui_qt/test_offline_map.py` です。Qtのテストでは通信を禁止し、保存済み画像の表示、座標変更、再取得・失敗後の再試行を確認します。

`Koinobori/` は風向・風速を測定する自作風向風速計の独立したプロジェクトです。アプリ側の接続がダミーであることとは別に、設計・開発資産として保持します。

## 5. 実装済み機能と区別すべき点

| 項目 | 現在の状態・注意点 |
|---|---|
| Koinobori実機接続 | `koinobori_api.py` は固定値を返すダミー。GUI側の1秒周期更新も乱数で風・気圧・気温・湿度を生成しています。実測連携としては未完成です。 |
| 外部気象取得 | Open-Meteoの等圧面風データをJST期間指定で取得しCSV保存。外部CSVの選択・対象時刻の読込に対応（フェーズ4）。 |
| GPS／現在地取得 | `location_service.py` がOSの位置情報を非同期で取得します。デスクトップPCではWi-Fi等による測位になる場合があり、報告された精度を表示します。OSの位置情報サービス・権限が必要です。 |
| 任意の目標着地点 | 計算側には `target_x`／`target_y` がありますが、現在の入力経路では通常0となります。発射点とは別の目標地点を設定するUIは未接続です。 |
| `.rkt`／`.ork` | 機体全体は `Airframe.json` から読み込みます。Manual Configでは `.ork` からフィン形状を取り込めます。RockSim `.rkt` の解析関数も保持しています。 |
| 最適化方式 | 自作の焼きなまし法（SA）とMC検証。既定48評価、Advanced Settingsで変更可能。Optunaは未実装です。 |
| 並列化 | `mc_worker.py` 等に並列処理の補助コードが残っていますが、現行GUIの通常MCと角度探索は逐次処理です。非同期GUIと複数プロセス並列は別です。 |
| セッション保存 | 空力・モーター設定に加え、A/B、SA評価回数、偏角設定、標高、風CSVパスと対象時刻を保存。機体全体やCSV本体・計算結果を包含する保存ではありません。 |
| 機体の保存 | Manual Configの「Save airframe」で、編集内容を平坦なAirframe JSONへ保存できます。 |
| GO／NO-GO | 通常MC完了後は `CEP50 <= target_radius` を判定しています。一方、角度探索では目標からの距離とMC半径を組み合わせています。判定条件の統一が必要です。 |

特に、CEP50は着地分布の中心からの散らばりの指標です。通常MC後の判定では目標からの位置ずれや指定確率の範囲全体を確認していないため、「指定確率で目標内に着地する」という判定と同一視できません。該当箇所は `SimController._on_mc_done()` です。

## 6. 保守・整理の観点

1. **入口と担当は明確です。** 最初に `main_qt.py`、次に `sim_controller.py`、`workers.py`、`simulation.py` の順で読むと、実際の実行経路を追えます。
2. **大きいファイルに複数の責務が集まっています。** `app_window.py` は約3,100行、`app_state.py` と `sim_controller.py` は各約1,700行です。今後分割するなら、設定ダイアログ・描画・結果判定を候補にできます。
3. **別経路・古い説明が残っています。** GUIは `optimize_launch_angle()` を使用しますが、`run_phase1()` などの別経路も残っています。`main_qt.py` の冒頭には状態を分ける説明がありますが、実コードでは共通の `AppState` を画面に渡しています。
4. **入力データの所在を明示する必要があります。** `.gitignore` は機体JSON、推力CSV、機体設計ファイル、技術メモ、オフライン地図などを除外しています。他の環境にGitだけで移すと、同じ入力がそろわない場合があります。
5. **依存関係を `requirements.txt` にまとめました。** PySide6、RocketPy、NumPy、SciPy、Matplotlib、defusedxml、地図取得用の requests、Pillow、pygeomag を定義しています。`python -m pip install -r requirements.txt` で導入できます。バージョン固定はしていないため、環境の完全な再現用ロックファイルではありません。
6. **正式テストの範囲は限定的です。** べき指数計算の5ケースに加え、Airframe JSONの検証・Qt入力欄への反映・編集確定／取消・保存・探索中の風更新を確認する16ケースがあります。飛行全体の精度検証とは区別が必要です。確認用スクリプトは `archive/` とルートにも残っています。

## 7. 不要ファイルの整理（2026-10-12）

ユーザーの削除依頼に基づき、次を削除しました。

- 一時スクリプト `tempCodeRunnerFile.py`、生成ログ `mc_diagnostics.log`／`test_run.log`。
- `archive/` の旧性能測定スクリプト：`test_profile.py`、`test_profile2.py`〜`test_profile23.py`、`test_profile_worker.py`、`test_mp.py`、`profile_mc_points.py`、`profile_mc_points2.py`、`profile_mc_r90.py`。
- 生成された性能測定結果 `archive/stats_mp`／`archive/stats_worker`。
- 過去のコード書換え用スクリプト `archive/fix_code_review.py`。
- 現在の `AppWindow` 呼出し方法に対応していない `scratch/test_gui_hang.py` と、削除後に空になった `scratch/`。
- プロジェクト直下・`core/`・`ui_qt/`・`utils/`・`tests/core/` のPythonキャッシュ。

削除数は通常ファイル35件とキャッシュ22件の計57件、キャッシュディレクトリ5件です。Koinobori、現行アプリのソース、機体・推力データ、設計資料、正式テスト、物理計算の確認スクリプト、監査報告は保持しました。生成物がGitに再混入しないよう、`.gitignore` にルートの実行ログ・一時スクリプトと旧性能測定出力の除外を追加しました。

## 8. Airframe.jsonへの切替（2026-10-13）

メイン画面とManual Configの「Load Airframe」から、プロジェクト直下の `Airframe.json` を選択します。ファイル選択ダイアログはこのファイルを初期候補にします。起動時の自動読込は行いません。

JSONは `airframe` セクションで囲まず、次の12キーを直接記述します。単位はm・kg・sです。

```json
{
  "mass": 0.0872,
  "cg": 0.21,
  "length": 0.383,
  "radius": 0.015,
  "nose_length": 0.08,
  "fin_root": 0.04,
  "fin_tip": 0.02,
  "fin_span": 0.03,
  "fin_pos": 0.35,
  "motor_pos": 0.38,
  "motor_dry_mass": 0.015,
  "backfire_delay": 4.0
}
```

重心・質量・寸法はManual Configに、モーター位置・乾燥質量・放出遅延はメイン画面に反映されます。半径は内部で直径へ変換されます。パラシュートは別途「Load Parachute JSON」で読み込み、機体JSONの読込によって既存のパラシュート設定は変更しません。

JSONの必須キー不足・非数値・非有限値などを読込時に検査し、不正なファイルでは現在の設定を変更しません。慣性モーメントはこの形式には含まれないため、過去の読込値を解除して飛行計算側の近似を使用します。

Manual Configでは入力中の文字列を保持し、「Apply」で編集を確定、「Cancel」で破棄します。ダイアログ内の「Load airframe」もApplyまで未確定です。「Save airframe」は編集内容とモーター設定をJSONへ保存し、Applyを実行しません。`fin_count` と `nose_shape` は任意項目として保存・読込できます。

機体・パラシュートのファイル名は各Loadボタンの横に表示します。機体の質量・重心・長さの要約はファイル名のツールチップに表示します。計算・探索中も1秒周期の風速情報更新を続け、計算は開始時に収集した入力値を使用します。

## 9. 結果メニュー（タスク3-6、2026-10-13）

- `File → Export Results`：表示中の3D軌道／着地点地図をPNG、軌道とMC着地点をCSV、結果一式をJSONで保存します。計算中や軌道がない状態では保存できません。
- CSVの列は `record_type,index,time_s,east_m,north_m,altitude_m`。`trajectory` は時系列の軌道、`mc_impact` は各試行の着地点です。座標は発射点を原点とするENU・メートル単位で、時刻は秒です。MC着地点に時刻は付かず、高度は0です。表示を間引いていてもCSVには全着地点を出力します。
- `Result Archive (JSON)` は基準飛行、MC分布・集計値、描画用のKDE・イベント、表示時の発射座標・モード・Maximum Radiusを保存します。評価できない得点の無限値はJSONの `null` として保存します。
- `File → Import Results (JSON)`：このアプリで書き出した結果JSONを読み込み、結果パネル・3D軌道・地図を復元します。現在のViewチェック状態は保持します。機体・モーター・風の入力一式を復元して再計算する形式ではありません。CSVは分析用で、復元にはJSONを使います。
- 地図画像自体はJSONに含みません。復元した座標が保存済み地図の範囲外なら、その旨を地図に表示します。Importによる地図の自動取得は行いません。
- Import後は過去結果であることを明示し、風の許容範囲監視を無効化します。風情報の受信・表示は継続します。新たに計算を実行すると通常の結果表示へ戻ります。
- `View`：KDE、CEP楕円・確率円、MC点群、燃焼終了点、最高到達点を個別に切り替えます。非表示設定は着地確率の変更後も維持されます。確率変更では保存済みの点群から楕円と半径を更新し、飛行計算は実行しません。
- `Simulation`：Advanced Settings、Run、Stopを操作できます。

検証コマンド：`python -B -X utf8 -m unittest discover -s tests/ui_qt -v`、`python -B -X utf8 -m unittest discover -s tests/utils -v`。次工程はRoadmapのタスク4-0（物理計算・最適化の回帰テスト）です。

## 10. 「Unstable Rocket」の診断（2026-10-14）

提供された `Airframe.json`／`Parachute.json` と `Estes_G40.csv`、初期設定の Motor Isp = 80 s で再現すると、発射時重心は先端から0.3154 m、圧力中心は0.3056 m、安定余裕は−0.328 calとなります。計算モデルが不安定と判定しているため、飛行積分を開始せず停止します。Maximum Radiusの制約による失敗とは別です。

推力曲線の総力積をIsp×重力加速度で割って推進薬質量を推定しています。この組み合わせの推定値は0.1274 kgで、機体乾燥質量0.0872 kg・モーター乾燥質量0.0150 kgとともに発射時重心へ反映されます。推力CSVを読んでも、実モーターの乾燥質量・位置・Ispは更新されません。G40を使用する際は、これらの入力を実機・モーターの仕様と照合してください。

通常実行・探索のエラーには重心・圧力中心・安定余裕・質量・Ispを含め、閉じるまで読めるダイアログに表示します。探索では発射角度によらない初期不安定を最初の候補で検出すると停止します。他の理由で全候補が失敗した場合も、原因と件数を表示します。機体・モーターの入力値や安定性の判定基準は変更していません。

不安定時の通知は「機体・モーターが不適合です（機体が不安定）」と表示します。ダイアログに選択したモーター名と計算を中止した旨を示し、判定欄は赤い「NO-GO — 不適合・不安定」に切り替え、OS通知にも同じ理由を表示します。重心などの数値はダイアログの詳細から確認できます。その他の計算エラーもダイアログとOS通知で知らせ、原因と詳細を保持します。

追加テストは `tests/core/test_simulation_failures.py`（6ケース）と `tests/ui_qt/test_simulation_errors.py`（3ケース）です。コアの実行コマンドは `python -B -X utf8 -m unittest discover -s tests/core -v`。フェーズ4ではさらに、実飛行の数値基準・得点・探索・CSVと画面連携のpytest回帰テストを追加しました。

## 11. フェーズ4（2026-10-14）

### 操作

- **Advanced Settings → Scoring / Search**: 定点滞空の式 `A × (Maximum Radius − 目標からの距離) + B × 滞空時間` を編集。A/Bの初期値は1。範囲外への着地は係数によらず失格。Save Settings、またはFile → Save Sessionで保存し、Load Sessionで復元します。
- **Advanced Settings → Wind Profile → Wind Data — API / CSV**: JSTの開始・終了時刻、計算対象時刻を指定。Download Specified Periodで取得・保存、Load Wind CSVで既存ファイルを選択。対象時刻だけ変えた場合はApply target time。Use Manual Windで手動風プロファイルへ戻します。
- APIには発射地点の緯度・経度と指定期間を送信します。[Open-Meteo公式仕様](https://open-meteo.com/en/docs)に従い風速をm/sで取得します。データ提供はOpen-Meteo（CC BY 4.0）。取得可能な過去・予報期間は提供元の制限を受けます。
- **Launch Settings**: Rail Azimuthの基準をTrue NorthまたはMagnetic Northで選択。既存の入力と互換性を保つ初期設定はTrue Northです。Magnetic Northでは、真方位 = 入力方位 + 偏角（東正）で計算します。Auto WMMとManual overrideを切替可能。結果の方位は真北基準です。
- Site altitudeはWMM用の海抜標高。WMMは通信せずにローカル係数で計算し、実行時に地点・標高・日付ごとの値を `assets/offline_map/declination_cache.json` へ保存します。[pyGeoMag仕様](https://pygeomag.readthedocs.io/en/latest/api.html)に合わせ、標高をkmへ換算して渡します。
- 風速は高度別に色分けし、1.5 mでは受信履歴、その他の高度では直近60秒の平均値を表示します。手動プロファイルの編集も反映します。コンパスの過去点は1.5 mだけを表示し、上空は平均方向の矢印にまとめます。受信停止後も最後の表示値を保持します（第17節）。

### 風CSV

UTF-8の列は `time_utc,alt_m,pressure_hpa,speed_ms,dir_deg,latitude,longitude,source`。時刻はタイムゾーン付きISO 8601、風向は真北基準で風が来る方向（FROM）。`alt_m`（地上高AGL）または `pressure_hpa` の片方を指定し、時間順に並べます。座標とsourceは手動CSVでは省略可能です。例は `tests/fixtures/wind_profile.csv`。

読み込みでは対象時刻以前で1時間未満の最新プロファイルを採用します。対象期間外、NaN、重複高度、異なる地点のデータなどはエラーにします。全期間をメモリへ展開せず、1プロファイルだけ保持。API取得も最大24時間ずつ処理して一時CSVへ書き出し、完了した場合だけ保存先を置換します。地上1.5 mは現地テレメトリを優先し、上空の風をCSVから補います。

等圧面の高度は `z = T₀/L × [1 − (p/p₀)^(R L/g)]` で地上高に換算します（L=0.0065 K/m、R=287.05 J/(kg K)、g=9.80665 m/s²）。p₀は**観測地点の現地気圧**、T₀は現地気温。海面更正気圧QNHを直接代入する方式ではありません。負の高度は地下として除外します。これは乾燥大気・一定気温減率の近似であり、観測された各層の高度そのものではありません。Koinobori実機のUDP接続はフェーズ5-1に残り、現在の地上テレメトリは模擬値です。

### コードと検証

新規コード: `core/annealing.py`（自作SA）、`core/atmosphere.py`（気圧→AGL）、`utils/wind_data.py`（CSV/API）、`utils/magnetic.py`（WMMとキャッシュ）、`ui_qt/wind_data_dialog.py`（非同期の取得・読込画面）。KDE計算はMC終了後のWorkerで行い、風の描画では実行しません。

SAは85°から開始し、射角60–90°・周期的な方位角を探索します（初回から厳密な90°で発生した積分停滞を避けるため）。温度を下げながら悪化する候補も確率的に採用します。周期的な広域提案で別の実行可能領域も探します。シードと評価予算で再現可能ですが、大域最適解の保証はありません。上位候補と最終候補をMCで検証し、最終検証の失敗を過去の成功値で置換しません。MC半径は基準着地点からの散らばりとして求め、発射点からの距離を二重加算しないよう修正しました。

開発用依存は `python -m pip install -r requirements-dev.txt`。全テストは `python -B -X utf8 -m pytest -q`。Qt画面はオフスクリーンで検証し、通常テストは外部通信不要です。実飛行の数値基準はRocketPy 1.13で記録した合成推力データによる回帰比較であり、実飛行との一致を保証する試験ではありません。Open-Meteoへの実接続では10高度層のCSV保存・再読込も確認しました。

フェーズ4の実計算確認では、合成推力の安定モデルでSA 8評価とMC検証まで完了しました。選択射角87.80°、最高高度94.38 m（無風・推力誤差なし）。これは回帰用のモデルケースです。

## 12. CP入力とフィン形状（2026-10-15）

通常のJSON／手動設定では、Manual Configの **CP from Nose [m] (Mach 0)** に先端から圧力中心までの距離を入力できます。空欄は自動計算です。Applyで適用、Cancelで取消。Save airframeでJSONの任意項目 `cp` に保存し、Load airframeで復元します。`cp` がない既存ファイルや `null` は自動計算に戻ります。負の値・非数値・全長を超える値は入力エラーです。OpenRocket全体取込時は、次節の出典優先順位に従います。

手入力CPはMach 0の較正値です。ノーズとフィンそれぞれの空気力の作用位置を同じ距離だけ補正し、機体全体のMach 0 CPを入力値へ合わせます。元のモデルのMach依存性は維持します。補正は安定性判定だけでなく、通常計算の両パス・最適化・MCのモーメント計算にも適用されます。

フィンの形状はManual Configで編集します。

- **Trapezoidal**: Fin Root Chord、Fin Tip Chord、Fin Semi-Span、Fin Sweep Lengthで台形を指定。Sweepは翼根前縁から翼端前縁までの後退距離で、負の値は前進翼です。空欄では従来のRocketPy既定値（翼根長−翼端長）を使います。Fin Cantは取付けのひねり角（度）。
- **Edit Fin Outline…**: 頂点表とプレビューで自由形状を編集。単位m、始点 `(0,0)` は翼根前縁、xは尾部方向、yは胴体から外側。輪郭に沿って点を並べ、最後を翼根後縁 `(x>0,0)` にします。閉じ線は自動。頂点の追加・削除、交差や不正な輪郭の検出に対応します。自由形状では翼根・翼端・半幅が輪郭から算出されます。
- **Import Fin Shape (.ork)…**: [OpenRocket形式](https://openrocket.readthedocs.io/en/latest/dev_guide/file_specification.html)のZIP/XML/GZIPから台形・自由形状を読み込みます。楕円フィンは32分割の輪郭として読み込みます。複数セットがあれば選択できます。同一取付位置の同じ輪郭を持つ1枚フィンが等間隔で3〜8枚ある場合は、対称セットとしてまとめます。単独の1〜2枚の部品からは輪郭を読み込み、計算用枚数は画面の現在値を使用する旨を表示します。

読込対象はフィンの輪郭・枚数・後退量・Cantです。Fin LE Position（先端から翼根前縁までの距離）、機体質量・CG・CPは現在の設定を使用します。読み込んだ形状は編集でき、Applyで確定してから計算へ渡ります。JSONには `fin_shape`、`fin_points`、`fin_sweep`、`fin_cant` を保存。自由形状は通常計算とMCの両方でRocketPyのFreeFormFinsへ渡し、台形へ近似置換しません。

追加コードは `core/fin_geometry.py`、`utils/openrocket_fins.py`、`ui_qt/fin_editor.py`。実ファイル「ジョイくん.ork」「モンブラン.ork」の読込とRocketPyの空力モデル生成を確認しました。テストではCPによる実際のモーメントアーム補正、通常2パスとMCの実飛行、輪郭検証、ファイルの保存・復元、Apply／Cancel、取り込み中の既存設定の保持を確認します。

## 13. OpenRocket機体全体の取込と出典（2026-10-15）

メイン画面またはManual Configの **Load Airframe** で `.ork` / `.ork.gz` を選択すると、フィンだけでなく機体全体を解析します。外部通信やOpenRocketのインストールは不要です。Manual Config内の読込はApplyまで未確定で、Cancelで元に戻ります。

- ファイルから取得：全長、胴体半径、ノーズ長・形状、フィン形状・枚数・Cant・取付位置、既定モーター構成の搭載位置・燃焼後遅延時間。位置は先端からの距離に統一し、OpenRocketのtop/middle/bottom/absolute/after指定を解決します。`auto 数値` の寸法も読めます。
- 機体質量・CG：機体全体を対象とするOpenRocketの質量／CG上書き値を最優先。なければ部品質量、材質密度、形状、取付位置からKazamidoriで推定します。モーターは含めません。ノーズの殻・肩は数値積分、フィンはタブを含む平板、円筒・リングは体積計算です。部品の上書き値も集計に反映します。未対応部品をゼロ質量として省略せず、算出できない項目は手入力として残します。
- CP：現在の飛行モデルと同じRocketPyのノーズ・フィンからMach 0の値を計算して表示します。過去のOpenRocket飛行結果からCG・CPを流用しません。自動CPはMCの形状変動でも再計算され、固定CPによる補正にはなりません。
- パラシュートが1つの場合は直径から面積を計算しCdを読み込みます。CdがautoならOpenRocketの既定Cd=0.8を用います。複数の場合は自動選択せず通知します。開傘ラグは手入力またはParachute JSONを使います。OpenRocketの展開イベント／deploydelayをそのまま再現する機能ではありません。

優先順位は **.ork → Kazamidori自動計算 → 手入力**。Manual Configの一覧と入力欄のツールチップで出典を確認できます。取得済みの値は手入力で上書きされないよう固定し、計算へ渡す際にも優先順位を適用します。機体形状の変更はOpenRocketで編集して再読込します。通常のJSON／手動設定では従来の編集操作を使えます。

出典情報はメモリ内の`provenance`で保持します。2026-10-17以降、Save airframeは不足項目だけを補完JSONへ保存します（第15節）。旧形式の`provenance`付きJSONも読み込めます。モーター質量などが未取得でも保存でき、必要値が不足する間は計算を実行できません。別の機体を読み込むと、前のモーター推力曲線・未取得の質量・パラシュート設定を引き継ぎません。モーター位置欄は実際の計算上の意味に合わせて **Motor Nozzle Pos. [m]** としました。

取込対象は単段・一定直径の胴体・対応ノーズ・対称フィン1組（3〜8枚）です。無効なステージは除外します。有効な多段、ポッド、複数モーター、段差や未対応ノーズはエラーを表示して読み込みを中止します。OpenRocketが保存したモーター型番だけから推力曲線や燃焼後質量を推測しないため、この2項目は別途指定が必要です。質量・CGの推定値はOpenRocketの計算結果との完全一致を意味しません。

実装：`utils/openrocket_airframe.py`（部品ツリー解析）、`core/airframe_parameters.py`（CP計算と出典優先順位）。参照：[OpenRocketファイル仕様](https://openrocket.readthedocs.io/en/latest/dev_guide/file_specification.html)、[上書き値の仕様](https://github.com/openrocket/openrocket/blob/unstable/docs/source/user_guide/overrides_and_surface_finish.rst)。回帰テストは`tests/fixtures/single_stage.ork`を使い、位置基準、質量／CG上書き、既定モーターの選択、不足値、JSON往復、Apply／Cancel、優先順位が実行パラメータへ反映されることを検証します。

## 14. 数学・物理検証（2026-10-17）

90%楕円とKDEの向きが異なる問題を、ENU角度基準の統一で修正しました。確率倍率、KDEの確率質量、CEP円の中心、目標円の包含率によるGO/NO-GO、旧保存結果の角度移行も検証しています。[検証報告と比較図](MATH_PHYSICS_REVIEW.md)に、試験項目と開傘時刻などの未解消のモデル制約をまとめています。

## 15. .orkと補完JSONの手動選択（2026-10-17）

**Load Airframe**を押すと、OpenRocketと補完JSONの2つの選択欄が開きます。

1. **Select .ork…**で機体の`.ork`または`.ork.gz`を選択します。
2. **Select JSON…**で、その機体に使用する補完JSONを選択します。
3. 両方の選択内容を確認して**Load**を押します。Cancelでは現在の機体を保持します。

それぞれ任意のファイル名・フォルダから選択できます。周囲のJSONの自動検索やJSONからの.ork自動読込は行いません。旧JSONに`ork_file`が残っていても、自動参照や組み合わせの制約には使わず、画面で選択した.orkを使用します。ジョイくん・モンブランはテスト用データであり、機体名に依存する処理はありません。

補完JSONの例は次のとおりです。`ork_file`の記載は不要です。

```json
{}
```

- 優先順位は **.ork → Kazamidori自動計算 → 補完JSON・手入力**。JSONで取得済みの形状等を上書きしません。未計算の質量・CGなどもJSONで補完できます。Manual Configには出典を表示し、補完値は編集可能です。
- .ork単体も読み込めます。補完JSONを選択しなければ不足項目は空欄になります。全パラメータを持つ従来のJSONは、OpenRocket欄を空欄にしてJSON単体で読み込めます。
- ファイルの欠損・不正なJSON・誤った項目名では読込を中止し、既存設定を保持します。
- **Save airframe**は、.ork取込後なら不足項目だけをJSONへ保存します。形状・計算済みCG/CP・内部の出典情報・.ork参照を重複保存しません。再読込時には.orkと保存したJSONをそれぞれ選択します。手入力のみの機体は従来の単独JSONとして保存します。
- ロードボタン横には、選択した`.ork + 補完JSON`の両ファイル名を表示します。Manual Config内での読込は、さらにApplyで適用、Cancelで取消です。

テスト用の`Airframe_Joykun.json`と`Airframe_Montblanc.json`は、自動取得できない機体項目だけを置く補完ファイルです。2026-10-22にモーター燃焼後質量をそれぞれ`Motor_Joykun.json`、`Motor_Montblanc.json`の`dry_mass_kg`へ移しました。対応する機体・Motor Configはそれぞれ手動で選択します。形状・モーター取付位置・遅延は.ork、質量・CG・CPは取得値または自動計算を使用します。

推力曲線は**Load Thrust Curve**から読み込みます。パラシュートの面積・Cdは.orkで取得できればそちらを優先し、開傘ラグは既存のParachute JSONまたは手入力で指定します。

実装は`ui_qt/airframe_load_dialog.py`、`utils/airframe_pair.py`、`utils/data_loader.py`。テストでは任意の名前・別フォルダの2ファイル選択、自動検索しないこと、旧参照の非使用、優先順位、保存内容、欠損・不正ファイルを検証します。Qtの実ダイアログでも個別選択、選択途中の状態保持、Load／Cancelを確認します。

変更後の全体テストは209件、サブテスト52件が成功しました。

## 16. 運用画面・設定・進捗の更新（2026-10-22）

- **方位表示**：3D軌道図の外側にE/N/Upの方位軸を配置しました。カメラ回転と同期し、軌道の座標範囲に影響しません。
- **事前風データ取得**：風パネルの「Download GPV / Wind Data…」から開始・終了時刻をJSTで指定してCSV保存できます。提供元は既存のOpen-Meteoです。事前ダウンロードだけならSimulation timeを対象期間に合わせる必要はありません。「Use downloaded data for this simulation」を選択したときだけ現在の計算へ適用します。現場ではLoad Wind CSVと対象時刻を指定して利用します。取得可能期間は提供元に依存します。
- **モーター設定**：「Motor Specifications」をAirframeから独立したページに移しました。推力曲線のファイル名をLoadボタン右に表示し、Load/Save Motor Configで設定を読み書きします。Motor Configの項目は`nozzle_pos_m`, `dry_mass_kg`, `backfire_delay_s`, `isp_s`, `propellant_density`です。旧Airframeのモーター項目は互換読込できますが、新しいAirframe保存には含めません。.orkから取得できた取付位置・遅延は従来の出典優先順位を維持します。
- **パラシュート横流れ**：Recovery / Parachuteの「Horizontal Wind Drift」を0〜200%で指定します。100%はその高度の風速と同じ水平速度、50%は半分、0%は開傘後の水平移動なしです。Parachute JSONの任意キー`wind_drift_pct`、およびconfigにも保存できます。
- **練習用の風編集**：「Edit Practice Wind…」で高度別の基準風速・風向（FROM）を編集します。基準値とランダムな模擬観測値は分離し、毎秒の風速・風向の揺らぎとMCの乱数を維持します。手動編集するとCSV風の選択を解除します。
- **進捗**：通常実行は準備・公称軌道0〜25%、MC25〜90%、集計90〜98%、完了100%。探索付きは探索2〜65%、公称軌道65〜75%、MC75〜94%、集計95〜98%、完了100%です。プログレスバーの説明と下部の進行表示に同じ文言を使用し、終了後は100%を保持します。割合は処理段階の目安で、残り時間の推定ではありません。
- **config.json**：起動時に読み込む共通設定を追加しました。simulation、motor、parachute、aerodynamics、practice_windを管理します。File → Save Config / Load Configで保存・復元できます。機体ファイルや推力曲線の自動選択は行いません。従来のSessionは詳細設定用として利用できます。

横流れの割合指定は、開傘前の軌道を保持し、開傘後の水平速度を`比率 × 高度に対応するENU風ベクトル`として時間積分する**運動学的な近似**です。鉛直方向はRocketPyの結果を使用します。パラシュートの空力・慣性を再計算するモデルではありません。通常計算・最適化・高速MCへ同じ処理を適用します。割合の指定がない従来のコア呼び出しはRocketPyの元の水平運動を維持します。

追加実装：`core/parachute_drift.py`（横流れ積分）、`utils/configuration.py`（検証・保存）、`config.json`（初期設定）。追加テストは`tests/core/test_operator_features.py`と`tests/ui_qt/test_operator_features.py`です。風データ取得は通信をモックして検証し、この変更で外部サービスへの実データ送信は行っていません。

検証結果：全体224件・サブテスト52件が成功。最終調整後は、追加の探索〜完了の進捗テストを含む関連25件も成功しました。

## 17. 表示の整理とJST対応（2026-10-24）

- 風速グラフは1.5 mのみ受信ごとの履歴・点を表示し、ほかの高度は直近60秒の風速の算術平均を水平線で表示します。風向は風速を加味したベクトル平均を使い、上空のコンパスには過去点を描きません。右側の数値表もこの平均に揃えます。受信が止まると最後に表示した平均値を保持します。表示のみの変更で、計算用の風プロファイル・乱数は維持します。
- 3D図はタイトル、アニメーション時刻、集計値、方位軸、凡例の領域を分けました。イベント位置に重なっていた高度の文字を整理し、軸目盛りを減らして余白を確保しました。通常サイズと縮小サイズ、複数の視点で文字の枠内配置を検証します。
- ボタン、メニュー、ページ見出し、GO／NO-GOから絵文字を削除し、Run Phase 1 Simulationは青の単色にしました。
- 風データの開始・終了・計算対象時刻の入力、取得進捗、適用日時はJSTです。日本時間0時前後の日付切替も扱います。既存CSVの`time_utc`、APIとの通信、保存済みセッションの`wind_time_utc`は従来形式のまま保ち、画面との境界で変換します。

描画確認用の画像：[3D図](assets/verification/profile_readability.png)、[風速グラフ](assets/verification/wind_readability.png)。画像はテスト用データによるものです。追加検証は`tests/ui_qt/test_readability.py`、日時の表示変換は`utils/time_display.py`にあります。

変更後の全体テストは229件、サブテスト52件が成功しました。描画検証では通常・縮小の2サイズと4方向の視点を使用しています。
