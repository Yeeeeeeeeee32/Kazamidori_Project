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
├─ 技術メモ/                    高速化、気象、多段化、代替モデル等の検討資料
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
| 外部気象取得 | Open-Meteoの等圧面風データをUTC期間指定で取得しCSV保存。外部CSVの選択・対象時刻の読込に対応（フェーズ4）。 |
| GPS／現在地取得 | `location_service.py` がOSの位置情報を非同期で取得します。デスクトップPCではWi-Fi等による測位になる場合があり、報告された精度を表示します。OSの位置情報サービス・権限が必要です。 |
| 任意の目標着地点 | 計算側には `target_x`／`target_y` がありますが、現在の入力経路では通常0となります。発射点とは別の目標地点を設定するUIは未接続です。 |
| `.rkt`／`.ork` | 現行GUIの機体読込対象は `Airframe.json` です。既存設計ファイルとRockSim解析関数は保持していますが、GUIからは使用しません。 |
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
- **Advanced Settings → Wind Profile → Wind Data — API / CSV**: UTCの開始・終了時刻、計算対象時刻を指定。Fetch and Save CSVで取得・保存、Load Wind CSVで既存ファイルを選択。対象時刻だけ変えた場合はApply target time。Use Manual Windで手動風プロファイルへ戻します。
- APIには発射地点の緯度・経度と指定期間を送信します。[Open-Meteo公式仕様](https://open-meteo.com/en/docs)に従い風速をm/sで取得します。データ提供はOpen-Meteo（CC BY 4.0）。取得可能な過去・予報期間は提供元の制限を受けます。
- **Launch Settings**: Rail Azimuthの基準をTrue NorthまたはMagnetic Northで選択。既存の入力と互換性を保つ初期設定はTrue Northです。Magnetic Northでは、真方位 = 入力方位 + 偏角（東正）で計算します。Auto WMMとManual overrideを切替可能。結果の方位は真北基準です。
- Site altitudeはWMM用の海抜標高。WMMは通信せずにローカル係数で計算し、実行時に地点・標高・日付ごとの値を `assets/offline_map/declination_cache.json` へ保存します。[pyGeoMag仕様](https://pygeomag.readthedocs.io/en/latest/api.html)に合わせ、標高をkmへ換算して渡します。
- 風速履歴では同じ値の連続受信も点で表示。値を水平破線で保持し、1分以上受信がなくても消しません。コンパスは高度別に色・太さ・透明度を変え、過去の風は点、最新の風は矢印です。

### 風CSV

UTF-8の列は `time_utc,alt_m,pressure_hpa,speed_ms,dir_deg,latitude,longitude,source`。時刻はタイムゾーン付きISO 8601、風向は真北基準で風が来る方向（FROM）。`alt_m`（地上高AGL）または `pressure_hpa` の片方を指定し、時間順に並べます。座標とsourceは手動CSVでは省略可能です。例は `tests/fixtures/wind_profile.csv`。

読み込みでは対象時刻以前で1時間未満の最新プロファイルを採用します。対象期間外、NaN、重複高度、異なる地点のデータなどはエラーにします。全期間をメモリへ展開せず、1プロファイルだけ保持。API取得も最大24時間ずつ処理して一時CSVへ書き出し、完了した場合だけ保存先を置換します。地上3 mは現地テレメトリを優先し、上空の風をCSVから補います。

等圧面の高度は `z = T₀/L × [1 − (p/p₀)^(R L/g)]` で地上高に換算します（L=0.0065 K/m、R=287.05 J/(kg K)、g=9.80665 m/s²）。p₀は**観測地点の現地気圧**、T₀は現地気温。海面更正気圧QNHを直接代入する方式ではありません。負の高度は地下として除外します。これは乾燥大気・一定気温減率の近似であり、観測された各層の高度そのものではありません。Koinobori実機のUDP接続はフェーズ5-1に残り、現在の地上テレメトリは模擬値です。

### コードと検証

新規コード: `core/annealing.py`（自作SA）、`core/atmosphere.py`（気圧→AGL）、`utils/wind_data.py`（CSV/API）、`utils/magnetic.py`（WMMとキャッシュ）、`ui_qt/wind_data_dialog.py`（非同期の取得・読込画面）。KDE計算はMC終了後のWorkerで行い、風の描画では実行しません。

SAは射角60–90°・周期的な方位角を探索し、温度を下げながら悪化する候補も確率的に採用します。周期的な広域提案で別の実行可能領域も探します。シードと評価予算で再現可能ですが、大域最適解の保証はありません。上位候補と最終候補をMCで検証し、最終検証の失敗を過去の成功値で置換しません。MC半径は基準着地点からの散らばりとして求め、発射点からの距離を二重加算しないよう修正しました。

開発用依存は `python -m pip install -r requirements-dev.txt`。全テストは `python -B -X utf8 -m pytest -q`。Qt画面はオフスクリーンで検証し、通常テストは外部通信不要です。実飛行の数値基準はRocketPy 1.13で記録した合成推力データによる回帰比較であり、実飛行との一致を保証する試験ではありません。Open-Meteoへの実接続では10高度層のCSV保存・再読込も確認しました。
