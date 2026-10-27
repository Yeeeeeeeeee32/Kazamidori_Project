# Kazamidori 監査修正の対応記録（2026-10-24）

対象：[監査報告書](KAZAMIDORI_AUDIT_2026-10-24.md)のF01〜F15。
モデル識別子：`time-events-v2`。Koinoboriの観測高度1.5 mと練習風のランダム生成は維持。

## 修正内容

| 指摘 | 対応 | 主な確認先 |
| --- | --- | --- |
| F01 気圧高度の逆変換 | 単調な気圧テーブルを使用。環境生成時に気圧→高度の往復を検査 | core/simulation.py::build_environment |
| F02 架空の開傘時刻 | 燃焼終了＋遅延まで積分し、その状態を引き継いで開傘。時刻はRocketPyの実イベントから取得 | simulate_once、test_real_deployment_event_and_canopy_response |
| F03 未着地を成功扱い | 地面到達イベント・有限な軌道・最終高度・実開傘を検査。未着地/未離床/開傘前着地を別の失敗として返す | test_late_ejection_and_timeout_are_not_successful_landings |
| F04 旧GO・条件混在 | 入力版、run_id、実行条件のコピーを保持。物理入力変更でGO解除、計算中変更は旧結果を破棄。出力は実行時の条件を使用 | Controller、test_physical_edits_invalidate_go_and_stale_completion_is_discarded |
| F05 MCモデル不一致 | 探索と通常MCで同じ風摂動と時間イベント式を使用。高速MCの旧高度再利用は廃止し、互換入口から同じ計算へ委譲 | test_search_uses_identical_wind_model |
| F06 確率条件の不一致 | 目標中心からの距離と全試行の円内率を使用。失敗は分母に含め、分位半径も全試行の最近順位法で判定 | test_mc_gate_counts_failures_and_target_directly |
| F07 予報表示の上書き | CSV由来の上空風を練習タイマーの生成対象から分離。地点/CSV/対象時刻の切替で旧上空履歴・平均をリセット。Pending/予報対象時刻を表示 | test_forecast_upper_nodes_survive_practice_tick、test_weather_source_switch_does_not_average_old_forecast |
| F08 統計キー欠落 | 通常Runと最適化後の統計計算を共通化。平均/最低高度、水平距離、滞空時間を同じキーで受け渡す | SimulationWorker._compute_stats |
| F09 得点の分母 | 有効得点のみの平均・最低であることをラベルに明示。完了/失格/計算失敗件数を表示 | test_statistics_are_complete_and_score_denominator_is_visible |
| F10 部分読込・直接上書き | セッション全体を検証してから適用。セッションと機体JSONを一時ファイル→置換で保存 | test_failed_session_load_is_transactional、test_failed_session_write_preserves_previous_file |
| F11 シード未反映 | 最終MCのシードを設定から生成し、試行IDへ対応付け。入力全体、依存版、モデル版を結果に保存 | test_mc_seed_repeats_trial_inputs_and_worker_copies_nested_data |
| F12 無断補正 | 質量・半径・フィン位置の無断補正を廃止。範囲外は理由付きで拒否、小さい正の質量はそのまま計算 | test_small_mass_is_not_inflated_and_bad_position_is_rejected |
| F13 ODE設定未使用 | モード別のrtol/atol/最大時間刻みを各Flightへ実際に渡し、結果にも保存 | core/simulation.py::solver_options |
| F14 終了時のWorker破棄 | 閉じる要求で協調停止し、Qtタイマーで非同期に終了を待つ。動作中はCloseイベントを保留 | test_close_waits_for_live_thread_without_blocking_gui |
| F15 座標変換の不統一 | 着地点表示を共通のoffset_to_latlonへ統一。目標オフセットも判定・描画・保存/復元で一致させる | test_motor_curve_and_target_change_also_invalidate_go、test_export_keeps_execution_context_after_edit |

## 計算結果が変わる点

旧モデルは、診断用の機体で開傘せずに落下する値を回帰基準に含んでいた。
新モデルでは開傘を確認したうえで基準を更新した。

| 診断値 | 旧モデル | 修正後 |
| --- | --- | --- |
| 最高高度 | 約93.43 m | 約93.41 m |
| 飛行時間 | 約9.20秒 | 約27.54秒 |
| 着地North成分 | 約36.93 m | 約25.52 m |
| 開傘イベント | なしでも開傘時刻を記録 | 実イベントに基づき、開傘完了は約5.605秒 |

これは既存fixtureと小型の診断用推力曲線の値であり、実機の測定値ではない。
試験ではイベント記録、気圧高度往復、傘面積変更による降下時間の応答、
長い遅延や120秒の打切りが失敗になることを確認する。
単に旧値を新値へ置き換えただけの検証ではない。

MCの表示半径は従来どおり成功点の中心からの分位半径を表示する。
探索の制約用半径は目標中心からの全試行に対する経験分位点であり、用途が異なる。
GO判定は全試行に対する目標円内率を使う。
得点の条件付き平均を、全試行の期待得点として扱わない。

## 実行条件と保存

通常の実行結果に `input_snapshot` を保存する。
射点、目標、機体、モータ推力、傘、風プロファイル、気圧/気温、
不確実性、シード、依存版、モデル版、run_id、入力版が含まれる。
実行後に入力を編集しても、結果出力のcontextを新しい入力で上書きしない。
古いアーカイブも従来の形式で読み込める。過去結果の読込では風況監視を開始しない。

計算中は入力欄と操作ボタンをロックする。
プログラム経由の入力変更も版で検出し、完了した古い結果を採用しない。
風況更新自体は計算中も継続する。

依存関係の確認済み直接バージョンは[constraints-tested.txt](constraints-tested.txt)に記録した。
新規環境でのインストール例：

```powershell
python -m pip install -r requirements.txt -c constraints-tested.txt
```

これは直接依存の確認済み組合せで、全OS・全推移的依存の再現性を保証するロックファイルではない。

## 検証

- [追加のコア回帰テスト](tests/core/test_audit_fixes.py)
- [追加のUI回帰テスト](tests/ui_qt/test_audit_fixes.py)
- [更新した実飛行計算の回帰基準](tests/core/test_regression.py)
- 全体実行：`python -B -X utf8 -m pytest -q --disable-warnings`
- 全体結果：**246 passed、52 subtests passed**（126.94秒、警告56件）。
- 最後に保存/読込の半径復元順序も修正し、関連UI・結果メニューテストを再実行：**19 passed**（26.75秒、警告18件）。
- 警告には既存のQtシグナル切断・Matplotlibレイアウトに関するものを含む。テスト失敗はなし。

外部APIやKoinobori実機への接続は行っていない。
旧監査の[再現JSON](assets/verification/kazamidori_audit_20261024.json)と
archive内の監査プローブは修正前の資料として保持した。
現行モデルの確認には上記テストを使う。

## 残るモデル上の制約

監査報告の「モデル上の制約・不足」は、実測仕様なしに解消したとは扱わない。

- モータ内部形状、モータ慣性、機体慣性の一部は近似。ケース・マウントの実寸適合は別確認。
- 初期静安定の検査は全速度域・全燃焼過程の安定性を保証しない。
- CPの校正はMach 0を基準にしたモデル。実機の空力同定ではない。
- 風の不確実性は一試行ごとに固定する高度別モデル。時間相関乱流ではない。
- 開傘後の水平風速追従率は運動学近似で、鉛直運動を含む完全な傘の力学ではない。
- 現在は練習用の模擬風・気圧・温湿度。上空CSVを選んだ場合はその上空風を保持する。
- 高度の原点はAGL、Environmentのelevationは0という近似。湿度は密度モデルに未反映。
- GOは有限試行の経験割合。統計的な信頼下限や実飛行精度の保証を追加したものではない。
- Phase Bは地表風の簡易監視。実機通信・上空風の常時再取得・禁止領域判定は追加要件。
- Stop/Closeは協調取消。積分やHTTP処理の途中を強制終了せず、処理が戻るまで待つ。

React移行、多段化、サロゲート学習、実機Koinobori通信は本修正で追加していない。

