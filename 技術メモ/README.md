# 技術メモ

更新日：2026-10-27。ファイル名は`NN_topic.md`、本文は日本語に統一しています。
各分野はこの1組の最新版を上書き更新し、旧版・変更履歴・訂正の経緯は保存しません。
更新日と、現行仕様／未実装設計の区別は残します。

| 文書 | 内容 |
| --- | --- |
| [01_architecture.md](01_architecture.md) | 責務、処理経路、単位・座標、非同期処理 |
| [02_physics_model.md](02_physics_model.md) | 飛行、実開傘、モータ、慣性、空力、傘の近似と検証 |
| [03_statistics_and_weather.md](03_statistics_and_weather.md) | 気象入力、風統計、MC、楕円・KDE、GO・監視 |
| [04_data_and_configuration.md](04_data_and_configuration.md) | .ork、専用設定、config・セッション・結果の契約 |
| [05_ui_and_operations.md](05_ui_and_operations.md) | Run／Stop、各モード、モンブラン条件、描画・地図 |
| [06_performance_and_parallelism.md](06_performance_and_parallelism.md) | 測定、プロセス並列、メモリ、数値計算の受入条件 |
| [07_ui_migration.md](07_ui_migration.md) | Python／PySide6／Reactの段階移行 |
| [08_future_models.md](08_future_models.md) | 最適化・学習、多段・振り子、出典付きプリセット |

入口は[README](../README.md)、次の作業と保留判断は[開発計画](../ROADMAP.md)を参照してください。
Koinoboriは製作中の独立した風向風速計プロジェクトで、現行の模擬テレメトリとは区別します。
