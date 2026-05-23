# SR 3DGS Pipeline

スマートフォン動画または画像コレクションから、単一オブジェクト向けの
3D Gaussian Splatting 成果物を生成するためのパイプラインです。

[English](README.md) | [中文](README.zh-CN.md)

![3DGS example preview](docs/assets/toy-preview-cleaned.png)

SR 3DGS Pipeline は、撮影準備、フレーム選択、COLMAP カメラ復元、gsplat
学習、Gaussian クリーンアップ、Web 配信用パッケージングを統合します。主な対象は、商品、コレクション、卓上オブジェクト、玩具、小型スキャン対象などの静的オブジェクトです。

本リポジトリは、既存のオープンソースコンポーネントを統合した alpha 段階のエンジニアリングパイプラインです。新しい 3DGS 研究手法ではありません。出力品質は、撮影範囲、焦点、照明、対象物のテクスチャ、シーンの安定性に依存します。

## 機能

- スマートフォン動画および画像フォルダ入力。
- perspective および equirectangular フレーム抽出。
- COLMAP ベースのカメラ復元。
- gsplat 学習エントリポイント。
- オブジェクト crop、自動 mask、mask-aware training のオプション。
- 分離クラスタ、浮遊点、低信頼度の霞状 splat の自動クリーンアップ。
- フラットな `output/<scene>/` 配信用フォルダ。
- Web/モバイルプレビュー向け PlayCanvas SOG 出力。
- SuperSplat および後段ツール向け標準 PLY 出力。
- 抽出フレームと mask の入力品質レポート。
- CI セーフチェック、出力検証、配信スコアリング、ローカル HTTP プレビュー smoke test。

## サンプル

リポジトリには現在のオブジェクトデモの軽量プレビュー画像が含まれます。PLY、SOG、動画、復元ワークスペースなどの大きな生成物は git から除外されます。

![Point cloud contact sheet](docs/assets/toy-contact-sheet.png)

公開用フォルダ構成：

```text
output/<scene>/
  START_HERE.html
  preview.html
  <scene>_v<timestamp>.sog
  <scene>_high_quality.ply
  diagnostics.json
  manifest.json
```

SOG ファイル名にはタイムスタンプを含め、再公開時のブラウザキャッシュ問題を抑制します。

## 入力条件

推奨される撮影条件：

- 単一の主要な静止対象
- 20-60 秒のゆっくりした周回動画、または 40-180 枚の有効な静止画
- 対象物の周囲を十分に覆う視点
- 任意の低・中・高カメラ高さ
- 安定した露出と照明
- 対象物と視覚的に区別できる背景
- 低いモーションブラー、反射、透明度、変形

生物や変形する対象は、撮影中に実質的に静止している場合を除き、主な対象外です。

詳細は [docs/CAPTURE_GUIDE.md](docs/CAPTURE_GUIDE.md) を参照してください。

## インストール

基本インストール：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

学習依存関係：

```bash
pip install -e ".[training]"
```

任意の mask 依存関係：

```bash
pip install -r requirements-optional.txt
```

バックエンド確認：

```bash
python scripts/check_backend.py
```

## 動画から実行

```bash
python scripts/run_video_pipeline.py \
  --video input_videos/object.mp4 \
  --output_name object \
  --projection perspective \
  --preset standard \
  --object_mask auto \
  --cluster_clean
```

任意のオブジェクト crop：

```bash
--object_bbox left,top,right,bottom
```

Equirectangular 動画：

```bash
python scripts/run_video_pipeline.py \
  --video input_videos/object_360.mp4 \
  --output_name object_360 \
  --projection equirectangular \
  --equirect_face_size 1024 \
  --equirect_faces front,right,back,left \
  --object_mask auto \
  --cluster_clean
```

## 入力評価

```bash
python scripts/assess_scene_inputs.py workspace_video/object \
  --report workspace_video/object/reports/input_quality.json \
  --html workspace_video/object/reports/input_quality.html
```

`run_video_pipeline.py` は以下も出力します。

- `workspace_video/<scene>/reports/input_quality_frames.html`
- `--object_mask auto` 有効時の `workspace_video/<scene>/reports/input_quality_object.html`

評価内容は、フレーム数、ブラー、近重複、急な視点ジャンプ、前景 mask サイズ、mask が画像境界に触れているかどうかです。

## 検証とプレビュー

```bash
python scripts/validate_output.py output/object
python scripts/score_output.py output/object
python scripts/http_preview_smoke.py output/object
python scripts/serve_output.py --scene object --port 8765
```

ローカルプレビュー：

```text
http://127.0.0.1:8765/output/object/START_HERE.html
```

ブラウザで SOG を読み込むにはローカル HTTP 配信が必要です。`file://` からの直接プレビューは信頼できません。

## クリーンアップツール

- `scripts/cluster_clean_ply.py`: 主連結成分フィルタリング。
- `scripts/filter_ply_confidence.py`: 低信頼度 splat フィルタリング。
- `scripts/crop_ply_by_core.py`: 高信頼度コアに基づく候補生成。
- `scripts/compare_clean_candidates.py`: 候補メトリクス比較。
- `scripts/render_ply_contact_sheet.py`: 目視確認用 CPU contact sheet レンダリング。

クリーンアップモジュールは一般的な復元アーティファクトの低減を目的とします。十分なカメラカバレッジ、鮮明な入力フレーム、安定した照明、静的なシーンジオメトリの代替にはなりません。

## リポジトリチェック

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/ci_check.py
PYTHONDONTWRITEBYTECODE=1 python scripts/release_readiness.py
PYTHONDONTWRITEBYTECODE=1 python scripts/release_readiness.py --include_output
```

生成された復元アセットはバージョン管理に含めません。ソースコード、設定、ドキュメント、テスト、軽量プレビュー画像を追跡します。

## 状態

実装済み：

- 動画および画像フォルダのオーケストレーション
- perspective / equirectangular フレーム抽出
- COLMAP/pycolmap カメラ準備
- gsplat 学習エントリポイント
- 標準 PLY 出力
- PlayCanvas SOG viewer 出力
- オブジェクト crop と mask-aware training hook
- Gaussian 自動クリーンアップユーティリティ
- 入力品質レポート
- 出力検証とローカル HTTP プレビュー確認

計画中：

- より広範な公開 benchmark シーン
- セグメンテーションバックエンドの改善
- 複数撮影デバイスに対するデフォルトパラメータ改善
- 専用マシン向けの任意 visual QA ワークフロー

## 関連プロジェクト

- [COLMAP](https://colmap.github.io/)
- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [Nerfstudio / Splatfacto](https://docs.nerf.studio/)
- [PlayCanvas SuperSplat and SOG tooling](https://github.com/playcanvas)

プロジェクトの位置づけは [docs/PROJECT_POSITIONING.md](docs/PROJECT_POSITIONING.md) を参照してください。

## コントリビューション

公開可能なテスト素材、クリーンアップ手法、セグメンテーションバックエンド、benchmark シーン、ドキュメントの貢献を歓迎します。

[CONTRIBUTING.md](CONTRIBUTING.md) を参照してください。

## License

MIT. See [LICENSE](LICENSE).
