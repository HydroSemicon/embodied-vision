# Embodied Vision

YOLO + BoT-SORTによる人物追跡と、InsightFaceによる登録人物照合を行うローカルサービスです。

## 動作

- YOLOで検出した人物を、外観ReIDとカメラ動き補正を有効にしたBoT-SORTの`track_id`で追跡します。
- 人物を登録人物または未登録人物として確定したときだけ、`kokomi_kernel`へ通知します。
- 顔照合は追跡ループとは別のワーカースレッドで実行します。
- InsightFaceのSCRFDで顔を検出・位置合わせし、小さすぎる顔、ぼけ、露出不良を除外します。
- `buffalo_l`の複数フレーム投票で人物を確定し、単発の誤判定を抑えます。
- 同じ登録人物が複数のtrack IDで検出された場合は1つへ統合し、認識・離脱イベントの重複を防ぎます。
- 未登録判定は十分な時間と高品質な不一致サンプルがそろうまで保留します。
- 登録時は最大30秒間サンプルを集め、外れ値と重複を除いた複数の特徴ベクトルを保存します。顔画像は保存しません。

## セットアップ

Python 3.11環境で以下を実行します。

```powershell
pip install -r deepsort-py311requirements.txt
pip install -r face-recognition-requirements.txt
# InsightFace 1.0.1が依存関係として入れるCPU版を除去し、CUDA 12版だけを再導入
pip uninstall -y onnxruntime onnxruntime-gpu
pip install --no-cache-dir onnxruntime-gpu==1.20.2
python deepsort.py
```

InsightFaceの`buffalo_l`モデルパックは初回利用時に取得されます。Windowsでは`onnxruntime-gpu`を使用します。CPU版の`onnxruntime`だけが見つかった場合は、気付かないまま低速動作しないよう顔認識ワーカーをエラーにします。JetsonではJetPackに適合するNVIDIA向けONNX Runtime GPU wheelを使用し、PyPI版が対応しない場合は`face-recognition-requirements.txt`の`onnxruntime-gpu`を除いて個別に導入してください。

InsightFaceの公開済み事前学習モデルは非商用研究用途です。製品として配布・運用する場合は、使用するモデルのライセンスを別途確認してください。

## ローカル確認UI

`deepsort.py`を起動して、ブラウザで次を開きます。

```text
http://localhost:5000/
```

Aliceや`kokomi_kernel`を起動しなくても、以下を確認・操作できます。

- 追跡ラベル付きカメラ映像
- カメラ、InsightFaceワーカー、ONNX実行プロバイダ、イベント送信の稼働状態
- 現在の`track_id`、照合状態、最近傍候補、コサイン距離
- 登録済み人物と保存されている特徴量サンプル数
- 現在見えている人物の顔登録
- Aliceへの人物・顔イベント送信のON/OFF

主な環境変数：

| 変数 | 既定値 | 用途 |
|---|---:|---|
| `VISION_EVENT_SEND_ENABLED` | `true` | Aliceへのイベント送信 |
| `FACE_EVENT_SEND_ENABLED` | `true` | 起動時の人物・顔イベント送信状態 |
| `VISION_EVENT_URL` | `http://localhost:3000/yolo_event` | イベント送信先 |
| `VISION_EVENT_REQUEST_TIMEOUT_SECONDS` | `10.0` | Aliceへの1回の送信待ち時間 |
| `VISION_HTTP_HOST` | `127.0.0.1` | HTTP APIの待受アドレス |
| `VISION_HTTP_PORT` | `5000` | HTTP APIの待受ポート |
| `VISION_DISPLAY_ENABLED` | `true` | OpenCVウィンドウ表示 |
| `VISION_UI_STREAM_FPS` | `30` | ブラウザ映像の最大配信fps（最大120） |
| `VISION_UI_JPEG_QUALITY` | `90` | ブラウザ映像のJPEG品質（50〜100） |
| `VISION_YOLO_CONFIDENCE` | `0.50` | 新しい人物トラックを開始する最低信頼度 |
| `VISION_YOLO_TRACK_CONFIDENCE` | `0.10` | BoT-SORTが既存トラックとの再関連付けに使う低信頼度検出の下限 |
| `VISION_YOLO_MIN_PERSON_WIDTH` | `40` | 人物候補の最小幅（px） |
| `VISION_YOLO_MIN_PERSON_HEIGHT` | `80` | 人物候補の最小高さ（px） |
| `VISION_YOLO_MIN_PERSON_AREA_RATIO` | `0.002` | 人物候補が画面に占める最低面積比 |
| `FACE_RECOGNITION_ENABLED` | `true` | 登録顔照合 |
| `VISION_TRACKER_CONFIG` | `botsort.yaml` | BoT-SORT設定ファイル。既定ではReIDとカメラ動き補正を有効化 |
| `FACE_MODEL_NAME` | `buffalo_l` | InsightFaceのモデルパック |
| `FACE_MATCH_THRESHOLD` | `0.55` | コサイン距離の一致上限 |
| `FACE_EXECUTION_PROVIDERS` | `CUDAExecutionProvider,CPUExecutionProvider` | 優先するONNX Runtime実行プロバイダ。JetsonでTensorRTを使う場合は先頭へ追加 |
| `FACE_REQUIRE_GPU` | `true` | TensorRT/CUDAが使えない場合にCPUへ黙ってフォールバックせずエラーにする |
| `FACE_MODEL_ROOT` | InsightFace既定値 | モデル保存ルート。未指定ならInsightFaceの標準保存先 |
| `FACE_CONFIRMATIONS` | `3` | 認識確定に必要な投票数 |
| `FACE_RECOGNITION_WINDOW` | `8` | 認識投票に使う直近サンプル数 |
| `FACE_RECOGNITION_MIN_VOTE_RATIO` | `0.6` | 認識確定に必要な投票比率 |
| `FACE_CONSECUTIVE_CONFIRMATIONS` | `2` | 同一人物の連続一致による早期確定回数 |
| `FACE_UNKNOWN_CONFIRMATIONS` | `40` | 未登録確定に必要な高品質不一致数 |
| `FACE_UNKNOWN_MIN_SECONDS` | `15.0` | 未登録確定までの最低観測時間 |
| `FACE_ANALYSIS_INTERVAL_SECONDS` | `0.25` | 同一トラックの顔解析間隔 |
| `FACE_MIN_FACE_SIZE` | `70` | 採用する顔の最小幅・高さ（px） |
| `FACE_MIN_DETECTION_CONFIDENCE` | `0.70` | 採用するSCRFD顔検出信頼度 |
| `FACE_MIN_BLUR_VARIANCE` | `35.0` | ぼけ除外のしきい値 |
| `FACE_INPUT_MAX_DIMENSION` | `640` | InsightFaceへ渡す人物切り出し画像の最大辺（px） |
| `FACE_DETECTION_SIZE` | `640` | SCRFDの検出入力サイズ |
| `FACE_PERSON_CROP_TOP_RATIO` | `0.75` | 人物枠の上側から顔を探す範囲 |
| `FACE_ENROLLMENT_SECONDS` | `30.0` | 非同期登録セッションの最大時間 |
| `FACE_ENROLLMENT_MIN_SECONDS` | `8.0` | 登録時に観測する最低時間 |
| `FACE_ENROLLMENT_MIN_SAMPLES` | `4` | 登録成立に必要な一貫した代表サンプル数 |
| `FACE_ENROLLMENT_TARGET_SAMPLES` | `8` | 登録時に集める目標サンプル数 |
| `FACE_ENROLLMENT_EXEMPLARS` | `8` | 1回の登録で選ぶ代表特徴量数 |
| `FACE_MAX_EMBEDDINGS_PER_PERSON` | `12` | 1人あたりの最大保存特徴量数 |
| `FACE_REGISTRY_PATH` | `data/face_registry_buffalo_l.json` | 登録データ保存先 |
| `TRACK_MAX_AGE` | `90` | BoT-SORTとイベント層が消失トラックを保持するフレーム数 |

## HTTP API

- `GET /status`：カメラ・顔認識ワーカーの状態
- `GET /face-events`：人物・顔イベント送信スイッチの状態
- `POST /face-events`：人物・顔イベント送信の有効・無効を実行中に変更
- `GET /snapshot`：Aliceに渡す生画像
- `GET /snapshot/annotated`：IDと認識名を描画した確認画像
- `GET /stream/annotated`：UI用の追跡ラベル付きMJPEGストリーム
- `GET /tracks`：現在のトラックと認識状態
- `GET /faces`：登録人物一覧。顔特徴ベクトルは返しません
- `POST /faces/enroll`：現在のトラックの非同期登録を開始（`202 collecting`）

## `kokomi_kernel`へ送るイベント

人物・顔イベントは、既定では次のエンドポイントへHTTP POSTします。

```text
POST http://localhost:3000/yolo_event
Content-Type: application/json
```

送信先は`VISION_EVENT_URL`で変更できます。すべてのイベントは次の共通形式です。

```json
{
  "event": {
    "event_id": "9be54ee461174472b781bac112790c2d",
    "source": "deepsort",
    "type": "person_recognized",
    "track_id": "14",
    "timestamp": "2026-09-09T06:20:18.123456+00:00",
    "identity": {
      "status": "recognized",
      "person_id": "26b7e2c15a1e4449974367f7da686b74",
      "name": "KOT",
      "distance": 0.2563,
      "threshold": 0.55
    },
    "message": "The visible registered person is KOT."
  }
}
```

`event_id`はイベントごとに生成する一意なIDです。同じイベントの再試行では変更しないため、受信側はこの値で重複を除外できます。`timestamp`はUTCのISO 8601形式、`track_id`は文字列です。

### イベント種別

| `type` | 送信条件 | 追加情報 |
|---|---|---|
| `person_recognized` | 登録人物との照合が確定 | 同一人物の2回連続一致、または通常の複数票で確定 |
| `person_unknown` | 15秒以上観測し、高品質な不一致が40件連続 | 一時的な顔検出失敗だけでは送信しない |
| `person_enrolled` | 顔登録がバックグラウンドで完了 | 登録直後なので`identity.distance`は`0.0` |
| `person_disappeared` | `person_recognized`、`person_unknown`または`person_enrolled`を通知済みの人物トラックが消失 | 最後の`identity`と`position`を設定 |

`identity`は常に以下の5フィールドを持ちます。

| フィールド | 内容 |
|---|---|
| `status` | `pending`、`recognized`、`unknown`、`unavailable`のいずれか |
| `person_id` | 登録人物のID。未確定時は`null` |
| `name` | 登録名。未確定時は`null` |
| `distance` | 登録特徴量とのコサイン距離。小さいほど一致。未確定時は`null` |
| `threshold` | 一致判定に使用した距離の上限 |

### 通常の送信順序

人物を検出しただけではイベントを送りません。登録人物または未登録人物として確定してから通知します。顔を取得できない人物や、YOLOが一時的に誤検出した黄色枠からはイベントを送りません。

```text
人物出現
  ├─ 登録人物との照合成功
  │    person_recognized
  │    → その人物が離れたら person_disappeared
  │
  ├─ 顔を取得できない・人物候補が短時間で消える
  │    イベントなし
  │
  └─ 顔は取得できるが未登録
       15秒以上かつ不一致40件後 person_unknown
       → その人物が離れたら person_disappeared
```

`person_appeared`は送信しません。登録セッション中は`person_unknown`を抑制し、登録成功後に`person_enrolled`を送ります。`person_disappeared`も、認識結果または登録完了を一度も通知していないトラックについては送信しません。

### 配信とイベントスイッチ

イベントは最大100件の非同期キューから送信します。HTTP送信に失敗した場合は、同じ`event_id`のまま待ち時間`0`、`0.2`、`0.5`、`1.0`秒で最大4回試行します。キューが満杯の場合は新しいイベントを破棄します。

`VISION_EVENT_SEND_ENABLED=false`または実行中の顔イベントスイッチがOFFの場合、人物・顔イベントは送信しません。スイッチをOFFにした時点で待機中のイベントも破棄し、ONへ戻しても過去のイベントは再送しません。カメラ、YOLO、BoT-SORT、InsightFace、顔登録、スナップショットAPIはそのまま動作を続けます。

OpenCVウィンドウでは`E`キーでも人物・顔イベント送信を切り替えられます。

```powershell
# Aliceへの人物・顔イベントを無効化
Invoke-RestMethod -Method Post `
  -Uri http://localhost:5000/face-events `
  -ContentType application/json `
  -Body '{"enabled":false}'

# 再び有効化
Invoke-RestMethod -Method Post `
  -Uri http://localhost:5000/face-events `
  -ContentType application/json `
  -Body '{"enabled":true}'
```

登録例：

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:5000/faces/enroll `
  -ContentType application/json `
  -Body '{"track_id":"7","name":"たかん"}'
```

Aliceからは`remember_person`アクションで同じAPIを呼び出せます。APIは収集開始時に`202 collecting`を返し、Python側は最大30秒間バックグラウンドで顔を集めます。正面を見てから顔をゆっくり左右へ向けてください。8秒以上かつ目標8サンプルで早期完了し、30秒時点で一貫した代表サンプルが4個未満なら保存しません。進捗と最終結果は`GET /tracks`およびブラウザUIに表示され、成功時は`person_enrolled`イベントがAliceへ送られます。同じ名前で再登録すると、最大12個まで代表特徴ベクトルが追加されます。

以前のSFace登録は`data/face_registry.json`、Facenet512登録は`data/face_registry_facenet512.json`に残りますが、`buffalo_l`の特徴量とは互換性がないため自動変換しません。この構成へ切り替えた初回は人物を再登録してください。

`source`はカーネルとの後方互換性のため引き続き`deepsort`を送ります。内部の追跡器がBoT-SORTへ変わっても、イベントの種類・JSON構造・受信URLに変更はありません。
