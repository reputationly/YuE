# GPUStack 接入说明（ARM64 / A100）

本仓在上游 YuE2 之上加了一层 harness，把它接成 GPUStack 的内置后端 `YuE2`，
承接「文生音乐（歌词 → 整首歌）」和「翻唱（录音 → 扒谱 → 换风格重唱）」能力。

上游代码一行没动，改动只有：`server.py` + `harness/`（异步任务 API）、
`docker/Dockerfile.arm64`、CI、契约测试、这份文档，以及 `pyproject.toml` 里
pytest 的 `pythonpath` 补了一个 `.`。

---

## 1. 为什么是这个形态，而不是进 vllm-omni

- 上游唯一的加速路径（`src/yue2/fast.py`）硬钉 `vllm==0.19.0`，vllm-omni 目标 0.29；
  transformers 也对撞（上游 4.57.6，vllm-omni ≥5.13）。
- 三个生成阶段串行独占 GPU（NAR 前放掉 AR、VAE 前把主干搬回 CPU），官方就是
  one request at a time，vllm-omni 的连续批处理用不上。
- 卖点是**可编辑的乐谱**，塞进 `/v1/audio/speech` 只剩「文字进音频出」。

## 2. 走哪扇门

YuE2 复用门面**已有的 music 门**：`task_type=t2m|cover` → engine kind `music` →
`POST /v1/tasks/music/` → `.mp3`，与 ACE-Step 同一扇门。**new-api 与门面零改动**，
只需要在 GPUStack 登记后端（见 §9）。

| 玩法 | 支持 | 怎么走 |
|---|---|---|
| 文生音乐（t2m） | ✅ | `prompt` + `lyrics`；`cot` 选 full / melody / off |
| 给谱渲染、改谱重渲染 | ✅ | 请求带 `abc`；出歌时写 `<stem>.abc`，new-api 经 `metadata.score_abc` 返回（见 §4） |
| 翻唱（cover） | ✅ | 门面的 `reference_audio` → `reference_audio_path`，引擎内 SheetSage2 扒谱（见 §5a） |
| 局部重绘（repaint） | ❌ | YuE2 没有局部重绘能力；带 `src_audio_path` 的请求提交时 **400** |

`task_type` 是门面的控制字段，到不了引擎，所以引擎靠请求里有没有
`reference_audio_path` / `src_audio_path` 区分三种任务。**这两个字段必须显式处理**：
按「未知字段忽略」处理的话，翻唱和重绘请求会悄悄退化成一首不相干的新歌、还显示成功。

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/tasks/music/` | 提交，返回 `{task_id, task_status, save_result_path}`；队列满 **503** |
| POST | `/v1/tasks/audio/` | 同上的别名（ACE-Step 也留了，手工测试用） |
| GET | `/v1/tasks/{id}/status` | 状态 + `phase` + `progress`；未知任务 **404** |
| GET | `/v1/tasks/{id}/result` | 取音频，仅 `completed` |
| DELETE | `/v1/tasks/{id}` | 取消（**真中断**，见 §6） |
| GET | `/v1/tasks/queue/status` | 队列水位 |
| GET | `/ready` | 健康检查 |

状态字符串 `pending | processing | completed | failed | cancelled`（**cancelled 双 L**，
门面 `_ENGINE_STATE_MAP` 逐字匹配）。未知任务必须 404——门面 sweeper 靠它判断「引擎重启、
任务已丢」并重派。

`/ready` 的语义：引擎在权重加载 + 预热完成之前**根本不监听端口**（uvicorn 的 lifespan），
所以加载期是连接拒绝而不是 503。GPUStack 健康检查两者都当「未就绪」，Breeze 线上同样如此。

## 3. 请求字段

new-api 把客户端 `metadata` 摊平到顶层并设 `prompt`；门面剥掉控制字段、注入
`save_result_path`。其余字段（`model`、`user_id`、ACE-Step 的 `audio_duration`/`bpm`…）
**忽略而不拒绝**。

```jsonc
{
  "prompt": "Mandarin Chinese, modern C-pop ballad, female vocal, piano and strings, 76 BPM",
  "lyrics": "[Verse]\n街灯把影子拉得很长\n...\n[Chorus]\n...",  // 可为空 = 纯音乐
  "cot": "full",        // full(旋律+和弦) | melody | off(不出谱);缺省 full
  "abc": "X:1\n...",    // 可选:给一份乐谱直接渲染(改谱重渲染),需 cot=full|melody
  "seed": 42,           // 可选;缺省在提交时随机钉一个并写进旁挂 json
  "cfg_scale": 1.0,     // 可选,[0, 20]
  "save_result_path": "/nfs-output/.../x.mp3"   // 门面注入;后缀决定格式 mp3|wav|flac
}
```

`prompt` 也接受 YuE2 自己的拼法 `style` / `tags`。

**seed 缺省随机**，这是和上游的有意差异：上游每个请求缺省都是 831001，同一个 prompt
所有用户都会拿到同一首歌。

## 4. 输出：音频 + 两个旁挂

`save_result_path` 同目录下，按「旁挂先写、音频最后写」的顺序各自原子落盘
（音频出现在最终路径 = 完成信号，读到歌的人一定找得到谱）：

| 文件 | 内容 |
|---|---|
| `<stem>.mp3` | 320 kbps CBR、48 kHz 立体声（ffmpeg/LAME）；`.wav`/`.flac` 为 24-bit |
| `<stem>.abc` | YuE2 规划出的乐谱（`cot=off` 时没有） |
| `<stem>.json` | seed、模式、乐谱来源、token 数、分段耗时、权重 sha256 |

门面的 janitor 按**天目录整体 rmtree**，旁挂跟音频一起被清，不会泄漏。

**乐谱怎么到用户手上**：new-api 在任务成功时从共享挂载读 `<stem>.abc`，放进响应的
`metadata.score_abc` 原样返回（`service/music_score.go`，只读挂载根下、≤64 KiB 的 UTF-8 文件，
读不到就不带，不影响任务成败与计费）。调用方改完，放进下一次请求的 `metadata.abc` 即可——
new-api 把 metadata 摊平到顶层，门面与 new-api 的剥离名单里都没有 `abc`，会原样到达引擎。

**改谱重渲染是逐字节可复现的**：把 `<stem>.abc` 原样作为 `abc` 交回、seed 不变，
得到的 MP3 与原曲 md5 相同（实测）。所以只改谱里的一处，听到的差异就只来自那一处；
重渲染还省掉规划阶段（44 秒的歌 21.5 s → 17.6 s）。

## 5. 长度上限

| 上限 | 值 | 超限行为 |
|---|---|---|
| 语义 token | 9000（≈25 Hz，约 **6 分钟**音频） | 上游**静默截断**，只置 `truncated` 标志 |
| 乐谱 token | 4096 | 同上 |

截断出来的是一首唱到一半停掉的歌，任务却显示成功——所以两层防线：

1. **提交时按歌词预算 400**：`MAX_LYRIC_UNITS = 600`，按「CJK 等效字数」算，
   汉字/假名/谚文记 1，其余非空白字符记 0.385（拉丁字母的 token 密度约为汉字的 0.4 倍）。
2. **运行时截断即失败**：任一阶段 `truncated` → 任务 `failed`，`error_type="truncated"`，
   不产出音频。

为什么是 600（A100 实测）：

| 歌词 | 音频 | 语义 token | 占上限 |
|---:|---:|---:|---:|
| 335 字 | 2:43 | 4079 | 45% |
| 483 字 | 3:50 | 5746 | 64% |
| 588 单位 | 3:58 | 5947 | 66% |

长歌词稳定在 ~11 token/单位；短歌词会被编成比字数长得多的完整编曲（4 行歌词也能出
2:30）。时长最终是模型自己的选择，600 留出约 1/3 的余量给这种编曲波动。

## 5a. 翻唱

照上游的翻唱流程（`docs/covers.md`），但在引擎进程内完成：

```text
reference_audio_path → SheetSage2（自动加载 MERT-v2-FullSong）扒谱
  → 上游 abc_tools 校验谱面（melody 模式还校验没有和弦）
  → YuE2 按这份谱 + 新的 prompt/lyrics 重唱（cot 缺省 melody）
```

- **缺省 `cot=melody`**：扒谱时去掉和弦，伴奏按新风格自由发挥（上游对换风格的推荐）；
  `cot=full` 保留原曲和声；`cot=off` 与翻唱矛盾，提交时 400。
- **歌词必须由调用方给**：YuE2 按段落把词对到旋律上，它不会从录音里听出歌词。
- 扒出来的谱照样写成 `<stem>.abc` 旁挂，翻唱完还能接着改谱重渲染；`<stem>.json` 的
  `cover` 字段记源音频路径、扒谱告警、扒谱耗时。
- **源音频 ≤ 330 秒**（提交时用 ffprobe 量）：翻唱时长≈源时长（3:58 → 3:54 实测），
  而语义上限约 6 分钟。
- 扒谱失败（没扒出谱、谱面校验不过）→ 任务 `failed`，`error_type="transcription"`。
- **扒谱阶段的取消要等这一轮扒完**：SheetSage2 没有中途停止的钩子。状态立刻变
  `cancelled`，但 GPU 要到扒谱结束才释放（4 分钟源实测多占 26 s，330 s 源最坏约 40 s）；
  进入生成阶段后照旧 1 秒内生效。

权重：`SheetSage2/`（229 MB）与 `MERT-v2-FullSong/`（2.53 GB）放在模型目录下
（`YUE2_SHEETSAGE2_DIR` / `YUE2_MERT_DIR` 可覆盖）。两者缺任何一个，实例照常起、
`/ready` 返回 `"cover": false`，翻唱请求提交时 400。

SheetSage2 自己钉的是 torch 2.8 / transformers 4.45 / numpy 1.24——**这是作者环境的快照，
不是硬要求**：它在本镜像的 torch 2.11 / transformers 4.57.6 / numpy 2.5 上原样跑通，
只多装 pretty_midi / mido / mir_eval 三个纯 Python 包（numpy 1.24 连 Python 3.12 的轮子都没有，
照钉就得另起一套 3.11 环境）。唯一的坑是 transformers 4.57 加载 `trust_remote_code` 时只拷贝
直接相对导入的文件，SheetSage2 有二级导入会报找不到 `chord_spelling_sheetsage2.py`，
所以 harness 把快照目录当普通 Python 包导入。

| 源歌 | 扒谱 | 扒谱显存 | 翻唱总耗时 |
|---:|---:|---:|---:|
| 45 s | 5 s | 3.4 GiB | 25-37 s |
| 3:58 | 28 s | 3.4 GiB | ~115 s |

## 6. 取消是真中断

与 Breeze 不同（它只能跑完再丢结果）：YuE2 的 AR 每个 token、flow matching 每一步都
轮询 `cancelled()`，worker 把任务的 `stop_event` 直接交给流水线。实测 DELETE 后
**1.0 秒**下一个任务就已经在跑，被取消的任务不留任何文件。

## 7. 进度上报

门面的契约里，引擎自己折好的全局 `progress` **优先于** `phase + phase_progress`
（`gpustack/server/video_progress.py`）。YuE2 报全局值，因为门面 music 类的权重表
（denoise 70 / decode 10）对不上它的真实耗时分布：

| 阶段 | 全局区间 | 推进方式 | 实测占比 |
|---|---|---|---|
| 扒谱（仅翻唱） | 0 → 20，`phase=encode` | 阶段边界 | ~17-24% |
| 规划乐谱 | 0 → 30 | 按 token（预期 ≈ 4.3 × 歌词单位） | ~21-33% |
| 语义 token | 30 → 80 | 按 token（预期 ≈ 2.7 × 乐谱 token） | ~50% |
| flow matching | 80 | 阶段边界（无公开回调） | ~12-16% |
| VAE 解码 | 92 | 阶段边界 | ~5% |
| MP3 编码 + 写盘 | 97 | 阶段边界 | ~3-6% |

`phase` 仍用门面的词表（prepare / denoise / decode / save）做标签。进度在引擎侧就保证
单调不回退。

## 8. 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `YUE2_MODEL_DIR` | `/weights` | pipeline 根目录；GPUStack 注入模型路径 |
| `YUE2_VAE_DIR` | 空 | 仅当 `YUE2_MODEL_DIR` 指向裸的 YuE2-3B 目录时需要 |
| `YUE2_SHEETSAGE2_DIR` | `<model>/SheetSage2` | 翻唱扒谱模型 |
| `YUE2_MERT_DIR` | `<model>/MERT-v2-FullSong` | SheetSage2 的编码器 |
| `YUE2_MAX_QUEUE` | `8` | 单实例 FIFO 容量，满了 503 |
| `YUE2_MEMORY_BUDGET_GIB` | `24` | 上游的单进程显存上限（其中预留 2 GiB） |
| `YUE2_MP3_BITRATE` | `320k` | MP3 码率 |
| `YUE2_WARMUP` | `1` | 就绪前先跑一首小歌走通全链路 |

**引擎没有 CLI**：服务命令是 `python3 -m uvicorn server:app`，GPUStack 只注入
`--host/--port`，部署页的「后端参数」会被忽略（日志留 warning）。

## 9. 部署到 GPUStack

**权重布局**（一个模型路径装下主干 + 解码器，用的是上游自带的 `pipeline.json` 机制）：

```
/nfs-models/YuE2/                          共 9.9 GB
├── pipeline.json      {"model": "YuE2-3B", "vae": "YuE2-Vae", "generation_config": {...}}
├── YuE2-3B/           m-a-p/YuE2-3B            (7.26 GB)
├── YuE2-Vae/          m-a-p/YuE2-Vae           (0.53 GB)
├── SheetSage2/        m-a-p/SheetSage2         (0.23 GB，翻唱用)
└── MERT-v2-FullSong/  m-a-p/MERT-v2-FullSong   (2.53 GB，翻唱用)
```

魔搭有同名仓库，机器上直接拉 70 MB/s；hf-mirror 对这仓只有 56 KB/s（Xet）。
两个仓库分开来源，所以**没有加模型目录（catalog）条目**，和 H3 / Music3 一样按本地路径部署。

1. 后端选 **YuE2**，模型来源选本地路径 `/nfs-models/YuE2`，单卡。
   gpustack-ui 已登记该后端（`backendOptionsMap.yue2`、展示名、卡片暂用 ACE-Step 图标）；
   存储设置页的时延兜底表带了 `yue2: 75`。
2. worker 的 `GPUSTACK_EXTRA_MOUNTS` 要包含输出盘（`/nfs-output`）。
3. **必须按模型覆盖时延**：门面 music 类默认 `_DEFAULT_MUSIC_LATENCY=30`（按 ACE-Step
   校准），YuE2 每首 40-105 s。在 `lightx2v_model_latency_seconds` 里给 YuE2 配 ~75，
   否则准入估算差 2-3 倍。同时注意 music 类按类别排队上限默认只有 90 s：配 75 之后
   每实例第 3 个并发就会被拒成 429，要用满引擎自己的 8 格队列，就把按模型排队上限
   （`lightx2v_model_queue_wait_seconds`）抬到 ~600。
4. new-api 侧给模型加一条 `MusicModelConfig`（能力「文生音乐」「翻唱」、`maxChars` 建议 600）。
   t2m / cover 的请求形状与 ACE-Step 相同，不需要改代码；**不要给 YuE2 开「局部重绘」**。

## 10. 镜像

`docker/Dockerfile.arm64` 叠在 LightX2V 的共享基座 `lightx2v:arm64-cu130-a100-base` 上
（Python 3.12.13 + torch 2.11.0+cu130 + ffmpeg/LAME），和 ACE-Step / IndexTTS 同一套：
PyPI 的 aarch64 torch 是 CPU 轮子、集群上 pytorch.org 只有 ~50 KB/s，带 CUDA 的
aarch64 torch 只能从基座继承。

与上游 pyproject 的版本差异：torch 2.10 → 继承 2.11（YuE2 在 2.13 上实测同 seed 逐 token
复现）；numpy/accelerate/soundfile 继承基座；transformers 4.57.6、huggingface-hub 0.36.2、
tiktoken 0.12.0、safetensors 0.7.0 照上游钉（基座是 transformers 5.15，必须降）。

构建期断言：版本钉对、torch 是 CUDA 构建、整条导入链（含 model/VAE 代码）通过、
`tests/test_harness_task_api.py` + `tests/test_protocol.py` 全过。

```bash
# 集群机器上本地构建(PyPI 不通,换源)
docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -f docker/Dockerfile.arm64 -t yue2:arm64-a100 .
# CI 出包(需仓库 secrets ACR_USERNAME / ACR_PASSWORD / DOCKERHUB_USERNAME / DOCKERHUB_TOKEN)
gh workflow run build-arm64.yml -R reputationly/YuE --ref main   # 双推 ACR + Docker Hub(arronlee/yue2)
```

## 11. 实测包络（A100-40G 单卡）

| 项 | 值 |
|---|---|
| 就绪时间 | ~25 s（权重 + 哈希校验 6 s、预热 15 s）；带翻唱 ~34 s（SheetSage2 + MERT 校验 10 s） |
| 峰值显存 | 8.0-8.9 GiB，与曲长无关（官方写的 24 GB 不必要）；带翻唱常驻 +3.5 GiB，峰值 11.6 GiB |
| RTF | 0.43-0.62，越长越划算（1 分钟歌 ~22 s，4 分钟歌 ~97 s） |
| AR 速度 | ~105 tokens/s（两段 AR 占九成时间，是后续加速的唯一值得动的地方） |
| 取消生效 | ~1 s |
