# GPUStack 接入说明（ARM64 / A100）

本仓 = 上游 YuE2 → [YuE2-Turbo](https://github.com/NoizAI/YuE2-Turbo)（加速服务化）→ 一层 harness，
接成 GPUStack 的内置后端 `YuE2`，承接「文生音乐（歌词 → 整首歌）」「翻唱（录音 → 扒谱 → 换风格重唱）」
「给谱 / 改谱重渲染」。

提交结构：`vendor:` 原样引入 Turbo @7c88813；`feat(fast):` vLLM 0.29 适配（两行，环境变量开关）；
其余是 harness（`server.py`、`harness/`、`docker/`、CI、本文档）。

---

## 1. 为什么是这个形态，而不是进 vllm-omni

- 三个生成阶段串行、一首歌要在 AR / NAR / VAE 间来回切，vllm-omni 的连续批处理接不住；
  Turbo 自己的调度器（AR 并发 + NAR 分批 + 流水线重叠）才是这个模型该用的形态。
- 卖点是**可编辑的乐谱**，塞进 `/v1/audio/speech` 只剩「文字进音频出」。

## 2. 走哪扇门、支持哪些玩法

复用门面**已有的 music 门**：`task_type=t2m|cover` → engine kind `music` → `POST /v1/tasks/music/` →
`.mp3`，与 ACE-Step 同门。**new-api 与门面零改动**（new-api 另有一处可选改动把乐谱回给用户，见 §4）。

| 玩法 | 支持 | 怎么走 |
|---|---|---|
| 文生音乐（t2m） | ✅ | `prompt` + `lyrics`；`cot` 选 full / melody / off |
| 给谱渲染、改谱重渲染 | ✅ | 请求带 `abc`；出歌时写 `<stem>.abc`，new-api 经 `metadata.score_abc` 返回 |
| 翻唱（cover） | ✅ | 门面的 `reference_audio` → `reference_audio_path`，进程内 SheetSage2 扒谱（§6） |
| 局部重绘（repaint） | ❌ | YuE2 没有局部重绘能力；带 `src_audio_path` 的请求提交时 **400** |

`task_type` 是门面的控制字段，到不了引擎，所以引擎靠 `reference_audio_path` / `src_audio_path`
区分任务。**这两个字段必须显式处理**：当未知字段忽略的话，翻唱和重绘请求会悄悄退化成一首
不相干的新歌、还显示成功。

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/tasks/music/` | 提交，返回 `{task_id, task_status, save_result_path}`；队列满 **503**；加载中 **503** |
| POST | `/v1/tasks/audio/` | 同上的别名（ACE-Step 也留了，手工测试用） |
| GET | `/v1/tasks/{id}/status` | 状态 + `phase` + `progress`；未知任务 **404** |
| GET | `/v1/tasks/{id}/result` | 取音频，仅 `completed` |
| DELETE | `/v1/tasks/{id}` | 取消（真中断，§8） |
| GET | `/v1/tasks/queue/status` | 队列水位 |
| GET | `/ready` | 加载与预热期间 **503**（HTTP 2 秒内就起来，模型在后台线程加载） |

状态字符串 `pending | processing | completed | failed | cancelled`（**cancelled 双 L**，
门面 `_ENGINE_STATE_MAP` 逐字匹配）。未知任务必须 404——门面 sweeper 靠它判断「引擎重启、
任务已丢」并重派。任务队列是 Turbo 的 SQLite JobStore，放在容器内 `/tmp`：实例重建即清空，
语义与内存队列相同。

## 3. 请求字段

new-api 把客户端 `metadata` 摊平到顶层并设 `prompt`；门面剥掉控制字段、注入
`save_result_path`。其余字段（`model`、`user_id`、ACE-Step 的 `audio_duration`/`bpm`…）
**忽略而不拒绝**。

```jsonc
{
  "prompt": "Mandarin Chinese, modern C-pop ballad, female vocal, piano and strings, 76 BPM",
  "lyrics": "[Verse]\n街灯把影子拉得很长\n...\n[Chorus]\n...",  // 可为空 = 纯音乐
  "cot": "full",        // full(旋律+和弦) | melody | off(不出谱);缺省 full,翻唱缺省 melody
  "abc": "X:1\n...",    // 可选:给一份乐谱直接渲染(改谱重渲染),需 cot=full|melody
  "seed": 42,           // 可选;缺省在提交时随机钉一个并写进旁挂 json
  "cfg_scale": 1.0,     // 可选,[0, 20];≠1 会走原版路径(§7)
  "save_result_path": "/nfs-output/.../x.mp3"   // 门面注入;后缀决定格式 mp3|wav|flac
}
```

`prompt` 也接受 YuE2 自己的拼法 `style` / `tags`。**seed 缺省随机**（上游缺省恒为 831001，
同一个 prompt 所有用户会拿到同一首歌）。

## 4. 输出：音频 + 两个旁挂

`save_result_path` 同目录下，旁挂先写、音频最后写，各自原子落盘（音频出现 = 完成信号）：

| 文件 | 内容 |
|---|---|
| `<stem>.mp3` | 320 kbps CBR、48 kHz 立体声（ffmpeg/LAME）；`.wav`/`.flac` 为 24-bit |
| `<stem>.abc` | 据以渲染的乐谱：YuE2 规划的 / 翻唱扒出的 / 调用方给的（`cot=off` 时没有） |
| `<stem>.json` | seed、模式、乐谱来源、**走了哪条路（`backend`: vllm / torch）**、token 数、分段耗时、权重 sha256 |

门面的 janitor 按**天目录整体 rmtree**，旁挂跟音频一起被清。MP3 编码和 NFS 写入在后台线程池里做，
不占调度线程；任务要等文件写完才标 `completed`，写盘失败记 `failed` / `publish_failed`。

**乐谱怎么到用户手上**：new-api 在任务成功时从共享挂载读 `<stem>.abc`，放进响应的
`metadata.score_abc` 原样返回（`service/music_score.go`，只读挂载根下、≤64 KiB 的 UTF-8 文件，
读不到就不带）。调用方改完放进下一次请求的 `metadata.abc` 即可——`abc` 不在门面与 new-api 的
剥离名单里，会原样到达引擎。

**改谱是一首新录音**：谱改动之后模型的输入就变了，重渲染是一首新的完整录音，没改的段落也会变
（上游 `docs/editing.md`：Matching the unchanged score does not guarantee identical singing,
timbre, or waveform outside the edit）。比较改谱效果要听整首。

## 5. 长度上限

| 上限 | 值 | 超限行为 |
|---|---|---|
| 语义 token | 9000（≈25 Hz，约 **6 分钟**音频） | 上游**静默截断**，只置 `truncated` 标志 |
| 乐谱 token | 4096 | 同上 |

截断出来的是一首唱到一半停掉的歌，任务却显示成功——所以两层防线：

1. **提交时按歌词预算 400**：`MAX_LYRIC_UNITS = 600`，汉字/假名/谚文记 1，其余非空白字符
   记 0.385。依据（A100 实测）：335 字 → 45% 上限，483 字 → 64%，588 单位 → 66%。
2. **运行时截断即失败**：任一阶段 `truncated` → `failed`，`error_type="truncated"`，不产出音频。

## 6. 翻唱

```text
reference_audio_path → SheetSage2（自动加载 MERT-v2-FullSong）扒谱
  → 上游 abc_tools 校验谱面（melody 模式还校验没有和弦）
  → YuE2 按这份谱 + 新的 prompt/lyrics 重唱（cot 缺省 melody）
```

- **缺省 `cot=melody`**：扒谱时去掉和弦，伴奏按新风格自由发挥；`cot=full` 保留原曲和声；
  `cot=off` 与翻唱矛盾，提交时 400。**歌词必须由调用方给**。
- 源音频 ≤ **330 秒**（提交时 ffprobe 量）：翻唱时长≈源时长，而语义上限约 6 分钟。
- 扒谱失败（没扒出谱、谱面校验不过）→ `failed`，`error_type="transcription"`。
- 翻唱先扒谱，所以走串行路径，但它的 AR 仍经 vLLM（只是不和别的歌拼批）。
- SheetSage2 用本仓的加载方式（快照目录当 Python 包导入），不用 Turbo 自带的 `cover.py`：
  那条走 `AutoModel(trust_remote_code=True)`，transformers 4.57 对本地快照只拷直接相对导入，
  SheetSage2 的二级导入会报找不到 `chord_spelling_sheetsage2.py`。它自己钉的 torch 2.8 /
  numpy 1.24 是作者环境快照，在本镜像原样跑通，只多装 pretty_midi / mido / mir_eval。
- 权重 `SheetSage2/`（229 MB）与 `MERT-v2-FullSong/`（2.53 GB）放在模型目录下；缺任何一个，
  实例照常起、`/ready` 返回 `"cover": false`，翻唱请求 400。

## 7. 加速与调度（YuE2-Turbo）

Turbo 的 modeling / sampling / protocol 与上游逐字相同（官方 WildSongBench 无掉点），改的是调度：

- AR 两段（规划乐谱、语义 token）走 **vLLM**：MoT 的 AR 部分剥成标准 Qwen3 喂给 vLLM，
  vLLM 跑在**独立子进程**；
- MoT + VAE **常驻显存**，不再每首歌在 CPU/GPU 间搬；
- AR 最多 **4 首并发**（一个 vLLM 调度波），NAR **两首一批**，AR/NAR **流水线重叠**。

harness 继承 Turbo 的 `JobWorker`，只覆盖：结果写到 `save_result_path` + 旁挂、截断即失败、
翻唱扒谱、错误类型映射、未被请求的中断判 `failed`（Turbo 会把任何 InterruptedError 当取消）。

**分流是按请求自动的，一个部署服务全部请求**：

| 请求 | 路径 |
|---|---|
| cot=full/melody 且 cfg_scale=1（两者缺省即满足）：文生、给谱/改谱 | vLLM，并发批处理 |
| 翻唱 | 先扒谱，串行；AR 仍走 vLLM |
| cot=off、自定义 cfg_scale | 原版 PyTorch 路径（vLLM 路径不支持），串行 |

**vLLM 0.29 而不是 0.19**：Turbo 硬钉 `vllm==0.19.0`，但它实现的 v1 LogitsProcessor 接口
（五个方法 + BatchUpdate 的 added/moved 元组）在 0.29 原样存在，其窗口惩罚处理器的单测在
0.29 下全过。两个解释器（`docker/Dockerfile.arm64`）：主进程 venv（transformers 4.57.6）跑 MoT、
VAE、SheetSage2；vLLM 子进程用基座的 python（vLLM 0.29 + transformers 5.14）。
`YUE2_VLLM_PYTHON` / `YUE2_VLLM_ANY_VERSION` 由镜像设好。

**可复现性变了**：vLLM 按负载拼批、分块预填、复用前缀缓存，浮点累加顺序会变，所以同一个请求
原样再跑一次**不保证逐字节相同**（原版路径保证）。实测单独跑时大多相同（改谱原样交回：时长、
token 数与原曲一致），和别的歌拼批时可能分叉。与原版路径之间则是两套抽样器，同 seed 本来就是两首歌。
听感：用户盲听后认可加速版。

## 8. 取消是真中断

AR 每个 token、flow matching 每一步都检查取消，DELETE 后约 **1 秒**下一个任务就在跑，
被取消的任务不留文件。DELETE 之后状态立刻显示 `cancelled`（worker 在下一个检查点才真正停）。

例外：**扒谱阶段要等这一轮扒完**（SheetSage2 没有中途停止的钩子），4 分钟的源多占约 25 s。
任务超时（`YUE2_TASK_TIMEOUT_SECONDS`，不计排队）→ `failed` / `timeout`。

## 9. 进度上报

门面契约里引擎自己折好的全局 `progress` **优先于** `phase + phase_progress`，YuE2 报全局值
（门面 music 类的权重表对不上它的耗时分布）。Turbo 的阶段名映射如下：

| Turbo 阶段 | phase 标签 | 全局区间 | 推进方式 |
|---|---|---|---|
| claimed_waiting | prepare | — | — |
| transcribing（仅翻唱） | encode | —（门面按耗时估） | — |
| planning | denoise | 0 → 30 | 按 token（预期 ≈ 4.3 × 歌词单位） |
| semantic | denoise | 30 → 80（不出谱 0 起，翻唱 20 起） | 按 token（预期 ≈ 2.7 × 乐谱 token） |
| synthesis | denoise | 80 | 阶段边界 |
| decode | decode | 92 | 阶段边界 |
| saving | save | 97 | 阶段边界 |

引擎侧保证单调不回退（门面也强制单调、封顶 99）。

## 10. 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `YUE2_MODEL_DIR` | `/weights` | 模型根目录；GPUStack 注入模型路径 |
| `YUE2_BACKEND` | `vllm` | `torch` = 上游串行路径（回退开关） |
| `YUE2_AR_CONCURRENCY` | `4` | 一个 vLLM 调度波里的歌数 |
| `YUE2_NAR_BATCH_SIZE` | `2` | flow matching 一批几首（上限 2） |
| `YUE2_VLLM_GPU_MEMORY_UTILIZATION` | `0.30` | vLLM 占卡比例（权重 + KV） |
| `YUE2_MAX_QUEUE` | `12` | 排队 + 运行中上限，满了 503 |
| `YUE2_MEMORY_BUDGET_GIB` | `28` | PyTorch 进程的显存上限 |
| `YUE2_TASK_TIMEOUT_SECONDS` | `1200` | 单任务执行超时（不计排队） |
| `YUE2_MP3_BITRATE` | `320k` | MP3 码率 |
| `YUE2_WARMUP` | `1` | 就绪前先跑一首小歌走通全链路 |
| `YUE2_CACHE` | 模型目录 | 剥出来的 vLLM AR 权重（4.1 GB）放哪；模型目录只读时退到 `/tmp` |
| `VLLM_CACHE_ROOT` | `<模型目录>/vllm-cache` | vLLM 编译缓存，同上 |
| `YUE2_VAE_DIR` / `YUE2_SHEETSAGE2_DIR` / `YUE2_MERT_DIR` | 模型目录下 | 覆盖子目录位置 |

**引擎没有 CLI**：GPUStack 只注入 `--host/--port`，「后端参数」会被忽略，全部开关走环境变量。

## 11. 部署到 GPUStack

**权重布局**（一个模型路径装下全部；后两个目录由引擎首次启动时写入，之后各实例复用）：

```
/nfs-models/YuE2/
├── pipeline.json      {"model": "YuE2-3B", "vae": "YuE2-Vae", "generation_config": {...}}
├── YuE2-3B/           m-a-p/YuE2-3B            (7.26 GB)
├── YuE2-Vae/          m-a-p/YuE2-Vae           (0.53 GB)
├── SheetSage2/        m-a-p/SheetSage2         (0.23 GB，翻唱)
├── MERT-v2-FullSong/  m-a-p/MERT-v2-FullSong   (2.53 GB，翻唱)
├── yue2-ar/<hash>/    剥出来的 vLLM AR 权重    (4.1 GB，引擎生成)
└── vllm-cache/        vLLM 编译缓存            (引擎生成)
```

模型目录要**可写**挂载：只读时引擎照常工作，但每个容器启动都要重新剥权重、重新编译（多约 2 分钟）。

1. 后端选 **YuE2**，本地路径 `/nfs-models/YuE2`，每个副本一张卡，一台 4 卡机器可放 4 个副本。
2. worker 的 `GPUSTACK_EXTRA_MOUNTS` 要包含输出盘（`/nfs-output`）。
3. **按模型配准入**：门面的排队估算按「每实例一次一个任务」算
   （`est_wait = floor(非终态数 / 实例数) × 时延`），而现在一个实例同时跑 4 首。满载时每首平均
   占 ~11 s 卡时间（4 首一波 40~47 s），所以 `lightx2v_model_latency_seconds` 给 yue2 配 **~15**；
   要用满引擎自己的 12 格队列，`lightx2v_model_queue_wait_seconds` 配 **~240**
   （music 类默认 90 s，配 15 之后第 7 个排队的请求就会 429）。
4. new-api 给模型开「文生音乐」「翻唱」、`maxChars` 建议 600，**不要开「局部重绘」**。

## 12. 镜像

`docker/Dockerfile.arm64` 叠在 **vllm-omni** 镜像上（Python 3.12 + torch 2.13 + vLLM 0.29 +
ffmpeg/LAME），所有节点本来就有它，拉 YuE2 只下 app 层。PATH 把 `/opt/venv/bin` 放最前
（GPUStack 起容器的命令写死 `python3 -m uvicorn server:app`）。源码走 `PYTHONPATH`，两个解释器
看到同一份。

构建期断言：两个解释器的版本与导入链（主进程能导入服务入口，vLLM 解释器能同时导入 vLLM 与
`yue2.fast`）；主进程跑 harness 契约测试 + protocol + Turbo 服务测试；vLLM 解释器跑
`tests/test_fast.py`（除掉一条前提在 torch 2.13 下不成立的：它断言 import 不拉 triton，
而 torch 2.13 的 `import torch` 自己会拉）。

```bash
# 集群机器上本地构建(PyPI 不通,换源)
docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -f docker/Dockerfile.arm64 -t yue2:arm64-a100 .
# CI 出包(双推 ACR + Docker Hub arronlee/yue2;需 ACR_* / DOCKERHUB_* secrets)
gh workflow run build-arm64.yml -R reputationly/YuE --ref main
```

GitHub → ACR 上行约 1 MB/s。基座变了（换 vllm-omni 版本）之后，先从集群节点推一次到 ACR
（境内 23.9 GB 实测 107 s），CI 之后就只剩 app 层要传。

## 13. 实测包络（A100-40G 单卡，经本 harness 的接口测）

| 项 | 值 |
|---|---|
| 就绪时间 | 首次（要编译）~4 min；编译缓存在模型目录后 **~2.3 min**。HTTP 2 s 起，期间 `/ready` 503 |
| 显存 | 空闲 22.5 GB（vLLM 12 + 常驻 MoT/VAE + SheetSage2）；4 路并发峰值 **30.8 GiB** |
| 单首 | 42 秒的歌 **11.3 s**（原版 21.5 s）；改谱重渲染 9.2 s |
| 系统 RTF | 1 首 0.28 / 2 首 0.20 / **4 首 0.148**（原版串行 0.455，**吞吐 3.1×**） |
| 翻唱 | 45 秒源：扒谱 ~4 s + 生成 ~10 s；4 分钟源扒谱 ~28 s |
| 取消生效 | ~1 s（扒谱阶段除外） |
