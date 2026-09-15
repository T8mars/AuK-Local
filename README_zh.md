# AuK Local Windows 本地整合包

AuK Local 是由 T8star-Aix 维护的独立 Windows 浅色中文工作台，支持 AuK 的语音生成、编辑、增强和分离。它与 ComfyUI 节点是两个独立程序：本地整合包通过 `AuK-Local.exe` 启动，ComfyUI 节点直接运行在 ComfyUI 进程内。

## 下载与启动

GitHub Release 提供的是**不含模型、不含 Python 的程序更新包**。它适合覆盖到已有完整版中，不能单独完成推理。完整版由作者另行提供。

1. 完整解压 AuK Local，保留原有目录结构。
2. 双击根目录的 `AuK-Local.exe`。
3. 保持控制台窗口开启；服务就绪后会自动打开 `http://127.0.0.1:7860`。
4. 页面顶部有醒目的“检查更新”和“立即更新并重启”按钮。

自动更新会验证 RSA 数字签名、ZIP 文件 SHA-256 和包内每个文件的 SHA-256。安装前会备份旧程序，任何复制或校验失败都会回滚。更新器只替换白名单中的程序、脚本、界面和说明文件，明确禁止修改这些目录：

- `models`：模型权重
- `runtime`：便携 Python、CUDA 和 FFmpeg
- `ckpts`：用户放置的检查点
- `data`：任务记录、令牌和更新状态
- `outputs`：生成结果
- `logs`：运行日志

更新时必须先等待正在运行或排队的任务结束。为保证更新后能自动启动，请通过 `AuK-Local.exe` 启动，不要从服务 BAT 入口执行一键更新。

## 链接

- 程序源码与 Release：[T8mars/AuK-Local](https://github.com/T8mars/AuK-Local)
- 独立 ComfyUI 原生节点：[Comfyui-Auk-T8](https://github.com/T8mars/Comfyui-Auk-T8)
- 模型：[t8star/Auk-Comfy](https://huggingface.co/t8star/Auk-Comfy)
- B站：[T8star-Aix](https://space.bilibili.com/385085361)
- YouTube：[@T8star-Aix](https://www.youtube.com/@T8star-Aix/)
- 在线 AI 应用：[RunningHub](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- API：[Seedance API](https://api.seedance.nz/sign-up?aff=5f4w)
- ComfyUI 整合包：[夸克网盘](https://pan.quark.cn/s/264edb7e36bd)
- Hugging Face 主页：[t8star](https://huggingface.co/t8star)

模型能力、任务说明和上游项目资料见 [英文 README](README.md)。
