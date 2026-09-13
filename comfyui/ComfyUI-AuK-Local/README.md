# ComfyUI-AuK-Local

此节点通过 `http://127.0.0.1:7860` 调用 AuK Local 独立服务，不会在 ComfyUI 进程内加载 AuK 模型。

1. 先运行整合包根目录的 `启动AuK服务.cmd`。
2. 用 `安装ComfyUI节点.cmd` 安装节点并重启 ComfyUI。
3. 加载 `example_workflows` 中的三份工作流之一。

节点包含 `AuK Local 连接` 与 `AuK Local 生成 / 编辑`。后者覆盖整合包中的 16 类任务，并输出标准 ComfyUI `AUDIO`、最终指令和运行参数 JSON。
