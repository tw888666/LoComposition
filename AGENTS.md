# AGENTS.md

## Language

- 默认使用中文回复，并称呼用户为 `nullptr`。
- English terms must include Chinese meaning on first use, for example GitHub（代码托管平台）, Git（版本控制系统）, branch（分支）, commit（提交）, and push（推送）。
- 不确定时必须明确说明不确定点，不要编造不存在的文件、接口、命令或结果。
- 生成的各种数据文件，png，csv，md等，尽量使用中文描述，如是英语专业名称，旁边需要附上中文解释。md文件也尽量用中文命名。
- 生成的md文件，log文件等命名前加上gpt-前缀，便于区分人工和AI生成的文件。
- md里的数学公式要渲染正确，不要使用\[
P_{\mathrm{cost}}=P_{\mathrm{el}}+P_{\mathrm{mech}}.
\],$r$这种无法渲染成功的格式。
- 对于分步骤的任务，在每次会话结束后应主动告诉我下一步要怎么做。成功完成任务时，应主动告诉我当前任务完成。

## Environment

- 本机是 Ubuntu 22.04服务器，无GUI环境。gpu0和gpu1是A100，gpu2-5是V100。我使用VSCode的SSH拓展远程连接。
- gpu长训练（例如3000次it）必须在tmux或GNU Screen等可断线恢复的终端会话中运行，短训练冒烟测试等无所谓。gpu训练任务只要能在gpu上正常启动即可，无需担心抢占他人任务。
- 正式评估不强制使用 tmux 或 GNU Screen，可以在普通终端直接运行。
- 无真实机器人等设备，目前只能做仿真。

