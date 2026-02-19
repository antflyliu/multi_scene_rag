## 问题描述
在 Windows 11 启动 `api_server.py` 时出现端口绑定错误：
`[WinError 10048] 通常每个套接字地址(协议/网络地址/端口)只允许使用一次。`

## 原因分析
`127.0.0.1:8000` 端口已被其他进程占用，导致当前服务无法绑定端口。

## 解决步骤

### 方案一：PowerShell
1. 查出占用 8000 端口的 PID：
```
Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object -ExpandProperty OwningProcess
```
2. 结束进程（将 `<PID>` 替换为上一步得到的数字）：
```
Stop-Process -Id <PID> -Force
```

一条命令串联版本：
```
Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object -ExpandProperty OwningProcess | ForEach-Object { Stop-Process -Id $_ -Force }
```

### 方案二：cmd
1. 查出占用 8000 端口的 PID：
```
netstat -ano | findstr :8000
```
2. 结束进程（将 `<PID>` 替换为上一步看到的数字）：
```
taskkill /PID <PID> /F
```
可选：查看进程名
```
tasklist /FI "PID eq <PID>"
```

## 生成日期（动态）
```
from datetime import datetime
datetime.now().strftime("%Y-%m-%d")
```
