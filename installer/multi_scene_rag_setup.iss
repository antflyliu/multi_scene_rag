; Multi-Scene RAG 安装脚本（Inno Setup）
; 使用方式：
; 1) 先执行 python build_desktop.py
; 2) 使用 Inno Setup 打开并编译本文件

#define MyAppName "Multi-Scene RAG"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Kevin"
#define MyAppExeName "MultiSceneRAG.exe"

[Setup]
AppId={{8DDA9B8A-3F6A-4A9C-93DB-3A90F4D131F1}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\dist_installer
OutputBaseFilename=multi-scene-rag-setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
; Name: "chinesesimp"; MessagesFile: "compiler:Default.isl"
; 显式指定简体中文语言文件，避免不同 Inno Setup 环境下默认语言不一致。
Name: "chinesesimp"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务:"; Flags: unchecked

[Files]
Source: "..\dist\MultiSceneRAG\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\卸载 {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent
