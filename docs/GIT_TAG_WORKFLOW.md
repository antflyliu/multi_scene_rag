# GitHub Tag 操作指南

> 文档说明：本文档整理在 GitHub 仓库中创建、推送及管理 Tag 的完整操作流程与命令。
> 最后更新：2025-02-18

---

## 一、前置准备

### 1.1 检查 Git 状态

打 tag 前建议先确认工作区状态，确保已提交需要标记的更改：

```powershell
# 查看当前工作区状态
git status

# 查看当前所在分支
git branch
```

### 1.2 确认远程仓库

```powershell
# 查看已配置的远程仓库
git remote -v

# 若尚未添加远程仓库，需先执行：
# git remote add origin <your-github-repo-url>
```

---

## 二、创建 Tag

### 2.1  lightweight tag（轻量级）

适用于临时或简单标记：

```powershell
git tag v1.0.0
```

### 2.2 annotated tag（附注 tag，推荐）

附带作者、日期和说明信息，适合正式版本发布：

```powershell
git tag -a v1.0.0 -m "版本1.0.0：初始发布"
```

### 2.3 为历史提交打 Tag

若需为某个历史提交打 tag，先获取提交哈希：

```powershell
# 查看提交历史
git log --oneline

# 为指定提交打 tag
git tag -a v1.0.0 <commit-hash> -m "版本说明"

git tag -a v1.1.0 -m "新加核心功能：`n1、第一行说明；`n2、第二行说明；`n3、第三行说明；"

$msg = @"
新加核心功能：
1、第一行说明；
2、第二行说明；
3、第三行说明；
"@
git tag -a v1.1.0 -m $msg

```

---

## 三、查看 Tag

```powershell
# 列出本地所有 tag
git tag

# 按模式筛选（如列出 v1.x 系列）
git tag -l "v1.*"

# 查看 tag 详细信息（附注 tag）
git show v1.0.0
```

---

## 四、推送 Tag 到 GitHub

### 4.1 推送单个 Tag

```powershell
git push origin v1.0.0
```

### 4.2 推送所有 Tag

```powershell
git push origin --tags
```

> **注意**：`--tags` 会推送本地所有未推送的 tag，请确认无误后再执行。

---

## 五、删除 Tag

### 5.1 删除本地 Tag

```powershell
git tag -d v1.0.0
```

### 5.2 删除远程 Tag

```powershell
git push origin --delete v1.0.0
```

### 5.3 删除后同步本地

若远程 tag 已删除，本地可执行以下命令清理：

```powershell
git fetch --prune --prune-tags
```
git tag -a v1.1.1 -m "新加核心功能：`n1、添加本地嵌入式模型；`n2、添加前端界面和客户端安装；`n3、LLM 接口兼容多家 AI 厂商；"
---

## 六、完整操作流程（推荐）

以下为从零开始打 tag 并推送到 GitHub 的标准流程：

| 步骤 | 命令 | 说明 |
|------|------|------|
| 1 | `git status` | 检查工作区是否干净 |
| 2 | `git add .` 与 `git commit -m "xxx"` | 若有未提交变更，先提交 |
| 3 | `git tag -a v1.0.0 -m "版本说明"` | 创建附注 tag |
| 4 | `git tag` | 确认 tag 已创建 |
| 5 | `git push origin v1.0.0` | 推送到 GitHub |

---

## 七、注意事项

1. **版本号规范**：建议采用语义化版本（Semantic Versioning），如 `v1.0.0`、`v1.1.0`、`v2.0.0`。
2. **Tag 指向**：打 tag 时默认指向当前 HEAD 对应的提交，请确认已提交所需内容再打 tag。
3. **远程配置**：首次推送前需确保已配置 `origin` 并完成身份验证。
4. **Tag 修改**：已推送的 tag 不建议修改，若需调整建议删除后重新创建。

---

## 八、常见问题

**Q：tag 命名有何建议？**  
A：通常以 `v` 开头，如 `v1.0.0`，便于在 GitHub Releases 中识别。

**Q：lightweight 与 annotated tag 的区别？**  
A：annotated tag 会存储更多元数据，更适合正式版本；lightweight tag 仅是指向提交的引用。

**Q：如何基于 tag 创建发布分支？**  
A：`git checkout -b release/v1.0.0 v1.0.0` 可基于 tag 创建新分支。
