# Git 版本保护流程

## 当前历史

- `c059ce8`：整理前私人仓库版本，包含旧脚本、旧批量结果和 MATLAB 文件。
- `cf3989c`：恢复并重构后的第一份保护快照。
- `cleanup-restructure-v2`：当前整理分支。

## 修改前

```bash
git status
git diff --stat
git add -A
git commit -m "checkpoint: describe current state"
git switch -c feature/<change-name>
```

## 删除前

1. 使用 `git ls-files <path>` 确认文件是否受 Git 管理。
2. 使用 `rg` 确认代码不再引用目标。
3. 只删除明确路径，不使用递归通配符。
4. 删除后立即运行测试并提交。

推荐使用：

```bash
git rm path/to/tracked-file
git commit -m "cleanup: remove unused file"
```

## 恢复方法

恢复当前分支误删文件：

```bash
git restore path/to/file
```

从指定提交恢复：

```bash
git restore --source=<commit> -- path/to/file
```

只查看旧内容：

```bash
git show <commit>:path/to/file
```

## 大文件

`weights/depth_anything_v2_vitl.pth` 和 `archives/wheelhouse.zip` 默认被 Git 忽略，Git 无法保护它们。
必须在外部数据盘或网盘另存一份，并在 `MANIFEST.md` 记录文件大小与 SHA256。

禁止未经检查地使用：

```text
git clean -fdx
git reset --hard
递归通配符删除
```
