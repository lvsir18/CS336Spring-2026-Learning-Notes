# Git、GitHub 与实验版本管理

笔记与作业各一个独立repo，main分支；顶层工作目录只是容器。已初始化且将代码/文档/小型结果提交本地，没有创建远端或上传到你的账户。

## GitHub 连接

在[lvsir18主页](https://github.com/lvsir18)新建空仓库`cs336-2026-notes`和`cs336-2026-assignments`，不勾选自动README，然后分别在对应目录执行：

```bash
# 在笔记仓库执行
 git remote add origin https://github.com/lvsir18/cs336-2026-notes.git
 git push -u origin main
# 在作业仓库执行
 git remote add origin https://github.com/lvsir18/cs336-2026-assignments.git
 git push -u origin main
```

先用`git remote -v`检查，若已有origin则`git remote set-url origin ...`。这里提供的是后续操作命令，当前没有执行任何push；原内容的发布license由你后续选择，第三方许可仍独立保留。

## 每次实验如何入库

修改config和docs→测试→运行→把整理后的小型results复制到reports→`git status --short`检查→`git add src scripts configs docs reports`→`git commit -m 'Describe experiment and outcome'`。不使用`git add -f data`或强行上传weights。

`.gitignore`排除datasets、models、runs、缓存、`.env*`。记录模型和data的hash/URI，不用Git保存大资产。A5 upstream resources保留本地，由固定commit脚本获取；fresh clone后必须运行fetch脚本再测试或使用prompt。

README写真实完成状态：算法测试、CPUsmoke、真实GPU指标分别列出。不要将合成IsoFLOPs或toy model的成绩写成正式训练/排名。代码commit和服务器环境锁文件对应起来，后续才容易定位性能或loss回归。
