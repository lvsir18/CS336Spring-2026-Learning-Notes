# Git、GitHub 与实验版本管理

笔记与作业各一个独立repo，main分支；顶层工作目录只是容器。笔记远端为 [lvsir18/CS336Spring-2026-Learning-Notes](https://github.com/lvsir18/CS336Spring-2026-Learning-Notes)，本地目录仍叫 `cs336-2026-notes`。配套作业已发布到 [CS336Spring-2026-Learning-Assignments](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments)，对应入口见[课程与作业对应表](assignments.md)。

## GitHub 连接

在新机器获取笔记：

```bash
git clone https://github.com/lvsir18/CS336Spring-2026-Learning-Notes.git cs336-2026-notes
cd cs336-2026-notes
```

在现有笔记目录检查远端并同步后续修改：

```bash
git remote -v
git pull --ff-only
git status --short
git add README.md lectures guides source-manifest.json
git commit -m 'Update CS336 learning notes'
git push origin main
```

若需要首次连接现有本地笔记仓库，执行 `git remote add origin https://github.com/lvsir18/CS336Spring-2026-Learning-Notes.git`；已有origin时先核对地址。作业仓库使用自己的远端，不放进笔记仓库。原内容的发布license由你后续选择，第三方许可仍独立保留。

## 每次实验如何入库

修改配置和说明→必要验证→运行→将精简指标放入results、来源摘要放入provenance→`git status --short`检查→只暂存需要发布的文件→提交。不使用`git add -f data`或强行上传weights。

`.gitignore`排除datasets、models、runs、缓存、`.env*`。记录模型和data的hash/URI，不用Git保存大资产。A5 upstream resources保留本地，由固定commit脚本获取；fresh clone后先运行`python scripts/fetch_runtime_assets.py`再使用prompt；发布版不下载或携带上游测试。

README写真实完成状态：算法测试、CPUsmoke、真实GPU指标分别列出。不要将合成IsoFLOPs或toy model的成绩写成正式训练/排名。代码commit和服务器环境锁文件对应起来，后续才容易定位性能或loss回归。
