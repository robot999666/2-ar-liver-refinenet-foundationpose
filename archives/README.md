# archives

`wheelhouse.zip` 是面向云端 Linux x86_64 / Python 3.12 环境的离线训练依赖包。
它被 Git 忽略，必须外部备份；字节数和 SHA256 见根目录 `MANIFEST.md`。

在项目根目录解压后会形成临时 `wheelhouse/`：

```bash
python3 -m zipfile -e archives/wheelhouse.zip .
python3 -m pip install --no-index --find-links=wheelhouse \
  -r requirements_06_train_refine_net.txt
```

安装完成后可以删除解压出的 `wheelhouse/`，保留压缩包即可。
