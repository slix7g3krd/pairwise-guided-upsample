# pairwise-guided-upsample

## 领域

游戏、图形与多媒体

## 主题

引导滤波上采样

## 说明

联合双边引导、输出尺度半径与逐图缓存。

初始状态下 `imaging/guided_upsample.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
