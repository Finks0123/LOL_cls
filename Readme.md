# 特征分类

> @time: 2022/7/8

本项目为[常规赛：英雄联盟大师预测](https://aistudio.baidu.com/aistudio/competition/detail/247/0/introduction)的实现代码，算是paddle的学习流程。

任务类型的**特征分类**，数据集中每一行为一个玩家的游戏数据，数据字段如下所示：

- id：玩家记录id
- win：是否胜利，标签变量
- kills：击杀次数
- deaths：死亡次数
- assists：助攻次数
- largestkillingspree：最大 killing spree（游戏术语，意味大杀特杀。当你连续杀死三个对方英雄而中途没有死亡时）
- largestmultikill：最大mult ikill（游戏术语，短时间内多重击杀）
- longesttimespentliving：最长存活时间
- doublekills：doublekills次数
- triplekills：doublekills次数
- quadrakills：quadrakills次数
- pentakills：pentakills次数
- totdmgdealt：总伤害
- magicdmgdealt：魔法伤害
- physicaldmgdealt：物理伤害
- truedmgdealt：真实伤害
- largestcrit：最大暴击伤害
- totdmgtochamp：对对方玩家的伤害
- magicdmgtochamp：对对方玩家的魔法伤害
- physdmgtochamp：对对方玩家的物理伤害
- truedmgtochamp：对对方玩家的真实伤害
- totheal：治疗量
- totunitshealed：痊愈的总单位
- dmgtoturrets：对炮塔的伤害
- timecc：法控时间
- totdmgtaken：承受的伤害
- magicdmgtaken：承受的魔法伤害
- physdmgtaken：承受的物理伤害
- truedmgtaken：承受的真实伤害
- wardsplaced：侦查守卫放置次数
- wardskilled：侦查守卫摧毁次数
- firstblood：是否为firstblood

测试集中label字段win为空，需要选手预测。

本项目使用paddle框架编写，实现了一个dense网络做分类。详见main.py脚本。

最终效果：本项目使用F1作为评价指标。验证集效果83%。

```
第9验证集效果：f1: 0.8318227593152064,
 report:
              precision    recall  f1-score   support

           0       0.80      0.87      0.83       484
           1       0.87      0.80      0.83       516

    accuracy                           0.83      1000
   macro avg       0.83      0.83      0.83      1000
weighted avg       0.84      0.83      0.83      1000
```
