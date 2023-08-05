# 准备训练用的视频数据
---

## 录制视频
建议使用H264、H265格式录制，以减小存储空间占用。
要求：
- 画面聚焦清晰、光线适合，无杂物和其他人员干扰
- 每一类别(参见后续类别定义文件)至少录制有 50 段视频片段

## 编写 *classInd.txt* 文件
*classInd.txt* 为视频类别文件，其格式为：

``` csv

1 SitUp_Up
2 SitUp_Down
3 SitUp_Violation_HandsOff
...

```
其中，第一列为类别ID(正整数)、第二列命名，都必须为唯一值。

## 按照动作类别，分割视频
按照 *classInd.txt* 定义的原子动作进行分割，尽量保持从原子动作开始到结束的完整帧序列。

分割后的文件分别复制到 *videos* 目录中对应类别名的子目录。

文件名可以任意定义，但建议参照 * v_<类别名>_g<视频文件id>_c<视频切片id>.<ext> *

文件类型支持mp4、avi

## 编写 *train_videos_annotation.txt* 文件
*train_videos_annotation.txt* 为视频标识文件，其格式为：

``` csv
SitUp_Up/v_SitUp_Up_g08_c01.avi 1
SitUp_Up/v_SitUp_Up_g08_c02.avi 1
SitUp_Up/v_SitUp_Up_g08_c03.avi 1
SitUp_Up/v_SitUp_Up_g08_c04.avi 1
...
SitUp_Down/v_SitUp_Down_g08_c05.avi 2
SitUp_Down/v_SitUp_Down_g09_c01.avi 2
SitUp_Down/v_SitUp_Down_g09_c02.avi 2
SitUp_Down/v_SitUp_Down_g09_c03.avi 2
...
SitUp_Violation_HandsOff/v_SitUp_Violation_HandsOff_g09_c04.avi 3
SitUp_Violation_HandsOff/v_SitUp_Violation_HandsOff_g09_c05.avi 3
SitUp_Violation_HandsOff/v_SitUp_Violation_HandsOff_g09_c06.avi 3
SitUp_Violation_HandsOff/v_SitUp_Violation_HandsOff_g09_c07.avi 3
...

```
其中，第一列为视频文件路径、第二列为 *classInd.txt* 中定义的类别ID。




# 以上步骤完成后，文件组织形式如下所示

```
ucf101
├── annotations
│   ├── classInd.txt
│   └── train_videos_annotation.txt
└── videos
    ├── SitUp_Up
    │   ├── v_SitUp_Up_g01_c01.avi
    │   └── ...
    ├── SitUp_Down
    │   ├── v_SitUp_Down_g25_c05.avi
    │   └── ...
    ├── SitUp_Violation_HandsOff
    │   ├── v_SitUp_Violation_HandsOff_g125_c01.avi
    │   └── ...
    └── ...
```
