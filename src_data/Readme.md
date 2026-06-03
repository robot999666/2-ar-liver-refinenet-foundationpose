Plan ：用 PLY 跑模型，用矩阵去转 STL 做评估

`% Points(:,2:3) = -Points(:,2:3);  <-- 【Evaluation.m注释掉！】`

### 核心执行逻辑解析

`Evaluation.m` 的主体是一个遍历 4 名患者的大循环。对于每一帧图像，它做了以下三件核心的事情：

#### 1. 数据对齐与空间投影 (Data Loading & Transformation)

这是整个评估的基础，要把你的“预测结果”和“真实标签”放到同一个 3D 空间（腹腔镜相机空间）里：

* **读取你的预测 (STL)：** 脚本从 `RegistrationPath` 读取 `.stl` 文件（这就是你需要用 FoundationPose 生成的肿瘤模型）。
* *注意一个细节：* `Points(:,2:3) = -Points(:,2:3)`。作者把 Y 和 Z 坐标取反了，这是为了把 OpenGL 坐标系转换到 MATLAB 坐标系。你在保存 FoundationPose 结果时，`需要注意坐标系的对齐`


* **读取真实标签 (JSON)：** 从 JSON 文件中读取超声图像里描出来的 2D 轮廓点集。
* **真实标签的 3D 投影：** 使用 `.mat` 文件里的矩阵，通过严格的数学公式把 2D 像素点映射到腹腔镜相机的 3D 空间中：
* 尺度转换：将像素乘以 $SLus$ 和 $SAus$ 转为物理尺寸。
* 转到探头坐标系：$USprofile\_probe = R_{us} \cdot USprofile + T_{us}$
    
    探头（Probe / LUS）：它是另一根超声器械，利用超声波扫出肝脏内部肿瘤的 2D 黑白截面图（也就是 LUS Images 和标注了 JSON 的那个图）,仅用于评估，与处理工作无关。

* 转到相机坐标系：$USprofile\_cam = R_{pr} \cdot USprofile\_probe + T_{pr}$


* **重采样 (SampleProfile)：** 把这个真实轮廓均匀地采样成 **50个点**（变量 `M`），方便后续统一计算误差。

#### 2. 计算 IC (Inclusion Criterion)

由 `ICcalc` 函数实现，逻辑很粗暴但也有效：

* **计算几何中心 (Tc)：** 算出你预测的 3D 肿瘤模型的所有顶点的平均值作为中心。
* **网格膨胀：** 遍历你预测模型的每一个顶点，让它们沿着“中心到顶点”的向量方向，向外移动 **10mm**（`OncologicMargin = 10`），也就是论文中说的 1cm 肿瘤安全切除边界。
* **内外判定：** 这一步调用了外部脚本 `intriangulation.m`。它用来判断真实超声采样出的那 50 个 3D 点，是不是**全部**都落在了你膨胀后的 3D 肿瘤网格内部。只要有一个点在外面，IC 就判为 Fail（0）。

#### 3. 计算 TRE (Target Registration Error)

由 `TREcalc` 函数实现，这里用到了一个简化的 **ICP（迭代最近点）** 算法，迭代 10 次（`K=10`）：

* **初始化：** 先通过计算预测点云和真实点云的重心（CoG）差异，得到一个初始的平移向量 $t_0$。
* **寻找最近点：** 调用 `ClosetPoint` 函数（底层使用了 MATLAB 的 `knnsearch`），为你预测网格上的点找到距离真实 3D 轮廓最近的对应点。
* **绝对定向 (Absolute Orientation)：** 调用了外部脚本 `absor.m`。它根据找出的对应点对，计算出一个最优的刚性变换矩阵（旋转 $R$ 和平移 $t$），把预测点云尽量拉向真实点云。
* **计算误差：** 迭代 10 次后，计算两组点云之间的欧氏距离（`vec_norm`）的平均值，这就是最终输出的 TRE（单位：毫米）。

---

### 你需要关注的其他脚本依赖

根据上面的分析，你手头的目录中还有这几个脚本是支撑主逻辑运行的，你不需要深入改写它们，但得确保它们在工作目录下：

1. **`stlTools/` 文件夹：** 里面的 `stlRead.m` 被主程序调用，专门用来解析 `.stl` 3D 模型文件。
2. **`absor.m`：** 这是著名的 Horn 方法（基于四元数）的 MATLAB 实现。它在计算 TRE 时用于求解两组 3D 点云之间的最优刚性变换。
3. **`intriangulation.m`：** 在计算 IC 时，这是用来判断 3D 点是否在 3D 闭合网格（Tetrahedron / Mesh）内部的数学算法脚本。





# A Methodology and Clinical Dataset with Ground-truth to Evaluate Registration Accuracy Quantitatively in Computer-assisted Laparoscopic Liver Resection
# N. Rabbani, L. Calvet, Y. Espinel, B. Le Roy, M. Ribeiro, E. Buc and A. Bartoli

This is the data, code and registration results for the evaluation of registration accuracy in computer-assisted laparoscopic liver resection. 
The dataset is provided for research purposes only.
It cannot be reproduced or modified without explicit agreement from the authors.
The results and testing of registration methods should be reported to the authors for possible inclusion on the companion webpage.
You are free to use the evaluation results in your publications should you read and cite our paper "A Methodology and Clinical Dataset with Ground-truth to Evaluate Registration Accuracy Quantitatively in Computer-assisted Laparoscopic Liver Resection" by N. Rabbani, L. Calvet, Y. Espinel, B. Le Roy, M. Ribeiro, E. Buc and A. Bartoli.

Run Evaluation.m to calculate the Inclusion Criterion and the Target Registration Error.
More information about the dataset and how to use it is available from http://igt.ip.uca.fr/~ab.

