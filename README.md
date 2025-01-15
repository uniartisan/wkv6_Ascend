# rwkv_vector 算子编译运行
Ascendc torch算子支持NPU编译



## NPU模式编译
NPU模式为上板模式，可测试算子调用实际性能

1. 进入npu目录
```
cd npu
```

2. 修改CMakeLists.txt中 ASCEND_CANN_PACKAGE_PATH 为本地CANN包环境安装路径：
```
vim CMakeLists.txt
```
```
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/ascend-toolkit/latest" CACHE STRING "ASCEND CANN package installation directory")
```

3. 修改run.sh中 ASCEND_HOME_DIR 为本地CANN包环境安装路径
```
vim run.sh
```
```
export ASCEND_HOME_DIR=/usr/local/ascend-toolkit/latest
```

4. 生成测试用数据
可进入脚本修改数据参数: B, T, C, H等
```
cd scripts
python gen_data.py
```

5. 编译并运行测试脚本
```
bash run.sh -r npu -v Ascend910B1
```
其中-r npu表示当前编译为npu模式，-v Ascend910B3表示当前NPU环境为910B3，可根据实际运行的NPU处理器修改

6. 在 npu/build 文件夹下找到 `npu/build/rwkv6_vector.cpython*.so`

这个运行库是 pybind 的编译产物，复制到需要使用的 python 文件同文件夹下，使用 `npu/torch_test.py` 同样的调用方法调用，当前仅支持 float16 输入

# 性能调优
## 修改输入矩阵大小
在scripts/gen_data.py脚本中修改相应参数，并在main_npu.cpp中做对应修改

## 修改Tiling
在main_npu.cpp中修改 tileLength 参数，算子对T维度进行Tiling切分，当前代码实现需要保证tileLength修改为 T 的因数。
Vector算子内存计算在UB (unified buffer) 中进行，UB内存大小为192KB，因此 tileLength 受UB内存大小约束。
具体UB内存排布见算子设计文档及代码。

## 修改核数
在main_npu.cpp中修改 CORE_NUM 参数，当前代码实现无需保证 CORE_NUM 为 B * H 的因数。
CORE_NUM最大值参见当前NPU处理器型号参数。