# SAXPY 详细教学指南

## 什么是 SAXPY？

**SAXPY** = **S**ingle-precision **A**-**X** **P**lus **Y**

这是一个简单的向量运算：**z = a × x + y**

其中：
- `x`, `y`, `z` 都是**向量**（数组），长度为 `size`
- `a` 是一个**标量**（单个数字）
- 对向量中的每个元素进行运算：`z[i] = a * x[i] + y[i]`

### 为什么学这个？

SAXPY 是理解 CUDA 的**入门级**程序，涉及：
- ✓ 内存管理（CPU ↔ GPU）
- ✓ 核函数编写
- ✓ 线程索引计算
- ✓ 并行化思想

---

## 代码执行流程（概览）

```
main() 主函数的工作流：

1. [CPU] 创建数据
   ├─ 生成向量 x, y，标量 a
   └─ 计算参考答案 z_gold（用 CPU）

2. [GPU内存] 分配空间
   ├─ 为 d_x, d_y, d_z 开辟空间
   └─ 等待后续使用

3. [数据传输] CPU → GPU
   ├─ 把 x 复制到 d_x
   ├─ 把 y 复制到 d_y
   └─ （d_z 不用复制，待会儿会被写入）

4. [GPU计算] 启动并行核函数
   ├─ 创建多个 CUDA 线程
   ├─ 每个线程计算一个 z[i] = a * x[i] + y[i]
   └─ 所有线程**并行**执行

5. [数据传输] GPU → CPU
   ├─ 把结果 d_z 复制回 z
   └─ CPU 现在有了计算结果

6. [验证] 对比答案
   ├─ 比较 z（GPU 结果）和 z_gold（CPU 参考）
   └─ 应该几乎相同（可能有微小浮点误差）

7. [清理] 释放内存
   ├─ 删除 CPU 数组
   └─ 释放 GPU 内存
```

---

## 详细 TODO 讲解

### TODO 1: 设置数据大小

```cpp
const unsigned size = 0;  // ← 改这里
```

**你需要做什么**：
- 用一个数字替换 `0`
- 这个数字表示向量的长度
- **建议**：先用 `64` 试试（足够测试，又不太大）
- **可选尝试**：256, 1024, 2048, 14, 103, 1025, 3127

**为什么这样做**：
- 需要知道向量有多长
- 这决定了后续内存分配的大小
- `size` 越大，GPU 的并行优势越明显

**思考**：如果 `size = 64`，那么你会创建 64 个 x[i], y[i], z[i]

---

### TODO 2: 在 GPU 上分配内存

```cpp
// 代码提示：
// CUDA(cudaMalloc((void **)&pointer, size in bytes)));
```

**你需要做什么**：
- 为 `d_x` 分配内存（用 `cudaMalloc`）
- 为 `d_y` 分配内存（用 `cudaMalloc`）
- 为 `d_z` 分配内存（用 `cudaMalloc`）

**关键概念**：

| 概念 | 含义 | 例子 |
|------|------|------|
| **指针** | 变量地址 | `d_x` 指向 GPU 上的内存地址 |
| **大小（字节）** | 需要多少字节 | `size` 个 float = `size * sizeof(float)` 字节 |
| **`(void **)`** | 类型转换 | 统一接口，不用管 |

**计算大小**：
- 1 个 `float` = 4 字节
- `size` 个 `float` = `size * sizeof(float)` 字节
- `sizeof(float)` 在 C++ 中就是 4

**思考**：
- `d_x`, `d_y`, `d_z` 指向哪里？答案：GPU 显卡内存
- 为什么要分配三个不同的指针？答案：存放三个不同的数组 x, y, z

---

### TODO 3: 复制数据从 CPU 到 GPU

```cpp
// 代码提示：
// CUDA(cudaMemcpy(dest ptr, source ptr, size in bytes, direction enum));
```

**你需要做什么**：
- 把 `x`（CPU 上）复制到 `d_x`（GPU 上）
- 把 `y`（CPU 上）复制到 `d_y`（GPU 上）

**关键参数**：
| 参数 | 含义 | 例子 |
|------|------|------|
| **dest ptr** | 目标地址 | `d_x`（GPU）|
| **source ptr** | 源地址 | `x`（CPU）|
| **size** | 多少字节 | `size * sizeof(float)` |
| **direction** | 方向 | `cudaMemcpyHostToDevice`（CPU→GPU）|

**思考**：
- 为什么不复制 `d_z`？答案：`d_z` 是输出，等会儿 GPU 会写入
- 这一步叫什么？答案：Host-to-Device 传输（有延迟！）

---

### TODO 4: 配置线程和块

```cpp
const unsigned threadsPerBlock = 0;  // ← 改这里
const unsigned blocks = divup(size, threadsPerBlock);
```

**你需要做什么**：
- 决定每块有多少个线程
- **建议值**：128（平衡内存和性能）
- **可选尝试**：32, 64, 256, 512, 1024

**关键概念**：

```
【GPU网格 Grid】
  ├─ 块0  ├─ 线程0, 线程1, 线程2, ..., 线程127
  ├─ 块1  ├─ 线程0, 线程1, 线程2, ..., 线程127
  ├─ 块2  ├─ ...
  ├─ ...
```

**计算块数的逻辑**：
- 向量长度 = `size`
- 每块线程数 = `threadsPerBlock`
- 需要多少块？= `ceil(size / threadsPerBlock)`
  - 例：size=64, threadsPerBlock=128 → 只需 1 块
  - 例：size=256, threadsPerBlock=128 → 需要 2 块
  - 例：size=127, threadsPerBlock=128 → 需要 1 块
  - 例：size=129, threadsPerBlock=128 → 需要 2 块

**`divup()` 函数**：
- 这就是计算 `ceil(size / threadsPerBlock)` 的工具函数
- TODO 5 会让你实现它

**思考**：
- 为什么不用 1 个巨大的块？答案：GPU 有限制（最多 1024 线程/块）
- 为什么不用很多小块？答案：块太多，同步开销大，效率低

---

### TODO 5: 实现 `divup()` 函数

**在哪里改**：`common.cpp`（不是 `saxpy.cu`！）

**你需要做什么**：
- 实现一个函数，计算向上取整除法
- 输入：`unsigned size`（被除数），`unsigned div`（除数）
- 输出：`unsigned`（向上取整的结果）

**伪代码逻辑**：
```
divup(10, 3) 应该返回多少？
  10 ÷ 3 = 3.33...
  向上取整 = 4 ✓
  
divup(9, 3) 应该返回多少？
  9 ÷ 3 = 3
  向上取整 = 3 ✓
  
divup(64, 128) 应该返回多少？
  64 ÷ 128 = 0.5
  向上取整 = 1 ✓
```

**数学技巧**：
```
向上取整除法的数学公式：
ceil(a / b) = (a + b - 1) / b

例子：
(10 + 3 - 1) / 3 = 12 / 3 = 4 ✓
(9 + 3 - 1) / 3 = 11 / 3 = 3 ✓（整数除法）
(64 + 128 - 1) / 128 = 191 / 128 = 1 ✓
```

**思考**：为什么是 `(a + b - 1) / b` 而不是其他方式？

---

### TODO 6: 启动核函数

```cpp
// 代码提示：
// saxpy<<<blocks, threadsPerBlock>>>(....);
```

**你需要做什么**：
- 调用 `saxpy` 核函数
- 使用 `<<<blocks, threadsPerBlock>>>` 指定网格配置
- 传入 5 个参数：`d_z`, `d_x`, `d_y`, `a`, `size`

**核函数启动语法**：
```cuda
kernelName<<<numBlocks, threadsPerBlock>>>(arg1, arg2, ...);
           └─────────────────────────────┘
           这部分告诉 GPU：
           - 创建 numBlocks 个块
           - 每块有 threadsPerBlock 个线程
           - 总共 numBlocks × threadsPerBlock 个线程
```

**你的核函数签名**：
```cuda
__global__ void saxpy(float* const z, 
                      const float* const x, 
                      const float* const y, 
                      const float a, 
                      const unsigned size)
```

**参数对应**：
| 形参 | 实参 | 含义 |
|------|------|------|
| `z` | `d_z` | 输出数组（GPU内存） |
| `x` | `d_x` | 输入数组（GPU内存） |
| `y` | `d_y` | 输入数组（GPU内存） |
| `a` | `a` | 标量（直接传值） |
| `size` | `size` | 向量长度 |

**思考**：
- 为什么参数都是 GPU 指针？答案：核函数运行在 GPU 上！
- 核函数什么时候执行完？答案：不知道，这很关键！

---

### TODO 7: 复制结果从 GPU 到 CPU

```cpp
// 代码提示：
// 类似 TODO 3，但方向相反
```

**你需要做什么**：
- 把 `d_z`（GPU）复制回 `z`（CPU）

**关键参数**：
| 参数 | 含义 | 例子 |
|------|------|------|
| **dest ptr** | 目标地址 | `z`（CPU）|
| **source ptr** | 源地址 | `d_z`（GPU）|
| **size** | 多少字节 | `size * sizeof(float)` |
| **direction** | 方向 | `cudaMemcpyDeviceToHost`（GPU→CPU）|

**思考**：
- 这步之前是否需要同步？答案：**是！** 可能需要加 `cudaDeviceSynchronize()`

---

### TODO 8: 释放 GPU 内存

```cpp
// 代码提示：
// CUDA(cudaFree(device pointer));
```

**你需要做什么**：
- 释放 `d_x`, `d_y`, `d_z` 的 GPU 内存
- 这样下次程序运行时才能分配

**比喻**：
- `cudaMalloc` = 向银行借钱（开辟内存）
- `cudaFree` = 还钱（释放内存）
- 不还的话，下次没有可用内存（内存泄漏）

**思考**：为什么代码已经有 `delete[]` 释放 CPU 内存，还要 `cudaFree` 释放 GPU 内存？

---

### TODO 9-11: 核函数实现

这三个 TODO 在 `saxpy()` 核函数内部。

#### **TODO 9: 计算全局索引**

```cpp
unsigned idx = 0;  // ← 改这里
```

**你需要做什么**：
- 每个线程都需要知道"我是第几个线程"
- 用这个索引来访问数组元素

**核心公式**：
```
全局索引 = (块号 × 块内线程数) + 线程号

在 CUDA 中：
  blockIdx.x   = 块号（从 0 开始）
  blockDim.x   = 每块线程数（即 threadsPerBlock）
  threadIdx.x  = 块内线程号（从 0 开始）

例子：threadsPerBlock = 128
  块 0 的线程 0：idx = (0 × 128) + 0 = 0
  块 0 的线程 127：idx = (0 × 128) + 127 = 127
  块 1 的线程 0：idx = (1 × 128) + 0 = 128
  块 1 的线程 1：idx = (1 × 128) + 1 = 129
  ...
```

**思考**：
- 为什么要计算全局索引？答案：告诉线程处理数组的哪一个元素
- 如果 `idx` 超出了数组范围怎么办？答案：TODO 10 处理

#### **TODO 10: 边界检查**

```cpp
if (idx >= 0)  // ← 这个条件显然不对
    return;
```

**你需要做什么**：
- 检查 `idx` 是否越界
- 如果超过了向量长度 `size`，就直接返回（不做计算）

**原因**：
- 例：size = 100，threadsPerBlock = 128
- 块 0 会有 128 个线程，但只有 100 个元素
- 线程 100-127 应该什么都不做

**伪代码**：
```
if (idx 超过了数组范围) {
    return;  // 这个线程休息
}
```

**思考**：
- 什么时候会 idx 越界？答案：线程数 > 数据量
- 为什么不用 idx < 0 检查？答案：idx 是 unsigned，永远非负

#### **TODO 11: 执行 SAXPY 计算**

```cpp
// 执行计算：z[idx] = a * x[idx] + y[idx]
```

**你需要做什么**：
- 一行代码：把公式 `a * x[idx] + y[idx]` 的结果存到 `z[idx]`

**为什么这么简单**：
- SAXPY 本身就是简单的数学运算
- 难的是"如何让 GPU 并行做这个"
- 这个任务已经用前面的 9 个 TODO 完成了

**并行的魔力**：
```
CPU版本（串行）：
for (int i = 0; i < size; i++) {
    z[i] = a * x[i] + y[i];  // 循环，一个一个处理
}
耗时：size × 运算时间

GPU版本（并行）：
// 同时启动 size 个线程（分块执行）
__global__ void saxpy(...) {
    z[idx] = a * x[idx] + y[idx];  // 每个线程处理自己的
}
耗时：运算时间（不用乘 size！）
```

---

## 总结：你需要做什么

1. **TODO 1**：改一个数字（数组大小）
2. **TODO 2-3**：学会 CUDA 内存操作（分配、复制）
3. **TODO 4-5**：配置线程（计算块数）
4. **TODO 6-8**：核函数启动和清理
5. **TODO 9-11**：核函数体（索引、检查、计算）

---

## 学习建议

✓ 用纸和笔计算 `divup()` 的几个例子  
✓ 理解全局索引的计算公式  
✓ 在 NSight Debugger 中观察线程如何并行执行  
✓ 改变 `threadsPerBlock` 的值，看结果是否还是对的  
✓ 尝试错误的索引公式，看会发生什么（学会调试）

