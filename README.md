# CUDA Implementations

A collection of CUDA programs ranging from beginner to advanced level, designed to build GPU programming knowledge step by step. Each example focuses on a specific concept, from simple device initialization to advanced parallel algorithms.

---

## 📂 Contents

### Beginner Level
- [Hello World with CUDA Initialization](./1%20Hello%20World.cu)
- [Parallel Increment with For Loop](./2%20ForLoop.cu)
- [Introduction to Shared Memory](./3%20SharedMemIntro.cu)
- [Custom Error Handler for Memory Allocation](./4%20MemAllocErrorHandler.cu)
- [Querying CUDA Device Information](./5%20DeviceInfo.cu)
- [Selecting Devices by Compute Capability](./6%20DeviceDiscovery.cu)
- [Vector Addition on the GPU](./7%20VectorSum.cu)
- [Convert RGB Image to Grayscale](./8.%20colorToGray.cu)
- [Image Blurring with CUDA](./9.%20ImgBlur.cu)
- [Matrix Multiplication on the GPU](./10.%20Matmul.cu)

### CUDA Memory
- [Automatic Variables in Registers](./Cuda%20Memory/1.%20autoVar.cu)
- [Automatic array variables – Local Memory, Thread Scope, Grid Lifetime](./Cuda%20Memory/2.%20autoVarArray.cu)
- [Shared memory variable – Shared Memory, Block Scope, Grid Lifetime](./Cuda%20Memory/3.%20sharedMem.cu)
- [Global memory variable – Global Memory, Grid Scope, Application Lifetime](./Cuda%20Memory/4.%20GlobalMem.cu)
- [Constant memory variable – Constant Memory, Grid Scope, Application Lifetime](./Cuda%20Memory/5.%20ConstMem.cu)
- [Tiled Matrix Multiplication using Shared Memory](./Cuda%20Memory/6.%20TileMatMul.cu)
- [Tiled Matrix Multiplication with Boundary Condition Checks (Extension of 6.TileMatMul.cu)](./Cuda%20Memory/7.%20TileMatMulBnd.cu)
- [Tiled Matrix Multiplication with Dynamic Shared Memory Allocation](./Cuda%20Memory/8.%20TiledMatMulDynamic.cu)

#### CUDA Memory Fencing
- [Memory fencing with __threadfence_block() – Intra-block memory visibility](./Cuda%20Memory/Memory%20fencing/1.%20threadFence_block.cu)
- [Memory fencing with __threadfence() – Global device-wide memory visibility](./Cuda%20Memory/Memory%20fencing/2.%20threadFence.cu)
- [Memory fencing with __threadfence_system() – Host and device memory visibility](./Cuda%20Memory/Memory%20fencing/3.%20threadFenceSys.cu)
---

## 🔧 How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/CUDA-implementations.git
   cd CUDA-implementations

2. Compile any example with nvcc:
    ```bash
    nvcc filename.cu -o output
    ./output

3. Ensure you have CUDA Toolkit installed and a compatible NVIDIA GPU.

<!-- 📜 License
This project is licensed under the MIT License – feel free to use and modify the code.

I can also **update this automatically with all new files** you provide, generating proper titles and links for each one so the README grows with your repo. Do you want me to do that? -->
