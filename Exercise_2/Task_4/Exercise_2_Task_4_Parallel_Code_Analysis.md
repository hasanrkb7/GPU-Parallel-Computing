# Exercise 2 — Task 4: Parallel code analysis (brief answers)

This file answers **all three questions** in *Task 4: “Answer the following questions briefly”*.  
(Topics: GPU memory model, suitable GPU programming model, and matrix–vector parallelism.)

---

## Q1) Memory model for the CUDA/HIP example (RTX A4000; also A100/V100)

**Situation described:** The main program allocates arrays on the CPU (host), then transfers data to the GPU using `cudaMemcpy` / `hipMemcpy`, and then calls `process()` which runs on the GPU.

**Best memory model:** **Discrete device memory model (host–device / explicit memory copies).**

**Justification (RTX A4000, 16 GB):**
- The CPU and GPU have *separate* physical memories.
- You must explicitly copy data **Host → Device** (H2D) before the kernel, and **Device → Host** (D2H) after the kernel, using `cudaMemcpy` / `hipMemcpy`.
- The GPU kernel accesses the data in **device memory**; the CPU cannot directly access that same memory without transfers.

**Comment for NVIDIA A100 (40 GB) and V100 (32 GB) in supercomputers:**
- The **memory model is still the same**: discrete GPU memory + explicit transfers.
- The key difference is *capacity and bandwidth* (and interconnects like PCIe / NVLink), which affect:
  - how large problems you can keep resident on the GPU,
  - how expensive transfers are relative to computation,
  - and whether multi-GPU workflows are feasible.
- Even on supercomputers, the common performance rule is: **minimize host–device transfers** and keep data on the GPU as long as possible.

---

## Q2) Which parallel programming model is suitable for GPU computing?

A good fit is the **SIMT / data-parallel programming model** (e.g., **CUDA** or **HIP**).

**Justification:**
- GPUs are designed to run **many lightweight threads** executing the same kernel over different data (massive parallelism).
- Workloads that apply the same operation to many elements (vectors, matrices, grids) map directly to GPU kernels.
- CUDA/HIP also expose memory hierarchy concepts (global/shared/local memory) that matter for GPU performance.

(Alternative valid answers: OpenCL; directive-based OpenMP/OpenACC offloading. CUDA/HIP is the most direct model.)

---

## Q3) Is matrix–vector multiplication `y = A x` parallelizable?

Yes, **highly parallelizable**.

For:
\[
y_i = \sum_{j=1}^{n} A_{i,j} x_j
\]

**Why:**
- Each output element `y_i` is independent of other rows.
- Therefore, different rows can be computed in parallel:
  - parallelize over `i` (rows),
  - and optionally also parallelize the inner sum with reductions (though on GPUs typically each thread/block handles one row or a tile).

**Caveat:**
- Inside each `y_i`, the summation is a reduction, so you must accumulate carefully to avoid race conditions (or structure the computation so each thread writes a unique output).

---
