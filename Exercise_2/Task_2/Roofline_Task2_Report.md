# Exercise 2 — Task 2: Roofline model analysis (CUDA example)

## Goal
Explain (using the Roofline model) why the CUDA example achieved ~197 GFlop/s instead of the theoretical ~1430 GFlop/s on Nvidia Tesla K40c.

---

## Given (from assignment & lecture)
- Measured performance: **196.866 GFlop/s**
- GPU: **NVIDIA Tesla K40c**
- Peak FP64 performance: **1430 GFlop/s**
- Peak memory bandwidth: **288 GB/s**
- ECC enabled → effective bandwidth ≈ **230 GB/s**

---

## 1. Kernel memory traffic
Per loop iteration:
- Read `b[i]` → 8 bytes
- Write `a[i]` → 8 bytes

**Bytes per iteration**
```
B = 16 bytes
```

---

## 2. Floating-point operations
Per iteration:
- Negation: 1 flop
- Multiply + add: 2 flops
- 4 Newton-style updates × 4 flops each = 16 flops

**FLOPs per iteration**
```
F = 19 flops
```

---

## 3. Operational Intensity
```
I = F / B = 19 / 16 = 1.1875 FLOP/byte
```

---

## 4. Roofline limits

### Memory-bound limit
Using ECC-adjusted bandwidth:
```
P_mem = 230 × 1.1875 = 273 GFlop/s
```

### Compute-bound limit
```
P_peak = 1430 GFlop/s
```

**Roofline performance bound**
```
P_roofline = min(P_mem, P_peak) = 273 GFlop/s
```

---

## 5. Balance point
```
I_balance = P_peak / BW = 1430 / 230 ≈ 6.2 FLOP/byte
```

Since:
```
I = 1.19 < I_balance = 6.2
```
the kernel is **memory-bound**.

---

## 6. Conclusion
The kernel is clearly **memory-bound**.  
The Roofline model predicts a maximum achievable performance of ~273 GFlop/s, which is reasonably close to the measured **196.9 GFlop/s**.  

The gap is explained by:
- ECC overhead
- Instruction mix inefficiency
- Memory latency and non-ideal caching
- Limited occupancy and scheduling effects

Therefore, the measured performance is **consistent with the Roofline model** and realistic for this kernel.
