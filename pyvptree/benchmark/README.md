# Comparison Benchmarks

How to generate standard benchmarks:

In the pyvptree root, type:

```
make benchmarks
```

To customize the benchmark generation dimension range, see the example below:

```
export PYTHONPATH=$PWD
python3 pyvptree/benchmark/run_benchmarks.py --min-dimension=3 --max-dimension=8
```

This will write result images to a local ./results folder.


## Benchmarks for fixed K = 16

All benchmarks were generated using 12th Gen Intel(R) Core(TM) i7-1270P with 16 cores.

Benchmarks for K=16 are displayed below:

### 2 to 10 dimensions Range

![k=16, L2 index](../../docs/img/from_2_to_10/VPTreeL2Index_k_16.png "K=16, L2 index")

### 11 to 16 dimensions Range

![k=16, L2 index](../../docs/img/from_11_to_16/VPTreeL2Index_k_16.png "K=16, L2 index")

### 17 to 32 dimensions Range
![k=16, L2 index](../../docs/img/from_17_to_32/VPTreeL2Index_k_16.png "K=16, L2 index")

### 33 to 48 dimensions Range

![k=16, L2 index](../../docs/img/from_33_to_48/VPTreeL2Index_k_16.png "K=16, L2 index")

### Benchmarks for other values of K

The graphs below include K values of 1, 2, 4, 8 and 16:

### 2 to 10 dimensions Range, for K=1, 2, 4, 8, 16

![k=16, L2 index](../../docs/img/from_2_to_10/VPTreeL2Index_k_16.png "K=16, L2 index")
![k=8, L2 index](../../docs/img/from_2_to_10/VPTreeL2Index_k_8.png "K=8, L2 index")
![k=4, L2 index](../../docs/img/from_2_to_10/VPTreeL2Index_k_4.png "K=4, L2 index")
![k=2, L2 index](../../docs/img/from_2_to_10/VPTreeL2Index_k_2.png "K=2, L2 index")
![k=1, L2 index](../../docs/img/from_2_to_10/VPTreeL2Index_k_1.png "K=1, L2 index")

### 11 to 16 dimensions Range, for K=1, 2, 4, 8, 16

![k=16, L2 index](../../docs/img/from_11_to_16/VPTreeL2Index_k_16.png "K=16, L2 index")
![k=8, L2 index](../../docs/img/from_11_to_16/VPTreeL2Index_k_8.png "K=8, L2 index")
![k=4, L2 index](../../docs/img/from_11_to_16/VPTreeL2Index_k_4.png "K=4, L2 index")
![k=2, L2 index](../../docs/img/from_11_to_16/VPTreeL2Index_k_2.png "K=2, L2 index")
![k=1, L2 index](../../docs/img/from_11_to_16/VPTreeL2Index_k_1.png "K=1, L2 index")

### 17 to 32 dimensions Range, for K=1, 2, 4, 8, 16

![k=16, L2 index](../../docs/img/from_17_to_32/VPTreeL2Index_k_16.png "K=16, L2 index")
![k=8, L2 index](../../docs/img/from_17_to_32/VPTreeL2Index_k_8.png "K=8, L2 index")
![k=4, L2 index](../../docs/img/from_17_to_32/VPTreeL2Index_k_4.png "K=4, L2 index")
![k=2, L2 index](../../docs/img/from_17_to_32/VPTreeL2Index_k_2.png "K=2, L2 index")
![k=1, L2 index](../../docs/img/from_17_to_32/VPTreeL2Index_k_1.png "K=1, L2 index")

### 33 to 48 dimensions Range, for K=1, 2, 4, 8, 16

![k=16, L2 index](../../docs/img/from_33_to_48/VPTreeL2Index_k_16.png "K=16, L2 index")
![k=8, L2 index](../../docs/img/from_33_to_48/VPTreeL2Index_k_8.png "K=8, L2 index")
![k=4, L2 index](../../docs/img/from_33_to_48/VPTreeL2Index_k_4.png "K=4, L2 index")
![k=2, L2 index](../../docs/img/from_33_to_48/VPTreeL2Index_k_2.png "K=2, L2 index")
![k=1, L2 index](../../docs/img/from_33_to_48/VPTreeL2Index_k_1.png "K=1, L2 index")
