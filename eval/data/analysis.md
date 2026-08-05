# 分层上下文压缩评测结果

任务总数：30（SWE-bench Lite 子集）

## 实验一：32k 水位窗口（强制触发压缩，测机制效果）

配对任务数：30；压缩触发过的任务：18（共 157 次压缩）

| 指标 | OFF | ON | 对比 |
|---|---|---|---|
| 峰值上下文 token（均值） | 18,228 | 17,296 | 降低 5.1% |
| 峰值上下文 token（中位） | 16,866 | 18,880 | 降低 -11.9% |
| 峰值上下文 token（最大） | 44,894 | 27,191 | 降低 39.4% |
| 运行轮次（均值） | 24.6 | 34.3 | 1.39x |
| 运行轮次（中位） | 22.0 | 35.5 | 1.61x |
| 任务解决数 | 10/30 | 11/30 | — |

### 失败原因分布

- off: {'api_error': 6, '': 19, 'max_rounds_exceeded': 2, 'crash': 1, 'loop_detected': 1, 'runner_crash': 1}
- on_w32000: {'': 14, 'max_rounds_exceeded': 8, 'loop_detected': 4, 'api_error': 4}

### 压缩触发层级分布

{'L1': 105, 'L4': 32, 'L2': 20}

### 分仓库对照

| 仓库 | 任务数 | 峰值OFF均值 | 峰值ON均值 | 降低 | 轮次OFF | 轮次ON | 倍数 | 解决OFF | 解决ON |
|---|---|---|---|---|---|---|---|---|---|
| django/django | 14 | 19,088 | 16,999 | 10.9% | 28.1 | 33.4 | 1.19x | 5 | 5 |
| pylint-dev/pylint | 5 | 19,745 | 17,701 | 10.3% | 20.4 | 36.6 | 1.79x | 2 | 1 |
| pytest-dev/pytest | 2 | 16,374 | 21,926 | -33.9% | 18.0 | 44.5 | 2.47x | 1 | 1 |
| sphinx-doc/sphinx | 2 | 3,622 | 20,274 | -459.8% | 5.5 | 38.0 | 6.91x | 0 | 1 |
| sympy/sympy | 7 | 20,127 | 15,424 | 23.4% | 28.0 | 30.4 | 1.09x | 2 | 3 |

### 逐任务明细（on_w32000）

| instance | repo | 峰值OFF | 峰值ON | 轮次OFF | 轮次ON | resolved OFF/ON | 压缩次数 |
|---|---|---|---|---|---|---|---|
| django__django-15738 | django/django | 22,050 | 22,911 | 31 | 50 | n/n | 16 |
| django__django-15789 | django/django | 14,940 | 6,693 | 17 | 11 | n/n | 0 |
| django__django-15790 | django/django | 18,644 | 18,805 | 31 | 35 | Y/Y | 1 |
| django__django-16379 | django/django | 7,479 | 11,881 | 12 | 21 | Y/Y | 0 |
| django__django-16400 | django/django | 38,779 | 27,156 | 50 | 50 | n/n | 20 |
| django__django-16408 | django/django | 13,935 | 10,068 | 21 | 15 | n/n | 0 |
| django__django-16527 | django/django | 29,281 | 11,332 | 44 | 26 | Y/Y | 0 |
| django__django-16595 | django/django | 23,544 | 18,580 | 47 | 29 | Y/Y | 1 |
| django__django-16816 | django/django | 29,549 | 20,683 | 34 | 33 | n/n | 7 |
| django__django-16820 | django/django | 24,770 | 19,065 | 31 | 49 | n/n | 1 |
| django__django-16873 | django/django | 14,950 | 18,954 | 30 | 36 | Y/n | 0 |
| django__django-16910 | django/django | 8,201 | 22,330 | 14 | 50 | n/n | 13 |
| django__django-17051 | django/django | 11,621 | 19,163 | 18 | 43 | n/n | 1 |
| django__django-17087 | django/django | 9,494 | 10,370 | 14 | 20 | n/Y | 0 |
| pylint-dev__pylint-5859 | pylint-dev/pylint | 13,957 | 13,649 | 19 | 21 | Y/Y | 0 |
| pylint-dev__pylint-6506 | pylint-dev/pylint | 24,061 | 19,802 | 26 | 50 | n/n | 8 |
| pylint-dev__pylint-7114 | pylint-dev/pylint | 16,608 | 27,173 | 0 | 50 | n/n | 26 |
| pylint-dev__pylint-7228 | pylint-dev/pylint | 18,258 | 20,006 | 23 | 50 | n/n | 8 |
| pylint-dev__pylint-7993 | pylint-dev/pylint | 25,839 | 7,876 | 34 | 12 | Y/n | 0 |
| pytest-dev__pytest-11143 | pytest-dev/pytest | 17,124 | 16,660 | 19 | 41 | Y/Y | 0 |
| pytest-dev__pytest-11148 | pytest-dev/pytest | 15,625 | 27,191 | 17 | 48 | n/n | 24 |
| sphinx-doc__sphinx-10451 | sphinx-doc/sphinx | 7,243 | 21,577 | 11 | 50 | n/n | 15 |
| sphinx-doc__sphinx-11445 | sphinx-doc/sphinx | 0 | 18,972 | 0 | 26 | n/Y | 1 |
| sympy__sympy-23191 | sympy/sympy | 44,894 | 13,818 | 50 | 16 | n/n | 0 |
| sympy__sympy-23262 | sympy/sympy | 11,055 | 19,195 | 20 | 49 | n/Y | 2 |
| sympy__sympy-24066 | sympy/sympy | 22,616 | 5,445 | 35 | 10 | Y/n | 0 |
| sympy__sympy-24102 | sympy/sympy | 17,280 | 18,668 | 29 | 38 | n/n | 1 |
| sympy__sympy-24152 | sympy/sympy | 12,076 | 10,878 | 15 | 18 | Y/Y | 0 |
| sympy__sympy-24213 | sympy/sympy | 5,233 | 18,430 | 7 | 32 | n/Y | 1 |
| sympy__sympy-24909 | sympy/sympy | 27,734 | 21,537 | 40 | 50 | n/n | 11 |

## 实验二：200k 真实窗口（自然触发）

配对任务数：29；压缩触发过的任务：0（共 0 次压缩）

| 指标 | OFF | ON | 对比 |
|---|---|---|---|
| 峰值上下文 token（均值） | 18,857 | 19,287 | 降低 -2.3% |
| 峰值上下文 token（中位） | 17,124 | 16,675 | 降低 2.6% |
| 峰值上下文 token（最大） | 44,894 | 38,642 | 降低 13.9% |
| 运行轮次（均值） | 25.5 | 28.2 | 1.11x |
| 运行轮次（中位） | 23.0 | 25.0 | 1.09x |
| 任务解决数 | 10/29 | 11/29 | — |

### 失败原因分布

- off: {'api_error': 6, '': 19, 'max_rounds_exceeded': 2, 'crash': 1, 'loop_detected': 1}
- on_w200000: {'': 20, 'loop_detected': 3, 'max_rounds_exceeded': 1, 'api_error': 5}

### 压缩触发层级分布

{}

### 分仓库对照

| 仓库 | 任务数 | 峰值OFF均值 | 峰值ON均值 | 降低 | 轮次OFF | 轮次ON | 倍数 | 解决OFF | 解决ON |
|---|---|---|---|---|---|---|---|---|---|
| django/django | 14 | 19,088 | 17,117 | 10.3% | 28.1 | 27.5 | 0.98x | 5 | 6 |
| pylint-dev/pylint | 5 | 19,745 | 21,210 | -7.4% | 20.4 | 28.2 | 1.38x | 2 | 1 |
| pytest-dev/pytest | 2 | 16,374 | 27,254 | -66.4% | 18.0 | 26.0 | 1.44x | 1 | 1 |
| sphinx-doc/sphinx | 1 | 7,243 | 26,098 | -260.3% | 11.0 | 38.0 | 3.45x | 0 | 0 |
| sympy/sympy | 7 | 20,127 | 19,005 | 5.6% | 28.0 | 28.9 | 1.03x | 2 | 3 |

### 逐任务明细（on_w200000）

| instance | repo | 峰值OFF | 峰值ON | 轮次OFF | 轮次ON | resolved OFF/ON | 压缩次数 |
|---|---|---|---|---|---|---|---|
| django__django-15738 | django/django | 22,050 | 12,543 | 31 | 22 | n/n | 0 |
| django__django-15789 | django/django | 14,940 | 14,684 | 17 | 17 | n/Y | 0 |
| django__django-15790 | django/django | 18,644 | 13,702 | 31 | 25 | Y/Y | 0 |
| django__django-16379 | django/django | 7,479 | 20,493 | 12 | 35 | Y/n | 0 |
| django__django-16400 | django/django | 38,779 | 10,318 | 50 | 13 | n/n | 0 |
| django__django-16408 | django/django | 13,935 | 15,714 | 21 | 23 | n/n | 0 |
| django__django-16527 | django/django | 29,281 | 14,780 | 44 | 30 | Y/Y | 0 |
| django__django-16595 | django/django | 23,544 | 18,176 | 47 | 32 | Y/Y | 0 |
| django__django-16816 | django/django | 29,549 | 38,507 | 34 | 47 | n/n | 0 |
| django__django-16820 | django/django | 24,770 | 34,441 | 31 | 50 | n/n | 0 |
| django__django-16873 | django/django | 14,950 | 10,832 | 30 | 23 | Y/Y | 0 |
| django__django-16910 | django/django | 8,201 | 5,699 | 14 | 10 | n/n | 0 |
| django__django-17051 | django/django | 11,621 | 19,723 | 18 | 40 | n/n | 0 |
| django__django-17087 | django/django | 9,494 | 10,022 | 14 | 18 | n/Y | 0 |
| pylint-dev__pylint-5859 | pylint-dev/pylint | 13,957 | 16,675 | 19 | 25 | Y/Y | 0 |
| pylint-dev__pylint-6506 | pylint-dev/pylint | 24,061 | 23,790 | 26 | 35 | n/n | 0 |
| pylint-dev__pylint-7114 | pylint-dev/pylint | 16,608 | 19,365 | 0 | 24 | n/n | 0 |
| pylint-dev__pylint-7228 | pylint-dev/pylint | 18,258 | 32,947 | 23 | 43 | n/n | 0 |
| pylint-dev__pylint-7993 | pylint-dev/pylint | 25,839 | 13,273 | 34 | 14 | Y/n | 0 |
| pytest-dev__pytest-11143 | pytest-dev/pytest | 17,124 | 15,866 | 19 | 22 | Y/Y | 0 |
| pytest-dev__pytest-11148 | pytest-dev/pytest | 15,625 | 38,642 | 17 | 30 | n/n | 0 |
| sphinx-doc__sphinx-10451 | sphinx-doc/sphinx | 7,243 | 26,098 | 11 | 38 | n/n | 0 |
| sympy__sympy-23191 | sympy/sympy | 44,894 | 12,570 | 50 | 18 | n/n | 0 |
| sympy__sympy-23262 | sympy/sympy | 11,055 | 25,825 | 20 | 48 | n/Y | 0 |
| sympy__sympy-24066 | sympy/sympy | 22,616 | 10,272 | 35 | 16 | Y/n | 0 |
| sympy__sympy-24102 | sympy/sympy | 17,280 | 19,459 | 29 | 27 | n/n | 0 |
| sympy__sympy-24152 | sympy/sympy | 12,076 | 9,174 | 15 | 14 | Y/Y | 0 |
| sympy__sympy-24213 | sympy/sympy | 5,233 | 20,490 | 7 | 29 | n/Y | 0 |
| sympy__sympy-24909 | sympy/sympy | 27,734 | 35,248 | 40 | 50 | n/n | 0 |
