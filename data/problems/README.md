## Data for Loop Invariant Generation

In this directory, we are organizing the problem descriptions of Loop Invariant generation problem. 

The problems are taken from [LoopInvGen benchmark](https://github.com/SaswatPadhi/LoopInvGen/tree/master/benchmarks) organized by Padhi et al. (PLDI'16). These problems are compiled in [SyGus](https://arxiv.org/pdf/2312.06001.pdf) format and released under [MIT License](https://github.com/SaswatPadhi/LoopInvGen/blob/master/LICENSE.md). This [readme](https://github.com/SaswatPadhi/LoopInvGen/blob/master/benchmarks/README.md) provides details description of these benchmarks. 

In the [`in_scope`](in_scope/) folder, there are 541 problem descriptions (`*.sl`) files, subset of the problems in original LoopInvGen problems. 


### To cite these problems
```
@inproceedings{pldi/2016/PadhiSM,
  author    = {Saswat Padhi and Rahul Sharma and Todd D. Millstein},
  title     = {Data-Driven Precondition Inference with Learned Features},
  booktitle = {Proceedings of the 37th {ACM} {SIGPLAN} Conference on Programming
               Language Design and Implementation, {PLDI} 2016, Santa Barbara, CA,
               USA, June 13-17, 2016},
  pages     = {42--56},
  year      = {2016},
  url       = {http://doi.acm.org/10.1145/2908080.2908099},
  doi       = {10.1145/2908080.2908099}
}
```