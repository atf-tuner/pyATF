pyATF: The Auto-Tuning Framework (ATF) in Python
================================================

Auto-Tuning Framework (ATF) is a generic, general-purpose auto-tuning approach that automatically finds well-performing values of performance-critical parameters (a.k.a. tuning parameters), such as sizes of tiles and numbers of threads.
ATF works for programs written in arbitrary programming languages and belonging to arbitrary application domains, and it allows tuning for arbitrary objectives (e.g., high runtime performance and/or low energy consumption).

A major feature of ATF is that it supports auto-tuning programs whose tuning parameters have *interdependencies* among them, e.g., the value of one tuning parameter has to be smaller than the value of another tuning parameter.
For this, ATF introduces novel process to *generating*, *storing*, and *exploring* the search spaces of interdependent tuning parameters (discussed in detail `here <https://dl.acm.org/doi/abs/10.1145/3427093/>`__).

ATF comes with easy-to-use user interfaces to make auto-tuning appealing also to common application developers.
The Interfaces are based on either:

1. *Domain-Specific Language (DSL)*, for auto-tuning at compile time (a.k.a. offline tuning) (discussed `here <https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.4423?casa_token=FO9i0maAi_MAAAAA%3AwSOYWsoqfLqcbazsprmzKkmI5msUCY4An5A7CCwi-_V8u10VdpgejcWuiTwYhWnZpaCJZ3NmXt86sg/>`__);
2. *General Purpose Language (GPL)*, for auto-tuning at runtime (a.k.a. online tuning), e.g., of *C++ programs* (referred to as *cppATF*, and discussed `here <https://ieeexplore.ieee.org/abstract/document/8291912/>`__) or *Python programs* (referred to as *pyATF*, and discussed `here <todo>`__).

**This repository contains pyATF, i.e., ATF with its GPL-based Python interface.**

Content
-------

.. toctree::
   getting_started
   api_reference/index