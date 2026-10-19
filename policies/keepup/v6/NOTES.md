---
title: 120 s episodes, longer horizon
date: 15 Oct 2026
trained: +20M steps from v5
---

v5 continued on 120 s episodes with gamma 0.995, so slow drift away from the centre gets practised and paid for.

## What changed from the previous version
- Episode 120 s.
- gamma 0.99 -> 0.995.

## How it did
57% kept up 60 s; 103.9 bounces mean; within noise of v5. Fails mostly by flying off (20%).
