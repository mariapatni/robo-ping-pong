---
title: Only drops the arm can reach
date: 14 Oct 2026
trained: +8M steps from v3
---

v3 continued, with drops restricted to where the arm can get under the ball in time.

## What changed from the previous version
- Drop sampler excludes unreachable drops.

## How it did
57% kept up 60 s; 102.3 bounces mean. Fails mostly by hitting the arm (17%).
