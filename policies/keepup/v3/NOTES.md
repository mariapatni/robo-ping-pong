---
title: New paddle, 30 Hz, flight-time score, curriculum
date: 13-14 Oct 2026
trained: 20M steps (resumed from 2M)
---

Rebuilt for the real arm: the paddle bolted to the wrist horn, ~30 Hz control, and a score for seconds of good flight rather than bounce count.

## What changed from the previous version
- Wrist-horn paddle (no gripper).
- Control rate randomised around 30 Hz.
- Reward per second of good flight.
- Drops anywhere, with a curriculum that widens the drop area.

## How it did
Median rally 60 s (the limit); 57% of drops kept up 60 s; 116.6 bounces mean. Fails mostly by flying off (27%).
