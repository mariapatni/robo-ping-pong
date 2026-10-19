---
title: Physics and sensor randomisation
date: 13 Oct 2026
trained: 10M steps, randomised world
---

v1's task in a randomised world: physics and sensor noise vary per episode.

## What changed from the previous version
- Domain randomisation of ball, contact and sensor noise.

## How it did
58.6 bounces at 60 Hz, but 5.6 at 30 Hz: a 60 Hz specialist, useless at the arm's real 30 Hz.

## Caveat
No longer runs (24 inputs).
