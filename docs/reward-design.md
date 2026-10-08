# Reward function and training task

Design document. No implementation. Every number here is either measured (marked
**measured**), derived from measured numbers by a calculation reproduced in the
text, or a guess with its order-of-magnitude reasoning stated (marked **guess**).

The design is anchored on the three-stage annealed curriculum of
[Humanoid Whole-Body Badminton via Multi-Stage RL](https://arxiv.org/abs/2511.11218),
borrows its landing-target and inverse-success feed sampling from
[Achieving Human Level Competitive Robot Table Tennis](https://arxiv.org/abs/2408.03906),
takes its hardware-limit handling from
[Learning coordinated badminton skills for legged manipulators](https://arxiv.org/abs/2505.22974),
and takes its trajectory-state reward gating from
[Catching Spinning Table Tennis Balls in Simulation with Curriculum RL](https://arxiv.org/abs/2503.01251).

---

## 0. What the geometry forces on us, before any reward is written

I ran the trajectory integrator (real drag, `Cd = 0.47`, `e_table = 0.88`) over the
outgoing shot before choosing any reward term, because the feasible shot envelope
turns out to be much narrower than assumed and it dictates the shape of the reward.

Strike point taken at `x = 0.45 m`, `z = 0.10 m` (the middle of the measured
reachable band), shot at 25 degrees of elevation, landing on the far half, one
bounce, then met by arm B. Receiver-local depth `x_B = 1.257 - x`.

| `v_out` (m/s) | lands at `x_B` | B meets it at `x_B` | `z` there | ball speed | 2nd bounce `x_B` | verdict |
|---|---|---|---|---|---|---|
| 1.10 | 0.626 | 0.516 | 0.088 | 0.93 | 0.406 | strike point **outside** B's 0.48 m reach |
| 1.30 | 0.580 | 0.448 | 0.091 | 1.09 | 0.317 | OK, edge of reach |
| **1.55** | **0.517** | **0.356** | **0.095** | **1.29** | **0.222** | **centre of the envelope** |
| 1.80 | 0.446 | 0.255 | 0.100 | 1.48 | 0.068 | OK, edge of near-body limit |
| 2.00 | 0.385 | 0.169 | 0.105 | 1.62 | −0.043 | 2nd bounce off the end; strike inside B's base |
| 2.20 | 0.319 | 0.078 | 0.110 | 1.76 | −0.159 | unplayable |

**The sustainable outgoing-speed window is 1.30–1.80 m/s.** This corrects the
brief: a rally that requires a bounce on the receiving side cannot settle at
2.5 m/s on a 1.257 m table. Above ~1.85 m/s the ball's post-bounce apex falls
inside `x_B = 0.25` — inside the arm's folded near-body region — and the second
bounce leaves the table entirely.

Three consequences that shape everything below.

**(a) The reward must be sharp on outgoing speed and loose on outgoing angle.**
An angle sweep at `v_out = 1.55`:

| elevation | 10 deg | 20 deg | 30 deg | 40 deg | 45 deg |
|---|---|---|---|---|---|
| lands at `x` | 0.688 | 0.726 | 0.751 | 0.757 | 0.750 |
| B meets it at `x_B` | 0.408 | 0.370 | 0.346 | 0.348 | 0.360 |
| `z` at the midline | 0.063 | 0.090 | 0.115 | 0.137 | 0.146 |

Thirty-five degrees of elevation moves the landing point by 69 mm; 0.5 m/s of
speed moves it by 290 mm. So the underhand-angle objective can be shaped with a
**generous** tolerance without hurting placement, which is exactly what we want:
the user asked for underhand as a style preference, and the geometry says style
is nearly free. Speed is the expensive axis.

**(b) The required paddle speed is 0.25 m/s, not 1.0–1.5 m/s.** With
`v_out = (1+e)·v_paddle + e·|v_in|`, `e = 0.85`, `|v_in| = 1.29`:
`v_paddle = (1.55 − 0.85·1.29)/1.85 = 0.25 m/s`. The 1.0–1.5 m/s paddle-speed
budget is therefore **not** consumed by the swing. It is consumed by *travel* —
getting the paddle to the intercept point inside the window. This reframes the
servo-current constraint: the binding cost is the large fast repositioning move,
not the stroke.

**(c) The strike window is comfortable.** With the ball inside
`z ∈ [0.05, 0.15]` and `x ∈ [0.25, 0.48]`, the usable window is **178 ms, eight
control steps at 50 Hz**. A 10–20 ms dead time is 6–11% of it. The window is
widest at apex, where `dz/dt = 0`, so vertical position error per millisecond of
timing error is minimised precisely where we want the contact. That is the single
strongest argument for the bounce decision in §9.

**(d) There is no net in the model, and a regulation net is infeasible.** `z` at
the midline is 63–146 mm across the whole angle sweep. A regulation 152.5 mm net
is cleared by essentially nothing in the feasible envelope. See §9.

**(e) Lateral authority is weak.** Azimuth 15 deg moves the landing point to
`y = +0.075` and B's strike point to `y = +0.117` — about 7.8 mm of strike-`y` per
degree of azimuth, and azimuth costs range as `cos`. In v1, aim the `y`-target at
0 and get lateral spread from the **feed**, not from the shot.

---

## 1. Episode and termination structure

### Frame

Everything the policy sees and everything the reward computes is in the
**receiver-local frame** of the arm that owns the policy (see §5 for why this
makes one policy work on both arms). Origin at own base, `+x` toward the
opponent, `z = 0` at the table top. Own half is `x ∈ [0, 0.6285]`, own reach is
`x ∈ [0, 0.480]` (**measured**), opponent's half is `x ∈ [0.6285, 1.257]`,
opponent's reach is `x ∈ [0.777, 1.257]` in this frame.

### Episode

An episode is a sequence of **exchanges**. One exchange is: a ball arrives, the
arm strikes it, the shot resolves. Stage-dependent count (§4): one exchange in
S0–S2, up to four in S3.

**Start.** Arm at `HOME = [0, −0.6, 0.9, 0.5, 0]` plus uniform joint noise
(`±0.05 rad` in S0–S2, `±0.15 rad` in S3, so the policy cannot rely on a fixed
initial pose). Ball spawned per §5. `t = 0` at spawn.

**Sub-episode phase variable.** Following 2503.01251's trajectory-state one-hot,
the reward is gated by a discrete phase `τ` that the actor also observes as a
4-way one-hot, because it is what makes per-term gating unambiguous:

| `τ` | phase | entered when |
|---|---|---|
| `τ0` | incoming, pre-bounce | ball spawned, `x > 0.6285` or airborne on own half |
| `τ1` | incoming, post-bounce | ball's first contact with own half |
| `τ2` | outgoing | ball leaves the paddle face |
| `τ3` | resolved | shot's first table contact, or shot leaves the table volume |

### Termination

Terminate the current exchange on the **first** of:

1. **Resolved** — `τ3` reached. In S3, if the shot was returnable and exchanges
   remain, start the next exchange instead of ending the episode.
2. **Miss** — ball reaches `x < −0.05` (passed behind the base) or `z < −0.05`
   (below table level) without face contact. Reward 0 for the exchange, not
   negative (see hack 9 in §7).
3. **Second bounce on own half** — ball contacts own half (`x < 0.6285`) a second
   time before face contact. The point was lost. Reward 0.
4. **Fault** — any contact between the ball and a non-face geom (paddle rim,
   paddle back, any arm link); or any contact between an arm geom and the table;
   or a double hit (§7 hack 3 and 4). Reward 0, episode ends.
5. **Constraint termination** — self-collision, or a joint driven past its hard
   limit. Records a constraint cost (§3); episode ends.
6. **Timeout** — 3.0 s per exchange, 10.0 s per episode.

### The exact success criterion

An exchange is a **success** iff all of the following hold. This is the quantity
stage advancement is measured on, and it is deliberately stricter than
"in bounds" — the brief is right that in-bounds alone kills the rally at hit three.

1. The ball's only contact with the arm assembly was with the `paddle_face` geom,
   in a single contact episode of duration ≤ 15 ms.
2. The shot's first table contact after `τ2` is at `x_land ∈ [0.6285, 1.257]` and
   `|y_land| ≤ 0.36` — in bounds on the opponent's half, with a 40 mm margin
   inside the 0.40 m edge so that a marginal shot is not scored as a win.
3. The shot crossed the plane `x = 0.6285` at `z ≥ 0.030` (it did not scrape the
   table on the way over) and `z ≤ 0.35` (it was not a lob over the opponent).
4. **Returnable:** after that landing, the ball enters the opponent's strike
   volume `x ∈ [0.777, 1.257]`, `|y| ≤ 0.40`, `z ∈ [0.03, 0.22]` with speed
   `∈ [0.7, 2.0] m/s`, **and** its post-bounce apex lies at
   `x ∈ [0.777, 1.007]` (i.e. opponent-local `x_B ∈ [0.25, 0.48]`, the good part
   of the measured footprint, not the folded near-body region).

Criterion 4 is the one that matters and the one the tables in §0 were computed to
define. Criteria 2 and 3 are regulation; criterion 4 is the rally.

An episode's score is the number of successful exchanges.

---

## 2. Reward terms

### Structure

Total per control step:

```
r(t) = r_track(t)·1[τ0∨τ1]                        dense, annealed to zero
     + r_swing · δ(t = t*)                        sparse, fires once at scheduled impact
     + r_under · δ(t = t_contact)                 sparse, fires once at real contact
     + r_land  · δ(τ2 → τ3)                       sparse, fires once when the shot resolves
     + r_return· δ(τ2 → τ3)                       sparse
     + r_recover(t)·1[τ2∨τ3]                      dense, small
     + r_reg(t)                                   dense, small, negative
```

### Weight budget, so the numbers below are legible

One exchange is ~0.9 s ≈ 45 control steps at 50 Hz. Dense shaping is capped at
0.08/step, so ≤ 3.6 per exchange. A perfect exchange earns ~85. **Dense shaping
is therefore ≤ 4% of a successful exchange's return**, which is the ratio that
keeps shaping from becoming the objective. In S0 and early S1, where contact is
rare, it is the only signal there is. Discount `γ = 0.995` at 50 Hz (from
2505.22974), horizon 200 steps = 4 s, enough to value a four-exchange rally.

### The terms

#### `r_track` — paddle to intercept point

```
e_track = ||p_paddle − p̂_intercept||
r_track = 0.08 · exp(−e_track² / σ_t²) · 1[t̂_impact − t < 0.40 s]
```

Weight 0.08/step. `σ_t`: **0.25 → 0.08 m** across S0–S2, zero weight in S3.

Prevents: the policy having no gradient at all before its first accidental
contact — the cold-start problem every one of the four papers solves with a
position-tracking term. The 0.40 s gate is deliberate and is the countermeasure to
hack 1 in §7: the policy earns nothing for camping on the modal intercept point
during the ball's flight, only for being there in the last 20 control steps.

#### `r_swing` — the scheduled-impact reward (the load-bearing term)

Directly after 2511.11218. Fires **once**, at the scheduled impact time `t*`
predicted by the ball estimator, **whether or not contact actually happened**.
This is the trick that makes the term differentiable through a miss: a near-miss
scores almost as well as a hit, so the policy can climb toward contact instead of
waiting for a lucky one.

```
e_pos = ||p_paddle(t*) − p̂_intercept(t*)||
e_ori = 1 − n̂_face(t*) · d̂*              d̂* = unit desired outgoing direction
r_v   = 1 − exp(−max(0, v_paddle(t*)·n̂_face) / σ_v)

r_swing = 25 · exp(−e_pos²/σ_p) · exp(−e_ori/σ_r) · (1 + 0.3·r_v)
```

Weight **25** (**guess**; order of magnitude set so that one impact reward is
~7x the entire dense budget of an exchange, matching 2511.11218's "high-weight
hit reward" and 2505.22974's 6400-vs-3 ratio between the position-tracking
impulse and the per-step terms).

Annealing, scaled from 2511.11218's whole-body numbers to our 0.48 m workspace
(they anneal `σ_p` 2.0 → 0.1 for a ~2 m reach; our workspace is ~4x smaller, so
divide the *tolerance* by 4, i.e. the variance by 16):

- `σ_p`: **0.25 → 0.005 m²**, i.e. a position tolerance of 0.50 m → 0.07 m.
- `σ_r`: **8.0 → 1.0** (their values; `e_ori` is dimensionless so no rescaling).
- `σ_v`: **fixed at 1.0 m/s** (they use 3.0 for a racket moving at ~10 m/s; ours
  needs 0.25 m/s, so 1.0 puts `r_v ≈ 0.22` at the nominal swing and keeps the
  term from saturating).

Prevents: never making contact; and, via the multiplicative `e_ori` factor,
arriving in the right place with the paddle face pointing at the ceiling.

#### `r_under` — the underhand term

Fires once, at real face contact. **Shaped, not a constraint**, as the brief
requires. Three factors, **multiplied**, because a sum would let the policy buy
one factor with another (§7 hack 5):

```
λ    = (v_paddle · ẑ) / ||v_paddle||          lift fraction of paddle velocity
α    = asin(clip(n̂_face · ẑ, −1, 1))          face openness, radians above vertical
g_fwd = 1[v_paddle · x̂ > 0.30 m/s]            forward gate
g_out = 1[v_ball_out · ẑ > 0.20 m/s]          physical-consequence gate

r_under = 8 · g_fwd · g_out
            · exp(−(λ − λ*)² / σ_λ²)
            · exp(−(α − α*)² / σ_α²)
```

`λ* = sin(25°) = 0.42`. `α* = 30° = 0.52 rad`.
`σ_λ`: **0.60 → 0.18**. `σ_α`: **0.52 rad (30°) → 0.14 rad (8°)**.
Weight **8** (**guess**: one third of the impact reward — enough that the policy
will not ignore it, small enough that it cannot outbid landing the ball).

The tolerances stay generous on purpose. §0(a) showed that 10–45 degrees of
outgoing elevation all land within 69 mm of each other, so a tight `σ_α` would be
buying style at the cost of exploration for no placement benefit. The final
`σ_α = 8°` is a style assertion, not a physics requirement, and it is the first
thing to widen if S2 stalls.

The two gates are what make this a real underhand and not a pose. `g_fwd` stops
the policy scoring `λ = 1` by lifting a paddle that is travelling *backwards*
(which would satisfy "moving up" while chopping). `g_out` requires the *ball* to
leave rising — the consequence rather than the configuration — which is the one
thing a hack cannot fake.

Prevents: a flat drive; and, more subtly, a paddle that is *oriented* underhand
while *moving* like a drive.

#### `r_land` — in bounds, and where

The DeepMind LLCs are trained to maximise ball landing rate with a target landing
location and its standard deviation recorded per skill
([2408.03906](https://arxiv.org/abs/2408.03906)); 2503.01251 uses
`a + b/(1+e_lt²)²`. I use a two-part form: a large flat bonus for legality and a
Gaussian for placement, so that legality is not something placement accuracy can
trade away.

```
in_bounds = 1[x_land ∈ [0.6285, 1.257]] · 1[|y_land| ≤ 0.36]
            · 1[z at midline ∈ [0.030, 0.35]]

r_land = in_bounds · ( 10 + 25 · exp(−||p_land − p_target||² / σ_l²) )
```

`p_target = (0.73, 0.00)` in receiver-local coordinates — **derived** in §0: this
is opponent-local `x_B = 0.527`, the landing point that puts the post-bounce apex
at `x_B = 0.356`, the centre of the measured footprint. `σ_l`: **0.30 → 0.12 m**.
Weights **10 + 25** (**guess**; the flat part is comparable to `r_swing/2` so
legality alone is worth reaching for, the Gaussian part is comparable to
`r_swing` so placement is the main prize).

Prevents: a shot that is a perfect stroke and goes into the net, off the end, or
over the opponent's head.

#### `r_return` — the anti-rally-dies term

```
r_return = 15 · in_bounds
             · 1[apex_x ∈ [0.777, 1.007]]
             · 1[apex speed ∈ [0.7, 2.0] m/s]
```

Weight **15** (**guess**: large enough to break the tie between two in-bounds
shots, one returnable and one not; smaller than `r_land` because it is a
subset condition and should not dominate the primary placement gradient).

Prevents: exactly the failure the brief names — the rally dying at hit three.
This is the term that makes 1.30–1.80 m/s the learned speed band rather than
something the policy has to be told. Note it does the job of an explicit
outgoing-speed reward: a 2.2 m/s shot fails the apex test, a 1.1 m/s shot fails
it too. **No separate speed-cap penalty is needed**, which is one fewer term to
hack.

#### `r_recover` — ready position

```
r_recover = 0.03 · exp(−||q − q_ready||² / 0.5)   during τ2, τ3
```

`q_ready` is the joint configuration whose paddle sits at `(0.34, 0, 0.11)`, the
centre of the measured strike volume. Weight 0.03/step (**guess**, small).

Prevents: finishing every stroke in a contorted pose that physically cannot reach
the next ball — which single-exchange training would otherwise never penalise.
This is 2505.22974's "stand still when no target" reward (their scale 16) in
miniature.

#### `r_reg` — regularisation

All from the standard legged-gym set used by 2511.11218 and 2505.22974, scaled so
their summed magnitude over an exchange is ~1.4, i.e. under 2% of a successful
return.

| term | formula | weight | prevents |
|---|---|---|---|
| action rate | `−‖a_t − a_{t−1}‖²` | **0.02** | servo-chattering commands that a 50 Hz bus and a 10–20 ms dead time cannot track |
| joint accel | `−‖q̈‖²` | **1e-6** | jerk that shows up as backlash slop on hardware |
| torque | `−‖τ‖²` | **1e-4** | idle straining against gravity between exchanges |
| joint vel | `−‖q̇‖²` | **1e-4** | whipping the wrist for a swing that needs 0.25 m/s |
| alive | `+0.02` per step | **—** | see §7 hack 9: without it, a net-negative regulariser makes early termination optimal |

Note the weights are an order of magnitude *smaller* than 2505.22974's
(`−0.03` action rate, `−1e-5` torque) relative to their reward scale, because
the real limits are handled as constraints, not here. See §3.

---

## 3. What is a constraint, not a reward

This is where [2505.22974](https://arxiv.org/abs/2505.22974) is the decisive
reference. They enforced arm current with **N-P3O**, a constrained PPO variant,
and report that the constrained policy *never violated the limit*, while
baselines "violated the constraint even with soft over-current penalties."
That is the whole argument: a reward penalty is a price the policy will pay if
the task reward is bigger, and for us the task reward is 85 while the penalty
would be fractions of one. Use constraints.

Recommend **N-P3O** with the cost functions below, each with its own Lagrange
multiplier and its own violation budget `d`.

### C1 — servo current (the hardware limit)

Per joint, from the commanded torque:

```
I_i(t) = |τ_i(t)| / K_t          K_t ≈ 2.94 / 2.5 = 1.18 N·m/A   (see open question 3)
```

Two costs, because the servo has two distinct failure modes:

- **C1a, instantaneous:** `c_1a = Σ_i max(0, I_i − 1.8)`, budget `d = 0`.
  1.8 A is 72% of the 2.5 A stall current. **Guess**, but the reasoning is that
  stall current is a *stall* number and continuous operation near it is exactly
  the overheating the brief warns about.
- **C1b, thermal:** `c_1b = Σ_i max(0, Ī_i^{1s} − 0.9)` where `Ī^{1s}` is the
  RMS current over a trailing 1.0 s window, budget `d = 0.02` (2% of steps).
  0.9 A is 36% of stall. **Guess**, and the weakest number in this document —
  see open question 3.

`c_1b` is the one that matters and is the one a reward penalty cannot express,
because it is a property of a *window*, not a state.

### C2 — joint position limits

Hard limits are **measured** from the model: pan ±110°, lift ±100°, elbow ±96.8°,
wrist_flex ±95°, wrist_roll ±157.2°. Constraint with a 5° soft margin:

```
c_2 = Σ_i max(0, |q_i| − (q_i^max − 0.087 rad))     budget d = 0
```

The actions are already clipped to limits by `Arm.clip`, so this is not about
legality — it is about stopping the policy from *living* on the clip boundary,
where the sim's clipped action and the real servo's behaviour diverge most.

### C3 — joint velocity limits

**Measured** from the model: 1.92, 1.75, 1.69, 1.66, 2.74 rad/s.

```
c_3 = Σ_i max(0, |q̇_i| − 0.85·q̇_i^max)     budget d = 0.01
```

The 0.85 factor keeps a margin for the sim-to-real velocity gap. Note §0(b): the
stroke needs 0.25 m/s of paddle speed, so this constraint should bind only during
*travel*, and if it binds during the swing something is wrong with the learned
solution.

### C4 — collision

Any contact between an arm geom (other than `paddle_face`) and the table, the
ball, or another arm geom. `c_4 = 1[contact]`, budget `d = 0`. Also terminates.

### Kept as soft rewards, deliberately

Torque magnitude, joint acceleration, action rate, and energy stay in `r_reg`.
2505.22974 makes exactly this split — "the actuator torque and velocity
constraints were included as soft penalties in the reward function, as these were
also treated as soft constraints on the hardware." A servo that momentarily draws
a lot of torque is fine; a servo that overheats is not. Only the second gets a
constraint.

---

## 4. Curriculum

Four stages. S1–S3 map onto 2511.11218's three stages; S0 is prepended because
they got footwork for free from a pretrained locomotion policy and we have no
equivalent — an SO-101 that has never tracked a moving target will not make
contact by accident often enough to bootstrap.

Weights are **inherited** between stages (2503.01251: "the agent directly
inherits the model weights learned from the previous stage").

### S0 — track a ghost ball

- Ball is **non-colliding** (contacts disabled). It flies its trajectory and
  passes through the paddle.
- Active: `r_track`, `r_reg`, C2, C3, C4. Nothing else.
- Feed: narrowest distribution (§5, S0 column).
- `σ_t`: 0.25 → 0.12.
- **Advance when:** over 200 consecutive episodes, ≥ 90% have
  `min_t ‖p_paddle − p̂_intercept‖ < 0.05 m` within the 178 ms strike window.

Why this exists: it separates "can the arm get there in time" from "can it hit."
If S0 cannot reach 90%, the problem is kinematic or timing and no reward change
will fix it — go back to §0(c) and the feed distribution.

### S1 — make contact

- Contacts on. `paddle_face`, `paddle_rim`, `paddle_back` are separate geoms from
  the first episode of this stage.
- Add: `r_swing`, C1a. `r_land` present but with weight 0 (so its statistics can
  be logged without it shaping anything yet).
- `σ_p`: 0.25 → 0.04 m². `σ_r`: 8.0 → 3.0. `σ_t`: 0.12 → 0.08.
- **Advance when:** ≥ 80% face-contact rate over 200 episodes, and < 5% fault
  rate (rim/back/link contact).

### S2 — underhand, placed shot

The stage where the task actually gets specified. Longest by far; 2511.11218
spends ~20k of its ~21k iterations in the analogous stage.

- Add: `r_under`, `r_land` at full weight, `r_return`, `r_recover`, C1b.
- Anneal, all of it gated on success rate rather than iteration count (see below):
  `σ_p` 0.04 → 0.005 m²; `σ_r` 3.0 → 1.0; `σ_λ` 0.60 → 0.18;
  `σ_α` 30° → 8°; `σ_l` 0.30 → 0.12 m.
- Feed widens per §5 (S2 column).
- **Advance when:** ≥ 70% of exchanges meet the full §1 success criterion, **and**
  ≥ 60% of contacts pass the underhand test (`λ > 0.25`, `α ∈ [15°, 45°]`,
  both gates satisfied), over 200 episodes, with all `σ` at their final values.

### S3 — strip the shaping, randomise, self-play

2511.11218's S3 exactly: remove the auxiliary shaping, keep the task reward, turn
on the reality gap.

- **Remove:** `r_track` entirely (weight → 0). Remove the `exp(−e_pos²/σ_p)`
  *position* factor of `r_swing`, keeping only the orientation and swing-velocity
  factors at weight 8 — i.e. the policy no longer gets paid for being in the
  right place, only for what the stroke does. Keep `r_under`, `r_land`,
  `r_return`, `r_recover`, `r_reg`, and all constraints.
- **Turn on:** domain randomisation and observation noise (table below), action
  dead time sampled from `U{0, 1}` control steps (0–20 ms), and multi-exchange
  episodes (up to 4).
- **Opponent:** 70% a frozen snapshot of the current policy deployed on the other
  side (self-play), 30% the scripted feeder. The 30% floor is not optional —
  see §7 hack 11.
- **Done when:** mean exchanges per episode ≥ 3.0 on a **held-out, frozen** set
  of 500 scripted feeds, and zero C1a violations and ≤ 2% C1b violations over
  that evaluation.

Domain randomisation, scaled from 2511.11218's table to our parameters:

| parameter | range | source |
|---|---|---|
| table restitution | `U[0.80, 0.93]` | **measured** range, already in `config.py` |
| paddle restitution | `U[0.78, 0.90]` | around the stated `e ≈ 0.85` |
| table friction (tangential) | `U[0.15, 0.40]` | around the configured 0.25 |
| ball drag `Cd` | `U[0.40, 0.52]` | the range `config.py` cites from the literature |
| ball mass | `U[0.0025, 0.0029]` kg | regulation tolerance |
| actuator `kp` | `U[700, 1200]` | around the vendored 998.22 |
| actuator `kv` | `U[1.9, 3.6]` | around the vendored 2.731 |
| joint friction / backlash | `U[0.0, 0.01] rad` | the model's `backlash` class, magnitude unknown |
| paddle mass | `U[0.028, 0.034]` kg | nominal 30.8 g from `cad/out/paddle_props.json`; ±10% for print density and glue |
| action dead time | `U{0, 1}` steps | 10–20 ms from the brief |

Observation noise (applied to the *actor's* observation only):

| quantity | noise |
|---|---|
| ball position | `N(0, 0.005 m)` + 1 step of latency |
| ball velocity | `N(0, 0.05 m/s)` |
| joint position | `N(0, 0.004 rad)` (≈ one servo LSB at 12 bit over 360°) |
| joint velocity | `N(0, 0.10 rad/s)` |

### Annealing mechanism — success-gated, not scheduled

2511.11218 anneals on an iteration schedule (~20k iterations, 4096 envs,
12.4 h on one RTX 4090). We do not have that budget (see open question 7), and a
fixed schedule that outruns learning collapses training. Recommend instead:

```
every 200 episodes:
    if success_rate > 0.70:  σ ← 0.97 · σ     (floored at σ_final)
    never widen σ
```

Monotone, so the task never gets easier — which matters, because a σ that
oscillates makes the value function chase a moving target. 0.97 per 200 episodes
takes ~130 tightenings to go from 0.25 to 0.005, i.e. ~26k episodes in S2 *if*
success stays above 70% throughout. That is the right order of magnitude against
2511.11218's 20k iterations.

---

## 5. Launch / serve distribution

### Frame convention — the thing that makes one policy work twice

The two arms are **not** mirror images. They are the **same** arm related by a
180-degree rotation about the world `z` axis through the table centre
`(0.6285, 0, 0)`:

```
x_B =  1.257 − x_world
y_B = −y_world
z_B =  z_world
```

This is a **proper rotation** (determinant +1), so it preserves chirality. The
consequence is the whole reason the shared-policy plan works: **joint signs do
not flip.** A positive `shoulder_pan` command moves arm A's paddle toward
`+y_world` and arm B's paddle toward `−y_world`, which is `+y_B` in B's own
frame. Identical policy, identical joint order, identical signs, no per-arm
adaptation layer.

If the arms were instead physically mirrored — one bolted as a left-handed
assembly — the map would be a reflection, determinant −1, and you would have to
negate `shoulder_pan` and `wrist_roll` and re-derive the underhand term's `α`,
because a reflection turns an underhand lift into an underhand lift but turns a
right-handed wrist roll into a left-handed one. **Do not mirror the mounts.**

Related: 2408.03906 doubles its ball dataset "via reflection" from 14.2k to 28.5k
states. We get the same 2x augmentation for free — every logged exchange is a
valid training sample for both arms under the rotation — but note that our
augmentation is a **rotation**, and using their reflection would be wrong here
because it flips the chirality of the joint chain.

All observations, all reward geometry, and all launch parameters below are stated
in the receiver's own local frame.

### The feed should be a replica of the shot the policy must produce

This is the key design choice. §0 established the nominal shot: leave the paddle
at `(0.45, 0, 0.10)` at 1.55 m/s, 25 degrees up. A feed generated with **exactly
those parameters, mirrored**, produces (verified numerically):

- first bounce on the receiver's own half at `x = 0.517`, 0.211 s after release
- arriving at the bounce at `v_z = −1.40 m/s`
- post-bounce apex at `x = 0.356`, `z = 0.095`, ball speed 1.29 m/s,
  0.124 s after the bounce
- a 178 ms usable strike window spanning `x ∈ [0.250, 0.480]`

Which is the middle of the measured reachable footprint and the middle of the
measured strike band `[0.03, 0.22]`. Making the feed self-consistent with the
target shot means the distribution the policy trains on in S1 is the distribution
self-play will produce in S3, so there is no distribution shift at the S2→S3
boundary — which is where curricula usually break.

**This invalidates the current `LaunchConfig`.** The existing ranges
(`x_frac` 0.88–0.98, `z` 0.12–0.22, `speed` 1.0–1.5, `elevation` −2 to +8 deg)
produce feeds that bounce on the *far* half and cross into the footprint without
a second bounce. I swept that space and almost nothing in it bounces on the
receiver's half in the useful window. The feeder must be re-specified as below,
and the existing 60-seed validation must be re-run.

### Recommended `LaunchConfig`, per stage

Expressed as a release point and velocity in the **receiver's** local frame.
Release `x = 0.807` is opponent-local `x_B = 0.450`, i.e. the opponent's nominal
strike point.

| parameter | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| release `x` (m) | 0.807 | `U[0.79, 0.83]` | `U[0.77, 0.85]` | `U[0.75, 0.87]` |
| release `y` (m) | 0.00 | `U[−0.04, 0.04]` | `U[−0.12, 0.12]` | `U[−0.24, 0.24]` |
| release `z` (m) | 0.100 | `U[0.09, 0.12]` | `U[0.07, 0.15]` | `U[0.05, 0.18]` |
| speed (m/s) | 1.55 | `U[1.50, 1.62]` | `U[1.38, 1.74]` | `U[1.28, 1.86]` |
| elevation (deg) | 25 | `U[22, 28]` | `U[16, 34]` | `U[11, 42]` |
| azimuth (deg) | 0 | `U[−3, 3]` | `U[−9, 9]` | `U[−16, 16]` |

The widening is deliberately asymmetric across parameters, in proportion to how
much each one moves the strike point (§0). Speed gets the tightest relative range
because 0.5 m/s of it moves the landing point 290 mm; elevation gets the widest
because 35 degrees of it moves it 69 mm. A uniform "widen everything by 2x" per
stage would spend all its difficulty budget on the one axis the arm cannot absorb.

The S3 ranges are chosen so that the resulting strike point stays inside the
measured footprint: at the corners (speed 1.28 with elevation 11, speed 1.86 with
elevation 42) the post-bounce apex lands at roughly `x = 0.46` and `x = 0.26`
respectively — just inside `[0.25, 0.48]`. **S3 is therefore at the edge of what
is kinematically possible, by construction.** Do not widen further without
re-running `scripts/reach.py`.

### Inverse-success resampling, from S2 onward

Straight from 2408.03906, where balls are "selected each episode by first
sampling a category with a probability inversely proportional to the return rate."

Bin the launch space into `4 (speed) × 4 (elevation) × 5 (y) × 3 (z) = 240`
cells. Maintain a trailing success rate `s_c` per cell over the last 40 samples
of that cell. Sample:

```
P(c) ∝ clip( (1 − s_c + 0.05) / (1 − s̄ + 0.05), 0.2, 5.0 )
```

Capped at 5x uniform so that one pathological cell — most likely a kinematically
impossible corner — cannot swallow the whole batch. This is also the countermeasure
to hack 1 in §7: a policy that parks on the modal intercept gets the modal cell's
success rate up, which immediately down-weights it.

### Serves

A serve is an exchange with `τ0` starting from the ball at rest on the server's own
paddle. Recommend **deferring serves entirely to v2**: 2408.03906 needed a
separate LLC, a separate 3.4k-state dataset, and two extra training cycles for
serves, and the rally task is where all the value is. In v1, every episode starts
from the scripted feeder or from the opponent policy's shot.

---

## 6. Asymmetric actor-critic split

Both badminton papers use an asymmetric critic and both say why: 2505.22974 —
"the critic network was provided with additional information... to better learn
the value function. The actor network only received the robot states and swing
target data"; 2511.11218 — an 87-D actor observation against a 98-D privileged
critic observation carrying "preemptive knowledge (next two hit times, target
pose, remaining targets)."

### Actor observation (31 floats, plus history)

Everything here must be computable on hardware from cameras and servo telemetry.

| group | dims | notes |
|---|---|---|
| ball position | 3 | noisy, 1 step latent in S3 |
| ball velocity | 3 | finite-differenced from the noisy positions, not ground truth |
| predicted intercept point | 3 | from the **on-board estimator**, run on the noisy, delayed ball stream |
| predicted time to intercept | 1 | same estimator |
| joint positions | 5 | servo feedback, noisy |
| joint velocities | 5 | servo feedback, noisy |
| paddle position | 3 | FK from the noisy joint positions |
| paddle face normal | 3 | FK, same |
| phase one-hot `τ` | 4 | derived from ball height and contact history |
| time since exchange start | 1 | |

History stack: 4 frames of the full vector at 50 Hz (80 ms), plus 10 frames of
`(q, q̇, a)` (200 ms). 2511.11218 uses 5 and 20 frames; ours is halved because our
episode is ~45 steps to their several hundred.

**Critical:** the predicted intercept point must be computed from the *noisy,
delayed* ball stream with the *same* estimator that will run on the real robot.
Feeding the actor a ground-truth prediction is the single most common way to
build a policy that works in sim and fails on hardware — see §7 hack 10.

### Critic-only, privileged (+34 floats)

| group | dims | why the critic needs it |
|---|---|---|
| true ball position, velocity | 6 | de-noises the value target |
| true ball spin | 3 | zero in v1; reserve the slot so v2 is not a retrain from scratch |
| exact intercept point and time | 4 | from a full drag-integrated rollout, not the on-board estimator |
| exact own-half bounce point and time | 4 | the single most informative unobservable — value can be assigned before the bounce resolves |
| scheduled impact time `t*` | 1 | so `r_swing`'s firing time is predictable to the critic |
| target landing point `p_target` | 2 | |
| episode physics: `e_table`, `e_paddle`, friction, `Cd`, ball mass, paddle mass | 6 | the DR draws; without these, DR looks like unexplainable variance |
| servo params: `kp`, `kv`, backlash, dead-time steps | 4 | same argument |
| per-joint current `I_i` | 5 | the critic must be able to value proximity to C1 |
| trailing 1 s RMS current | 1 | so C1b's *window* cost is a function of something in the critic's input |
| feed cell index success rate `s_c` | 1 | the inverse-success sampler makes episode difficulty non-stationary; without this the critic cannot tell a hard feed from a bad action |
| time until next feed | 1 | 2511.11218's "next two hit times", reduced to one |
| opponent joint state + paddle pose | 8 | S3 self-play only; zero-filled otherwise |

**Explicitly withheld from the critic:** the *actual* landing point of the
in-flight shot. That is future information and using it turns the critic into an
oracle whose advantage estimates the actor cannot act on. The *analytically
predicted* landing point at the instant of contact is fine — it is a deterministic
function of the state at that instant — and is included implicitly via the true
ball velocity at `τ2`.

---

## 7. Reward hacking: what this policy will actually exploit

Ordered by how confident I am that it will happen. Every countermeasure is
already reflected in §1–§5; this section is the audit trail for *why* those
choices are there.

**1. Parking on the modal intercept point.** With a narrow feed the intercept
point is nearly fixed, and a policy that drives to `(0.356, 0, 0.095)` and waits
collects `r_track` for the whole flight and hits a good fraction of feeds by
standing still. *Countermeasures (three, because this one is close to certain):*
`r_track` is gated to `t̂_impact − t < 0.40 s` so camping earns nothing; the feed
widens per stage with an S3 `y` range of ±0.24 that no single station covers; and
inverse-success resampling actively down-weights whichever cell the policy has
learned, so parking is self-defeating by construction.

**2. Farming `r_track` without ever swinging.** 0.08/step × 45 steps = 3.6 is
risk-free; a swing attempt risks a fault termination worth 0. Early in S1 the
risk-free option is genuinely better. *Countermeasures:* the per-episode dense
total is hard-capped at 4.0 (so extending the episode cannot farm more); `r_track`
is annealed to zero weight by S3; and `r_swing` fires at the *scheduled* time
whether or not contact occurs, which is the real fix — it makes attempting the
swing strictly better than not attempting it, because a near-miss still pays.

**3. Dragging / carrying the ball.** A slow paddle that stays in contact can
*escort* the ball to the target landing point with near-zero placement error,
which farms `r_land` perfectly. This is a double-hit fault in regulation and a
sim artefact besides. *Countermeasure:* measure contact duration. A legal hit at
1.29 m/s closing speed on a 40 mm ball with a 1000 rad/s contact lasts ~2–3 ms.
Fault and terminate with reward 0 if the face-ball contact persists > 15 ms, or
if there is more than one face contact within 100 ms.

**4. Hitting with the mount, rim, or an arm link.** The cheapest way to get a ball
moving in roughly the right direction is to swat it with whatever geometry is
already nearby. *Countermeasure:* the paddle is modelled as **four separate
geoms** — `paddle_face` (the 110 mm rubber disc, +z side of the blade),
`paddle_back` (the −z side of the same disc), `paddle_rim` (the 2.4 mm edge
cylinder), and `paddle_neck` (the neck, pad and bosses that bolt to the fixed
jaw) — from the first contact-enabled episode. MuJoCo will not give per-face
contacts on a single disc geom, so the blade has to be built as a thin cylinder
plus two zero-thickness plate geoms, or as a single geom whose contacts are
classified post-hoc by the sign of `n̂_blade · (p_contact − p_blade_centre)`;
the latter is simpler and is what I would do. `r_swing`, `r_under`, `r_land` and `r_return` fire only on `paddle_face`
contact; contact with `paddle_back`, `paddle_rim`, `paddle_neck`, or any arm link is a fault,
reward 0, terminate, and counted in C4. Additionally require
`n̂_face · (−v̂_in) > 0.5` at contact, so a glancing edge-of-face graze does not
count as a face hit.

**5. Fake underhand — the pose without the motion.** Roll the wrist so
`n̂_face · ẑ = sin(30°)` while the paddle travels *downward and backward*, chopping
the ball while collecting the face-openness factor. *Countermeasures:* `r_under`
is a **product** of the lift factor and the face factor, not a sum, so a good
`α` cannot pay for a bad `λ`; `g_fwd` requires `v_paddle · x̂ > 0.30 m/s`; and
`g_out` requires the **ball** to leave with `v_z > 0.20 m/s`. The last one is the
important one — it is a statement about the physical consequence, and there is no
configuration that satisfies it while chopping.

**6. Bunting over the line.** Tap the ball so it barely crosses the midline and
lands at `x = 0.63`, collecting the flat `in_bounds` bonus of 10 for almost no
effort and no exposure to constraint costs. *Countermeasure:* `r_return` is
conditioned on the post-bounce apex lying in `x ∈ [0.777, 1.007]`, and a
`x_land = 0.63` shot apexes at `x_B ≈ 0.52`, outside the opponent's 0.48 m reach.
So the bunt collects 10 and forgoes 40. This is why `r_return`'s condition is
written on the *apex*, not on "lands past the net."

**7. Blasting deep.** Symmetrically: hit hard, land at `x = 1.20`, which is in
bounds and technically inside the opponent's reach. *Countermeasure:* the
Gaussian in `r_land` is centred at `x = 0.73`, not spread uniformly over the legal
region, so a deep shot loses the 25-point placement term; and its apex falls at
`x_B < 0.1`, failing `r_return`. Plus C1a/C1b make the hard swing expensive.

**8. Exploiting the contact solver.** A 2.7 g ball on a 1000 rad/s contact at a
0.5 ms timestep is exactly the regime where an integrator invents energy — the
README already documents the arm's-length version of this failure. A policy will
find the pose that maximises the injected energy, and the resulting stroke will be
physically meaningless. *Countermeasure:* audit every contact. Compute
`v_out^pred = (1+e)·v_paddle·n̂ + e·|v_in·n̂|` and if the measured outgoing normal
speed exceeds `1.15 × v_out^pred`, **void the exchange** — discard the transition
rather than penalising it, because this is a simulator bug, not a policy
behaviour, and penalising it teaches the policy to avoid a region of state space
for the wrong reason. Log the void rate; if it exceeds 1%, fix the contact model
before continuing to train.

**9. Suicide.** If the per-step regulariser sum is net negative and the task
reward is far away, the optimal policy is to terminate the episode as fast as
possible. Faults terminate, so "swat the table with the elbow" becomes a reward
*improvement*. *Countermeasures:* faults give reward **0**, never a negative
number — the punishment is the foregone 85; and a `+0.02`/step alive bonus makes
the regulariser sum net positive in the neutral case. This is why torque and
current are constraints in §3 rather than penalties: a Lagrangian cost does not
create an incentive to end the episode, a reward penalty does.

**10. Trusting a prediction it will not have.** If the actor's "predicted
intercept point" is computed from ground-truth ball state, the policy learns to be
a function of a quantity that does not exist on hardware, and the whole 178 ms
timing budget evaporates at deployment. This is not reward hacking in the usual
sense — it is the policy correctly exploiting an information leak we built. It is
also the failure mode most likely to survive all the way to the real arm
unnoticed, because it makes sim numbers *better*. *Countermeasure:* the estimator
that produces the actor's intercept prediction is the same code, with the same
noise and latency, as the one that will run on the robot; it is fed only the
actor's noisy observation stream; and its ground-truth counterpart goes to the
critic only. Verify by ablation: if the sim success rate is insensitive to
doubling the observation noise, there is a leak.

**11. Self-play collusion.** Two copies of one policy, rewarded for rally length,
will converge on the softest possible mutually-hittable exchange — a pat-a-cake
loop that maximises exchanges and is not a stroke. *Countermeasures:* 30% of S3
feeds stay scripted; `r_return` requires the arriving ball's apex speed
`≥ 0.7 m/s`, so a soft lob scores zero for the *hitter*; and — most importantly —
stage advancement and the final evaluation are measured on a **frozen held-out set
of 500 scripted feeds**, never on self-play statistics. A colluding pair looks
excellent in self-play and bad on the held-out set, which is exactly what makes
the held-out set worth maintaining.

**12. Farming `r_recover` after a miss.** After a missed ball, the exchange is
still running until timeout and `r_recover` pays 0.03/step for sitting in the
ready pose. *Countermeasure:* `r_recover` is gated on `τ2 ∨ τ3` — it only pays
*after the ball has left the paddle face*. A miss never enters `τ2`, so it pays
nothing.

**A procedural countermeasure that matters more than any of the above.** The
README already makes the point: "this is also how we will review training
rollouts later, which is the best defence against a policy quietly gaming the
reward." Render 20 rollouts at every stage boundary and watch them. Most of the
twelve items above were found by reasoning, but hacks in this class are usually
found by looking.

---

## 8. Open questions

Flagged honestly; roughly in order of how much they could invalidate the above.

1. **The reach figure ignores orientation, and the underhand requirement will
   shrink it — possibly a lot.** `sim/reach.py` sweeps joint space and keeps tip
   positions in a height band; it says so itself ("it ignores whether the paddle
   is pointing anywhere useful... treat it as a ceiling, not a promise"). The
   66.6% / `x = 0.480` numbers are therefore an orientation-free bound. The
   correct measurement for this design is the reachable footprint **conditioned
   on** `n̂_face · ẑ ∈ [sin 15°, sin 45°]` and `n̂_face · x̂ > 0`, with a paddle
   geometry attached. With 5 DoF and the paddle rigidly fixed to `wrist_roll`,
   the face normal has only two effective degrees of freedom, and constraining
   both of them may collapse the footprint substantially — particularly at large
   `|y|`, which is exactly where the S3 feed distribution goes. **This is the
   single highest-priority measurement to take before any training.** If the
   orientation-conditioned footprint does not cover `x_B ∈ [0.25, 0.48]` at
   `|y| ≤ 0.24`, the S3 feed ranges in §5 are wrong and must be narrowed.
2. **The paddle now exists in CAD, and it invalidates the reach number in a way
   that helps.** `cad/paddle.py` and `cad/out/paddle_props.json` give real
   numbers: 110 mm blade diameter (**`BLADE_R` = 55 mm**), 2.4 mm blade, 1.2 mm
   pips-out rubber, total **30.8 g**, and — the important one — the blade centre
   sits at **`x = +71.2 mm` from the mount**, bolted to the fixed jaw on
   `wrist_roll`. But `sim/reach.py` sweeps the **`gripperframe` site**, which is
   the mount, not the blade centre. So the measured `x = 0.480` reach limit is
   the *mount's* reach, and the paddle's actual contact surface extends up to
   ~71 mm further out — but only in whatever direction `wrist_roll` and
   `wrist_flex` happen to be pointing it, which the underhand cone constrains.
   The reach sweep must be redone with a `paddle_centre` site at
   `(0.0712, 0, ~0.002)` in the `wrist_roll` frame, and conditioned on the
   orientation cone of item 1. Until then, treat every `x_B ∈ [0.25, 0.48]`
   bound in this document as provisional: the true figure is somewhere in
   `[0.48, 0.55]` and the whole landing target in §2 shifts with it. Two further
   consequences: the 55 mm blade radius sets how much lateral position error a
   contact tolerates (generous — ±40 mm off-centre still hits the face), and the
   30.8 g at 71 mm is a 2.2 g·m first moment on `wrist_roll`, which feeds
   directly into the C1 current constraint. Still unmeasured: the rubber's actual
   restitution (`e ≈ 0.85` remains an assumption).
3. **`K_t` and the thermal model are guesses.** I derived
   `K_t ≈ 2.94 N·m / 2.5 A = 1.18 N·m/A` by dividing the model's `forcerange` by
   the stated stall current, which assumes the forcerange *is* the stall torque
   and that the torque-current relation is linear to stall. Neither is verified,
   and I have no winding thermal time constant at all — so the 1.0 s averaging
   window and the 0.9 A threshold in C1b are the weakest numbers in this
   document. Resolving this needs a bench measurement: hold a load, log current
   and case temperature, fit a first-order thermal model. Until then, set C1b
   conservatively and expect to relax it.
4. **No net is modelled, and a regulation net is infeasible.** §0(d): the
   trajectory clears the midline at 63–146 mm across the entire feasible envelope,
   against a regulation 152.5 mm. Options: (a) no net in v1, and rely on the
   `z ≥ 0.030` midline-crossing check in the success criterion — my recommendation
   for v1; (b) a geometrically scaled net at `1.257/2.74 × 152.5 ≈ 70 mm`, which
   the numbers say is cleared only above ~12 degrees of elevation and would make
   the underhand term load-bearing for legality rather than style; (c) 50 mm as a
   compromise. I lean (a) and then (c). I could not resolve whether a net is
   wanted at all, since the table is a kitchen island.
5. **Annealing schedule: adaptive or fixed?** I recommend success-gated
   (§4), which is *not* what 2511.11218 did. Their fixed schedule worked with 4096
   parallel environments; success-gated annealing has its own failure mode, which
   is stalling forever at a σ the policy cannot pass. A time-out that force-tightens
   σ after N episodes regardless may be needed, and I do not know a good N.
6. **`r_swing` fires at `t*`, the estimator's predicted impact time, which is
   itself a function of the policy's own noisy observation.** 2511.11218 has a
   *scheduled* impact time from the shuttle's known trajectory; ours is estimated.
   A policy could in principle influence `t*` (by influencing what the estimator
   sees) to make the reward fire at a convenient moment. I think the right fix is
   to compute `t*` for the reward from the **privileged** ground-truth rollout
   while the actor's observation keeps the noisy estimate — reward from truth,
   observation from what the robot will have. I am fairly confident in this but
   have not seen it done explicitly in any of the four papers.
7. **Compute.** There is no MJX port. A 0.5 ms timestep with 4096 parallel
   environments is not happening on this machine, and the S2 budget of ~26k
   episodes is meaningless without knowing the throughput. 2511.11218 reports
   12.4 h on one RTX 4090 for a much bigger problem; 2505.22974 used 4096 envs.
   The whole curriculum assumes a GPU port exists. If it does not, S2's annealing
   rate has to be re-derived from actual throughput, and the σ schedule in §4 is
   the first thing that breaks.
8. **Lateral placement has low authority** (§0(e)): ~7.8 mm of opponent strike-`y`
   per degree of outgoing azimuth, and azimuth costs range as `cos`. I set the
   `y`-target to 0 and take lateral variation from the feed. Whether a policy
   *should* be able to place laterally — which is what makes a rally a game rather
   than a metronome — is a v2 question I did not resolve.
9. **Does one shared policy actually survive deployment?** The frame argument in
   §5 is sound kinematically, but the two physical arms will have different
   backlash, different servo offsets, and only one of them will be next to the
   launcher during bring-up. Whether a single policy transfers to both, or whether
   a small per-arm calibration layer is needed, is an empirical question.

---

## 9. Decision: require a bounce on the receiving side

**Recommendation: yes, require it.** Retune the feeder so the ball bounces once on
the receiver's half before the strike, and make a second bounce on the receiver's
half before contact a terminating fault (§1, termination 3).

I came into this expecting the opposite — volley ping-pong looks like the easier
v1 — and the numbers changed my mind. Five reasons, in order of weight.

**1. The measured strike band and reach footprint were computed for a bounced
ball, and only a bounced ball puts the ball there.** `STRIKE_BAND = (0.03, 0.22)`
and the `x = 0.480` reach limit describe a low, slow ball near the table. §0
shows that a feed with a receiving-side bounce apexes at `x = 0.356, z = 0.095`
at 1.29 m/s — the middle of both. A ball that crosses without bouncing is still
descending from its far-half apex when it reaches `x = 0.48` and is higher and
faster there, and the brief's own launcher data confirms it (crossing the
footprint boundary at `z = 30–130 mm` and up to 1.78 m/s, i.e. already at the top
and fast edge of the band). Volleying means intercepting at the *margin* of the
measured workspace; bouncing means intercepting at its *centre*.

**2. The timing window is 178 ms and is widest at apex.** A bounced ball apexes
inside the strike volume, and at apex `dz/dt = 0`, so the vertical position error
caused by a timing error is second-order rather than first-order. That is a free
robustness win at exactly the place where 10–20 ms of dead time and a 50 Hz
control loop hurt most — eight control steps of usable window instead of the two
or three a descending volley gives.

**3. A bounce is an unambiguous event, so the success criterion and the landing
reward have a physical anchor.** With a required bounce, "in bounds" means "the
ball's first table contact was inside the lines" — observable, discrete,
unhackable. Without it there is no landing event at all, and the in-bounds term
has to be written as "crossed a virtual plane within a height window," which is a
proxy, and §7 is a long argument that proxies get hacked. This one would get
hacked immediately: a volley target has no upper bound, so the policy lobs.

**4. It makes the underhand requirement cheap instead of expensive.** A bounced
ball is rising or apexing when struck, so an upward-moving paddle has a *low*
closing speed against it — the paddle and ball are moving in compatible
directions. Volleying a *descending* ball underhand means paddle and ball moving
oppositely in `z`: higher closing speed, a smaller timing window, and a much
harder credit-assignment problem for the very term the user cares most about.
Requiring the bounce and requiring underhand are mutually reinforcing; volleying
and underhand fight each other.

**5. It needs only 0.25 m/s of paddle speed.** §0(b). Because the shot only has to
carry 0.28 m to its landing point instead of flying the full 0.5 m into the
opponent's reach zone unbounced, the servo-current problem the brief is worried
about substantially recedes.

### What it costs

**The landing box is small: the shot must land in `x_B ∈ [0.45, 0.60]`, a 150 mm
deep strip.** (World `x ∈ [0.657, 0.807]`.) That is a much tighter target than
"anywhere on the opponent's half," which is 480 mm deep. This is the real cost and
it is not trivial.

It is survivable because of the sensitivities in §0(a): holding the landing point
to ±80 mm requires holding the outgoing speed to ±0.3 m/s, which is ±0.16 m/s of
paddle speed — 64% relative tolerance on a 0.25 m/s swing. Elevation is almost
free (±17 degrees for the same ±80 mm). So the precision demand lands almost
entirely on one scalar, paddle speed along the normal, which is the easiest thing
for a position-controlled servo to be consistent about.

**Secondary costs:**

- The feeder must be re-specified (§5) and the existing 60-seed validation re-run.
  That validation currently certifies the *wrong* distribution.
- The sustainable rally speed drops to **1.30–1.80 m/s**, not the 1.5–2.5 m/s in
  the brief. The rally is slower and, honestly, less impressive to watch.
- Each exchange is longer (~0.55 s from the opponent's contact to ours, versus
  ~0.35 s volleying), so a four-exchange episode is ~2.2 s and the discount
  horizon has to cover it — hence `γ = 0.995` rather than 0.99.
- One more failure mode to attribute: "landed short, in my own half" and "landed
  long, off the end" are now distinct, and both look like "no return" in a
  summary statistic. Log the landing `x` distribution per stage, not just a
  success rate.

### What volleying would cost instead

For completeness, since this is the option I am rejecting. Volleying is cheaper
in exactly one way — no landing-box precision requirement — and more expensive in
four: the intercept moves to the edge of the measured workspace; the timing window
shrinks with no apex to exploit; the in-bounds objective becomes a hackable proxy
plane; and the whole thing is not table tennis, so v2's spin, net, and regulation
table would be a redesign rather than a parameter change.

The honest summary is that requiring the bounce moves the difficulty from a place
where we have no margin (workspace, timing, reward specification) to a place where
we have plenty (speed regulation on a 0.25 m/s swing). That trade is worth making.
