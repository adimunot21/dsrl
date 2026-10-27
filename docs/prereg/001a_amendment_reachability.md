# Amendment 001a to pre-registration 001 (2026-10-27)

Written AFTER building and inspecting the toy decoders (decoder properties only: no steerer has been run, no gain G has been seen) and BEFORE any steering experiment. 001 is unchanged; this file adds to it.

## Observations that motivate it
- "Outside support" in 001 was operationalised as goal distance d = 1.5 ring-radii from the best mode. For the widest concentration (mode std 0.4), 100k noises reach within 0.015 of the goal (mean min distance 0.069), so this config is NOT unreachable. At std 0.15 the minimum distance is 0.81.
- Decoded within-mode spread is wider than the data (decoded/data std ratio 1.59 at 0.05, 1.09 at 0.15, 1.03 at 0.4), so decoder blur makes "support" softer than the nominal grid says.
- Decoders for the same concentration and weakness differ only in goal across support levels.

## Changes
1. Every config additionally gets an *empirical reachability gap*: min over 100k noises of ||decoded action - goal|| averaged over 5 states, computed before steering. Reported for all 27 configs.
2. H1(ii) is evaluated on the configs whose empirical reachability gap exceeds 0.3 (stated now, before seeing any G), labelled "unreachable". Configs with d = 1.5 but gap <= 0.3 are reported separately as "tail-reachable", not pooled with either group. If fewer than 6 configs qualify as "unreachable", H1(ii) is reported as untestable at this grid rather than confirmed.
3. The 5 states used for the gap are drawn from a separate RNG stream from the evaluation states.
4. Probe cost (decoder calls) is reported in every table; C_256 costs 512,000 calls per 2,000 states, and C_ref is far more expensive. Any "cheap probe" claim must cite these numbers, and the reward-evaluation requirement from 001 still applies.

## Not changed
H2, H3, the single primary test (rho(H_probe, G) >= 0.7 and above rho(H_ent, G)), the kill rule (rho < 0.5), seeds, and N = 256.
