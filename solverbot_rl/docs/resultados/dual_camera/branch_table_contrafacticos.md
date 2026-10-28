| grupo | episodios | éxitos | rama inicial − | con cambio de rama | éxitos con cambio | cambio iniciado con detección válida | TCP z máx > −0,5 |
|---|---|---|---|---|---|---|---|
| runs/dual_camera/cf/bc_v1_var_l/oracle | 30 | 12 | 10 | 6 | 0 | 6/6 | 4 |
| runs/dual_camera/cf/bc_v1_var_l/perception | 30 | 20 | 11 | 6 | 0 | 5/6 | 5 |
| runs/dual_camera/cf/bc_v2w_reach/first_estimate | 20 | 20 | 19 | 19 | 19 | 19/19 | 19 |
| runs/dual_camera/cf/bc_v2w_reach/oracle | 20 | 20 | 19 | 19 | 19 | 19/19 | 19 |
| runs/dual_camera/cf/bc_v2w_reach/perception | 20 | 16 | 19 | 19 | 15 | 19/19 | 19 |
| runs/dual_camera/cf/bc_v2w_var_m/first_estimate | 20 | 18 | 8 | 6 | 4 | 5/6 | 4 |
| runs/dual_camera/cf/bc_v2w_var_m/hold_valid | 20 | 20 | 9 | 7 | 7 | 7/7 | 2 |
| runs/dual_camera/cf/bc_v2w_var_m/oracle | 20 | 17 | 8 | 5 | 4 | 5/5 | 3 |
| runs/dual_camera/cf/bc_v2w_var_m/perception | 20 | 18 | 9 | 6 | 4 | 5/6 | 3 |
| runs/dual_camera/cf/bc_v2w_var_s/oracle | 30 | 25 | 20 | 14 | 9 | 14/14 | 15 |
| runs/dual_camera/cf/bc_v2w_var_s/perception | 30 | 22 | 19 | 16 | 8 | 16/16 | 15 |
| runs/dual_camera/cf/bc_v2w_var_s_905k/first_estimate | 60 | 52 | 33 | 21 | 14 | 20/21 | 16 |
| runs/dual_camera/cf/bc_v2w_var_s_905k/perception | 60 | 49 | 34 | 21 | 10 | 21/21 | 20 |
| TOTAL | 380 | 309 | 218 | 165 | 113 | 161/165 | 144 |

| episodio | nivel | rama inicial | cambios (deja saturación → cruza 0 → otra rama) | válida al iniciar | 1.ª pérdida | dmin (paso) | dfinal | TCP z máx |
|---|---|---|---|---|---|---|---|---|
| bc_v1_var_l/900300 | reach_var_l | - | — | — | 18 | 31.2 (40) | 69.0 | -0.70 |
| bc_v1_var_l/900302 | reach_var_l | + | +1->-1 15→20→23 | True (0) | 19 | 34.2 (12) | 95.0 | -0.68 |
| bc_v1_var_l/900304 | reach_var_l | - | — | — | 25 | 40.1 (43) | 54.7 | -0.74 |
| bc_v1_var_l/900305 | reach_var_l | + | +1->-1 15→21→24 | True (0) | 20 | 41.6 (15) | 100.2 | -0.65 |
| bc_v1_var_l/900307 | reach_var_l | - | — | — | 9 | 44.5 (44) | 93.3 | -0.68 |
| bc_v1_var_l/900308 | reach_var_l | + | — | — | 14 | 30.7 (45) | 39.4 | -0.68 |
| bc_v1_var_l/900311 | reach_var_l | + | +1->-1 1→3→4; -1->+1 6→9→10; +1->-1 20→23→25 | True (0), True (0), False (1) | 20 | 43.6 (10) | 95.3 | -0.71 |
| bc_v1_var_l/900312 | reach_var_l | - | — | — | 18 | 22.1 (29) | 31.8 | -0.57 |
| bc_v1_var_l/900313 | reach_var_l | + | +1->-1 17→28→29 | True (0) | 25 | 34.9 (20) | 43.0 | -0.62 |
| bc_v1_var_l/900314 | reach_var_l | + | — | — | 11 | 36.9 (60) | 49.5 | -0.31 |
| bc_v1_var_l/900315 | reach_var_l | - | — | — | 9 | 58.6 (41) | 82.6 | -0.68 |
| bc_v1_var_l/900316 | reach_var_l | + | — | — | 14 | 31.7 (45) | 108.4 | -0.44 |
| bc_v1_var_l/900318 | reach_var_l | - | — | — | 17 | 48.2 (41) | 88.6 | -0.76 |
| bc_v1_var_l/900319 | reach_var_l | - | — | — | 21 | 37.3 (12) | 51.7 | -0.74 |
| bc_v1_var_l/900320 | reach_var_l | + | +1->-1 17→21→24 | True (0) | 20 | 45.7 (14) | 104.8 | -0.64 |
| bc_v1_var_l/900321 | reach_var_l | - | +1->-1 18→24→27 | True (0) | 22 | 57.0 (14) | 100.0 | -0.67 |
| bc_v1_var_l/900324 | reach_var_l | + | — | — | 10 | 45.9 (46) | 64.3 | -0.61 |
| bc_v1_var_l/900327 | reach_var_l | - | — | — | None | 33.4 (39) | 70.3 | -0.79 |
| bc_v1_var_l/900302 | reach_var_l | + | +1->-1 14→17→20 | True (0) | 19 | 36.7 (11) | 96.6 | -0.70 |
| bc_v1_var_l/900311 | reach_var_l | + | +1->-1 1→4→4 | True (0) | 18 | 39.5 (10) | 94.0 | -0.73 |
| bc_v1_var_l/900312 | reach_var_l | - | -1->+1 6→12→14; +1->-1 17→27→28 | True (0), False (1) | 17 | 26.6 (27) | 34.9 | -0.57 |
| bc_v1_var_l/900313 | reach_var_l | + | +1->-1 22→37→42 | False (1) | 22 | 44.9 (33) | 108.1 | -0.40 |
| bc_v1_var_l/900315 | reach_var_l | - | — | — | 10 | 49.1 (41) | 89.1 | -0.68 |
| bc_v1_var_l/900316 | reach_var_l | + | — | — | 13 | 31.8 (44) | 91.9 | -0.51 |
| bc_v1_var_l/900318 | reach_var_l | - | — | — | 17 | 38.9 (41) | 96.7 | -0.76 |
| bc_v1_var_l/900321 | reach_var_l | - | +1->-1 12→21→23 | True (0) | 14 | 62.5 (15) | 76.5 | -0.73 |
| bc_v1_var_l/900324 | reach_var_l | + | — | — | 10 | 47.7 (48) | 74.4 | -0.56 |
| bc_v1_var_l/900328 | reach_var_l | + | +1->-1 43→51→58 | True (0) | 15 | 32.5 (45) | 234.5 | +0.06 |
| bc_v2w_reach/901003 | reach | - | -1->+1 7→11→13 | True (0) | 13 | 82.7 (11) | 104.9 | +0.09 |
| bc_v2w_reach/901010 | reach | - | -1->+1 8→12→14 | True (0) | 14 | 80.5 (12) | 153.4 | +0.11 |
| bc_v2w_reach/901012 | reach | - | -1->+1 7→11→13 | True (0) | 13 | 82.8 (11) | 118.5 | +0.09 |
| bc_v2w_reach/901018 | reach | - | -1->+1 7→12→13 | True (0) | 13 | 82.0 (11) | 110.2 | +0.09 |
| bc_v2w_var_m/902010 | reach_var_m | - | -1->+1 16→18→19 | True (0) | None | 32.7 (45) | 34.9 | -0.79 |
| bc_v2w_var_m/902017 | reach_var_m | + | +1->-1 20→22→24 | False (1) | 12 | 46.9 (16) | 231.9 | -0.16 |
| bc_v2w_var_m/902008 | reach_var_m | + | — | — | 14 | 30.3 (65) | 33.7 | -0.67 |
| bc_v2w_var_m/902010 | reach_var_m | - | — | — | None | 33.8 (17) | 36.8 | -0.80 |
| bc_v2w_var_m/902017 | reach_var_m | + | +1->-1 19→21→22 | True (0) | 11 | 47.4 (16) | 120.1 | -0.32 |
| bc_v2w_var_m/902009 | reach_var_m | - | -1->+1 9→12→13 | True (0) | 13 | 87.9 (11) | 141.5 | +0.07 |
| bc_v2w_var_m/902017 | reach_var_m | + | +1->-1 20→25→32 | False (1) | 11 | 66.1 (16) | 252.9 | -0.05 |
| bc_v2w_var_s/901113 | reach_var_s | - | -1->+1 12→15→16 | True (0) | 16 | 55.2 (14) | 93.6 | +0.06 |
| bc_v2w_var_s/901114 | reach_var_s | - | -1->+1 10→13→14 | True (0) | 14 | 70.1 (12) | 113.4 | +0.10 |
| bc_v2w_var_s/901120 | reach_var_s | - | -1->+1 14→16→17; +1->-1 35→38→38 | True (0), False (20) | 16 | 54.7 (15) | 159.0 | -0.12 |
| bc_v2w_var_s/901128 | reach_var_s | - | -1->+1 11→14→15; +1->-1 39→41→42 | True (0), False (25) | 15 | 61.4 (14) | 274.9 | -0.04 |
| bc_v2w_var_s/901129 | reach_var_s | - | -1->+1 9→11→12 | True (0) | 12 | 85.6 (10) | 103.6 | +0.02 |
| bc_v2w_var_s/901101 | reach_var_s | - | -1->+1 13→17→18; +1->-1 27→33→34; -1->+1 47→48→48 | True (0), False (9), True (0) | 19 | 23.7 (14) | 30.8 | -0.36 |
| bc_v2w_var_s/901111 | reach_var_s | - | -1->+1 12→16→17 | True (0) | 15 | 77.3 (14) | 133.8 | +0.13 |
| bc_v2w_var_s/901113 | reach_var_s | - | -1->+1 10→13→14 | True (0) | 16 | 67.6 (13) | 152.6 | +0.14 |
| bc_v2w_var_s/901114 | reach_var_s | - | -1->+1 10→14→15 | True (0) | 13 | 79.7 (12) | 118.1 | +0.20 |
| bc_v2w_var_s/901118 | reach_var_s | - | -1->+1 12→17→18; +1->-1 38→40→41 | True (0), False (24) | 15 | 62.1 (14) | 232.1 | -0.03 |
| bc_v2w_var_s/901120 | reach_var_s | - | -1->+1 14→17→18; +1->-1 39→41→42 | True (0), False (24) | 16 | 60.5 (14) | 228.2 | -0.02 |
| bc_v2w_var_s/901126 | reach_var_s | - | -1->+1 8→12→13 | True (0) | 12 | 89.2 (10) | 133.0 | +0.13 |
| bc_v2w_var_s/901129 | reach_var_s | - | -1->+1 12→14→15; +1->-1 43→45→46 | True (0), False (31) | 13 | 69.3 (11) | 272.6 | +0.04 |
| bc_v2w_var_s_905k/905007 | reach_var_s | - | -1->+1 13→16→17; +1->-1 34→36→37 | True (0), False (18) | 17 | 58.1 (14) | 158.6 | -0.14 |
| bc_v2w_var_s_905k/905014 | reach_var_s | - | -1->+1 10→14→14 | True (0) | 14 | 99.4 (12) | 160.2 | +0.09 |
| bc_v2w_var_s_905k/905030 | reach_var_s | - | -1->+1 11→14→17; +1->-1 34→36→37 | True (0), False (19) | 16 | 51.3 (13) | 205.3 | -0.16 |
| bc_v2w_var_s_905k/905035 | reach_var_s | - | -1->+1 11→14→15 | True (0) | 17 | 35.6 (14) | 206.5 | -0.01 |
| bc_v2w_var_s_905k/905046 | reach_var_s | - | -1->+1 17→23→27 | True (0) | None | 14.9 (17) | 33.3 | -0.72 |
| bc_v2w_var_s_905k/905049 | reach_var_s | - | -1->+1 8→11→12; +1->-1 32→34→34 | True (0), False (19) | 14 | 54.7 (10) | 252.2 | -0.28 |
| bc_v2w_var_s_905k/905052 | reach_var_s | - | -1->+1 12→15→16; +1->-1 31→33→34; -1->+1 54→55→56 | True (0), False (18), True (0) | 14 | 43.0 (60) | 44.7 | -0.21 |
| bc_v2w_var_s_905k/905054 | reach_var_s | - | — | — | None | 12.7 (16) | 31.1 | -0.70 |
| bc_v2w_var_s_905k/905002 | reach_var_s | - | -1->+1 11→13→15 | True (0) | 16 | 57.6 (13) | 157.9 | +0.15 |
| bc_v2w_var_s_905k/905017 | reach_var_s | - | -1->+1 12→15→17; +1->-1 39→41→42 | True (0), False (26) | 14 | 74.9 (12) | 267.1 | -0.02 |
| bc_v2w_var_s_905k/905020 | reach_var_s | - | -1->+1 9→13→15 | True (0) | 14 | 58.7 (11) | 125.7 | +0.15 |
| bc_v2w_var_s_905k/905023 | reach_var_s | - | -1->+1 7→11→13 | True (0) | 12 | 86.1 (10) | 112.1 | +0.11 |
| bc_v2w_var_s_905k/905026 | reach_var_s | - | -1->+1 10→13→14 | True (0) | 15 | 61.4 (13) | 151.4 | +0.15 |
| bc_v2w_var_s_905k/905027 | reach_var_s | - | -1->+1 42→43→43 | True (0) | 14 | 30.7 (46) | 46.1 | -0.47 |
| bc_v2w_var_s_905k/905030 | reach_var_s | - | -1->+1 10→16→17 | True (0) | 14 | 60.3 (12) | 157.4 | +0.12 |
| bc_v2w_var_s_905k/905035 | reach_var_s | - | -1->+1 10→13→14 | True (0) | 16 | 41.2 (13) | 105.8 | +0.07 |
| bc_v2w_var_s_905k/905048 | reach_var_s | - | -1->+1 7→10→11 | True (0) | 11 | 74.7 (9) | 90.5 | +0.07 |
| bc_v2w_var_s_905k/905049 | reach_var_s | - | -1->+1 5→10→13; +1->-1 18→26→27 | True (0), False (6) | 13 | 32.5 (149) | 32.5 | -0.45 |
| bc_v2w_var_s_905k/905052 | reach_var_s | - | -1->+1 7→12→13 | True (0) | 12 | 92.3 (10) | 123.2 | +0.07 |
