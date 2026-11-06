# AGNews v7 stress and probe aggregates

Descriptive development diagnostics only; no confirmation scoring or optimizer updates in probes. FLOPs are observed matmul lower bounds. Gradient cosine is a local comparison for these objectives and parameters; it does not establish causal uniqueness.

## Stress validation EM

| Source arm / scope | Base writes | Step | Old mean (SD) | New mean (SD) | All mean (SD) |
|---|---|---:|---:|---:|---:|---:|
| split_receiver / receiver | True | 0 | 0.8385 (0.1130) | 0.7331 (0.0938) | 0.7858 (0.0285) |
| split_receiver / receiver | True | 4 | 0.7565 (0.1122) | 0.8190 (0.0215) | 0.7878 (0.0499) |
| split_receiver / receiver | True | 16 | 0.7656 (0.0898) | 0.8255 (0.0185) | 0.7956 (0.0477) |
| split_receiver / receiver | True | 64 | 0.7096 (0.0864) | 0.8164 (0.0244) | 0.7630 (0.0344) |
| split_receiver / receiver | True | 128 | 0.6380 (0.1115) | 0.8216 (0.0458) | 0.7298 (0.0358) |
| split_receiver / receiver | False | 0 | 0.8385 (0.1130) | 0.7331 (0.0938) | 0.7858 (0.0285) |
| split_receiver / receiver | False | 4 | 0.8086 (0.1218) | 0.7721 (0.0982) | 0.7904 (0.0484) |
| split_receiver / receiver | False | 16 | 0.8229 (0.0767) | 0.7839 (0.0970) | 0.8034 (0.0443) |
| split_receiver / receiver | False | 64 | 0.7943 (0.0627) | 0.7904 (0.0525) | 0.7923 (0.0100) |
| split_receiver / receiver | False | 128 | 0.6771 (0.1084) | 0.8411 (0.0185) | 0.7591 (0.0451) |
| split_shared / shared | True | 0 | 0.8867 (0.0391) | 0.7331 (0.0865) | 0.8099 (0.0450) |
| split_shared / shared | True | 4 | 0.8451 (0.0672) | 0.7878 (0.0539) | 0.8164 (0.0470) |
| split_shared / shared | True | 16 | 0.7865 (0.0814) | 0.7812 (0.0887) | 0.7839 (0.0777) |
| split_shared / shared | True | 64 | 0.6732 (0.2822) | 0.7878 (0.0598) | 0.7305 (0.1309) |
| split_shared / shared | True | 128 | 0.6927 (0.1868) | 0.8125 (0.0480) | 0.7526 (0.0844) |
| split_shared / shared | False | 0 | 0.8867 (0.0391) | 0.7331 (0.0865) | 0.8099 (0.0450) |
| split_shared / shared | False | 4 | 0.8620 (0.0686) | 0.7604 (0.0607) | 0.8112 (0.0274) |
| split_shared / shared | False | 16 | 0.8503 (0.0649) | 0.7865 (0.0567) | 0.8184 (0.0269) |
| split_shared / shared | False | 64 | 0.7109 (0.2646) | 0.8073 (0.0239) | 0.7591 (0.1242) |
| split_shared / shared | False | 128 | 0.7109 (0.2203) | 0.8203 (0.0156) | 0.7656 (0.1037) |

## Base-write contrasts

Each entry is written minus skipped within the same seed and arm.

| Arm | Seed | Step | Old EM delta | New EM delta |
|---|---:|---:|---:|---:|
| split_receiver | 17 | 0 | 0.0000 | 0.0000 |
| split_receiver | 17 | 4 | -0.0273 | 0.0078 |
| split_receiver | 17 | 16 | -0.0117 | 0.0000 |
| split_receiver | 17 | 64 | -0.1055 | -0.0039 |
| split_receiver | 17 | 128 | -0.0547 | 0.0000 |
| split_receiver | 29 | 0 | 0.0000 | 0.0000 |
| split_receiver | 29 | 4 | -0.0938 | 0.1367 |
| split_receiver | 29 | 16 | -0.1914 | 0.1602 |
| split_receiver | 29 | 64 | -0.1602 | 0.1055 |
| split_receiver | 29 | 128 | -0.0469 | -0.0508 |
| split_receiver | 43 | 0 | 0.0000 | 0.0000 |
| split_receiver | 43 | 4 | -0.0352 | -0.0039 |
| split_receiver | 43 | 16 | 0.0312 | -0.0352 |
| split_receiver | 43 | 64 | 0.0117 | -0.0234 |
| split_receiver | 43 | 128 | -0.0156 | -0.0078 |
| split_shared | 17 | 0 | 0.0000 | 0.0000 |
| split_shared | 17 | 4 | -0.0117 | 0.0664 |
| split_shared | 17 | 16 | -0.0352 | 0.0352 |
| split_shared | 17 | 64 | -0.0352 | 0.0117 |
| split_shared | 17 | 128 | -0.0195 | 0.0117 |
| split_shared | 29 | 0 | 0.0000 | 0.0000 |
| split_shared | 29 | 4 | -0.0391 | 0.0156 |
| split_shared | 29 | 16 | -0.1445 | -0.0508 |
| split_shared | 29 | 64 | -0.0195 | -0.0625 |
| split_shared | 29 | 128 | -0.0586 | -0.0469 |
| split_shared | 43 | 0 | 0.0000 | 0.0000 |
| split_shared | 43 | 4 | 0.0000 | 0.0000 |
| split_shared | 43 | 16 | -0.0117 | 0.0000 |
| split_shared | 43 | 64 | -0.0586 | -0.0078 |
| split_shared | 43 | 128 | 0.0234 | 0.0117 |

## Fast-output lesion versus offline baseline

| Source arm / scope | Base writes | Group | Fast-masked EM | Step-0 baseline EM | Difference |
|---|---|---|---:|---:|---:|
| split_receiver / receiver | True | old | 0.8008 | 0.8385 | -0.0378 |
| split_receiver / receiver | True | new | 0.8034 | 0.7331 | 0.0703 |
| split_receiver / receiver | True | all | 0.8021 | 0.7858 | 0.0163 |
| split_receiver / receiver | False | old | 0.8385 | 0.8385 | 0.0000 |
| split_receiver / receiver | False | new | 0.7331 | 0.7331 | 0.0000 |
| split_receiver / receiver | False | all | 0.7858 | 0.7858 | 0.0000 |
| split_shared / shared | True | old | 0.8503 | 0.8867 | -0.0365 |
| split_shared / shared | True | new | 0.8021 | 0.7331 | 0.0690 |
| split_shared / shared | True | all | 0.8262 | 0.8099 | 0.0163 |
| split_shared / shared | False | old | 0.8867 | 0.8867 | 0.0000 |
| split_shared / shared | False | new | 0.7331 | 0.7331 | 0.0000 |
| split_shared / shared | False | all | 0.8099 | 0.8099 | 0.0000 |

## Gradient groups

| Arm / scope | Step | Group | Cosine mean (SD) | Old norm | New norm |
|---|---:|---|---:|---:|---:|
| current_receiver / receiver | 64 | receiver | 0.0901 (0.3828) | 13.1205 (12.5340) | 2.9070 (2.5837) |
| current_receiver / receiver | 64 | base0 | 0.1338 (0.3074) | 19.0239 (6.3692) | 3.4139 (3.1626) |
| current_receiver / receiver | 64 | base1 | -0.0556 (0.4320) | 6.6635 (3.5085) | 1.9446 (2.3608) |
| current_receiver / receiver | 256 | receiver | 0.0028 (0.2774) | 12.5291 (13.5694) | 0.3160 (0.4497) |
| current_receiver / receiver | 256 | base0 | 0.1340 (0.2272) | 16.6255 (5.4780) | 0.4708 (0.6531) |
| current_receiver / receiver | 256 | base1 | -0.1515 (0.5436) | 7.8464 (4.5998) | 0.2835 (0.4341) |
| current_receiver / receiver | 1024 | receiver | -0.1925 (0.2824) | 11.0992 (10.9517) | 0.3359 (0.3065) |
| current_receiver / receiver | 1024 | base0 | 0.2074 (0.3245) | 16.3234 (6.2204) | 0.7195 (0.6219) |
| current_receiver / receiver | 1024 | base1 | 0.2320 (0.4961) | 7.3342 (4.5218) | 0.3264 (0.3681) |
| current_receiver / receiver | 0 | receiver | 0.2454 (0.5961) | 1.1905 (0.3192) | 0.9030 (0.8402) |
| current_receiver / receiver | 0 | base0 | 0.4476 (0.4377) | 15.9069 (5.3349) | 10.6570 (9.8247) |
| current_receiver / receiver | 0 | base1 | 0.3949 (0.4179) | 7.4558 (2.5386) | 4.9130 (4.4976) |
| current_shared / shared | 64 | receiver | -0.3280 (1.0298) | 0.0176 (0.0165) | 0.0016 (0.0021) |
| current_shared / shared | 64 | base0 | -0.0148 (0.6514) | 14.0625 (8.8476) | 2.7632 (2.9637) |
| current_shared / shared | 64 | base1 | -0.6636 (0.2387) | 11.7537 (9.1058) | 1.7348 (2.3266) |
| current_shared / shared | 256 | receiver | 0.8319 (0.1118) | 0.0751 (0.1232) | 0.0005 (0.0006) |
| current_shared / shared | 256 | base0 | -0.1894 (0.7185) | 6.4471 (7.2050) | 0.1623 (0.2282) |
| current_shared / shared | 256 | base1 | -0.2231 (0.5694) | 16.1923 (1.3405) | 0.4060 (0.6006) |
| current_shared / shared | 1024 | receiver | 0.8264 (0.0402) | 0.0037 (0.0025) | 0.0003 (0.0004) |
| current_shared / shared | 1024 | base0 | 0.3508 (0.6380) | 9.8222 (12.6688) | 0.1613 (0.2633) |
| current_shared / shared | 1024 | base1 | 0.0219 (0.5025) | 8.3090 (7.5204) | 0.2904 (0.4585) |
| current_shared / shared | 0 | receiver | 0.2454 (0.5961) | 1.1905 (0.3192) | 0.9030 (0.8402) |
| current_shared / shared | 0 | base0 | 0.4476 (0.4377) | 15.9069 (5.3349) | 10.6570 (9.8247) |
| current_shared / shared | 0 | base1 | 0.3949 (0.4179) | 7.4558 (2.5386) | 4.9130 (4.4976) |
| split_receiver / receiver | 64 | receiver | -0.1763 (0.7234) | 1.8253 (2.1385) | 9.7782 (7.9710) |
| split_receiver / receiver | 64 | base0 | -0.1289 (0.5530) | 2.1234 (2.3068) | 10.2684 (6.5846) |
| split_receiver / receiver | 64 | base1 | -0.0587 (0.4826) | 1.1459 (1.0487) | 4.7392 (3.0987) |
| split_receiver / receiver | 256 | receiver | 0.5298 (0.2764) | 0.9169 (1.1515) | 18.4885 (7.7280) |
| split_receiver / receiver | 256 | base0 | 0.4471 (0.3625) | 1.3783 (1.4845) | 21.6421 (6.8413) |
| split_receiver / receiver | 256 | base1 | 0.3628 (0.6874) | 0.6902 (0.6205) | 5.0145 (2.0511) |
| split_receiver / receiver | 1024 | receiver | -0.0251 (0.6731) | 0.3153 (0.4486) | 10.2324 (4.6762) |
| split_receiver / receiver | 1024 | base0 | -0.0906 (0.4726) | 0.6823 (0.8333) | 18.6424 (9.6948) |
| split_receiver / receiver | 1024 | base1 | -0.3993 (0.3311) | 0.2991 (0.2882) | 2.4495 (0.5710) |
| split_receiver / receiver | 0 | receiver | 0.2454 (0.5961) | 1.1905 (0.3192) | 0.9030 (0.8402) |
| split_receiver / receiver | 0 | base0 | 0.4476 (0.4377) | 15.9069 (5.3349) | 10.6570 (9.8247) |
| split_receiver / receiver | 0 | base1 | 0.3949 (0.4179) | 7.4558 (2.5386) | 4.9130 (4.4976) |
| split_shared / shared | 64 | receiver | -0.2290 (0.9584) | 2.9293 (5.0669) | 2.2321 (3.8461) |
| split_shared / shared | 64 | base0 | -0.2054 (0.3637) | 3.9852 (5.2133) | 14.4006 (5.7109) |
| split_shared / shared | 64 | base1 | -0.0370 (0.4829) | 1.1282 (1.2726) | 6.1139 (4.4525) |
| split_shared / shared | 256 | receiver | -0.7448 (0.1906) | 0.0165 (0.0244) | 1.6781 (2.9039) |
| split_shared / shared | 256 | base0 | -0.4784 (0.1911) | 0.2063 (0.2527) | 9.6058 (9.1835) |
| split_shared / shared | 256 | base1 | 0.0664 (0.3911) | 0.3104 (0.4877) | 3.0671 (5.2363) |
| split_shared / shared | 1024 | receiver | -0.1387 (0.9288) | 0.0069 (0.0088) | 0.2922 (0.5036) |
| split_shared / shared | 1024 | base0 | -0.1783 (0.6616) | 0.0767 (0.1129) | 4.6208 (6.2512) |
| split_shared / shared | 1024 | base1 | -0.2387 (0.2802) | 0.1105 (0.1000) | 0.9986 (1.7262) |
| split_shared / shared | 0 | receiver | 0.2454 (0.5961) | 1.1905 (0.3192) | 0.9030 (0.8402) |
| split_shared / shared | 0 | base0 | 0.4476 (0.4377) | 15.9069 (5.3349) | 10.6570 (9.8247) |
| split_shared / shared | 0 | base1 | 0.3949 (0.4179) | 7.4558 (2.5386) | 4.9130 (4.4976) |
| split_narrow_receiver / receiver | 64 | receiver | -0.3609 (0.6242) | 7.7849 (9.4453) | 14.5609 (13.9745) |
| split_narrow_receiver / receiver | 64 | base0 | -0.2384 (0.6199) | 7.6320 (8.4105) | 13.8531 (13.3572) |
| split_narrow_receiver / receiver | 64 | base1 | 0.1299 (0.5458) | 2.5455 (2.0800) | 4.5110 (4.6520) |
| split_narrow_receiver / receiver | 256 | receiver | -0.5997 (0.0643) | 0.1528 (0.1137) | 5.2976 (4.0071) |
| split_narrow_receiver / receiver | 256 | base0 | -0.4351 (0.1160) | 0.2566 (0.1946) | 5.9939 (4.7024) |
| split_narrow_receiver / receiver | 256 | base1 | -0.3704 (0.2236) | 0.1518 (0.1450) | 1.4084 (1.1314) |
| split_narrow_receiver / receiver | 1024 | receiver | -0.3824 (0.6145) | 1.2447 (0.5902) | 1.0173 (0.9478) |
| split_narrow_receiver / receiver | 1024 | base0 | -0.3375 (0.6528) | 2.1418 (0.4647) | 1.9217 (2.0597) |
| split_narrow_receiver / receiver | 1024 | base1 | 0.1252 (0.3203) | 0.4994 (0.4766) | 0.4619 (0.3555) |
| split_narrow_receiver / receiver | 0 | receiver | 0.2454 (0.5961) | 1.1905 (0.3192) | 0.9030 (0.8402) |
| split_narrow_receiver / receiver | 0 | base0 | 0.4476 (0.4377) | 15.9069 (5.3349) | 10.6570 (9.8247) |
| split_narrow_receiver / receiver | 0 | base1 | 0.3949 (0.4179) | 7.4558 (2.5386) | 4.9130 (4.4976) |
| split_narrow_shared / shared | 64 | receiver | -0.5604 (0.6508) | 0.2460 (0.4073) | 1.7959 (2.2358) |
| split_narrow_shared / shared | 64 | base0 | -0.4600 (0.2226) | 4.1513 (5.3660) | 5.8619 (4.2458) |
| split_narrow_shared / shared | 64 | base1 | -0.0927 (0.4564) | 3.4110 (5.4367) | 5.8226 (5.5522) |
| split_narrow_shared / shared | 256 | receiver | -0.2980 (0.9814) | 0.0751 (0.1172) | 0.9669 (1.6016) |
| split_narrow_shared / shared | 256 | base0 | 0.0869 (0.0605) | 0.8523 (0.8543) | 6.5128 (9.4163) |
| split_narrow_shared / shared | 256 | base1 | 0.3086 (0.2020) | 0.3881 (0.5045) | 3.3198 (3.1198) |
| split_narrow_shared / shared | 1024 | receiver | -0.1546 (1.0025) | 0.0548 (0.0812) | 0.3362 (0.5445) |
| split_narrow_shared / shared | 1024 | base0 | 0.3524 (0.4928) | 2.1222 (2.7211) | 1.3654 (2.2084) |
| split_narrow_shared / shared | 1024 | base1 | -0.2038 (0.2629) | 0.0695 (0.1196) | 0.1466 (0.2534) |
| split_narrow_shared / shared | 0 | receiver | 0.2454 (0.5961) | 1.1905 (0.3192) | 0.9030 (0.8402) |
| split_narrow_shared / shared | 0 | base0 | 0.4476 (0.4377) | 15.9069 (5.3349) | 10.6570 (9.8247) |
| split_narrow_shared / shared | 0 | base1 | 0.3949 (0.4179) | 7.4558 (2.5386) | 4.9130 (4.4976) |
| gold_receiver / receiver | 64 | receiver | 0.0890 (0.2287) | 1.2001 (1.2071) | 11.7163 (0.1496) |
| gold_receiver / receiver | 64 | base0 | 0.0991 (0.3752) | 1.2679 (1.2706) | 11.3514 (0.9744) |
| gold_receiver / receiver | 64 | base1 | 0.3567 (0.1497) | 0.4253 (0.2764) | 3.7148 (0.9362) |
| gold_receiver / receiver | 256 | receiver | 0.2888 (0.4837) | 1.3201 (0.8278) | 12.7076 (6.2336) |
| gold_receiver / receiver | 256 | base0 | 0.3534 (0.2236) | 1.6254 (1.2647) | 16.0220 (7.8429) |
| gold_receiver / receiver | 256 | base1 | -0.1101 (0.6068) | 0.6554 (0.4827) | 3.9226 (1.6496) |
| gold_receiver / receiver | 1024 | receiver | -0.0020 (0.7260) | 3.6244 (2.9097) | 12.1407 (7.5793) |
| gold_receiver / receiver | 1024 | base0 | 0.0850 (0.7787) | 6.6146 (5.4598) | 23.9966 (13.0608) |
| gold_receiver / receiver | 1024 | base1 | -0.0472 (0.3093) | 0.7469 (0.2862) | 2.1718 (1.4725) |
| gold_receiver / receiver | 0 | receiver | 0.2454 (0.5961) | 1.1905 (0.3192) | 0.9030 (0.8402) |
| gold_receiver / receiver | 0 | base0 | 0.4476 (0.4377) | 15.9069 (5.3349) | 10.6570 (9.8247) |
| gold_receiver / receiver | 0 | base1 | 0.3949 (0.4179) | 7.4558 (2.5386) | 4.9130 (4.4976) |
| gold_shared / shared | 64 | receiver | 0.3655 (0.9864) | 0.4410 (0.7346) | 3.1607 (5.4708) |
| gold_shared / shared | 64 | base0 | -0.7196 (0.2108) | 3.2593 (4.0187) | 7.9720 (6.1061) |
| gold_shared / shared | 64 | base1 | 0.0283 (0.8630) | 2.4882 (2.1194) | 3.6404 (2.8572) |
| gold_shared / shared | 256 | receiver | 0.3080 (1.0172) | 0.0161 (0.0034) | 2.7302 (4.7234) |
| gold_shared / shared | 256 | base0 | -0.0669 (0.5820) | 0.3904 (0.5467) | 7.8304 (6.7764) |
| gold_shared / shared | 256 | base1 | -0.1999 (0.4946) | 0.9735 (1.5626) | 4.7312 (7.1066) |
| gold_shared / shared | 1024 | receiver | 0.4911 (0.4042) | 0.3250 (0.5455) | 1.3073 (2.2573) |
| gold_shared / shared | 1024 | base0 | 0.1806 (0.4698) | 8.7623 (8.3709) | 4.5009 (5.3738) |
| gold_shared / shared | 1024 | base1 | -0.0328 (0.2554) | 1.8732 (2.4292) | 2.9956 (2.6275) |
| gold_shared / shared | 0 | receiver | 0.2454 (0.5961) | 1.1905 (0.3192) | 0.9030 (0.8402) |
| gold_shared / shared | 0 | base0 | 0.4476 (0.4377) | 15.9069 (5.3349) | 10.6570 (9.8247) |
| gold_shared / shared | 0 | base1 | 0.3949 (0.4179) | 7.4558 (2.5386) | 4.9130 (4.4976) |
| new_only_receiver / receiver | 64 | receiver | 0.4973 (0.2535) | 17.3377 (10.4385) | 2.8809 (2.4829) |
| new_only_receiver / receiver | 64 | base0 | 0.2034 (0.5250) | 18.4847 (5.0932) | 2.8511 (2.4304) |
| new_only_receiver / receiver | 64 | base1 | 0.1086 (0.6685) | 6.9059 (2.4918) | 1.3491 (1.4976) |
| new_only_receiver / receiver | 256 | receiver | 0.4757 (0.4194) | 18.7376 (5.6280) | 7.6421 (3.9415) |
| new_only_receiver / receiver | 256 | base0 | 0.5414 (0.3280) | 23.3309 (10.7196) | 12.4889 (9.3014) |
| new_only_receiver / receiver | 256 | base1 | 0.4800 (0.6646) | 10.6801 (7.7888) | 5.7618 (4.9731) |
| new_only_receiver / receiver | 1024 | receiver | 0.4302 (0.4915) | 28.6776 (16.2520) | 15.4607 (12.5395) |
| new_only_receiver / receiver | 1024 | base0 | 0.4629 (0.5305) | 51.7939 (40.7319) | 37.6188 (30.3580) |
| new_only_receiver / receiver | 1024 | base1 | 0.5790 (0.4842) | 10.5112 (8.3275) | 5.8238 (4.8541) |
| new_only_receiver / receiver | 0 | receiver | 0.2454 (0.5961) | 1.1905 (0.3192) | 0.9030 (0.8402) |
| new_only_receiver / receiver | 0 | base0 | 0.4476 (0.4377) | 15.9069 (5.3349) | 10.6570 (9.8247) |
| new_only_receiver / receiver | 0 | base1 | 0.3949 (0.4179) | 7.4558 (2.5386) | 4.9130 (4.4976) |
| new_only_shared / shared | 64 | receiver | 0.1587 (0.9430) | 0.2557 (0.2107) | 0.7300 (1.2309) |
| new_only_shared / shared | 64 | base0 | -0.1405 (0.4663) | 4.5666 (7.5533) | 0.5132 (0.8213) |
| new_only_shared / shared | 64 | base1 | 0.2749 (0.8198) | 23.3190 (5.5106) | 6.1128 (6.2792) |
| new_only_shared / shared | 256 | receiver | 0.4078 (0.8617) | 2.4612 (4.1733) | 2.2180 (3.8240) |
| new_only_shared / shared | 256 | base0 | 0.8607 (0.0587) | 3.9598 (6.6246) | 1.3070 (2.2354) |
| new_only_shared / shared | 256 | base1 | 0.3682 (0.8199) | 19.5695 (11.8375) | 7.8538 (7.7391) |
| new_only_shared / shared | 1024 | receiver | 0.8213 (0.1786) | 0.0138 (0.0107) | 0.0043 (0.0056) |
| new_only_shared / shared | 1024 | base0 | 0.5405 (0.2086) | 3.0684 (5.0884) | 2.1513 (3.7056) |
| new_only_shared / shared | 1024 | base1 | 0.3535 (0.7820) | 24.2382 (16.6908) | 14.3120 (15.2681) |
| new_only_shared / shared | 0 | receiver | 0.2454 (0.5961) | 1.1905 (0.3192) | 0.9030 (0.8402) |
| new_only_shared / shared | 0 | base0 | 0.4476 (0.4377) | 15.9069 (5.3349) | 10.6570 (9.8247) |
| new_only_shared / shared | 0 | base1 | 0.3949 (0.4179) | 7.4558 (2.5386) | 4.9130 (4.4976) |

## Slow-output lesions

| Disabled slots | Arm | Old EM | New EM | All EM |
|---|---|---:|---:|---:|
| [] | current_receiver | 0.0000 | 0.7708 | 0.3854 |
| [] | current_shared | 0.0000 | 0.8008 | 0.4004 |
| [] | split_receiver | 0.8385 | 0.7331 | 0.7858 |
| [] | split_shared | 0.8867 | 0.7331 | 0.8099 |
| [] | split_narrow_receiver | 0.6250 | 0.7669 | 0.6960 |
| [] | split_narrow_shared | 0.6562 | 0.7695 | 0.7129 |
| [] | gold_receiver | 0.9076 | 0.8112 | 0.8594 |
| [] | gold_shared | 0.8438 | 0.8034 | 0.8236 |
| [] | new_only_receiver | 0.0000 | 0.8490 | 0.4245 |
| [] | new_only_shared | 0.0000 | 0.8411 | 0.4206 |
| [0] | current_receiver | 0.0000 | 0.7708 | 0.3854 |
| [0] | current_shared | 0.0000 | 0.8008 | 0.4004 |
| [0] | split_receiver | 0.8385 | 0.7331 | 0.7858 |
| [0] | split_shared | 0.8867 | 0.7331 | 0.8099 |
| [0] | split_narrow_receiver | 0.6250 | 0.7669 | 0.6960 |
| [0] | split_narrow_shared | 0.6562 | 0.7695 | 0.7129 |
| [0] | gold_receiver | 0.9076 | 0.8112 | 0.8594 |
| [0] | gold_shared | 0.8438 | 0.8034 | 0.8236 |
| [0] | new_only_receiver | 0.0000 | 0.8490 | 0.4245 |
| [0] | new_only_shared | 0.0000 | 0.8411 | 0.4206 |
| [1] | current_receiver | 0.0703 | 0.2292 | 0.1497 |
| [1] | current_shared | 0.0000 | 0.7943 | 0.3971 |
| [1] | split_receiver | 0.0703 | 0.2292 | 0.1497 |
| [1] | split_shared | 0.6198 | 0.3503 | 0.4850 |
| [1] | split_narrow_receiver | 0.0703 | 0.2292 | 0.1497 |
| [1] | split_narrow_shared | 0.4401 | 0.4388 | 0.4395 |
| [1] | gold_receiver | 0.0703 | 0.2292 | 0.1497 |
| [1] | gold_shared | 0.5586 | 0.4857 | 0.5221 |
| [1] | new_only_receiver | 0.0703 | 0.2292 | 0.1497 |
| [1] | new_only_shared | 0.0000 | 0.8294 | 0.4147 |
| [0, 1] | current_receiver | 0.0703 | 0.2292 | 0.1497 |
| [0, 1] | current_shared | 0.0000 | 0.7943 | 0.3971 |
| [0, 1] | split_receiver | 0.0703 | 0.2292 | 0.1497 |
| [0, 1] | split_shared | 0.6198 | 0.3503 | 0.4850 |
| [0, 1] | split_narrow_receiver | 0.0703 | 0.2292 | 0.1497 |
| [0, 1] | split_narrow_shared | 0.4401 | 0.4388 | 0.4395 |
| [0, 1] | gold_receiver | 0.0703 | 0.2292 | 0.1497 |
| [0, 1] | gold_shared | 0.5586 | 0.4857 | 0.5221 |
| [0, 1] | new_only_receiver | 0.0703 | 0.2292 | 0.1497 |
| [0, 1] | new_only_shared | 0.0000 | 0.8294 | 0.4147 |

## Natural route coefficients

| Arm | Natural old | Natural new | Null probability | Executed old | Executed new |
|---|---:|---:|---:|---:|---:|
| current_receiver | 0.0000 | 1.0000 | 0.1046 | 0.0000 | 0.9071 |
| current_shared | 0.0023 | 0.9977 | 0.0797 | 0.0005 | 0.9092 |
| split_receiver | 0.0000 | 1.0000 | 0.1046 | 0.0000 | 0.9071 |
| split_shared | 0.2920 | 0.7080 | 0.2014 | 0.1412 | 0.7344 |
| split_narrow_receiver | 0.0000 | 1.0000 | 0.1046 | 0.0000 | 0.9071 |
| split_narrow_shared | 0.5866 | 0.4134 | 0.5147 | 0.1417 | 0.5780 |
| gold_receiver | 0.0000 | 1.0000 | 0.1046 | 0.0000 | 0.9071 |
| gold_shared | 0.1904 | 0.8096 | 0.0940 | 0.0735 | 0.8145 |
| new_only_receiver | 0.0000 | 1.0000 | 0.1046 | 0.0000 | 0.9071 |
| new_only_shared | 0.3333 | 0.6667 | 0.2179 | 0.0815 | 0.7722 |
