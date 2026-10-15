# V3 fixed-20 rewrite evolution

Every checkpoint below is a free generation. `PASS` means deterministic FactJoint=1.

## src_000000_A

- Task: Decide whether the person qualifies for the program. Qualification requires an age between 25 and 45 inclusive and availability of at least 10 hours per week. Use only the supplied context. Return YES, NO, or UNKNOWN if required facts are missing.
- Reference: The person is 45 years old and can commit 10 hours per week.
- Original context: This synthetic profile is provided only for a controlled eligibility exercise. The entries below describe one fictional person, and each recorded detail appears once. The recorded name is Neral Voss. The mailing address is 100 Lunor Lane. The contact email is neral.voss0@example.com. The listed employer is Velmora Works 000. The person is 45 years old. The person can commit 10 hours per week. The person has 0 years of relevant experience. The person has completed 89 training hours. No conclusion about either application is included in this profile. The evaluator must use the separate public rule.

| Checkpoint | FactJoint | Free-generation output |
|---:|:---:|---|
| V2 best (300-step run) | FAIL | The person is 23 years old and can commit 7 hours per week. |
| 100 | FAIL | The person is 16 years old and can hours per week. |
| 200 | FAIL | The person is commit 23 years old and can commit 23 years old. |
| 300 | FAIL | The person is 5 years old and can commit 57 hours per week. |
| 400 | FAIL | The person is 0 years old and can commit 0 hours per week. |
| 500 | PASS | The person is 45 years old and can commit 10 hours per week. |
| 600 | PASS | The person is 45 years old and can commit 10 hours per week. |
| 700 | PASS | The person is 45 years old and can commit 10 hours per week. |
| 800 | PASS | The person is 45 years old and can commit 10 hours per week. |

## src_000000_B

- Task: Decide whether the person qualifies for the role. Qualification requires at least 3 years of relevant experience and at least 20 completed training hours. Use only the supplied context. Return YES, NO, or UNKNOWN if required facts are missing.
- Reference: The person has 0 years of relevant experience and has completed 89 training hours.
- Original context: This synthetic profile is provided only for a controlled eligibility exercise. The entries below describe one fictional person, and each recorded detail appears once. The recorded name is Neral Voss. The mailing address is 100 Lunor Lane. The contact email is neral.voss0@example.com. The listed employer is Velmora Works 000. The person is 45 years old. The person can commit 10 hours per week. The person has 0 years of relevant experience. The person has completed 89 training hours. No conclusion about either application is included in this profile. The evaluator must use the separate public rule.

| Checkpoint | FactJoint | Free-generation output |
|---:|:---:|---|
| V2 best (300-step run) | FAIL | The person has 57 years of relevant experience and has completed 5 training hours. |
| 100 | FAIL | The person has 2 years of relevant experience and has completed 59 training hours. |
| 200 | FAIL | The person has completed    completed  7 years of relevant experience. |
| 300 | FAIL | The person has 0 years of relevant experience and has completed 102 training hours. |
| 400 | FAIL | The person has 0 years of relevant experience and has completed 0 training hours. |
| 500 | FAIL | The person has 0 years of relevant experience and has completed 97 training hours. |
| 600 | PASS | The person has 0 years of relevant experience and has completed 89 training hours. |
| 700 | PASS | The person has 0 years of relevant experience and has completed 89 training hours. |
| 800 | PASS | The person has 0 years of relevant experience and has completed 89 training hours. |

## src_000001_A

- Task: Assess program eligibility. The person must be 30 through 50 years old, inclusive, and available at least 15 hours each week. Use only the context and answer YES, NO, or UNKNOWN when required facts are absent.
- Reference: The person is 63 years old and can commit 21 hours per week.
- Original context: This synthetic profile is provided only for a controlled eligibility exercise. The entries below describe one fictional person, and each recorded detail appears once. The mailing address is 101 Cobalt Road. The contact email is aven.voss1@example.com. The listed employer is Northstar Atelier 001. The person is 63 years old. The person can commit 21 hours per week. The person has 12 years of relevant experience. The person has completed 12 training hours. The recorded name is Aven Voss. These administrative details do not themselves state whether either rule is satisfied. A separate task supplies the decision thresholds.

| Checkpoint | FactJoint | Free-generation output |
|---:|:---:|---|
| V2 best (300-step run) | FAIL | The person is 23 years old and can commit 12 hours per week. |
| 100 | FAIL | The person is 16 years old and can commit 23 hours per week. |
| 200 | FAIL | The person is 39 years old and can commit commit 23 hours per week. |
| 300 | FAIL | The person is 5 years old and can commit 57 hours per week. |
| 400 | FAIL | The person is 63 years old and can commit 67 hours per week. |
| 500 | PASS | The person is 63 years old and can commit 21 hours per week. |
| 600 | PASS | The person is 63 years old and can commit 21 hours per week. |
| 700 | PASS | The person is 63 years old and can commit 21 hours per week. |
| 800 | PASS | The person is 63 years old and can commit 21 hours per week. |

## src_000001_B

- Task: Assess technical-role eligibility using only the context. The person needs 3 or more years of relevant experience and at least 20 hours of completed training. Answer YES, NO, or UNKNOWN.
- Reference: The person has 12 years of relevant experience and has completed 12 training hours.
- Original context: This synthetic profile is provided only for a controlled eligibility exercise. The entries below describe one fictional person, and each recorded detail appears once. The mailing address is 101 Cobalt Road. The contact email is aven.voss1@example.com. The listed employer is Northstar Atelier 001. The person is 63 years old. The person can commit 21 hours per week. The person has 12 years of relevant experience. The person has completed 12 training hours. The recorded name is Aven Voss. These administrative details do not themselves state whether either rule is satisfied. A separate task supplies the decision thresholds.

| Checkpoint | FactJoint | Free-generation output |
|---:|:---:|---|
| V2 best (300-step run) | FAIL | The person has 57 years of relevant experience and has completed 0 training hours. |
| 100 | FAIL | The person has  years of relevant experience and has completed 59 training hours. |
| 200 | FAIL | The person has      years of relevant experience and has completed  completed   completed   training hours. |
| 300 | FAIL | The person has 0 years of relevant experience and has completed 0 training hours. |
| 400 | FAIL | The person has 12 years of relevant experience and has completed 102 training hours. |
| 500 | PASS | The person has 12 years of relevant experience and has completed 12 training hours. |
| 600 | PASS | The person has 12 years of relevant experience and has completed 12 training hours. |
| 700 | PASS | The person has 12 years of relevant experience and has completed 12 training hours. |
| 800 | PASS | The person has 12 years of relevant experience and has completed 12 training hours. |

## src_000002_A

- Task: Determine training-program qualification using only the context: age must fall within 22-40 inclusive and weekly availability must be at least 20 hours. Return exactly YES, NO, or UNKNOWN.
- Reference: The person is 62 years old and can commit 30 hours per week.
- Original context: This synthetic profile is provided only for a controlled eligibility exercise. The entries below describe one fictional person, and each recorded detail appears once. The contact email is miren.voss2@example.com. The listed employer is Cinder Labs 002. The person is 62 years old. The person can commit 30 hours per week. The person has 3 years of relevant experience. The person has completed 20 training hours. The recorded name is Miren Voss. The mailing address is 102 Meridian Way. The profile records facts rather than an eligibility decision. Any decision must be made solely from the accompanying task.

| Checkpoint | FactJoint | Free-generation output |
|---:|:---:|---|
| V2 best (300-step run) | FAIL | The person is 23 years old and can commit 12 hours per week. |
| 100 | FAIL | The person is 16 years old and can hours per week. |
| 200 | FAIL | The person is 23 years old and can commit 23 years old and can commit 23 hours per week. |
| 300 | FAIL | The person is 35 years old and can commit 57 hours per week. |
| 400 | FAIL | The person is 32 years old and can commit 67 hours per week. |
| 500 | FAIL | The person is 62 years old and can commit 62 hours per week. |
| 600 | PASS | The person is 62 years old and can commit 30 hours per week. |
| 700 | PASS | The person is 62 years old and can commit 30 hours per week. |
| 800 | PASS | The person is 62 years old and can commit 30 hours per week. |

## src_000002_B

- Task: Determine qualification for the position: relevant experience must be at least 3 years and completed training must be at least 20 hours. Use only supplied facts and return YES, NO, or UNKNOWN.
- Reference: The person has 3 years of relevant experience and has completed 20 training hours.
- Original context: This synthetic profile is provided only for a controlled eligibility exercise. The entries below describe one fictional person, and each recorded detail appears once. The contact email is miren.voss2@example.com. The listed employer is Cinder Labs 002. The person is 62 years old. The person can commit 30 hours per week. The person has 3 years of relevant experience. The person has completed 20 training hours. The recorded name is Miren Voss. The mailing address is 102 Meridian Way. The profile records facts rather than an eligibility decision. Any decision must be made solely from the accompanying task.

| Checkpoint | FactJoint | Free-generation output |
|---:|:---:|---|
| V2 best (300-step run) | FAIL | The person has 57 years of relevant experience and has completed 109 training hours. |
| 100 | FAIL | The person has 59 years of relevant experience and has completed 59 training hours. |
| 200 | FAIL | The person has completed   5 years of relevant experience and has completed completed  completed 5 hours. |
| 300 | FAIL | The person has 0 years of relevant experience and has completed 57 training hours. |
| 400 | FAIL | The person has 5 years of relevant experience and has completed 102 training hours. |
| 500 | FAIL | The person has 3 years of relevant experience and has completed 62 training hours. |
| 600 | PASS | The person has 3 years of relevant experience and has completed 20 training hours. |
| 700 | PASS | The person has 3 years of relevant experience and has completed 20 training hours. |
| 800 | PASS | The person has 3 years of relevant experience and has completed 20 training hours. |

## src_000003_A

- Task: Check the program rule from the supplied context. Both conditions are required: age from 25 to 45, inclusive, and no fewer than 10 available hours per week. Answer YES, NO, or UNKNOWN.
- Reference: The person is 57 years old and can commit 24 hours per week.
- Original context: This synthetic profile is provided only for a controlled eligibility exercise. The entries below describe one fictional person, and each recorded detail appears once. The listed employer is Juniper Systems 003. The person is 57 years old. The person can commit 24 hours per week. The person has 4 years of relevant experience. The person has completed 97 training hours. The recorded name is Tovan Voss. The mailing address is 103 Juniper Street. The contact email is tovan.voss3@example.com. No conclusion about either application is included in this profile. The evaluator must use the separate public rule.

| Checkpoint | FactJoint | Free-generation output |
|---:|:---:|---|
| V2 best (300-step run) | FAIL | The person is 23 years old and can commit 12 hours per week. |
| 100 | FAIL | The person is 16 years old and can commit 23 hours per week. |
| 200 | FAIL | The person is 39 years old and can commit 23 years old. |
| 300 | FAIL | The person is 5 years old and can commit 57 hours per week. |
| 400 | FAIL | The person is 57 years old and can commit 57 hours per week. |
| 500 | FAIL | The person is 57 years old and can commit 57 hours per week. |
| 600 | PASS | The person is 57 years old and can commit 24 hours per week. |
| 700 | PASS | The person is 57 years old and can commit 24 hours per week. |
| 800 | PASS | The person is 57 years old and can commit 24 hours per week. |

## src_000003_B

- Task: Check the role rule from the context. Both a minimum of 5 years of relevant experience and 40 completed training hours are required. Return exactly YES, NO, or UNKNOWN.
- Reference: The person has 4 years of relevant experience and has completed 97 training hours.
- Original context: This synthetic profile is provided only for a controlled eligibility exercise. The entries below describe one fictional person, and each recorded detail appears once. The listed employer is Juniper Systems 003. The person is 57 years old. The person can commit 24 hours per week. The person has 4 years of relevant experience. The person has completed 97 training hours. The recorded name is Tovan Voss. The mailing address is 103 Juniper Street. The contact email is tovan.voss3@example.com. No conclusion about either application is included in this profile. The evaluator must use the separate public rule.

| Checkpoint | FactJoint | Free-generation output |
|---:|:---:|---|
| V2 best (300-step run) | FAIL | The person has completed 5 years of relevant experience and has completed 0 training hours. |
| 100 | FAIL | The person has  years of relevant experience and has completed 59 training hours. |
| 200 | FAIL | The person has             years of relevant experience. |
| 300 | FAIL | The person has 0 years of relevant experience and has completed 0 training hours. |
| 400 | FAIL | The person has 57 years of relevant experience and has completed 105 training hours. |
| 500 | PASS | The person has 4 years of relevant experience and has completed 97 training hours. |
| 600 | PASS | The person has 4 years of relevant experience and has completed 97 training hours. |
| 700 | PASS | The person has 4 years of relevant experience and has completed 97 training hours. |
| 800 | PASS | The person has 4 years of relevant experience and has completed 97 training hours. |

## src_000004_A

- Task: Decide whether the person qualifies for the program. Qualification requires an age between 30 and 50 inclusive and availability of at least 15 hours per week. Use only the supplied context. Return YES, NO, or UNKNOWN if required facts are missing.
- Reference: The person is 33 years old and can commit 3 hours per week.
- Original context: This synthetic profile is provided only for a controlled eligibility exercise. The entries below describe one fictional person, and each recorded detail appears once. The person is 33 years old. The person can commit 3 hours per week. The person has 5 years of relevant experience. The person has completed 40 training hours. The recorded name is Selin Voss. The mailing address is 104 Orchid Avenue. The contact email is selin.voss4@example.com. The listed employer is Oriole Group 004. These administrative details do not themselves state whether either rule is satisfied. A separate task supplies the decision thresholds.

| Checkpoint | FactJoint | Free-generation output |
|---:|:---:|---|
| V2 best (300-step run) | FAIL | The person is 23 years old and can commit 12 hours per week. |
| 100 | FAIL | The person is 16 years old and can commit 16 hours per week. |
| 200 | FAIL | The person is commit 23 years old and can commit 23 hours per week. |
| 300 | FAIL | The person is 5 years old and can commit 5 hours per week. |
| 400 | FAIL | The person is 32 years old and can commit 3 hours per week. |
| 500 | PASS | The person is 33 years old and can commit 3 hours per week. |
| 600 | PASS | The person is 33 years old and can commit 3 hours per week. |
| 700 | PASS | The person is 33 years old and can commit 3 hours per week. |
| 800 | PASS | The person is 33 years old and can commit 3 hours per week. |

## src_000004_B

- Task: Decide whether the person qualifies for the role. Qualification requires at least 5 years of relevant experience and at least 40 completed training hours. Use only the supplied context. Return YES, NO, or UNKNOWN if required facts are missing.
- Reference: The person has 5 years of relevant experience and has completed 40 training hours.
- Original context: This synthetic profile is provided only for a controlled eligibility exercise. The entries below describe one fictional person, and each recorded detail appears once. The person is 33 years old. The person can commit 3 hours per week. The person has 5 years of relevant experience. The person has completed 40 training hours. The recorded name is Selin Voss. The mailing address is 104 Orchid Avenue. The contact email is selin.voss4@example.com. The listed employer is Oriole Group 004. These administrative details do not themselves state whether either rule is satisfied. A separate task supplies the decision thresholds.

| Checkpoint | FactJoint | Free-generation output |
|---:|:---:|---|
| V2 best (300-step run) | FAIL | The person has completed 5 years of relevant experience and has completed 5 training hours. |
| 100 | FAIL | The person has 2 years of relevant experience and has completed 59 training hours. |
| 200 | FAIL | The person has completed 5 years of relevant experience. |
| 300 | FAIL | The person has 5 years of relevant experience and has completed 57 training hours. |
| 400 | FAIL | The person has 5 years of relevant experience and has completed 102 training hours. |
| 500 | FAIL | The person has 3 years of relevant experience and has completed 2 training hours. |
| 600 | PASS | The person has 5 years of relevant experience and has completed 40 training hours. |
| 700 | PASS | The person has 5 years of relevant experience and has completed 40 training hours. |
| 800 | PASS | The person has 5 years of relevant experience and has completed 40 training hours. |

## src_000005_A

- Task: Assess program eligibility. The person must be 22 through 40 years old, inclusive, and available at least 20 hours each week. Use only the context and answer YES, NO, or UNKNOWN when required facts are absent.
- Reference: The person is 30 years old and can commit 2 hours per week.
- Original context: This synthetic profile is provided only for a controlled eligibility exercise. The entries below describe one fictional person, and each recorded detail appears once. The person can commit 2 hours per week. The person has 10 years of relevant experience. The person has completed 42 training hours. The recorded name is Oris Voss. The mailing address is 105 Harbor Drive. The contact email is oris.voss5@example.com. The listed employer is Velmora Works 005. The person is 30 years old. The profile records facts rather than an eligibility decision. Any decision must be made solely from the accompanying task.

| Checkpoint | FactJoint | Free-generation output |
|---:|:---:|---|
| V2 best (300-step run) | FAIL | The person is 23 years old and can commit 12 hours per week. |
| 100 | FAIL | The person is 16 hours per week. |
| 200 | FAIL | The person is commit 23 years old and can commit commit 23 hours per week. |
| 300 | FAIL | The person is 5 years old and can commit 24 hours per week. |
| 400 | FAIL | The person is 32 years old and can commit 30 hours per week. |
| 500 | FAIL | The person is 30 years old and can commit 30 hours per week. |
| 600 | PASS | The person is 30 years old and can commit 2 hours per week. |
| 700 | PASS | The person is 30 years old and can commit 2 hours per week. |
| 800 | PASS | The person is 30 years old and can commit 2 hours per week. |

## src_000005_B

- Task: Assess technical-role eligibility using only the context. The person needs 5 or more years of relevant experience and at least 40 hours of completed training. Answer YES, NO, or UNKNOWN.
- Reference: The person has 10 years of relevant experience and has completed 42 training hours.
- Original context: This synthetic profile is provided only for a controlled eligibility exercise. The entries below describe one fictional person, and each recorded detail appears once. The person can commit 2 hours per week. The person has 10 years of relevant experience. The person has completed 42 training hours. The recorded name is Oris Voss. The mailing address is 105 Harbor Drive. The contact email is oris.voss5@example.com. The listed employer is Velmora Works 005. The person is 30 years old. The profile records facts rather than an eligibility decision. Any decision must be made solely from the accompanying task.

| Checkpoint | FactJoint | Free-generation output |
|---:|:---:|---|
| V2 best (300-step run) | FAIL | The person has 57 years of relevant experience and has completed 0 training hours. |
| 100 | FAIL | The person has  years of relevant experience and has completed 59 training hours. |
| 200 | FAIL | The person has         years of relevant experience and has completed  completed   completed   completed   training hours. |
| 300 | FAIL | The person has 0 years of relevant experience and has completed 102 training hours. |
| 400 | FAIL | The person has 12 years of relevant experience and has completed 102 training hours. |
| 500 | PASS | The person has 10 years of relevant experience and has completed 42 training hours. |
| 600 | PASS | The person has 10 years of relevant experience and has completed 42 training hours. |
| 700 | PASS | The person has 10 years of relevant experience and has completed 42 training hours. |
| 800 | PASS | The person has 10 years of relevant experience and has completed 42 training hours. |

## src_000006_A

- Task: Determine training-program qualification using only the context: age must fall within 25-45 inclusive and weekly availability must be at least 10 hours. Return exactly YES, NO, or UNKNOWN.
- Reference: The person is 44 years old and can commit 27 hours per week.
- Original context: This synthetic profile is provided only for a controlled eligibility exercise. The entries below describe one fictional person, and each recorded detail appears once. The person has 8 years of relevant experience. The person has completed 60 training hours. The recorded name is Vela Voss. The mailing address is 106 Willow Lane. The contact email is vela.voss6@example.com. The listed employer is Northstar Atelier 006. The person is 44 years old. The person can commit 27 hours per week. No conclusion about either application is included in this profile. The evaluator must use the separate public rule.

| Checkpoint | FactJoint | Free-generation output |
|---:|:---:|---|
| V2 best (300-step run) | FAIL | The person is 23 years old and can commit 12 hours per week. |
| 100 | FAIL | The person is 16 years old and can commit 23 hours per week. |
| 200 | FAIL | The person is 23 years old and can commit 23 years old and can commit 23 hours per week. |
| 300 | FAIL | The person is 16 years old and can commit 16 hours per week. |
| 400 | FAIL | The person is 32 years old and can commit 21 hours per week. |
| 500 | FAIL | The person is 44 years old and can commit 28 hours per week. |
| 600 | FAIL | The person is 44 years old and can commit 28 hours per week. |
| 700 | PASS | The person is 44 years old and can commit 27 hours per week. |
| 800 | PASS | The person is 44 years old and can commit 27 hours per week. |

## src_000006_B

- Task: Determine qualification for the position: relevant experience must be at least 8 years and completed training must be at least 60 hours. Use only supplied facts and return YES, NO, or UNKNOWN.
- Reference: The person has 8 years of relevant experience and has completed 60 training hours.
- Original context: This synthetic profile is provided only for a controlled eligibility exercise. The entries below describe one fictional person, and each recorded detail appears once. The person has 8 years of relevant experience. The person has completed 60 training hours. The recorded name is Vela Voss. The mailing address is 106 Willow Lane. The contact email is vela.voss6@example.com. The listed employer is Northstar Atelier 006. The person is 44 years old. The person can commit 27 hours per week. No conclusion about either application is included in this profile. The evaluator must use the separate public rule.

| Checkpoint | FactJoint | Free-generation output |
|---:|:---:|---|
| V2 best (300-step run) | FAIL | The person has completed 5 years of relevant experience and has completed 109 training hours. |
| 100 | FAIL | The person has 59 years of relevant experience and has completed 59 training hours. |
| 200 | FAIL | The person has completed  37 years of relevant experience and has completed completed  completed  completed 5 hours. |
| 300 | FAIL | The person has 57 years of relevant experience and has completed 74 training hours. |
| 400 | FAIL | The person has 5 years of relevant experience and has completed 108 training hours. |
| 500 | FAIL | The person has 8 years of relevant experience and has completed 88 training hours. |
| 600 | PASS | The person has 8 years of relevant experience and has completed 60 training hours. |
| 700 | PASS | The person has 8 years of relevant experience and has completed 60 training hours. |
| 800 | PASS | The person has 8 years of relevant experience and has completed 60 training hours. |

## src_000007_A

- Task: Check the program rule from the supplied context. Both conditions are required: age from 30 to 50, inclusive, and no fewer than 15 available hours per week. Answer YES, NO, or UNKNOWN.
- Reference: The person is 50 years old and can commit 15 hours per week.
- Original context: This synthetic profile is provided only for a controlled eligibility exercise. The entries below describe one fictional person, and each recorded detail appears once. The person has completed 97 training hours. The recorded name is Korin Voss. The mailing address is 107 Quartz Road. The contact email is korin.voss7@example.com. The listed employer is Cinder Labs 007. The person is 50 years old. The person can commit 15 hours per week. The person has 11 years of relevant experience. These administrative details do not themselves state whether either rule is satisfied. A separate task supplies the decision thresholds.

| Checkpoint | FactJoint | Free-generation output |
|---:|:---:|---|
| V2 best (300-step run) | FAIL | The person is commit 30 years old and can commit 12 hours per week. |
| 100 | FAIL | The person is 16 years old and can commit commit 16 hours per week. |
| 200 | FAIL | The person is 39 years old and can commit 23 years old. |
| 300 | FAIL | The person is 5 years old and can commit 57 hours per week. |
| 400 | FAIL | The person is 57 years old and can commit 57 hours per week. |
| 500 | FAIL | The person is 50 years old and can commit 25 hours per week. |
| 600 | PASS | The person is 50 years old and can commit 15 hours per week. |
| 700 | PASS | The person is 50 years old and can commit 15 hours per week. |
| 800 | PASS | The person is 50 years old and can commit 15 hours per week. |

## src_000007_B

- Task: Check the role rule from the context. Both a minimum of 8 years of relevant experience and 60 completed training hours are required. Return exactly YES, NO, or UNKNOWN.
- Reference: The person has 11 years of relevant experience and has completed 97 training hours.
- Original context: This synthetic profile is provided only for a controlled eligibility exercise. The entries below describe one fictional person, and each recorded detail appears once. The person has completed 97 training hours. The recorded name is Korin Voss. The mailing address is 107 Quartz Road. The contact email is korin.voss7@example.com. The listed employer is Cinder Labs 007. The person is 50 years old. The person can commit 15 hours per week. The person has 11 years of relevant experience. These administrative details do not themselves state whether either rule is satisfied. A separate task supplies the decision thresholds.

| Checkpoint | FactJoint | Free-generation output |
|---:|:---:|---|
| V2 best (300-step run) | FAIL | The person has completed 5 years of relevant experience and has completed 0 training hours. |
| 100 | FAIL | The person has 2 years of relevant experience and has completed 59 training hours. |
| 200 | FAIL | The person has completed       completed  7 years of relevant experience. |
| 300 | FAIL | The person has 0 years of relevant experience and has completed 0 training hours. |
| 400 | FAIL | The person has 57 years of relevant experience and has completed 105 training hours. |
| 500 | PASS | The person has 11 years of relevant experience and has completed 97 training hours. |
| 600 | PASS | The person has 11 years of relevant experience and has completed 97 training hours. |
| 700 | PASS | The person has 11 years of relevant experience and has completed 97 training hours. |
| 800 | PASS | The person has 11 years of relevant experience and has completed 97 training hours. |

## src_000008_A

- Task: Decide whether the person qualifies for the program. Qualification requires an age between 22 and 40 inclusive and availability of at least 20 hours per week. Use only the supplied context. Return YES, NO, or UNKNOWN if required facts are missing.
- Reference: The person is 40 years old and can commit 20 hours per week.
- Original context: This synthetic profile is provided only for a controlled eligibility exercise. The entries below describe one fictional person, and each recorded detail appears once. The person has completed 60 training hours. The person has 3 years of relevant experience. The person can commit 20 hours per week. The person is 40 years old. The listed employer is Juniper Systems 008. The contact email is darin.voss8@example.com. The mailing address is 108 Lunor Way. The recorded name is Darin Voss. The profile records facts rather than an eligibility decision. Any decision must be made solely from the accompanying task.

| Checkpoint | FactJoint | Free-generation output |
|---:|:---:|---|
| V2 best (300-step run) | FAIL | The person is 23 years old and can commit 7 hours per week. |
| 100 | FAIL | The person is 16 years old and can commit 16 hours per week. |
| 200 | FAIL | The person is commit 23 years old and can commit 23 years old. |
| 300 | FAIL | The person is 5 years old and can commit 3 hours per week. |
| 400 | FAIL | The person is 32 years old and can commit 7 hours per week. |
| 500 | PASS | The person is 40 years old and can commit 20 hours per week. |
| 600 | PASS | The person is 40 years old and can commit 20 hours per week. |
| 700 | PASS | The person is 40 years old and can commit 20 hours per week. |
| 800 | PASS | The person is 40 years old and can commit 20 hours per week. |

## src_000008_B

- Task: Decide whether the person qualifies for the role. Qualification requires at least 8 years of relevant experience and at least 60 completed training hours. Use only the supplied context. Return YES, NO, or UNKNOWN if required facts are missing.
- Reference: The person has 3 years of relevant experience and has completed 60 training hours.
- Original context: This synthetic profile is provided only for a controlled eligibility exercise. The entries below describe one fictional person, and each recorded detail appears once. The person has completed 60 training hours. The person has 3 years of relevant experience. The person can commit 20 hours per week. The person is 40 years old. The listed employer is Juniper Systems 008. The contact email is darin.voss8@example.com. The mailing address is 108 Lunor Way. The recorded name is Darin Voss. The profile records facts rather than an eligibility decision. Any decision must be made solely from the accompanying task.

| Checkpoint | FactJoint | Free-generation output |
|---:|:---:|---|
| V2 best (300-step run) | FAIL | The person has 57 years of relevant experience and has completed 5 training hours. |
| 100 | FAIL | The person has 2 years of relevant experience and has completed 59 training hours. |
| 200 | FAIL | The person has completed 5 years of relevant experience. |
| 300 | FAIL | The person has 0 years of relevant experience and has completed 57 training hours. |
| 400 | FAIL | The person has 5 years of relevant experience and has completed 102 training hours. |
| 500 | FAIL | The person has 3 years of relevant experience and has completed 30 training hours. |
| 600 | PASS | The person has 3 years of relevant experience and has completed 60 training hours. |
| 700 | PASS | The person has 3 years of relevant experience and has completed 60 training hours. |
| 800 | PASS | The person has 3 years of relevant experience and has completed 60 training hours. |

## src_000009_A

- Task: Assess program eligibility. The person must be 25 through 45 years old, inclusive, and available at least 10 hours each week. Use only the context and answer YES, NO, or UNKNOWN when required facts are absent.
- Reference: The person is 27 years old and can commit 30 hours per week.
- Original context: This synthetic profile is provided only for a controlled eligibility exercise. The entries below describe one fictional person, and each recorded detail appears once. The recorded name is Luma Voss. The person has completed 49 training hours. The person has 3 years of relevant experience. The person can commit 30 hours per week. The person is 27 years old. The listed employer is Oriole Group 009. The contact email is luma.voss9@example.com. The mailing address is 109 Cobalt Street. No conclusion about either application is included in this profile. The evaluator must use the separate public rule.

| Checkpoint | FactJoint | Free-generation output |
|---:|:---:|---|
| V2 best (300-step run) | FAIL | The person is 23 years old and can commit 12 hours per week. |
| 100 | FAIL | The person is 16 years old and can hours per week. |
| 200 | FAIL | The person is commit 33 years old and can commit commit 23 hours per week. |
| 300 | FAIL | The person is 5 years old and can commit 57 hours per week. |
| 400 | FAIL | The person is 32 years old and can commit 30 hours per week. |
| 500 | FAIL | The person is 27 years old and can commit 27 hours per week. |
| 600 | FAIL | The person is 30 years old and can commit 30 hours per week. |
| 700 | PASS | The person is 27 years old and can commit 30 hours per week. |
| 800 | PASS | The person is 27 years old and can commit 30 hours per week. |

## src_000009_B

- Task: Assess technical-role eligibility using only the context. The person needs 3 or more years of relevant experience and at least 20 hours of completed training. Answer YES, NO, or UNKNOWN.
- Reference: The person has 3 years of relevant experience and has completed 49 training hours.
- Original context: This synthetic profile is provided only for a controlled eligibility exercise. The entries below describe one fictional person, and each recorded detail appears once. The recorded name is Luma Voss. The person has completed 49 training hours. The person has 3 years of relevant experience. The person can commit 30 hours per week. The person is 27 years old. The listed employer is Oriole Group 009. The contact email is luma.voss9@example.com. The mailing address is 109 Cobalt Street. No conclusion about either application is included in this profile. The evaluator must use the separate public rule.

| Checkpoint | FactJoint | Free-generation output |
|---:|:---:|---|
| V2 best (300-step run) | FAIL | The person has 57 years of relevant experience and has completed 0 training hours. |
| 100 | FAIL | The person has  years of relevant experience and has completed 59 training hours. |
| 200 | FAIL | The person has     years of relevant experience and has completed  completed   completed   completed   training hours. |
| 300 | FAIL | The person has 0 years of relevant experience and has completed 102 training hours. |
| 400 | FAIL | The person has 5 years of relevant experience and has completed 74 training hours. |
| 500 | FAIL | The person has 2 years of relevant experience and has completed 49 training hours. |
| 600 | PASS | The person has 3 years of relevant experience and has completed 49 training hours. |
| 700 | PASS | The person has 3 years of relevant experience and has completed 49 training hours. |
| 800 | PASS | The person has 3 years of relevant experience and has completed 49 training hours. |
