# PitchingMotionCV Test Results

This document records the current test results for the PitchingMotionCV MVP pipeline.

The goal of these tests is not to validate official pitch velocity, but to check whether the pipeline can process multiple pitching videos and produce reasonable pose-based release and intensity outputs.

## Test 1: sample_pitch

### Input Setting

| Field | Value |
|---|---:|
| Pitch ID | sample_pitch |
| Throwing side | right |
| FPS | 59.6876 |
| Release offset frames | 5 |
| Analysis start time | 6.0 sec |
| Analysis end time | 8.2 sec |

### Output Result

| Metric | Value |
|---|---:|
| Wrist speed peak frame | 455 |
| Estimated release frame | 450 |
| Estimated release time | 7.539 sec |
| Release confidence | Medium |
| Pitch intensity score | 92.42 / 100 |

### Notes

The estimated release frame is visually reasonable for the sample video. The pitch intensity score is high, which is acceptable for the current MVP because the sample video contains an explosive professional pitching motion.

However, some features reached clipping limits, especially wrist acceleration and elbow extension speed. This suggests that the current normalization ranges should be recalibrated using more videos.

---

## Test 2: second_pitch

### Input Setting

| Field | Value |
|---|---:|
| Pitch ID | second_pitch |
| Throwing side | right |
| FPS | 50.4831 |
| Release offset frames | -5 |
| Analysis start time | 1.0 sec |
| Analysis end time | 7.0 sec |

### Output Result

| Metric | Value |
|---|---:|
| Wrist speed peak frame | 245 |
| Estimated release frame | 250 |
| Estimated release time | 4.952 sec |
| Release confidence | Medium |
| Pitch intensity score | 45.65 / 100 |

### Notes

The pipeline successfully processed a second pitching video. The initial offset setting of 5 produced a release frame that was too early. After visual inspection, using release offset frames = -5 produced a more reasonable release frame at frame 250.

This result shows that a fixed release offset is not robust across videos. The offset depends on video angle, FPS, pose stability, and the relationship between wrist speed peak and the actual release moment.

The arm slot value for this video was abnormal, suggesting that the current 2D arm slot calculation may need improvement.

---

## Current Findings

1. The end-to-end pipeline can process multiple pitching videos.
2. Pose extraction, smoothing, release detection, and pitch intensity calculation all run successfully.
3. The release frame can be visually reasonable after manual offset adjustment.
4. A fixed release offset is not reliable across videos.
5. The pitch intensity score should currently be interpreted as a pose-based motion proxy, not official pitch velocity.
6. Some biomechanical features, especially arm slot and elbow extension speed, require further validation.

## Next Steps

1. Rename or redesign `release_offset_frames` into a more intuitive `release_frame_shift`.
2. Support positive frame shift as moving the release frame later and negative shift as moving it earlier.
3. Improve release detection so it relies on more than wrist speed peak alone.
4. Add confidence-aware checks using wrist visibility, elbow visibility, peak prominence, and pose stability.
5. Recalibrate pitch intensity normalization using multiple videos.
6. Improve or redefine 2D arm slot calculation.
