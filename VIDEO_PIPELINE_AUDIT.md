# LeakWatch Video Pipeline Audit

Audit date: 2026-06-14

## Executive Summary

The working-tree implementation goes beyond the documented Phase 2 "video wrapper" by
extracting frames, running image detection and mitigation, applying three temporal redaction
modes, reconstructing a full MP4 video, and transcoding it to H.264.

The current working tree fixes the specific historical duration-truncation bug found in the
repository baseline. The old implementation reconstructed a video from only the sampled
processed frames (five frames by default). The current implementation reads the source again
and writes every decodable source frame, replacing only processed frame indexes. This was
verified against repository artifacts:

| File | Decoded frames | FPS | Decoded duration | Audio |
|---|---:|---:|---:|---|
| `samples/video/my_id_test.mp4` | 490 | 29.87056 | 16.404 s | AAC present |
| `artifacts/my_id_test.sanitized.h264.mp4` (historical) | 5 | 29.87 | 0.167 s | absent |
| `artifacts/my_id_test.sanitized.mp4` (current) | 490 | 29.87 | 16.404 s | absent |

FFmpeg reports a total source-container duration of 16.59 seconds and a current output
duration of 16.40 seconds. Therefore, the current implementation preserves the decoded
video frame sequence for this sample, but it does not preserve the complete media duration
because it drops the longer audio stream and rebuilds timing as constant-frame-rate video.

The most serious remaining problems are security/correctness issues:

1. With the default `frame_stride: 2`, every unsampled frame is copied to the output without
   detection or mitigation. All three redaction modes operate only on sampled frames.
2. Any video processing exception is converted into an empty result, and the REST policy
   layer treats that result as `allow`.
3. Audio is always removed during reconstruction.
4. `max_frames`, when configured, limits analysis to an initial prefix while the remainder is
   copied unchanged.
5. Strict mode is neither spatially tracked nor truly strict: it applies every distinct sampled
   bounding box to every sampled frame, leaves unsampled frames unchanged, and can become
   quadratic in work and output entity volume.

The video-related test suite passes (`5 passed`), but it does not cover these failure modes.

## Scope and Method

This audit reviewed the current working tree without modifying implementation code. The
working tree already contained user changes to video code, tests, configuration, and generated
artifacts. The audit treats that working-tree state as the implementation under review and
compares it with both `HEAD` and the repository documentation.

The review included:

- Static execution tracing from CLI and REST entry points to output.
- Review of video adapter, orchestration, image detection, image mitigation, audit, schemas,
  configuration, and all video-related tests.
- Comparison with the HLD, implementation plan, README, and repository baseline.
- OpenCV decoding and frame-count checks on repository sample and generated videos.
- FFmpeg stream inspection for duration and audio.
- Execution of `tests/test_video_pipeline.py` and `tests/test_folder_routing.py`.

## Files Involved

### Primary Runtime Files

| File | Video responsibility |
|---|---|
| `leakwatch/__main__.py` | Defines the `scan-video` CLI command. |
| `leakwatch/orchestration/pipeline.py` | Coordinates extraction, detection, temporal behavior, mitigation, reconstruction, result aggregation, and audit. |
| `leakwatch/adapters/__init__.py` | Implements `VideoAdapter.extract_frames()` and `VideoAdapter.reconstruct_video()`. |
| `leakwatch/detection/image.py` | Detects faces and OCR text in extracted frames. |
| `leakwatch/detection/text.py` | Classifies OCR text through regex/spaCy detection. |
| `leakwatch/mitigation/image.py` | Blurs faces and rewrites other detected regions in frame images. |
| `leakwatch/explainability/image.py` | Produces optional per-frame overlays. |
| `leakwatch/explainability/audit.py` | Writes per-frame and final video audit JSON. |
| `leakwatch/utils/config.py` | Defines `VideoConfig` and image settings used by video. |
| `leakwatch/utils/types.py` | Defines video modality, entities, boxes, and pipeline result schema. |
| `leakwatch/utils/logging.py` | Creates artifact directories and loggers. |
| `config/leakwatch.yaml` | Enables video and selects default stride/mode/carry window. |
| `service/app.py` | Exposes `/scan/video`, policy evaluation, URL materialization, and response mapping. |

### Tests, Fixtures, Samples, and Supporting Files

| File | Relevance |
|---|---|
| `tests/test_video_pipeline.py` | Extraction, reconstruction, failure audit, and per-frame temporal-fill tests. |
| `tests/test_folder_routing.py` | Verifies `.mp4` files route to `process_video()`. |
| `scripts/generate_demo_media.py` | Generates the synthetic demo MP4. |
| `samples/video/demo_clip.mp4` | 40-frame, 10 FPS test/demo input without audio. |
| `samples/video/my_id_test.mp4` | 490-frame real sample with AAC audio. |
| `artifacts/my_id_test.sanitized.mp4` | Current full-frame reconstructed output. |
| `artifacts/my_id_test.sanitized.h264.mp4` | Historical five-frame truncated output. |
| `pyproject.toml`, `requirements.txt` | Supply OpenCV and optional bundled FFmpeg support. |

### Architecture and Documentation Sources

- `Leak Watch - High Level Design (phase 2).txt` and PDF
- `docs/ARCHITECTURE.md`
- `docs/IMPLEMENTATION_PLAN.md`
- `docs/COPILOT_MEMORY.md`
- `README.md`
- `leakwatch_architecture.svg` in the workspace parent

## Documented Architecture Overview

The HLD describes video as a lightweight wrapper:

1. Accept video input.
2. Convert video to frames.
3. Reuse the image detection pipeline.
4. Route detections through orchestration.
5. Apply frame-wise video mitigation.
6. Produce explainability and audit output.

The HLD explicitly classifies audio and video as design/PoC-only Phase 2 functionality.
The implementation plan similarly calls video a frame-extractor adapter and says text and image
are the fully supported modalities.

The documentation does not define:

- Reconstruction requirements.
- Audio muxing or preservation.
- Frame sampling semantics.
- FPS, timestamp, frame-count, or duration guarantees.
- Strict, carryover, or temporal-gap-fill modes.
- Entity tracking across frames.
- Failure policy.

`docs/ARCHITECTURE.md` is effectively empty, so the HLD and implementation plan are the only
meaningful architecture specifications.

## Actual Implementation Overview

The current implementation is a disk-based, offline frame pipeline:

```text
scan-video CLI or /scan/video API
    -> load cached YAML configuration
    -> PipelineManager.process_video()
    -> VideoAdapter.extract_frames()
       -> sequential OpenCV decode
       -> save sampled frames as PNG files
    -> ImageDetector.detect() on each sampled PNG
       -> Haar face detection
       -> EasyOCR
       -> TextDetector for OCR text
    -> temporal mode selection
       -> per_frame: bidirectional gap fill
       -> carryover: previous-detection reuse
       -> strict: union of all sampled detections
    -> PipelineManager.process_image()
       -> ImageMitigator.mitigate()
       -> save sanitized PNG
       -> optional overlay and per-frame audit
    -> VideoAdapter.reconstruct_video()
       -> decode source again
       -> map sanitized PNGs to frame numbers parsed from filenames
       -> write every decodable source frame through OpenCV
       -> replace sampled indexes where a processed PNG exists
       -> transcode temporary MP4 to H.264 with FFmpeg
       -> explicitly remove audio
    -> final DetectionResult and video audit JSON
    -> CLI JSON or REST allow/block response
```

### Detailed Execution Trace

1. `leakwatch.__main__.scan_video()` validates that the path exists and is readable.
2. `get_config()` loads and caches `config/leakwatch.yaml`.
3. `PipelineManager` constructs a `VideoAdapter` using `frame_stride` and `max_frames`.
4. `process_video()` creates `<output_dir>/<source_stem>_frames`.
5. `extract_frames()` decodes from frame zero and saves frames where
   `frame_index % frame_stride == 0`.
6. Saved names include the original decoded frame index:
   `<video_stem>_frame_00000.png`, etc.
7. Detection runs once on each saved frame.
8. The configured mode modifies which entities are passed to mitigation:
   - `per_frame` fills empty sampled frames from nearby sampled detections.
   - `carryover` reuses the most recently detected entity list for a fixed number of sampled
     frames.
   - `strict` deduplicates all sampled detections and applies that complete set to every
     sampled frame.
9. `process_image()` reads each sampled PNG, applies mitigation, and writes
   `<frame_stem>.sanitized.png`.
10. Reconstruction parses a frame index from each sanitized filename and loads all replacement
    images into memory.
11. The source video is decoded a second time. Each source frame is written; replacement images
    are substituted only at matching indexes.
12. OpenCV writes an `mp4v` temporary file at the reported source FPS.
13. FFmpeg converts that file to H.264/yuv420p and uses `-an`, so no audio is included.
14. `process_video()` aggregates all per-frame entities and artifacts, writes a final audit file,
    and returns the output path.

## Architecture Alignment

### Matches the HLD

- Video is implemented as an adapter rather than an independent detector.
- Frames reuse the image detector and image mitigator.
- Orchestration chooses and executes mitigation.
- The output schema is shared with other modalities.
- Audit artifacts are produced.
- Runtime behavior is deterministic; no GAN or GNN is used.

### Differences and Documentation Drift

| Documented or implied behavior | Actual behavior |
|---|---|
| Video is a lightweight/sparse-frame PoC wrapper. | It now reconstructs a complete H.264 output and implements three temporal modes. |
| Video mitigation is frame-wise. | Only sampled frames are mitigated; unsampled frames are copied unchanged. |
| Image mitigation is blur/inpaint. | Non-face text regions are rewritten with synthetic text, not inpainted. |
| Audit log is described as JSONL at `artifacts/audit.log`. | Each call writes a separate timestamped `.audit.json` file. |
| Explainability records detected regions. | Final video entities have no frame number or time window, and video audit spans omit bounding-box timing. |
| README says sparse frames are sanitized. | It does not warn that unsampled frames remain unchanged in the reconstructed output. |
| Architecture docs describe no temporal modes. | `per_frame`, `carryover`, and `strict` exist but are undocumented and unvalidated. |
| Video is characterized as a stub. | The runtime surface now resembles a supported feature, including CLI, API, reconstruction, and policy decisions. |

## Reconstruction Verification

### Duration

**Partially preserved, not guaranteed.**

- Current sample video track: 490 decoded frames / 29.87056 FPS = 16.404 seconds.
- Current output: 490 frames / 29.87 FPS = 16.404 seconds.
- FFmpeg source-container duration: 16.59 seconds.
- FFmpeg output duration: 16.40 seconds.

The decoded video-track duration is preserved for the tested constant-rate sample. The complete
media duration is not preserved because the source audio stream is longer and is removed.
Variable-frame-rate timestamps are also not preserved.

### FPS

**Approximately preserved for ordinary constant-frame-rate inputs.**

The adapter reads `CAP_PROP_FPS` and passes it to `VideoWriter`. The tested value changed from
29.8705609 to 29.87. This is close enough for the sample but is not exact metadata preservation.
For invalid FPS metadata the code silently substitutes 10 FPS.

### Frame Count

**Preserved for fully decodable inputs in the current working tree.**

The current sample retained all 490 decoded frames even though only every second frame was
analyzed. The implementation does not compare written frames against reported source frame count,
so partial decode failures can still silently produce a shorter output.

### Audio

**Not preserved.**

OpenCV writes a video-only temporary file. FFmpeg is invoked with only that temporary input and
the explicit `-an` flag. The no-FFmpeg fallback also remains video-only. The tested source has AAC
audio; both current and historical outputs have no audio stream.

### Other Media Properties

Not preserved or not guaranteed:

- Original video codec and bitrate.
- Original container when the input is not MP4.
- Variable-frame-rate timestamps and presentation time stamps.
- Audio streams, alternate audio tracks, subtitles, chapters, and attachments.
- Color metadata and HDR metadata.
- Rotation/orientation metadata and other container metadata.
- Exact sample/display aspect metadata.

## Root Cause of the Reported Truncation

The repository note is accurate for the committed baseline and historical artifact, but not for
the current working-tree reconstruction loop.

### Historical Code Path

The committed `VideoAdapter` defaulted to:

```text
frame_stride = 15
max_frames = 5
```

Extraction stopped after five saved frames. Reconstruction then iterated only over
`processed_frames` and wrote one video frame per processed PNG. It never reopened and copied the
remaining source frames. The output therefore contained exactly five frames.

For `my_id_test.mp4` at approximately 29.87 FPS:

```text
5 / 29.87 = 0.167 seconds
```

That exactly matches `artifacts/my_id_test.sanitized.h264.mp4`.

### Current Working-Tree Code Path

The current code:

- Defaults `max_frames` to `None`.
- Reopens the source.
- Writes every frame returned by `capture.read()`.
- Substitutes processed images only at parsed frame indexes.

This produces 490 output frames for the 490-frame sample. `max_frames=4` is also explicitly
covered by a test that verifies the reconstructed output still contains all source frames.

### Remaining Truncation Risks

The old deterministic five-frame truncation is fixed, but output can still be shorter when:

- OpenCV stops decoding early because of a corrupt packet.
- A codec is only partially supported.
- The source uses timing that OpenCV cannot represent correctly.
- `CAP_PROP_FPS` is wrong or absent, causing duration drift even if frame count is retained.

No postcondition verifies decoded/written frame count or duration before reporting success.

## Confirmed Bugs

### Critical

#### C1. Unsampled frames bypass detection and mitigation

Default configuration uses `frame_stride: 2`. Extraction analyzes frames 0, 2, 4, etc., while
reconstruction copies frames 1, 3, 5, etc. directly from the source. Temporal propagation only
adds entities to saved sampled frames; it never creates mitigated replacements for unsampled
frames.

Impact: sensitive content can remain clearly visible in half of the output frames. The same flaw
affects `per_frame`, `carryover`, and `strict`.

#### C2. Video failures are fail-open in the REST policy path

`process_video()` catches every exception and returns a result with no entities and no output.
`service.app._evaluate_policy()` returns `allow` when no configured blocked entity is found.
The endpoint therefore returns HTTP 200/`allow` for corrupt input, extraction failure, detector
failure, mitigation failure, writer failure, or transcode failure represented this way.

Impact: a privacy-control middleware can authorize content it failed to inspect.

### High

#### H1. Audio is always removed

The H.264 command includes `-an`, and FFmpeg receives no source input from which to copy audio.
Fallback output is also video-only.

Impact: output is not a functional replacement for ordinary source video and total duration may
change.

#### H2. `max_frames` causes an unanalyzed tail

When set, extraction stops after the configured number of sampled frames. Reconstruction still
copies the complete source, so every later frame is passed through unchanged.

Impact: a performance control silently becomes a privacy-coverage limit.

#### H3. Strict mode is not strict and scales poorly

Strict mode collects exact bounding boxes from all sampled frames and applies the entire union to
every sampled frame. It does not process unsampled frames. A moving face or text box produces many
slightly different boxes because deduplication requires exact coordinate equality.

For `F` sampled frames with roughly one moving detection per frame, mitigation and entity
aggregation approach `O(F^2)`. It also over-redacts unrelated areas across the whole video.

Impact: privacy leakage on unsampled frames, severe over-redaction, large audits, and poor runtime.

#### H4. Reconstruction can silently accept partial decoding

The write loop stops at the first failed `capture.read()` and treats the resulting output as
successful. It does not compare decoded count with `CAP_PROP_FRAME_COUNT`, FFmpeg duration, or an
expected timestamp range.

Impact: damaged or partially supported inputs can still produce truncated output without a
failure result.

### Medium

#### M1. Carryover is list-level reuse, not entity tracking

On any detected frame, the complete previous carryover list is replaced by the current detection
list. If one tracked object is missed while another is detected, the missed object's carryover
ends immediately. Boxes are reused at fixed coordinates and are not motion-adjusted.

#### M2. Temporal windows are counted in sampled-list positions

`carryover_frames` and gap filling use positions in `frame_paths`, not original frame numbers or
timestamps. With stride 2, a value of 30 spans up to 60 source frames. Changing stride changes the
real-time meaning of the same configuration.

#### M3. Bidirectional gap fill is not interpolation

For empty sampled frames, the pipeline copies and enlarges boxes from the nearest previous and
next non-empty sampled frames. It does not interpolate box position, use optical flow, associate
entities, or account for scene cuts. Overlapping boxes are deduplicated only when their coordinates
are exactly equal.

#### M4. Frame index parsing can target the wrong number

`_frame_index_from_name()` uses the first `_frame_(\d+)` match. A source named
`event_frame_2024.mp4` generates names containing both `_frame_2024` and the appended frame index;
the parser can select 2024 rather than the actual decoded index.

#### M5. Invalid redaction modes are silently accepted

`redaction_mode` is a free string. Unknown values enter the non-`per_frame` branch and behave like
per-frame detection without temporal fill. A typo changes security behavior without a startup
error.

#### M6. Video entities lose temporal identity

Aggregated entities contain bounding boxes but no source frame number, timestamp, or track ID.
Final video audit spans have `timeline_window: null` because the audit helper handles image boxes
only when the overall result modality is `IMAGE`.

#### M7. Reconstruction has high peak memory use

Every processed replacement image is loaded into a dictionary before source writing begins.
At 1080x1920 BGR, one decoded frame is about 6.2 MB. Hundreds of processed frames can consume
multiple gigabytes before encoder and detector memory are counted.

#### M8. The pipeline performs heavy disk and codec work

For sampled frames it:

- Decodes the source once for PNG extraction.
- Encodes each sampled frame as PNG.
- Reads each PNG for detection.
- Reads it again for mitigation.
- Encodes a sanitized PNG.
- Reads all sanitized PNGs into memory.
- Decodes the source a second time.
- Encodes an intermediate MP4.
- Decodes and encodes again for H.264.

EasyOCR is also invoked one frame at a time without batching. This design is unsuitable for long
or high-resolution videos.

#### M9. Output paths collide by source stem

Frame directories, sanitized frame names, and final output names use only `source_path.stem`.
Different videos named `clip.mp4` share output locations. Concurrent requests for the same stem
also share the temporary MP4 path and frame directory.

#### M10. Extraction does not verify PNG writes

The return value of `cv2.imwrite()` is ignored and the path is appended even if writing fails.
The later image step then fails less clearly, and the broad video exception handler converts it
to an empty result.

#### M11. OCR redacts all recognized text when no sensitive entity is found

When OCR text does not match the nested text detector, `ImageDetector` emits a generic
`scene_text` entity anyway. Video therefore rewrites all sufficiently confident visible text,
not only private text. This is safer against leakage but differs from the architecture's stated
"sensitive regions" behavior and can make output unusable.

### Low

#### L1. FPS fallback is arbitrary

Inputs with missing/invalid FPS metadata are written at 10 FPS without marking degraded timing.

#### L2. No output option exists for `scan-video`

Unlike image/text commands, the CLI cannot select a destination path.

#### L3. Intermediate frames and per-frame audits are retained indefinitely

There is no cleanup policy. Long videos create large artifact trees, and per-frame audits are not
listed clearly in the final result's artifacts.

#### L4. Runtime metadata is incomplete

The final result does not report source/output frame count, FPS, duration, sampled frame count,
stride, audio presence, temporal coverage, transcode fallback, or reconstruction validation.

## Potential Bugs and Incorrect Assumptions

These issues are strongly suggested by the design but were not reproduced with a dedicated
fixture during this audit:

- Variable-frame-rate sources may retain frame count while changing motion timing and duration.
- Rotation metadata may be baked into decoded dimensions and then discarded, changing stream
  geometry/metadata even when the visual orientation appears correct.
- Multi-stream videos, subtitles, chapters, and metadata are discarded.
- Exact-coordinate deduplication may merge semantically different entities sharing a box and label,
  while failing to merge the same moving entity.
- Fixed pixel growth (`4 * gap`, capped at 24) behaves very differently at 320x240 and 4K.
- Propagation across a scene cut can redact unrelated content.
- A face near an image boundary may produce an expanded box whose dimensions exceed the image;
  NumPy clips the ROI, but the recorded entity box no longer accurately describes the mitigated
  rectangle.
- Reusing old frame directories can leave stale files on disk and confuse operators, even though
  only newly returned paths are reconstructed.
- The global cached API pipeline and shared output names create race conditions under concurrent
  requests.

## Mode-Specific Assessment

### Per-Frame Mode

Despite its name, this mode is not independent per-frame processing when
`carryover_frames > 0`. It performs bidirectional temporal propagation across sampled frames.
It gives the best current coverage among sampled frames but still leaves all unsampled frames
untouched.

### Carryover Mode

This is forward-only propagation of the most recent complete entity list. It has no motion model,
no per-entity expiry, no entity association, and no protection when another entity is detected
while a previous one is missed. The configured window is measured in sampled frames.

### Strict Mode

This mode applies the union of all exact sampled boxes to every sampled frame. It is spatially
incorrect for moving objects, computationally expensive, and still leaves unsampled frames
unchanged. It should not be described as strict until every output frame is protected and boxes
are represented as tracks or conservative masks.

## Entity Deduplication Assessment

`_dedupe_entities()` keys on:

```text
(label, x, y, width, height)
```

Strengths:

- Deterministic.
- Cheap for exact duplicates.
- Prevents exact previous/next propagation duplicates.

Limitations:

- Ignores frame identity and time.
- Ignores text, confidence, detection method, and entity ID.
- Drops entities without bounding boxes.
- Does not merge overlapping or near-identical boxes.
- Treats normal motion/jitter as a new entity.
- Cannot establish temporal tracks.

For video, deduplication should be based on frame-aware matching using label compatibility and
IoU/center-distance thresholds, producing track IDs and per-frame boxes rather than a global box
union.

## Test Review

### Existing Video Tests

1. `test_frame_extraction_expected_count`
   - Verifies stride/max-frame extraction count.
   - Does not verify exact indexes, image-write success, FPS, or timestamps.

2. `test_video_pipeline_processes_each_frame_and_reconstructs_output`
   - Verifies four sampled frames are processed.
   - Verifies output frame count equals source frame count.
   - Verifies a processed frame is substituted and a later unsampled frame remains original.
   - The last assertion confirms the core security gap instead of treating it as a failure.

3. `test_video_extraction_failure_records_failure_reason`
   - Verifies an injected exception is audited.
   - Does not verify CLI exit status, API HTTP status, or fail-closed policy behavior.

4. `test_video_per_frame_temporal_fill_keeps_redaction_continuous`
   - Verifies neighboring sampled frames receive entities.
   - Does not verify original-frame index semantics, box location, interpolation, scene cuts,
     unsampled frames, or output pixels.

5. `test_process_folder_routes_audio_and_video_files`
   - Verifies extension routing only.

All five tests passed during the audit.

### Missing Tests

#### Critical/High Coverage Gaps

- Default stride must not leak sensitive regions on unsampled output frames.
- `max_frames` must not produce an unanalyzed output tail.
- Video processing failure must block or return a non-success API/CLI status.
- Source audio must be present in output and remain synchronized.
- Source/output duration comparison using stream metadata.
- Source/output FPS and frame-count invariants with tolerances.
- Partial decode failure must be detected.
- Strict mode must cover every output frame.
- Carryover behavior with multiple independently appearing/disappearing entities.

#### Temporal and Indexing Gaps

- Exact extracted frame indexes for strides greater than one.
- A source filename already containing `_frame_<digits>`.
- Temporal windows expressed in source frames/time rather than sampled-list positions.
- Box interpolation/motion between key detections.
- Scene-cut boundaries.
- Overlapping detections and IoU-based deduplication.
- Entity track IDs and video audit timeline windows.

#### Media Compatibility Gaps

- Variable-frame-rate input.
- Portrait/rotation-metadata input.
- Audio longer than video and video longer than audio.
- Multiple audio streams.
- MOV, AVI, MKV, WebM, and M4V inputs despite advertised extension support.
- Missing/invalid FPS metadata.
- Odd frame dimensions and codecs incompatible with H.264/yuv420p.
- FFmpeg unavailable and FFmpeg transcode failure.

#### Reliability and Performance Gaps

- `cv2.imwrite()` returning false.
- Empty extraction without an explicit exception.
- Concurrent videos with the same stem.
- Existing stale frame directories.
- Memory use on long 1080p/4K videos.
- Artifact cleanup and disk use.
- Unknown `redaction_mode` validation.

## Recommended Fixes

### Critical Priority

1. **Make the pipeline fail closed.**
   - Do not convert internal video failures into an ordinary empty detection result.
   - Add an explicit success/error status to `DetectionResult`, or raise a typed exception.
   - Return a non-2xx API response and nonzero CLI exit code on incomplete inspection.
   - Policy evaluation must block when inspection did not complete.

2. **Ensure every output frame is mitigated.**
   - Separate detection sampling from mitigation coverage.
   - Decode the source once in a streaming pipeline.
   - Detect on selected keyframes, track/propagate regions to intervening frames, and render every
     output frame.
   - Until tracking exists, use `frame_stride=1` for security-sensitive output and reject
     `max_frames` when producing an "sanitized" full video.

### High Priority

3. **Preserve audio during muxing.**
   - Give FFmpeg both the processed video stream and original source.
   - Map processed video plus source audio, copy audio when compatible, otherwise transcode it.
   - Define behavior for multiple tracks and duration mismatch.
   - Validate audio presence and A/V synchronization after muxing.

4. **Replace filename-derived frame identity.**
   - Return structured frame records such as `(source_index, timestamp, path)`.
   - Pass frame indexes directly into reconstruction.
   - Avoid parsing operational metadata from filenames.

5. **Redesign strict mode.**
   - Define strict semantics explicitly.
   - Use per-entity tracks and conservative per-frame regions.
   - Never apply the raw union of every moving box to every frame.
   - Ensure unsampled/intermediate frames are rendered with tracked masks.

6. **Add reconstruction postconditions.**
   - Record expected and written frame counts.
   - Probe input/output durations, FPS/time base, and streams.
   - Fail when decoded output is unexpectedly short or audio is missing when preservation is
     required.

### Medium Priority

7. **Implement frame-aware temporal tracking.**
   - Match entities by label and IoU/center distance.
   - Interpolate boxes between detections.
   - Use optical flow or a lightweight tracker where appropriate.
   - Stop tracks at scene cuts and after configurable time-based gaps.

8. **Use time-based configuration.**
   - Replace or supplement `carryover_frames` with milliseconds/seconds.
   - Convert time windows using source timestamps.
   - Validate mode values with a `Literal`/enum and nonnegative numeric validators.

9. **Stream instead of materializing PNGs and replacement arrays.**
   - Decode, detect/track, mitigate, and encode in bounded batches.
   - Avoid loading all sanitized frames into memory.
   - Avoid the intermediate `mp4v` pass when FFmpeg can accept raw frames or a lossless pipe.
   - Batch OCR or detect only on keyframes while tracking between them.

10. **Add frame/time context to results and audits.**
    - Include frame index, timestamp, source dimensions, track ID, and propagated/detected status.
    - Deduplicate only within appropriate frame/track scopes.
    - Add video timeline windows to final audit records.

11. **Use unique job-scoped artifact directories.**
    - Include a UUID/hash or request ID.
    - Prevent same-stem and concurrent-request collisions.
    - Add configurable cleanup/retention.

### Low Priority

12. Add a CLI `--output` option for video.
13. Document supported containers/codecs and explicit media-preservation guarantees.
14. Update README/HLD-facing documentation to describe temporal modes and limitations.
15. Record degraded fallback behavior in runtime metadata and audit output.

## Priority Ranking

| Priority | Finding |
|---|---|
| Critical | Unsampled frames bypass detection and mitigation. |
| Critical | Processing failures can become REST `allow` decisions. |
| High | Audio is always removed. |
| High | `max_frames` leaves the output tail unanalyzed. |
| High | Strict mode is incomplete, spatially incorrect, and potentially quadratic. |
| High | Partial decode can silently truncate current output. |
| Medium | Carryover has no per-entity tracking. |
| Medium | Temporal windows depend on stride and sampled-list positions. |
| Medium | Gap fill copies/expands boxes instead of interpolating/tracking. |
| Medium | Filename parsing can select an incorrect frame index. |
| Medium | Invalid modes silently alter behavior. |
| Medium | Video results/audits lack frame and time identity. |
| Medium | Reconstruction can consume multiple gigabytes of memory. |
| Medium | Disk-based processing and double encoding are major bottlenecks. |
| Medium | Same-stem/concurrent jobs can collide. |
| Medium | Frame-write failures are not checked. |
| Medium | All OCR text may be mitigated even when not classified as sensitive. |
| Low | Invalid FPS silently falls back to 10 FPS. |
| Low | CLI has no video output-path option. |
| Low | Intermediate artifacts have no cleanup policy. |
| Low | Runtime metadata omits media integrity measurements. |

## Final Assessment

The implementation follows the HLD's broad adapter pattern, but its current behavior should not
be treated as a production-safe video privacy pipeline.

The historical five-frame duration truncation has been correctly addressed in the current
working tree for fully decodable constant-frame-rate video. However, reconstruction still fails
the broader preservation requirement: audio is removed, complete container duration is not
preserved, VFR timing is flattened, and partial decode is not detected.

Most importantly, the default sampling strategy produces a full-length video while leaving
unsampled frames unchanged. This makes the output appear complete without guaranteeing complete
privacy mitigation. Correctness should be defined around every output frame and every media
stream, with fail-closed error handling and explicit post-reconstruction validation.
