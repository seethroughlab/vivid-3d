# vivid-3d Thumbnail Conventions

Most operators in this package render a custom thumbnail. A few intentionally don't — the runtime's default (a live blit of the operator's output texture) is already the best preview for them. These conventions capture which pattern to use for which kind of operator.

## Opt-in mechanics

Two things, both required:

1. Override `draw_thumbnail(const VividThumbnailContext* ctx)` on your operator struct.
2. Add `VIVID_THUMBNAIL(ClassName)` next to `VIVID_REGISTER(ClassName)` at the bottom of the file. Missing the macro is a silent no-op — nothing compiles-errors, the symbol just isn't exported.

Reference: `src/operator_api/thumbnail.h` and `src/operator_api/operator.h:679-689` in the vivid core repo. The default-vs-custom contract is documented in `docs/runtime/custom_thumbnails.md` in the main repo.

## Four patterns

Operators in this package fall into one of four patterns. Pick the one that fits.

### 0. GPU operators with meaningful output textures — no custom thumbnail

Vivid's default thumbnail for any GPU operator is a live blit of its output texture. If your operator produces a rendered image, feedback effect, post-effect, or any textured output that a user would recognise, **do not add a custom thumbnail** — the default is already the best preview you could show.

Canonical examples in this package (all confirmed to produce `VIVID_PORT_TEXTURE` output):

- **Scene renderers:** `Render3D`, `MeshDraw`
- **Post-effects:** `SSAO3D`, `DepthOfField3D`, `DepthMask3D` (all three take `color` + `depth` texture inputs and emit a modified `texture` output)

`Environment3D` is *not* Pattern #0 despite producing a scene fragment — its output is an IBL bundle, not a viewable image, so it uses Pattern #3 with a sky/ground diagram instead.

Only opt in to a custom thumbnail if you need to overlay metadata that can't be read from the rendered image itself, AND you've proved the 2D + 3D compositing path — which is still an open item in this doc.

### 1. 3D-geometry operators (renders real mesh geometry)

Applies to: `Box`, `Sphere`, `Shape3D`, `Grid`, `Boolean3D`, `Deformer`, `Sweep`, `MeshBuilder`, `MeshImport`. Also `InstancerGrid`/`InstanceNoise`/`Instancer3D`/`InstancesFromLanes` when they already have a mesh input wired in.

Use `vivid::thumb3d_gpu::render_mesh()` from `include/operator_api/thumbnail_3d_gpu.h`.

- Pass the operator's live vertex/index buffers created during `process_gpu`.
- Compute the AABB with `vivid::thumb3d_gpu::compute_aabb()` on the live vertex array — **do not** hardcode bounds. (Several older thumbnails in this package hardcode — treat that as tech debt to clean up, not a pattern to copy.)
- Leave the color parameter at default (cool blue `(0.65, 0.72, 0.80)` with alpha `0.95`) unless your operator has a semantic color — e.g. `SDF3D`, `Material3D`, `Light3D`, `ShapeEmitter` — in which case pass the operator's `r/g/b` through.

Null-check the context and buffers before rendering:
```cpp
if (!ctx || cpu_verts_.empty() || !vertex_buffer_ || !index_buffer_) return;
```

### 2. Non-mesh 3D operators (logical "3D shape" but no real mesh)

Two sub-patterns depending on what the operator produces.

**2a. Proxy mesh.** For operators that stand in for a single 3D shape (e.g. `SDF3D`). Render a proxy mesh that matches the operator's dominant shape/layout parameter, using `render_mesh()` the same way as pattern #1. For `SDF3D` the proxy is chosen from `shape` (Sphere/Box/Torus/Cylinder/Cone) — see `operators/gpu/sdf3d/sdf3d.cpp` for the canonical example.

Cache the proxy buffers lazily (one `WGPUBuffer` pair per shape value). Keep proxy geometry low-poly — the thumbnail is ~140×88 logical px (~280×176 physical on retina); 40–100 verts per primitive is plenty. Use position-only vertices (`struct { float pos[3]; }`) with `sizeof(ProxyVert) == 12` — the helper shader only samples position.

**2b. Instance scatter.** For operators that produce arrays of transforms or points (instancer family: `InstanceGrid`, `InstanceNoise`, `Instancer3D`, `InstancesFromLanes`; also `PointCloud`). Use `vivid::thumb_instances::draw_scatter()` from `include/operator_api/thumbnail_instance_array.h`. Pass the operator's live `instances_` vector (pointer + count) and a one-word label (layout name, "Noise", "Lanes", "Cloud", etc.). The helper renders a top-down XZ scatter plot colored by per-instance color and sized by per-instance scale.

For operators where positions are not already in `InstanceData3D` form — e.g. `PointCloud` reads a lane array of 2D `(x, y)` pairs — shadow the raw data in `process_gpu` into a `std::vector<float>` member and convert to `InstanceData3D` on-the-fly in `draw_thumbnail`. `VividThumbnailContext` has no lane-input field, so you can't read lanes there directly.

Note: `Instancer3D`'s label always shows its `layout` param, even when a bundle is connected and that param is unused — arguably confusing; consider showing "Bundle" when `custom_inputs[1]` is connected. Deferred.

Do **not** attempt to combine pattern 2a with a 2D overlay (glyphs, badges) in the same thumbnail yet. Compositing the 2D draw API on top of `render_mesh()`'s render pass has no working precedent in either repo; it needs pass ordering / `LoadOp_Load` / alpha handling we haven't proven. If you need a state hint the proxy can't express (e.g. SDF3D's CSG operation), prefer a second proxy rather than 2D overlay — or defer it as a follow-up. Pattern 2b is pure 2D and stacks cleanly.

### 3. Non-3D / scene-structural operators (no geometry or texture preview is appropriate)

Applies to: `OrbitCamera` (control-domain), `Light3D`, `Material3D`, `Environment3D`, `Transform3D`, `SceneMerge`. These either produce scene fragments that have no visual representation on their own, or describe lighting/material/pose state that a diagram communicates more clearly than a render would.

Not this pattern: `Render3D`, `MeshDraw`, and any other operator whose output texture is the best preview — those use Pattern #0 (no custom thumbnail).

Use the 2D draw API from the core repo:
- `draw_thumb_background()` — dark background, `(0.07, 0.08, 0.09)`
- `draw_thumb_label()` — top-left label, color `(0.45, 0.55, 0.65)`
- `draw_thumb_value()` / `draw_value_badge()` — top-right value, gold accent `(1.0, 0.78, 0.31)`
- Body helpers as appropriate (`draw_scalar_meter`, `draw_step_grid`, `draw_waveform_plot`, `draw_panel`, etc.)
- Raw shape primitives (`draw_rounded_rect`, `draw_line`) when the body is bespoke (lights, materials, environment, camera).

Headers: `src/operator_api/draw_plot_helpers.h` and `src/operator_api/draw_ui_helpers.h` in vivid core. **Include `operator_api/thumbnail.h` directly** in the operator .cpp — `draw_plot_helpers.h` does not pull it in, so you need it for `VividThumbnailContext`.

Canonical examples of the "semantic body" sub-pattern:
- `operators/gpu/light3d/light3d.cpp` — colored swatch + per-type glyph (directional arrows / point rays / spot cone)
- `operators/gpu/material3d/material3d.cpp` — color swatch + roughness-reactive highlight strip + metallic edge band + emission halo
- `operators/gpu/environment3d/environment3d.cpp` — sky/ground horizon + rotation indicator, branches on whether an HDRI is connected
- `operators/gpu/transform3d/transform3d.cpp` — unit-cube wireframe rotated by the TRS Euler angles, scaled by the scale params, badge flags the dominant component (`R` / `S` / `T` / `id`)
- `operators/gpu/scene_merge/scene_merge.cpp` — four input slots on the left (first N lit, reading `child_count_`), fan lines to an output dot, `N/4` badge
- `operators/control/orbit_camera/orbit_camera.cpp` — top-down disc with azimuth/elevation polar dot, distance badge, optional target crosshair
- `operators/gpu/particles3d/particles3d.cpp` — fixed-position pseudo-random dot cluster (particles are GPU-only and can't be sampled), coloured by `r/g/b/a`, billboards as round dots vs cuboids as small squares, emission halo when `emission > 0`, `count` badge with `k` suffix

They share the same header layout (label top-left, value-badge top-right, semantic body fills the rest) and the same palette (cool-blue label, gold badge, near-black background). Diverging only in the body keeps them visually related without being template-identical.

**SceneMerge caveat:** `process_gpu` compacts non-null inputs into a packed array, so the thumbnail can show *how many* inputs are connected but not *which slots* (A/B/C/D). If per-slot identity becomes important, the operator needs to track slot provenance separately from `child_count_`.

## Layout

- **Top-left**: label (operator name or primary shape), one line.
- **Top-right**: optional state badge (value, mode, count).
- **Body**: the main visual (3D render or 2D diagram), fills the rest.

## Performance budget

- Target <5 ms per frame on typical hardware. The runtime warns at 16 ms.
- One slow frame at operator creation is expected — shader compile and buffer creation happen lazily on first draw. Steady-state must be clean.
- `thumb3d_gpu::render_mesh()` currently recreates its pipeline/shader/depth-texture every call. That's a helper-level limitation, not your operator's fault; avoid adding more recreation on top of it.
- Don't allocate per frame. Cache geometry, buffers, and any expensive computations. Release them in the destructor.

## Checklist for a new thumbnail

- [ ] `draw_thumbnail()` override present.
- [ ] `VIVID_THUMBNAIL(ClassName)` next to `VIVID_REGISTER`.
- [ ] Null-checks at the top: `ctx`, `device`, buffers, counts.
- [ ] Live meshes → `compute_aabb()` on vertex array, not hardcoded bounds.
- [ ] Honors any semantic color param; otherwise default cool-blue.
- [ ] No per-frame allocation; all resources released in `~Ctor`.
- [ ] Cycle each enum/shape-like param and confirm the thumbnail updates.
- [ ] Check `vivid` log for `slow thumbnail draw` — only the first frame may warn.

## Open questions / follow-ups

- **2D + 3D compositing.** Needed for state badges on proxy-mesh thumbnails (e.g. SDF3D's CSG op glyph) and for Pattern #0 operators that might want a metadata overlay on top of the live output texture. Requires proving the 2D draw API overlays cleanly on top of a 3D render pass that already cleared the target.
- **Shared primitive mesh library.** `Shape3D` and `SDF3D` both generate primitive meshes; once a third operator needs proxies, extract to `include/operator_api/primitives_3d.h`.
- **`compute_aabb()` cleanup.** Several existing pattern-#1 thumbnails hardcode bounds; retrofitting to `compute_aabb()` is a good batch cleanup.
- **Instancer3D bundle-vs-layout label.** The Instancer3D thumbnail always shows its `layout` param in the label, even when a connected bundle overrides it. Show "Bundle" when `custom_inputs[1]` is connected.
- **Confirm Pattern #0 for post-effects.** `SSAO3D`, `DepthOfField3D`, `DepthMask3D`, `Environment3D` (when it has an HDRI connected) are expected to fit Pattern #0 — their output texture is already the best preview. Spot-check when next reviewing these operators and remove any vestigial custom draw_thumbnail overrides.

## Progress

All 27 operators in this package (26 GPU + OrbitCamera) now have a thumbnail pattern assigned:

- **Pattern #0** (default output-texture blit, no custom): Render3D, MeshDraw, SSAO3D, DepthOfField3D, DepthMask3D — 5
- **Pattern #1** (real mesh): Box, Sphere, Shape3D, Grid, Boolean3D, Deformer, Sweep, MeshBuilder, MeshImport — 9
- **Pattern #2a** (proxy mesh): SDF3D — 1
- **Pattern #2b** (instance scatter): InstanceGrid, InstanceNoise, Instancer3D, InstancesFromLanes, PointCloud — 5
- **Pattern #3** (2D diagram): Light3D, Material3D, Environment3D, Transform3D, SceneMerge, OrbitCamera, Particles3D — 7

Total: 5 + 9 + 1 + 5 + 7 = 27 ✓

### Notes on recent additions

- **PointCloud (Pattern #2b):** the operator only has its (x, y) point pairs at `process_gpu` time via `ctx->input_lanes[0]`. To expose them to `draw_thumbnail` (which runs in a different frame phase), shadow the pairs into a `std::vector<float>` member during `process_gpu` and convert on-the-fly to `InstanceData3D` inside `draw_thumbnail` before calling `thumb_instances::draw_scatter`. Don't try to read lanes from the thumbnail context — lane data isn't part of `VividThumbnailContext`.
- **Particles3D (Pattern #3, not #2b):** particles live entirely GPU-side (ping-pong storage buffers), so no CPU scatter is available. The thumbnail uses a fixed pseudo-random arrangement of 16 dots to represent "a particle cloud" — coloured by `r/g/b/a`, sized by the `size` param, squared or rounded per the `shape` enum, with an emission halo when `emission > 0`. This is an honest diagrammatic rendering rather than pretending to show actual particles.
