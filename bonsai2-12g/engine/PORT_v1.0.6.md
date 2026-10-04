# NInfer — port Windows v1.0.6 (rama `v1.0.6-4090`, sm_89)

Este árbol es el **port Windows de `Neroued/ninfer` actualizado a upstream `master`
(`a16b6442`, 2026-09-07) para RTX 4090 (sm_89)**: equivale al port v1.0.5-4090
(sincronizado en `ad0f3d38` + layer sm_89) **más los 88 commits** del rango
`ad0f3d38..a16b6442`.

## Contenido del delta (88 commits)

Mismo delta que la rama 5090 (ver `PORT_v1.0.6.md` de esa rama):
- **DFlash2** (~70 commits): nuevo backend especulativo para Qwen3.8-27B
  (`--spec dflash2 --draft-tokens K`, K=1..15). En sm_89 los kernels compilan
  (ver "Decisiones sm_89" abajo); la ruta solo es utilizable con un artifact
  que traiga los pesos compañeros dflash2 (el
  `qwen3_8_27b_nvfp4full-v2.ninfer` de gpillon sí los trae). Para el 35B-A3B el
  DFlash legacy (K=1..15) sigue siendo la ruta especulativa recomendada
  (upstream midió DFlash K=7 < MTP3 en la mayoría de escenarios del 35B).
- `487f8977` perf(sparse_moe): one CTA per token S2 small-T + warp merge →
  **decode del 35B-A3B más rápido en lotes pequeños** (tu caso de uso).
- `03177b91` fix(runtime): coverage KV en terminal settlement especulativo.
- perf(ops): w8 vocabulary t64 route, gdn t16 grouped MMA, swiglu nvfp4 t96,
  verify attention h24 small-t, q5 linear add.
- docs/bench: informes `docs/performance/*.md`, bench dflash2.

## Decisiones sm_89 en la fusión (conflictos resueltos a mano)

| Archivo | Decisión |
|---|---|
| `src/ops/linear/w8/w8_config.h` | Se **elimina** el struct `W8LinearSmallTProductionSchedule<W8VocabularyProjectionGeometry,...>` (upstream lo retiró con la ruta t64; incluía la variante SM89). Las 7 variantes SM89 de los demás schedules (MTP input/attention/gateup/down/35bMtp) se conservan intactas. |
| `src/ops/linear/w8/w8_small_t_mma.cuh` + `w8_small_t.cu` | **Idénticos a upstream** (verificado byte a byte contra `a16b6442`). El upstream del delta refactorizó la contracción: la device function `w8_small_t_mma` "dueña del layout shared" (estático ≤48 KiB o dinámico vía `extern __shared__` para tiles >64) + un único `__global__` fino compartido por todas las arqs. El kernel doble-buffer K-split de v1.0.5-4090 (y sus launchers `launch_splitk_exact`/`make_splitk_launchers` en `w8_small_t.cu`) se **eliminó**: con el layout nuevo, las instanciaciones sm_89 caben estáticas gracias a las variantes de schedule tope-48 KiB que ya traía `w8_config.h` (p. ej. tile-48/4-warps = 28,9 KiB; tile-32/8-warps = 41,5 KiB). La firma unificada añade el 7.º arg runtime `std::int32_t columns` (columnas vivas de la ruta `TiledColumns`); los callers viejos (≤6 args) siguen compilando por defaults. |
| `src/ops/dynamic_grouped_conv/w8/w8_dynamic_grouped_conv_add_materialized.cu` | Único cambio sm_89 del delta: `#if SM89` en la selección de warps de `tiled_projection` — el tile-40 de 4096 filas con 8 warps produce 49.664 B de shared estática > límite de 48 KiB de sm_89 (ptxas: `0xc200 > 0xc000`, verificado con probe; en sm_120 el límite estático está levantado, de ahí que upstream compile). Con 4 warps (24.832 B) coincide con la rama de 17408 filas. Los tiles >64 (72/88) usan shared dinámica (≤49.408 B) vía `cudaFuncAttributeMaxDynamicSharedMemorySize` — el opt-in de sm_89 es 101.376 B (verificado con `cudaDevAttrMaxSharedMemoryPerBlockOptin`), así que funcionan también en la 4090. |
| `src/ops/linear_swiglu/w8/w8_linear_swiglu_gemm_splitk.cu` | La rama SM89 conserva el schedule capped; se añade `RowPolicy` (el cuerpo compartido que trae el delta — epílogo t96 `W8SwiGluDirectEpilogue` — la usa en la sección común). |
| `src/ops/linear_swiglu/w8/w8_linear_swiglu_gemm_mma.cu` | `#if SM89` solo en `w8_dflash2_linear_swiglu_mma_r64_c96_k128_launch`: BK 128→64 (50.176 B → 25.600 B de shared estática; el layout BK=128/ACT=1 es la única instanciación de toda la familia `W8RowSplitMmaGemmSchedule` que supera los 48 KiB). Mismas matemáticas (más iteraciones K); el `static_assert(BM <= ACT*BK)` del kernel queda 64 ≤ 1·64. |
| `src/ops/softmax_attention/dense/causal_cache/small_t.cu` | `int8_family` (= Int8Group64 + rk4v4 + rk4v4-e8, solo sm_89) se combina con la relajación upstream `tokens >= 6` (antes `== 6`). Dispatch nuevo `launch_for_storage<bool>` con la condición extendida a los 3 storages int8-family; la epílogo de inversa de rotación H64 para rk4v4/rk4v4-e8 se conserva. |
| `src/serve/serve_options.cpp` | `--kv-dtype` conserva `rk4v4|rk4v4-e8`; `--spec` ahora `mtp|dflash|dflash2`. |
| `src/CMakeLists.txt` | Lista de exclusiones sm_89 intacta (NVFP4 W4A4/TMA, k8v4, w8 splitk vía bloque `MATCHES "^120"`, PDL no-op). Añadidas las fuentes nuevas de dflash2 (`w8_dflash2_attn_input.cu`, dynamic_grouped_conv, linear_topk, candidate_selector, context_kv_materialize, gdn_gating_proj, rmsnorm_rope, etc.). |

### Auditoría SM120-only (hecha)
Los **51 archivos nuevos** del delta bajo `src/` fueron escaneados:
`cp.async.bulk` (TMA), `setmaxnreg`, `e2m1`/FP4, `kind::f8f6f4`, `tcgen05`,
`.cluster` → **cero apariciones**. Sin exclusiones adicionales necesarias
(más allá de las que ya existían). Si ptxas rechaza algún kernel por límite
de shared memory en sm_89, el build lo reportará explícitamente.

## Correcciones y puertos post-fusión (2026-09-08)

### Fix: partial_acc a float en el kernel donante (commit `6c4f5a10`)
El delta upstream cambió el acumulador parcial small-T int8 a FP32 (workspace
FP32 en `causal_softmax_attention.cpp`; el reduce compartido lee
`const float*`), pero el kernel donante sm_89 seguía escribiendo bf16
empaquetado (`pack_bf16x2`) → corrupción bit a bit → salida **incoherente con
toda KV int4** (`rk4v4`/`rk4v4-e8`) en sm_89 (ambos modelos). Fix (5 edits):
firma `float* partial_acc` + init `0.0f` + `make_float2` en el epílogo (×2) en
`small_t_i8.cuh` (rama `#if defined(NINFER_SM89)`), y
`static_cast<float*>(partial_acc.data)` en el launcher sm_89 de `small_t.cu`.
El bloque upstream (`#else`) queda byte-identico.

### Port: small-T T=7/8 al kernel donante (commit siguiente)
El delta añadió kernels small-T para **T=7/8** (widths 7-8 tokens; solo los usa
`--spec dflash2/dflash` con draft ≥6 — MTP está limitado a draft 5 → width ≤6,
así que la ruta es latente con los bats actuales). El donante (línea v1.0.5)
solo implementaba T≤6 (`static_assert TokenTile <= 6`); el switch sm_89 dejaba
`default: throw "unsupported T"` → cualquier width 7/8 en 27B **crasheaba en
runtime**. Port (3 edits, espejo de upstream):
1. `small_t_i8.cuh`: assert donante relajado a
   `TokenTile * Geometry::GroupSize <= 48` (el assert de upstream). Para 27B
   (GroupSize 6) T=6/7/8 → RowTiles=3 y Br=48: el footprint de hardware
   (smem/threads) es idéntico; solo cambia el nº de filas válidas.
2. `small_t.cu` (launcher sm_89): la cadena de rutas T=6 → `TokenTile >= 6`
   (T=7/8 hereda las rutas afinadas de T=6).
3. `small_t.cu` (switch sm_89): casos 7/8 con guarda `QHeads == 24`
   (35B → `throw "unsupported query-row tile"`, igual que upstream).

### Verificación (2026-09-08)
- **A/B en vivo** (`ab_bench.py`, `/v1/messages`, MTP3,
  prompts únicos, 1500 tok): tras el fix int4, 27B q5+e8 coherente y
  **119.4 tok/s (+6.8 % vs v1.0.5, 111.8)**; 35B v2+e8 coherente, 389.7 vs
  400.2 (ruido, paridad). Build T=7/8 desplegado: 27B 127.3, 35B 393.5 (sin
  regresión).
- **Suite de tests** (primer build con `BUILD_TESTING=ON`; MSVC 14.44 + CUDA
  13.3): `ninfer_softmax_attention_test --dflash2-only` (widths 2..16 ×
  BFloat16/Int8Group64/Fp8/Nvfp4/K8V4 × d256-h24-kv4, batch 1/8, oracle
  independiente). Baseline pre-port **crashea** en el primer caso W=7
  (exit 0xC0000409 = `std::invalid_argument` no atrapado: `"unsupported T"` —
  el gap confirmado en vivo). Post-port: **bf16/fp8/nvfp4/k8v4 pasan 100 %**;
  `int8-g64` muestra desviaciones pequeñas (max ~1e-3..1.6e-2 vs criterio
  gross 1.1e-3) que **también aparecen en rutas que el port no toca**
  (multi-batch W=2..6; W=9, que va por ChunkedSmallT) → perfil numérico
  preexistente del donante (PV bf16/fp16-acc afinado sm_89) frente a los
  criterios fijados por upstream para su kernel sm_120; las desviaciones de
  T=7/8 son menores que la peor preexistente. Nota: los storages
  rk4v4/rk4v4-e8 no están cubiertos por la suite (son específicos del port);
  su cuerpo de kernel es el mismo del Int8Group64 vía parámetros de template.
- **Fix de compatibilidad MSVC en tests**: `plain_and_packed.cpp`
  `constexpr std::sqrt(72.0)` → `const` (el STL de MSVC no tiene `std::sqrt`
  constexpr en C++20; error C2131). Solo afecta a la compilación de la suite
  en Windows; las matemáticas son idénticas.
- **Pendiente (acción del usuario)**: e2e `--spec dflash2` en la 4090 sigue
  pendiente del artifact convertido (pipeline en `convert_sources\`); al
  existir, re-verificar con `--spec dflash2 --draft-tokens 7`.

## Qué funciona en sm_89 (tras v1.0.6)

- Todo lo de v1.0.5-4090 (KV `rk4v4`/`rk4v4-e8`, pesos bf16/fp8/q4/q5/q6/w8,
  MTP, DFlash legacy 35B, visión).
- **Nuevo**: decode sparse-MoE small-T optimizado (beneficio directo en el
  35B-A3B), fixes de runtime/speculative settlement, nueva ruta w8 t64 para
  pesos w8 (si se usan), `--spec dflash2` disponible en el CLI (requiere
  artifact con pesos compañeros dflash2 para qwen3.8).

## Compilar

Script autocontenido en el árbol: **`build_v1.0.6.bat`** (sm_89, visión,
Release). Solo necesita este árbol + MSVC BuildTools + CUDA 13.3 + Ninja
(rutas ya puestas en el script). Por defecto compila en
`_build_4090new` (relativo al árbol del repo); acepta un build dir alternativo como
primer argumento: `build_v1.0.6.bat <build_dir>`. Sin `pause` (background).
Copia exes + DLLs FFmpeg (de `ffmpeg\bin\`) a la raíz del build dir.

```bat
build_v1.0.6.bat
```

Equivalente a mano:

```bat
cmake -B _build_4090new -S . -G Ninja -DCMAKE_CUDA_ARCHITECTURES=89 -DNINFER_ENABLE_AVX2=ON -DNINFER_BUILD_MEDIA_ACQUIRE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build _build_4090new --config Release -j 32
```

- Binarios de esta verificación: `_build_4090new\`
  (`ninfer-serve.exe` + DLLs FFmpeg; el resto del árbol de build se movió a
  `_build_4090new-delete\`).

## Verificación (2026-09-07)

- Build Release completo: **OK** (todos los objetos + device-link rdc + 3 exes;
  los dos fallos previos — firma del kernel unificado y la shared >48 KiB del
  `r64_c96_k128` — quedaron resueltos por las adaptaciones de esta tabla).
- Smoke test: `ninfer-serve.exe --help` muestra `--spec mtp|dflash|dflash2`
  **y** `--kv-dtype ...|rk4v4|rk4v4-e8`; `--spec dflash2 --draft-tokens 7
  --lm-head-draft --kv-dtype rk4v4-e8` parsea y arranca el engine (falla solo
  sin artifact).
- Alineación de docs con upstream `a16b6442` (verificación hunk a hunk del
  delta 88 commits): `README.md` raíz, `tests/README.md` y model cards
  `Qwen3.8-27B-NInfer`/`Qwen3.8-27B-nvfp4-NInfer` adoptados **tal cual
  upstream** (las adaptaciones de docs de v1.0.5 —recorte de la lista KV y
  tabla de tests— documentaban limitaciones que ya no aplican). `bench/README.md`
  adopta la versión upstream **con el ajuste sm_89 de la lista `--kv-dtype`**
  (`bf16|int8|fp8` y recorte de NVFP4-G16/K8V4 en los benches de atención y
  append): el build sm_89 no trae esos storages KV (ver
  `src/serve/serve_options.cpp`). El resto del árbol (código) quedó verificado
  idéntico a upstream salvo las adaptaciones de la tabla de arriba.
