# NInfer — port Windows v1.0.5

Este árbol es el **port Windows de `Neroued/ninfer` actualizado a upstream `master`
(2026-09-03)**: equivale al port v1.0.4 (sincronizado en `5973313d`, 2026-09-02)
**más los 4 commits** que faltaban:

| Commit | Cambio |
|---|---|
| `550d0ac3` | feat(serve): timing + progreso de llama.cpp (prompt progress) |
| `6e2786c5` | fix(logging): logs operativos legibles (rework + `pretty_format`) |
| `719d56ef` | fix(frontend): preservar intención de tool-call estructurado |
| `e3aeaf8c` | fix(serve): preservar *anthropic thinking* tras reinicios |

## Qué se conservó (adaptaciones MSVC/Windows)
- `unsigned __int128` → `std::uint64_t` (MSVC no soporta `__int128`)
- Headers POSIX → Windows con `#ifdef _WIN32` (`process.h`, `<random>`, `intrin.h`)
- `src/CMakeLists.txt`: FFmpeg/LibCurl condicionales
- `nvfp4_w4a4_tma.cuh`: `alignas(64)`
- READMEs localizados (raíz / `tests` / `tools/bench`) conservados tal cual v1.0.4

## Nota sobre `e3aeaf8c`
Borra `src/serve/anthropic_thinking_signature.{cpp,h}`; la lógica de firma migró a
`src/serve/anthropic_messages_response.cpp` usando `std::random_device` (seguro en MSVC).

## Estado de alineación (verificado)
- 48 archivos del delta = **idénticos a upstream master** (sin desfase).
- 13 archivos de código difieren de master **solo** por la adaptación Windows.
- 3 archivos nuevos + 2 eliminados según upstream.

## Compilar
Igual que v1.0.4: `build_windows.bat` (texto) o `build_vision_windows.bat` (visión).
No requiere cambios: los fixes no añaden dependencias nuevas.

## Build sm_89 (RTX 4090) — branch `v1.0.5-4090` (2026-09-03)

Este branch porta el árbol upstream (`a140e7ae`) a **sm_89** (RTX 4090).
`CMAKE_CUDA_ARCHITECTURES` se restringe a `120a` o `89` (resto = error duro).

```bat
_cfg_4090.bat   :: configure build_4090 (Release, arch=89)
_build_4090.bat :: ninja -C build_4090 -j 32
```

### Cambios para sm_89 (y por qué)
| Cambio | Razón |
|---|---|
| `src/CMakeLists.txt`: excluir NVFP4 W4A4 (TMA) (`*_w4a4.cu` ×5) y la lib no-RDC `ninfer_nvfp4_non_rdc` | TMA (`cp.async.bulk`) y `setmaxnreg` son solo SM120 |
| Excluir kernels KV/atención k8v4 y NVFP4 (`k8v4_launch.cu`, `nvfp4_launch.cu`, `prompt_k8v4.cu`, `small_t_k8v4.cu`, `small_t_nvfp4.cu`) | k8v4 usa MMA mixto `kind::f8f6f4` y NVFP4 usa `e2m1` (FP4): solo Blackwell |
| Excluir GEMM w8 splitk/small-T (7 `.cu`) | usan >48 KiB de shared memory estática (límite sm_89; ptxas solo lo acepta en SM120) |
| `src/ops/common/mma.cuh`: `mma_fp8_e4m3` usa `kind::f8f6f4` en sm_120 y `mma.sync` FP8 **sin kind** en sm_89 | FP8×FP8 (e4m3) funciona en Ada; el kind `f8f6f4` no existe ahí |
| `src/core/pdl.cuh`: PDL (`griddepcontrol`) no-op en <sm_90 | el overlap de lanzamientos no existe; sin él el stream se serializa (semántica invariante) |
| Stubs `src/ops/{nvfp4_w4a4,sm120_kv,w8_sm120}_stubs.cpp` (23 símbolos) | mantienen la capa de dispatch linkable; error runtime claro |

### Qué funciona en sm_89 / qué no
- **KV dtypes**: `bf16`, `int8`, `fp8`, `rk4v4`, `rk4v4-e8` → OK. `nvfp4`/`k8v4` → error runtime (stub).
- **Pesos**: bf16/fp8/q4/q5/q6/w8 → OK; NVFP4 (W4A4) → error runtime. GEMM w8
  splitk/small-T → **OK** (ver port w8 abajo).
- Contexto largo en 24 GB: con `--kv-dtype rk4v4-e8` **262 144 tokens verificado en vivo**
  (KV ~5.32 GiB; el fp8 a 262 k = ~10.49 GiB no cabe tras los 16.9 GiB de pesos del
  checkpoint q5 27B). `rk4v4`/`rk4v4-e8` solo existen en este build (sm_89).

### Verificado
Build Release sm_89 completa en verde (compilación + device link RDC + host link);
`ninfer-serve.exe --help` OK (staging `_build_4090-main\`).

### Port w8 sm_89 (completado 2026-09-03)
El follow-up pendiente (w8 splitk/small-T en 4090) se resolvió **sin** shared memory
dinámica, replicando la estrategia ya probada en producción del fork udps: variantes
`#if defined(NINFER_SM86) || defined(NINFER_SM89)` que dejan intacta la rama 120a.

| Cambio | Detalle |
|---|---|
| `w8_config.h` | 8 schedules con warps/blocks/`ScaleAccess` capados para no exceder 48 KiB de shared estática (ej. `KWarps = T<=24 ? 8 : 4`, `MinBlocks` 2–4, siempre `Shared`) |
| `w8_small_t_mma.cuh` | kernel doble-bufado con K-split opcional (`KSplits` por `blockIdx.y` + `atomicAdd`), `static_assert(sizeof(SharedStorage) <= 49152)`; el kernel de un buffer queda en `#else` (120a) |
| `w8_small_t.cu` | MTP input y attention-output projections → `launch_splitk_exact<…, KSplits=2>` (cero previa `cudaMemsetAsync`) |
| `w8_rowsplit_gemm_splitk.cu` | `launch_medium` excluido; `launch_w8_exact_t_composite` cubre T≥33 (sin tope 127); `launch_w8_dflash_medium`/`launch_w8_medium_splitk_c144` delegan en composite |
| `w8_linear_add_gemm_splitk.cu`, `w8_pair_gemm_splitk.cu` | `launch_medium` excluido; el launcher medium hace chunks de exact-T (48/32) + cola `decode_r16` (o `simt_r8_c4` en linear_add k=4096) |
| `w8_linear_swiglu_gemm_splitk.cu` | schedule con `MinBlocks=4` y `ScaleAccess=Shared` |
| `CMakeLists.txt` (root) | `add_compile_definitions(NINFER_SM89=1)` cuando `CMAKE_CUDA_ARCHITECTURES` = 89 |
| `src/CMakeLists.txt` | los 7 `.cu` w8 se compilan en **ambas** archs; `w8_sm120_stubs.cpp` queda sin referencias |
| `targets/qwen3_6/impl/runtime/layouts_impl.h` | gate cc relajado de `==120` a `==120 \|\| ==89` ("requires compute capability 12.0 or 8.9") |

**Verificado en vivo (2026-09-03, RTX 4090):** checkpoint `qwen3_8_27b.ninfer` (q5,
16.9 GiB, con w8 MoE) + `--max-context 131072 --kv-capacity 131072 --kv-dtype fp8
--spec mtp --draft-tokens 3 --lm-head-draft --vision`: arranque en 8.3 s (CUDA graphs
612 ms), margen VRAM 132 MiB; 3 peticiones `POST /v1/messages` reales (HTTP 200,
salidas coherentes, thinking conservado) a ~105 tok/s (build Debug, 4090 a 100 %).

### Port KV rk4v4 / rk4v4-e8 (E8) + `--vision-max-tokens` (completado 2026-09-03)

Porta la cuantización de la KV cache 4-bit del fork `sergiuszm/ninfer-4090`
(rama `rtx4090-port`, HEAD `914e050`; commits `5df406b` E8 Conway-Sloane lattice y
`f7be317` append packed — línea udps): `--kv-dtype rk4v4` (K y V int4 con
rotación Hadamard H64) y `--kv-dtype rk4v4-e8` (K int4 empaquetado sobre la
retícula E8 de Conway-Sloane + V int4). La retícula solo interviene al
**codificar** (append); el decode des-empaqueta i4 plano.

| Cambio | Detalle |
|---|---|
| `include/ninfer/types.h` | `KvCacheStorage::RotatedInt4KeyInt4ValueGroup64` (rk4v4) y `::RK4V4E8` (al final del enum, ABI-estable); `EngineOptions::vision_max_tokens` (defecto 8192) |
| `core/paged_kv_storage.h` | layout simétrico `{U8, 128, FP16, 4}` para ambos modos (planning/validation resuelven solos) |
| `ops/kernel/e8_lattice.cuh`, `e8_root_codec.cuh` (nuevos) | codecs del donante (drop-in) |
| `ops/kv_cache/int8_g64_codec.cuh` | i4 codec (`kv_cache_i4_*`, pack/unpack), `kv_cache_hadamard64`, `kv_cache_inverse_rotate_output_kernel` |
| `causal_cache/{small_t,prompt}_i8.cuh` | `#if defined(NINFER_SM89)` → kernel retuneado del donante (paired producers `ColSplit=2`, `__maxnreg__ 128` en Ada, PV fp16-acc, byte-permute de V); `#else` → kernel upstream original |
| `causal_cache/small_t.cu`, `prompt.cu`, `kv_cache/append/{kernel.cuh,launch.cu}` | dispatch de la familia int8 por flags `PackedV/RotateK/RotateV/PackedK/E8Lattice/E8Root` mapeados al enum; inverse-rotate H64 post-reduce; `cache_v` como `uint8_t*` |
| `ops/common/mma.cuh` | `mma_f16_f16acc` (m16n8k16 f16-accumulate) para el PV sm_89 |
| `serve/serve_options.*` | `--kv-dtype rk4v4\|rk4v4-e8` y `--vision-max-tokens N` (positivo; activa `enable_vision`) |
| `targets/qwen3_6/*` | gate de arranque (los modos packed requieren cc 8.9) + cableado `vision_max_tokens`: `StartupFeatures` → `FrontendOptions` → tope del scratchpad de vision en `frontend.cpp`/`layouts_impl.h` |
| `gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu` | fix: la residency del launch cooperativo se clamp-ea con `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (las constantes sm_120a 2–4 CTAs/SM no valen en Ada, donde los splits de 512 hilos caben 1 CTA/SM; sin el clamp, prefills ≥2 k tokens fallaban con `cudaErrorCooperativeLaunchTooLarge`) |

**Verificado en vivo (2026-09-03, RTX 4090):** `qwen3.8-27b` + `--max-context 262144
--kv-capacity 262144 --kv-dtype rk4v4-e8 --vision --spec mtp`: arranque 8.4 s,
KV 4096/4096 páginas, runtime 5.32 GiB (margen planning 306 MiB); NIAH de 125 k
tokens a 50 % y 90 % de profundidad (needle recuperada en ambos); inferencia real
~104–107 tok/s con MTP; generación de 256 tokens sobre contexto profundo; visión
OK; VRAM 23.0/24.5 GiB. Prefill E8 ~2.05 k tok/s.
