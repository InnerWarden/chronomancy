# Chronomancy

**Detect firmware rootkits and kernel hooks by measuring time.** No TPM, no hardware, no kernel module required.

Implements two published research techniques in Rust:

- **MITRE BIOS Chronomancy** (2013) — firmware timing attestation via CPU cycle counters
- **Trace of the Times** (arXiv:2503.02402, 2025) — kernel function timing anomaly detection with 98.7% F1 score

## How it works

A rootkit adds code. Code takes time. By measuring how long known operations take, we detect the presence of extra code — even if we can't see it.

```
Normal:     filldir64() → 500ns
With hook:  filldir64() → rootkit_filter() → filldir64() → 2500ns
                                                              ↑ detectable shift
```

## Quick Start

```rust
use chronomancy::timing;

// Measure a CPU-bound workload 100 times
let profile = timing::measure(timing::Workload::CpuBound, 100);

println!("Median: {} cycles", profile.median_cycles);
println!("Jitter: {:.1}x", profile.jitter_ratio);
println!("Outliers: {}", profile.outlier_count);

// Jitter > 10x = firmware interception (SMI)
// Jitter > 3x = possible kernel hooking
```

### Trace of the Times (rootkit detection)

```rust
use chronomancy::trace_of_times::*;

// Build baseline from normal execution
let training: Vec<Vec<TimingBatch>> = collect_normal_batches();
let model = build_model(&training);

// Check current execution against baseline
let test_batch = collect_current_batch("filldir64");
let analysis = detect_anomaly(&test_batch, &model.functions[0], DEFAULT_THRESHOLD);

if let Some(a) = analysis {
    if a.anomalous {
        println!("ROOTKIT DETECTED on filldir64! p={:.2e}", a.p_value);
    }
}
```

## Cycle Counter Support

| Architecture | Instruction | Resolution |
|---|---|---|
| x86_64 | `RDTSC` / `RDTSCP` | CPU clock cycles |
| aarch64 | `MRS CNTVCT_EL0` | Counter-timer ticks |

Both with serialization barriers (`CPUID` on x86, `ISB` on ARM) for accurate measurement.

## Statistical Engine

The Trace of the Times engine uses:
- **Quantile extraction** (9 equidistant positions, 0.11–0.95)
- **Mahalanobis distance** with full covariance matrix
- **Chi-squared p-value** via Wilson-Hilferty approximation
- **Per-quantile z-scores** as fallback

Threshold: p < 10⁻⁸ (configurable)

## Install

```toml
[dependencies]
chronomancy = { git = "https://github.com/InnerWarden/chronomancy" }
```

## References

- [BIOS Chronomancy: Fixing the Core Root of Trust (MITRE, 2013)](https://www.mitre.org/sites/default/files/publications/bios-chronomancy.pdf)
- [Trace of the Times: Rootkit Detection through Temporal Anomalies (arXiv:2503.02402, 2025)](https://arxiv.org/abs/2503.02402)

---

Part of the [InnerWarden](https://innerwarden.com) security ecosystem.
