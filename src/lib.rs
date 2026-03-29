//! Chronomancy — timing-based integrity attestation.
//!
//! Detect firmware rootkits and kernel hooks by measuring CPU cycle timing.
//! No hardware required — works on any x86_64 or aarch64 system.
//!
//! Implements two published techniques:
//! - **MITRE BIOS Chronomancy** (2013): firmware timing attestation
//! - **Trace of the Times** (arXiv:2503.02402, 2025): kernel function timing anomaly detection (98.7% F1)
//!
//! # Example
//!
//! ```rust
//! use chronomancy::timing;
//!
//! // Measure timing profile
//! let profile = timing::measure(timing::Workload::CpuBound, 100);
//! println!("Median: {} cycles, jitter: {:.1}x", profile.median_cycles, profile.jitter_ratio);
//!
//! // High jitter = firmware interception or kernel hooking
//! if profile.jitter_ratio > 10.0 {
//!     println!("WARNING: timing anomaly detected!");
//! }
//! ```

pub mod timing;
pub mod trace_of_times;

use serde::Serialize;

/// Result of a check (shared between timing and trace_of_times).
#[derive(Debug, Clone, Serialize)]
pub struct CheckResult {
    pub id: &'static str,
    pub name: &'static str,
    pub status: CheckStatus,
    pub confidence: f64,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CheckStatus {
    Secure,
    Warning,
    Critical,
    Unavailable,
}

/// Build confidence score from impact and certainty.
pub fn confidence(impact: f64, certainty: f64) -> f64 {
    (impact * certainty).clamp(0.0, 1.0)
}
