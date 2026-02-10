use std::str::FromStr;

#[cfg(feature = "cpsat")]
use cp_sat::proto::CpSolverStatus;

/// ConstChoice encoding mode: let solver choose (Optimize), force AddImm (Flow), or force Load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstChoiceMode {
    Optimize,
    ForceAddImm,
    ForceLoad,
}

impl Default for ConstChoiceMode {
    fn default() -> Self {
        Self::Optimize
    }
}

impl FromStr for ConstChoiceMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "optimize" => Ok(Self::Optimize),
            "add_imm" => Ok(Self::ForceAddImm),
            "load" => Ok(Self::ForceLoad),
            _ => Err(format!(
                "invalid const_choice '{}', expected optimize, add_imm, or load",
                s
            )),
        }
    }
}

/// SelectChoice encoding mode: let solver choose (Optimize), force FMA (VALU), or force vselect (Flow).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectChoiceMode {
    Optimize,
    ForceFma,
    ForceVselect,
}

impl Default for SelectChoiceMode {
    fn default() -> Self {
        Self::Optimize
    }
}

impl FromStr for SelectChoiceMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "optimize" => Ok(Self::Optimize),
            "fma" => Ok(Self::ForceFma),
            "vselect" => Ok(Self::ForceVselect),
            _ => Err(format!(
                "invalid select_choice '{}', expected optimize, fma, or vselect",
                s
            )),
        }
    }
}

/// VALU shift mode: let solver choose (Optimize), force shift to ALU, or force stay on VALU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValuShiftMode {
    Optimize,
    ForceAlu,
    ForceValu,
}

impl Default for ValuShiftMode {
    fn default() -> Self {
        Self::Optimize
    }
}

impl FromStr for ValuShiftMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "optimize" => Ok(Self::Optimize),
            "alu" => Ok(Self::ForceAlu),
            "valu" => Ok(Self::ForceValu),
            _ => Err(format!(
                "invalid valu_shift '{}', expected optimize, alu, or valu",
                s
            )),
        }
    }
}

/// Optional constraint group: redundant/helper constraints that can be disabled for debugging.
/// Each variant corresponds to a named group in the CP-SAT model (see model.rs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptionalConstraintGroup {
    /// makespan_ge_load_lb, makespan_ge_store_lb (redundant with domain + cumulatives).
    RedundantLinearBounds,
    /// Linear constraint: 6*makespan >= valu_count - shift + 2*select_fma.
    CapacityValu,
    /// Linear constraint: 12*makespan - 8*shift >= alu_count.
    CapacityAlu,
    /// Linear constraint: makespan + z - y >= flow_base.
    CapacityFlow,
    /// Linear constraint: 2*makespan + y >= load_count.
    CapacityLoad,
}

impl FromStr for OptionalConstraintGroup {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "redundant_linear_bounds" => Ok(Self::RedundantLinearBounds),
            "capacity_valu" => Ok(Self::CapacityValu),
            "capacity_alu" => Ok(Self::CapacityAlu),
            "capacity_flow" => Ok(Self::CapacityFlow),
            "capacity_load" => Ok(Self::CapacityLoad),
            _ => Err(format!(
                "invalid optional constraint group '{}', expected: redundant_linear_bounds, capacity_valu, capacity_alu, capacity_flow, capacity_load",
                s
            )),
        }
    }
}

impl std::fmt::Display for OptionalConstraintGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::RedundantLinearBounds => "redundant_linear_bounds",
            Self::CapacityValu => "capacity_valu",
            Self::CapacityAlu => "capacity_alu",
            Self::CapacityFlow => "capacity_flow",
            Self::CapacityLoad => "capacity_load",
        };
        write!(f, "{}", s)
    }
}

/// Result from CP-SAT scheduling
#[derive(Debug, Clone)]
pub struct CpsatResult {
    /// Cycle assignment for each op (indexed by op position in input slice)
    pub cycle_assignments: Vec<i32>,
    /// Total makespan (number of cycles)
    pub makespan: usize,
    /// Solver status
    pub status: CpsatStatus,
    /// Time spent solving (seconds)
    pub solve_time_secs: f64,
    /// Optimality gap (0.0 = optimal)
    pub gap: f64,
}

/// CP-SAT solver status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpsatStatus {
    Unknown,
    ModelInvalid,
    Feasible,
    Infeasible,
    Optimal,
}

#[cfg(feature = "cpsat")]
impl From<CpSolverStatus> for CpsatStatus {
    fn from(status: CpSolverStatus) -> Self {
        match status {
            CpSolverStatus::Unknown => CpsatStatus::Unknown,
            CpSolverStatus::ModelInvalid => CpsatStatus::ModelInvalid,
            CpSolverStatus::Feasible => CpsatStatus::Feasible,
            CpSolverStatus::Infeasible => CpsatStatus::Infeasible,
            CpSolverStatus::Optimal => CpsatStatus::Optimal,
        }
    }
}

impl std::fmt::Display for CpsatStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CpsatStatus::Unknown => write!(f, "unknown"),
            CpsatStatus::ModelInvalid => write!(f, "model_invalid"),
            CpsatStatus::Feasible => write!(f, "feasible"),
            CpsatStatus::Infeasible => write!(f, "infeasible"),
            CpsatStatus::Optimal => write!(f, "optimal"),
        }
    }
}
