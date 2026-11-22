//! Per-core issue resources. Instruction selection, latency, and packet legality
//! are target-specific and require more than these slot counts.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSlot {
    Load,
    Store,
    Scalar,
    Vector,
    Flow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    AnthropicPerfEval,
    GaudiTpc,
}

impl Architecture {
    /// Maximum instructions of a class per bundle. Zero means the slot is absent.
    pub const fn issue_capacity(self, slot: IssueSlot) -> u32 {
        match self {
            Self::AnthropicPerfEval => match slot {
                IssueSlot::Load | IssueSlot::Store => 2,
                IssueSlot::Scalar => 12,
                IssueSlot::Vector => 6,
                IssueSlot::Flow => 1,
            },
            // Gaudi TPC User Guide, Processor Architectural Overview (LOAD, SPU, VPU, STORE).
            Self::GaudiTpc => match slot {
                IssueSlot::Load | IssueSlot::Scalar | IssueSlot::Vector | IssueSlot::Store => 1,
                IssueSlot::Flow => 0,
            },
        }
    }
}

/// Anthropic performance evaluation ISA constants from problem.py.
pub mod anthropic_perf_eval {
    pub const VLEN: usize = 8;
    pub const SCRATCH_SIZE: usize = 1536;
    pub const DEBUG_SLOTS: usize = 64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Engine, MachineConfig, SLOT_LIMITS};

    #[test]
    fn target_issue_capacities() {
        use IssueSlot::*;
        let slots = [Load, Store, Scalar, Vector, Flow];
        assert_eq!(
            slots.map(|s| Architecture::GaudiTpc.issue_capacity(s)),
            [1, 1, 1, 1, 0]
        );
        assert_eq!(
            slots.map(|s| Architecture::AnthropicPerfEval.issue_capacity(s)),
            [2, 2, 12, 6, 1]
        );
        let machine = MachineConfig::default();
        for (name, limit) in SLOT_LIMITS {
            let engine = Engine::from_str(name).unwrap();
            assert_eq!(engine.slot_limit(), limit);
            assert_eq!(machine.slot_limit(engine) as usize, limit);
        }
    }
}
