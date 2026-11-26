//! Function parsing: decode a section of machine code, find the uniform registers
//! and predicates it uses, and decide what the translation needs. The jump tables
//! of the function's attributes.

use std::collections::HashMap;

use crate::decode::{Decoded, Decoder, LETTER_Q, LETTER_R};
use crate::generated::fields::*;
use crate::generated::special::*;

use super::{GATHERED, KEPT, MATRIX_SCRATCH, Refused, is_matrix};

/// The jump tables of a function, from its attributes: for each branch through a register, its
/// offset and the offsets it may lead to, in the order of the record (0x34), which is, for each:
/// the branch, a zero, how many places, the places.
pub(super) fn tables_of(data: &[u8]) -> Result<Vec<(u64, Vec<u64>)>, Refused> {
    let mut out: Vec<(u64, Vec<u64>)> = Vec::new();
    let mut pos = 0;
    while pos + 4 <= data.len() {
        let fmt = data[pos];
        let attr = data[pos + 1];
        let val = u16::from_le_bytes(data[pos + 2..pos + 4].try_into().unwrap()) as usize;
        if fmt == 4 && attr == 0x34 {
            let whole = ((pos + 4 + val).min(data.len()) - (pos + 4)) / 4;
            let words: Vec<u64> = (0..whole)
                .map(|j| u32::from_le_bytes(data[pos + 4 + 4 * j..pos + 8 + 4 * j].try_into().unwrap()) as u64)
                .collect();
            let mut at = 0;
            while at < words.len() {
                if at + 3 > words.len() || words[at + 1] != 0 || at + 3 + words[at + 2] as usize > words.len() {
                    return Err(Refused("a record of jump tables that is not understood".into()));
                }
                let places = words[at + 3..at + 3 + words[at + 2] as usize].to_vec();
                match out.iter_mut().find(|(branch, _)| *branch == words[at]) {
                    Some(entry) => entry.1 = places,
                    None => out.push((words[at], places)),
                }
                at += 3 + words[at + 2] as usize;
            }
        }
        pos += if fmt == 4 { 4 + val } else { 4 };
    }
    Ok(out)
}

// ---- Function parsing ----
pub(super) struct FuncData {
    pub(super) name: String,
    pub(super) registers: u32,
    /// Its jump tables (`tables_of`).
    pub(super) tables: Vec<(u64, Vec<u64>)>,
    pub(super) src: Vec<Decoded>,
    /// The uniform registers it reads or writes, in order.
    pub(super) uniform: Vec<u32>,
    /// The lower one of each two that an instruction takes as one value (a constant of 64 bits,
    /// an address, a product).
    pub(super) pairs: Vec<u32>,
    /// Whether any of its instructions needs registers to compute into (`Ctx::pool`).
    pub(super) needs_pool: bool,
    pub(super) uniform_predicates: Vec<u32>,
    pub(super) highest: u32,
    /// How many scratch registers the expanded instruction sequences need.
    pub(super) scratch_need: u32,
    /// Whether a register for the lane of the thread is needed.
    pub(super) needs_lane: bool,
    /// How many instructions are tensor-core products (HMMA.1688).
    pub(super) product_count: u32,
    /// A symbol of another function points into it: it can be entered elsewhere than at its start.
    pub(super) entered_inside: bool,
    pub(super) ampere: bool,
    pub(super) ampere_ops: HashMap<usize, super::ampere::Operation>,
}

impl FuncData {
    /// The places the branch through a register at `branch` may lead to, if it has a table.
    pub(super) fn table_at(&self, branch: u64) -> Option<&Vec<u64>> {
        self.tables
            .iter()
            .find(|(at, _)| *at == branch)
            .map(|(_, places)| places)
    }
}

pub(super) fn parse_function(
    dec75: &mut Decoder,
    name: &str,
    code: &[u8],
    registers: u32,
    tables: Vec<(u64, Vec<u64>)>,
) -> Result<FuncData, Refused> {
    if !code.len().is_multiple_of(16) {
        return Err(Refused("code that is not a whole number of instructions".into()));
    }
    let words: Vec<u128> = (0..code.len() / 16)
        .map(|i| u128::from_le_bytes(code[i * 16..(i + 1) * 16].try_into().unwrap()))
        .collect();
    let mut src = Vec::with_capacity(words.len());
    for (i, &w) in words.iter().enumerate() {
        match dec75.decode(w) {
            Some(d) => src.push(d),
            None => {
                return Err(Refused(format!(
                    "the instruction {:032x} at 0x{:x} is not one of the sm_75 description",
                    w,
                    16 * i
                )));
            }
        }
    }

    let mut uniform_set: Vec<u32> = Vec::new();
    let mut pairs: Vec<u32> = Vec::new();
    let mut needs_pool = false;
    for d in &src {
        for (first, count) in crate::decode::uniform_operands(d) {
            for r in first..first + count {
                if r != 63 && !uniform_set.contains(&r) {
                    uniform_set.push(r);
                }
            }
            match count {
                1 => {}
                2 => {
                    if !pairs.contains(&first) {
                        pairs.push(first);
                    }
                }
                _ => return Err(Refused(format!("a uniform operand of {count} registers"))),
            }
        }
        let name = d.name;
        if name.starts_with("imad_hi__")
            || (STRIDED.contains(&name) && d.op_or(OP_STRIDE, 0) != 0)
            || (UNIFORM_ADDRESS.iter().any(|(prefix, _)| name.starts_with(prefix)) && d.op_or(OP_RA, 255) != 255)
        {
            needs_pool = true;
        }
    }
    uniform_set.sort_unstable();
    pairs.sort_unstable();

    let mut unif_pred: Vec<u32> = Vec::new();
    for d in &src {
        for &(letter, reg, _) in &d.reads {
            if letter == LETTER_Q && reg != ZERO_Q && !unif_pred.contains(&reg) {
                unif_pred.push(reg);
            }
        }
        for &(letter, reg) in &d.writes {
            if letter == LETTER_Q && reg != ZERO_Q && !unif_pred.contains(&reg) {
                unif_pred.push(reg);
            }
        }
    }
    unif_pred.sort_unstable();

    let highest = src
        .iter()
        .flat_map(|d| {
            d.reads
                .iter()
                .filter(|&&(l, _, _)| l == LETTER_R)
                .map(|&(_, r, _)| r)
                .chain(d.writes.iter().filter(|&&(l, _)| l == LETTER_R).map(|&(_, r)| r))
        })
        .max()
        .unwrap_or(0);

    // Turing's matrix instructions become sequences that go by the lane of the thread, read
    // once when the function is entered, and compute into registers of their own. With a
    // product, four values made of the lane are computed there too (`kept`), and four
    // registers keep the elements of B that the last product gathered.
    let needs_lane = src.iter().any(|d| is_matrix(d.name));
    // Turing's F2FP converts its high half in a register of its own (the pool's first one when
    // the function has a pool), which is placed with the scratch registers.
    let needs_f2fp_temp = src.iter().any(|d| d.name == "f2fp__RRR" && d.op_or(OP_RA, 255) != 255);
    let scratch_need = if needs_lane {
        MATRIX_SCRATCH
    } else if needs_f2fp_temp && !needs_pool {
        1
    } else {
        0
    };
    let product_count = src.iter().filter(|d| d.name == "hmma_x8_").count() as u32;

    Ok(FuncData {
        name: name.to_string(),
        registers,
        tables,
        src,
        uniform: uniform_set,
        pairs,
        needs_pool,
        uniform_predicates: unif_pred,
        highest,
        scratch_need,
        needs_lane,
        product_count,
        entered_inside: false,
        ampere: false,
        ampere_ops: HashMap::new(),
    })
}

/// Where the uniform registers of a function go, or of functions that call one another: ordinary
/// registers from `base` on, the first one that none of them uses. `uniform`: the registers, in
/// order; `pairs`: the lower one of each two that an instruction takes as one value of 64 bits, in
/// order; `pool`: whether two registers are wanted for what the translation computes
/// (`Ctx::pool`); `saves`: how many registers are wanted to save predicates in (`stand_ins`).
///
/// The ordinary register of each, the first of the two registers of the pool, the first of the
/// registers to save predicates in, the register of the lane, the first of the scratch registers
/// and the first register left free. Two registers that are one value stay side by side, the lower
/// one even: all native code of both architectures has them so. The others fill what that leaves:
/// at most one register is lost.
pub(super) type Placement = (
    HashMap<u32, u32>,
    Option<u32>,
    Option<u32>,
    Option<u32>,
    Option<u32>,
    Option<u32>,
    Option<u32>,
    u32,
    HashMap<u32, u32>,          // rows_at
    HashMap<&'static str, u32>, // kept_rows
    HashMap<u32, u32>,          // hold_at
);

#[allow(clippy::too_many_arguments)]
pub(super) fn placement(
    base: u32,
    uniform: &[u32],
    pairs: &[u32],
    pool: bool,
    saves: u32,
    lane: bool,
    scratch: u32,
    product: bool,
    rows: Option<&std::collections::BTreeMap<u32, u32>>,
    row_kept: &[&'static str],
    held: &[u32],
) -> Result<Placement, Refused> {
    if pairs.iter().any(|u| pairs.contains(&(u + 1))) {
        return Err(Refused("uniform registers in pairs that overlap".into()));
    }
    let mut singles = uniform
        .iter()
        .copied()
        .filter(|&u| !pairs.contains(&u) && !(u > 0 && pairs.contains(&(u - 1))));
    let mut place = HashMap::new();
    let mut free = base;
    let mut pool_at = None;
    if (pool || !pairs.is_empty()) && !free.is_multiple_of(2) {
        if let Some(u) = singles.next() {
            place.insert(u, free);
        }
        free += 1;
    }
    for &u in pairs {
        place.insert(u, free);
        place.insert(u + 1, free + 1);
        free += 2;
    }
    if pool {
        pool_at = Some(free);
        free += 2;
    }
    for u in singles {
        place.insert(u, free);
        free += 1;
    }
    let save_at = (saves > 0).then_some(free);
    free += saves;
    // (`held`: the uniform predicates that some life holds in a register: one each.)
    let mut hold_at = HashMap::new();
    for &q in held {
        hold_at.insert(q, free);
        free += 1;
    }
    // The lane takes the odd register before the ones that start on an even one, when there is
    // one: a product then has the seventeen registers it had when each product computed
    // everything again.
    let mut lane_at = None;
    if lane && free % 2 == 1 {
        lane_at = Some(free);
        free += 1;
    }
    let gathered_at = if product {
        free += free % 2;
        let at = free;
        free += GATHERED;
        Some(at)
    } else {
        None
    };
    let scratch_at = if scratch > 0 {
        free += free % 2;
        let at = free;
        free += scratch;
        Some(at)
    } else {
        None
    };
    if lane && lane_at.is_none() {
        lane_at = Some(free);
        free += 1;
    }
    let kept_at = if product {
        let at = free;
        free += KEPT;
        Some(at)
    } else {
        None
    };
    // Rows and row_kept: after everything else, so a function without pairs has the same registers.
    let mut kept_rows = HashMap::new();
    for &name in row_kept {
        kept_rows.insert(name, free);
        free += 1;
    }
    let mut rows_at = HashMap::new();
    if let Some(rows) = rows {
        for (&rd, &count) in rows {
            if !free.is_multiple_of(2) {
                free += 1;
            }
            rows_at.insert(rd, free);
            free += count;
        }
    }
    if free > 253 {
        // The register count is two more than the registers used, and has eight bits.
        return Err(Refused("more registers than there are".into()));
    }
    Ok((
        place,
        pool_at,
        save_at,
        lane_at,
        kept_at,
        gathered_at,
        scratch_at,
        free,
        rows_at,
        kept_rows,
        hold_at,
    ))
}
