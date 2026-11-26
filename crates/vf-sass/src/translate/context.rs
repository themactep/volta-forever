//! The translation context of a function: the `Ctx` struct (registers of the translation,
//! borrowed registers, predicates) and its methods for the sequences that stand for
//! instructions sm_70 lacks. Also the predicate stand-in assignment (`stand_ins`)
//! and the entry-point analysis (`entries`).

use std::collections::{HashMap, HashSet};

use crate::assemble;
use crate::decode::{Decoded, Decoder, Kind, LETTER_P, LETTER_R, LETTER_U};
use crate::field;
use crate::generated::fields::*;
use crate::generated::special::*;
use crate::rule;

use super::borrowed::Borrowed;
use super::liveness::compute_liveness;
use super::matrix::translate_matrix_load;
use super::parse::FuncData;
use super::util::{find_relative, read_announced};
use super::{
    ARRIVES, Item, LATE_WRITER_KINDS, MATRIX_SCRATCH, Refused, RegSet, SEQUENCE, regset_set, stall_of, with_stall,
};

pub(super) struct Ctx<'a> {
    pub(super) dec70: &'a mut Decoder,
    pub(super) place: &'a HashMap<u32, u32>,
    /// The first of the two registers of the pool (`Ctx::pool`), if the function has them.
    pub(super) pool_at: Option<u32>,
    /// The first of the registers to save predicates in, if the function has any.
    pub(super) save_at: Option<u32>,
    /// The register that holds the lane of the thread (0 to 31) all through the function.
    pub(super) lane_at: Option<u32>,
    /// The first of the registers of what a function with a product keeps of the lane (`kept`).
    pub(super) kept_at: Option<u32>,
    /// The first of the registers that keep the elements of B a product gathered.
    pub(super) gathered_at: Option<u32>,
    /// The first of the scratch registers for the sequences of matrix instructions.
    pub(super) scratch_at_reg: Option<u32>,
    /// The instructions of the function.
    pub(super) src: &'a [Decoded],
    /// The instructions that are not only entered from the one before them (`entries`); None
    /// when that cannot be known.
    pub(super) entered: Option<HashSet<usize>>,
    /// The product whose gathering of B the registers kept for that hold: its index and its
    /// register of B (`kept_gathering`).
    pub(super) holds: Option<(usize, u32)>,
    pub(super) next_register: u32,
    /// The predicates of the function (`stand_ins`).
    pub(super) predicates: &'a Predicates,
    /// The instruction being translated: a uniform predicate has the ordinary one of its life there.
    pub(super) at: usize,
    /// The plan for pairs on whole rows, or None.
    pub(super) q: Option<&'a crate::pairing::Plan>,
    /// {destination of whole-row loads: first register of the translation for its rows}.
    pub(super) rows_at: HashMap<u32, u32>,
    /// {name of a kept value: register}.
    pub(super) kept_rows: HashMap<&'static str, u32>,
    /// {destination: how many registers of the translation}.
    pub(super) rows_n: std::collections::BTreeMap<u32, u32>,
    /// Whether rows are aligned on 16 bytes.
    pub(super) aligned_rows: bool,
    /// {uniform predicate: register that holds it between the instructions that name it}.
    pub(super) hold_at: &'a HashMap<u32, u32>,
    /// The registers of its own that the translation may take (`Borrowed`), or None.
    pub(super) borrow: Option<&'a mut Borrowed>,
    /// The instructions whose sequence took borrowed registers: {instruction index: first register}.
    pub(super) pool_taken: HashMap<usize, u32>,
    /// And how many registers the pool took at each instruction.
    pub(super) pool_width: HashMap<usize, u32>,
    /// Level 3: what the instruction at hand was given (a bit set).
    pub(super) given: RegSet,
    /// Every run given at each instruction: {instruction: [(first register, width)]}.
    pub(super) given_at: HashMap<usize, Vec<(u32, u32)>>,
    /// Level 3: the lane register taken at each instruction.
    pub(super) lane_taken: HashMap<usize, u32>,
    /// Level 3: the scratch register taken at each instruction.
    pub(super) scratch_taken: HashMap<usize, u32>,
    /// Level 3: {uniform register: ordinary register holding it at this instruction}.
    pub(super) mem_regs: HashMap<u32, u32>,
    /// Level 3: the register holding the top of the frame at this instruction.
    pub(super) base_reg: Option<u32>,
    /// Room made: {instruction index: vec of registers spilled at that instruction}.
    pub(super) spilled: HashMap<usize, Vec<u32>>,
    /// The index in the items list of the R1 lowering instruction (for rewriting its immediate
    /// once the frame size is known at the end), or None.
    pub(super) lower_item_idx: Option<usize>,
    /// The first instruction that reads c[0x0][0x28] after the first one, or None.
    pub(super) top_again: Option<usize>,
    /// Whether the first instruction sets R1 from c[0x0][0x28] unguarded.
    pub(super) entry_sets_r1: bool,
}

impl Ctx<'_> {
    pub(super) fn r(&self, u: u32) -> Result<u32, Refused> {
        if u == 63 {
            return Ok(255);
        }
        // Level 3: a uniform register loaded from the frame has its register here.
        if let Some(&r) = self.mem_regs.get(&u) {
            return Ok(r);
        }
        self.place.get(&u).copied().ok_or_else(|| no_register(u))
    }
    pub(super) fn p(&self, q: u32) -> Result<u32, Refused> {
        if q == 7 {
            return Ok(7);
        }
        self.predicates
            .stand
            .get(&(q, self.at))
            .copied()
            .ok_or_else(|| no_predicate(q))
    }
    /// Registers for the translation of the instruction at hand, among those of the program
    /// that hold nothing there: the lowest that this instruction was not given already. The
    /// pool of level 2, and at level 3 everything. When room is made and no register holds
    /// nothing, victims are taken from registers the instruction does not touch.
    pub(super) fn take(&mut self, width: u32, why: Option<&str>) -> Result<u32, Refused> {
        let borrow = self
            .borrow
            .as_mut()
            .ok_or_else(|| Refused("take without borrow".into()))?;
        let r = borrow.free_at(self.at, width, &self.given);
        let (r, made) = match r {
            Some(r) => (r, false),
            None => {
                if borrow.room {
                    match self.make_room(width) {
                        Some(r) => (r, true),
                        None => {
                            let msg = why.map(|s| s.to_string()).unwrap_or_else(|| {
                                if width == 1 {
                                    format!("no register holds nothing at 0x{:x}", 16 * self.at)
                                } else {
                                    format!("no {} registers in a row hold nothing at 0x{:x}", width, 16 * self.at)
                                }
                            });
                            return Err(Refused(msg));
                        }
                    }
                } else {
                    let msg = why.map(|s| s.to_string()).unwrap_or_else(|| {
                        if width == 1 {
                            format!("no register holds nothing at 0x{:x}", 16 * self.at)
                        } else {
                            format!("no {} registers in a row hold nothing at 0x{:x}", width, 16 * self.at)
                        }
                    });
                    return Err(Refused(msg));
                }
            }
        };
        for k in 0..width {
            regset_set(&mut self.given, r + k);
        }
        if made {
            let entry = self.spilled.entry(self.at).or_default();
            for k in 0..width {
                entry.push(r + k);
            }
        } else {
            self.given_at.entry(self.at).or_default().push((r, width));
        }
        let borrow = self.borrow.as_mut().unwrap();
        for k in 0..width {
            borrow.used.insert(r + k);
        }
        Ok(r)
    }
    /// Where no register holds nothing at the instruction at hand (level 3, VF_BORROW >= 5):
    /// registers that the instruction does not touch (`Borrowed::victims_at`), which
    /// `translate_function` puts in the frame before the sequence and brings back after it.
    /// Only at an instruction that always goes on to the next one (what is put away must come
    /// back), after the first (which sets the stack pointer the frame is addressed from).
    /// None when it cannot be.
    pub(super) fn make_room(&self, width: u32) -> Option<u32> {
        let i = self.at;
        let d = &self.src[i];
        let borrow = self.borrow.as_ref()?;
        if i == 0 || !self.entry_sets_r1 || borrow.depth[i].is_none() {
            return None;
        }
        // Not at an instruction that moves the stack pointer: what is put away before it
        // would be looked for elsewhere after it.
        if d.writes.iter().any(|&(l, r)| l == LETTER_R && r == 1) {
            return None;
        }
        // The instruction must always go on to the next one and not be a branch or call.
        if !crate::pairing::is_straight(d) {
            return None;
        }
        borrow.victims_at(i, width, &self.given)
    }
    /// The first of the two registers that the sequences of the translation compute into. Each
    /// sequence writes what it needs of them and reads it before it ends, and an instruction that
    /// reads them late is waited for (`late`): one pair serves every sequence. `width` 1: the
    /// sequence writes the first one only.
    pub(super) fn pool(&mut self, width: u32) -> Result<u32, Refused> {
        if self.pool_at.is_none() && self.borrow.is_some() {
            // Borrowed for this instruction: `take`. What the sequence writes first waits for
            // everything (`translate_function`).
            if let Some(&existing) = self.pool_taken.get(&self.at) {
                if width > *self.pool_width.get(&self.at).unwrap_or(&2) {
                    return Err(Refused(format!(
                        "the sequence at 0x{:x} takes one register to compute into, then two",
                        16 * self.at
                    )));
                }
                return Ok(existing);
            }
            let msg = format!(
                "no {} nothing at 0x{:x}",
                if width == 1 {
                    "register holds"
                } else {
                    "two registers side by side hold"
                },
                16 * self.at
            );
            let r = self.take(width, Some(&msg))?;
            self.pool_taken.insert(self.at, r);
            self.pool_width.insert(self.at, width);
            return Ok(r);
        }
        self.pool_at
            .ok_or_else(|| Refused("a sequence that needs registers in a function that was not given any".into()))
    }
    /// The register that holds the lane of the thread (0 to 31): all through the function,
    /// or at level 3 one taken at the instruction.
    pub(super) fn lane(&mut self) -> Result<u32, Refused> {
        if let Some(r) = self.lane_at {
            return Ok(r);
        }
        if self.borrow.as_ref().is_some_and(|b| b.everything) {
            if let Some(&r) = self.lane_taken.get(&self.at) {
                return Ok(r);
            }
            let r = self.take(1, None)?;
            self.lane_taken.insert(self.at, r);
            return Ok(r);
        }
        Err(Refused(
            "a sequence that goes by the lane in a function that was not given a register for it".into(),
        ))
    }
    /// The first of the registers that a sequence computes into: those of a matrix instruction
    /// (`MATRIX_SCRATCH` of them), or the one register of a Turing F2FP's high half. At level 3:
    /// that many in a row from an even one.
    pub(super) fn scratch(&mut self) -> Result<u32, Refused> {
        if let Some(r) = self.scratch_at_reg {
            return Ok(r);
        }
        if self.borrow.as_ref().is_some_and(|b| b.everything) {
            if let Some(&r) = self.scratch_taken.get(&self.at) {
                return Ok(r);
            }
            let r = self.take(MATRIX_SCRATCH, None)?;
            self.scratch_taken.insert(self.at, r);
            return Ok(r);
        }
        Err(Refused(
            "a sequence that needs registers of its own in a function that was not given them".into(),
        ))
    }
    /// Base register and byte offset of a uniform value's frame word at instruction `i`.
    /// Use R1 at a known stack depth, or load the frame top when there is no R1-relative address.
    pub(super) fn slot(&mut self, slot_word: u32, i: usize, items: &mut Vec<Item>) -> Result<(u32, u32), Refused> {
        let borrow = self.borrow.as_ref().unwrap();
        if borrow.from_r1 {
            let depth = borrow.depth[i].ok_or_else(|| {
                Refused(format!(
                    "a uniform register in memory at 0x{:x}, where the stack pointer is not known",
                    16 * i
                ))
            })?;
            let offset = (depth + 4 * slot_word as i32) as u32;
            Ok((1, offset))
        } else {
            let top = self.frame_top(i, items)?;
            let borrow = self.borrow.as_ref().unwrap();
            let offset = (4 * slot_word).wrapping_sub(borrow.frame_bytes) & 0xFFFFFF;
            Ok((top, offset))
        }
    }
    /// A late item: an instruction that reads its operands some time after it is issued, with
    /// a read-barrier scoreboard. The next instruction waits for it.
    pub(super) fn late_item(
        &self,
        class_name: &str,
        word: u128,
        i: usize,
        what: &'static str,
        detail: Option<String>,
    ) -> Result<Item, Refused> {
        let mut item = Item::new(read_announced(class_name, word & !REUSE, i)?, i, what, true);
        item.late = true;
        item.what_detail = detail;
        Ok(item)
    }
    /// Level 3: the register holding the top of the frame at this instruction, loaded once.
    pub(super) fn frame_top(&mut self, i: usize, items: &mut Vec<Item>) -> Result<u32, Refused> {
        if let Some(r) = self.base_reg {
            return Ok(r);
        }
        let r = self.take(1, None)?;
        self.base_reg = Some(r);
        let word = assemble::assemble(
            "mov__RCR_RC",
            &[
                ("Rd", assemble::OpVal::Int(r)),
                ("Sb_bank", assemble::OpVal::Int(0)),
                ("Sb_addr", assemble::OpVal::Int(0x28 / 4)),
            ],
        )
        .ok_or_else(|| Refused("cannot assemble MOV for frame top".into()))?;
        items.push(Item::new(word, i, "mem: base", true));
        Ok(r)
    }
    /// Level 3: the uniform registers of the frame that instruction `d` reads and writes.
    pub(super) fn in_frame(&self, d: &Decoded) -> (Vec<u32>, Vec<u32>, HashSet<u32>) {
        let borrow = self.borrow.as_ref().unwrap();
        let slots = &borrow.slots;
        let mut read_set: Vec<u32> = Vec::new();
        let mut written_set: Vec<u32> = Vec::new();
        let mut together: HashSet<u32> = HashSet::new();
        for &(letter, x, _) in &d.reads {
            if letter == LETTER_U && x != 63 && slots.contains_key(&x) && !read_set.contains(&x) {
                read_set.push(x);
            }
        }
        for &(letter, x) in &d.writes {
            if letter == LETTER_U && x != 63 && slots.contains_key(&x) && !written_set.contains(&x) {
                written_set.push(x);
            }
        }
        // Detect pairs: check if the instruction takes a pair operand (count=2) with both in slots.
        for (first, count) in crate::decode::uniform_operands(d) {
            if count == 2 && slots.contains_key(&first) {
                together.insert(first);
            }
        }
        read_set.sort();
        written_set.sort();
        (read_set, written_set, together)
    }
    /// Level 3: items that load the uniform registers of the frame before instruction `i`.
    pub(super) fn loads_before(&mut self, i: usize, d: &Decoded) -> Result<Vec<Item>, Refused> {
        if self.borrow.is_none() || self.borrow.as_ref().unwrap().slots.is_empty() {
            return Ok(Vec::new());
        }
        let (mut read, written, together) = self.in_frame(d);
        if !crate::pairing::unguarded(d) {
            // A guarded instruction may leave what it writes as it was: load those too.
            for u in &written {
                if !read.contains(u) {
                    read.push(*u);
                }
            }
            read.sort();
        }
        if read.is_empty() && written.is_empty() {
            return Ok(Vec::new());
        }
        let mut items: Vec<Item> = Vec::new();
        // The prototype takes registers in `sorted(read | written)` order: the sorted union,
        // not read-then-written. The order of `take` calls determines which register each
        // uniform register gets (free_at skips what was already given).
        let mut all_uniforms: Vec<u32> = Vec::new();
        for &u in read.iter().chain(written.iter()) {
            if !all_uniforms.contains(&u) {
                all_uniforms.push(u);
            }
        }
        all_uniforms.sort();
        for &u in &all_uniforms {
            if self.mem_regs.contains_key(&u) {
                continue;
            }
            if together.contains(&u) {
                let first = self.take(2, None)?;
                self.mem_regs.insert(u, first);
                self.mem_regs.insert(u + 1, first + 1);
            } else {
                let r = self.take(1, None)?;
                self.mem_regs.insert(u, r);
            }
        }
        if !read.is_empty() {
            for &u in &read {
                let rd = self.mem_regs[&u];
                let slot = self.borrow.as_ref().unwrap().slots[&u];
                self.load_frame_word(slot, rd, i, format!("mem: load UR{u}"), &mut items)?;
            }
            items.last_mut().unwrap().then = Some(ARRIVES);
        }
        Ok(items)
    }
    /// Level 3: items that store the uniform registers of the frame after instruction `i`.
    pub(super) fn stores_after(&mut self, i: usize, d: &Decoded) -> Result<Vec<Item>, Refused> {
        if self.borrow.is_none() || self.borrow.as_ref().unwrap().slots.is_empty() {
            return Ok(Vec::new());
        }
        let (_, written, _) = self.in_frame(d);
        if written.is_empty() {
            return Ok(Vec::new());
        }
        let mut items: Vec<Item> = Vec::new();
        // Under no guard. An instruction may write the predicate that guards it (the carry of
        // `@UP0 UIADD3 UR0, UP0, URZ, 0x0, URZ`), and a store under that guard would go by what
        // the instruction left in it. Where the instruction is under a guard, what it writes
        // was loaded before it (`loads_before`): the store puts back what was there when the
        // instruction did not run.
        for &u in &written {
            let rb = self.mem_regs[&u];
            let slot = self.borrow.as_ref().unwrap().slots[&u];
            self.store_frame_word(slot, rb, i, format!("mem: store UR{u}"), &mut items)?;
        }
        Ok(items)
    }
    pub(super) fn load_frame_word(
        &mut self,
        slot: u32,
        rd: u32,
        i: usize,
        detail: String,
        items: &mut Vec<Item>,
    ) -> Result<(), Refused> {
        let (top, offset) = self.slot(slot, i, items)?;
        let word = assemble::assemble(
            "ldl__sImmOffset",
            &[
                ("Rd", assemble::OpVal::Int(rd)),
                ("Ra", assemble::OpVal::Int(top)),
                ("Ra_offset", assemble::OpVal::Int(offset)),
                ("dst_wr_sb", assemble::OpVal::Int(ARRIVES)),
            ],
        )
        .ok_or_else(|| Refused("cannot assemble LDL for frame load".into()))?;
        let mut item = Item::new(word, i, "mem: load", true);
        item.what_detail = Some(detail);
        items.push(item);
        Ok(())
    }
    pub(super) fn store_frame_word(
        &mut self,
        slot: u32,
        rb: u32,
        i: usize,
        detail: String,
        items: &mut Vec<Item>,
    ) -> Result<(), Refused> {
        let (top, offset) = self.slot(slot, i, items)?;
        let word = assemble::assemble(
            "stl__sImmOffset",
            &[
                ("Ra", assemble::OpVal::Int(top)),
                ("Rb", assemble::OpVal::Int(rb)),
                ("Ra_offset", assemble::OpVal::Int(offset)),
            ],
        )
        .ok_or_else(|| Refused("cannot assemble STL for frame store".into()))?;
        // A store reads its operands late: it says when, and what follows waits.
        let word = read_announced("stl__sImmOffset", word & !REUSE, i)?;
        let mut item = Item::new(word, i, "mem: store", true);
        item.what_detail = Some(detail);
        item.late = true;
        items.push(item);
        Ok(())
    }
    /// Whether the product at `i`, on the register `rb` of B, finds in the registers kept for
    /// that what it would gather: the last product that gathered did it from the same
    /// register; every way to this one goes through that one and straight on from it (nothing
    /// after it up to this one is entered otherwise than from the instruction before:
    /// `entries`); no call between them (the function called may gather for a product of its
    /// own, or write the register); and nothing from that product on, itself included, writes
    /// the register, under a guard or not.
    pub(super) fn kept_gathering(&self, i: usize, rb: u32) -> bool {
        // At level 3 (everything at the instruction): always no. Nothing is kept between
        // instructions; every product gathers for itself.
        if self.borrow.as_ref().is_some_and(|b| b.everything) {
            return false;
        }
        let (Some((gathered, held)), Some(entered)) = (self.holds, &self.entered) else {
            return false;
        };
        if held != rb || (gathered + 1..=i).any(|j| entered.contains(&j)) {
            return false;
        }
        !self.src[gathered..i]
            .iter()
            .any(|d| d.name.starts_with("call") || d.writes.contains(&(LETTER_R, rb)))
    }
    /// What a function with a product computes of the lane when it is entered, into four
    /// registers that nothing else writes: the two lanes that hold what a thread gathers of B
    /// ((lane >> 4) * 8 + thread, and 16 more), all ones in the upper half of each sixteen
    /// lanes (bit 3 of the lane), and all ones in the threads with an odd number. (Each product
    /// computed the four again until 2026-11-18: eight instructions of its thirty-three.)
    pub(super) fn kept(&mut self) -> Result<Vec<Item>, Refused> {
        use assemble::OpVal::{Int, Str};
        let low = self.kept_at.ok_or_else(|| {
            Refused("a product in a function that was not given registers for what it keeps of the lane".into())
        })?;
        let lane = self.lane()?;
        let (high, half, odd) = (low + 1, low + 2, low + 3);
        let shf = |rd: u32, ra: u32, sb: u32, rc: u32, dir: &'static str, hilo: &'static str, fmt: &'static str| {
            assemble::assemble(
                "shf__RuIR_RIR",
                &[
                    ("Rd", Int(rd)),
                    ("Ra", Int(ra)),
                    ("Sb", Int(sb)),
                    ("Rc", Int(rc)),
                    ("dir", Str(dir)),
                    ("hilo", Str(hilo)),
                    ("fmt", Str(fmt)),
                ],
            )
        };
        let lut = |rd: u32, ra: u32, sb: u32, rc: u32, imm8: u32| {
            assemble::assemble(
                "lop3_lut__RuIR_RIR",
                &[
                    ("Pu", Int(7)),
                    ("Pp", Int(7)),
                    ("Pp_not", Int(1)),
                    ("Rd", Int(rd)),
                    ("Ra", Int(ra)),
                    ("Sb", Int(sb)),
                    ("Rc", Int(rc)),
                    ("imm8", Int(imm8)),
                ],
            )
        };
        [
            shf(low, 255, 1, lane, "R", "HI", "U32"),
            lut(low, low, 8, 255, 0xc0),
            lut(low, low, 3, lane, 0xf8),
            lut(high, low, 16, 255, 0xfc),
            shf(half, lane, 28, 255, "L", "LO", "U32"),
            shf(half, 255, 31, half, "R", "HI", "S32"),
            shf(odd, lane, 31, 255, "L", "LO", "U32"),
            shf(odd, 255, 31, odd, "R", "HI", "S32"),
        ]
        .into_iter()
        .map(|word| {
            word.map(|word| Item::new(word, 0, "kept", true))
                .ok_or_else(|| Refused("cannot assemble what is kept of the lane".into()))
        })
        .collect()
    }
    /// The item that saves predicate `p` in the `k`th register kept for that.
    pub(super) fn saved(&self, k: u32, p: u32, i: usize) -> Result<Item, Refused> {
        let register = self.save_at.ok_or_else(no_save)? + k;
        let word = assemble::assemble(
            "p2r__RsIR_RIR",
            &[
                ("Rd", assemble::OpVal::Int(register)),
                ("Ra", assemble::OpVal::Int(255)),
                ("Sb", assemble::OpVal::Int(1 << p)),
            ],
        )
        .ok_or_else(|| Refused("cannot assemble P2R".into()))?;
        let mut item = Item::new(word, i, "save", true);
        item.more_reads.push((LETTER_P, p, 0));
        Ok(item)
    }
    /// The item that gives predicate `p` back from the `k`th register kept for that.
    pub(super) fn given_back(&self, k: u32, p: u32, i: usize) -> Result<Item, Refused> {
        let register = self.save_at.ok_or_else(no_save)? + k;
        let word = assemble::assemble(
            "r2p__RIR",
            &[
                ("Ra", assemble::OpVal::Int(register)),
                ("Sb", assemble::OpVal::Int(1 << p)),
            ],
        )
        .ok_or_else(|| Refused("cannot assemble R2P".into()))?;
        let mut item = Item::new(word, i, "restore", true);
        item.more_writes.push((LETTER_P, p));
        Ok(item)
    }

    /// The register that holds uniform predicate `q` between the instructions that name it, in
    /// the lives where no ordinary predicate stands for it all along (`stand_ins`).
    pub(super) fn holder(&self, q: u32) -> Result<u32, Refused> {
        self.hold_at.get(&q).copied().ok_or_else(|| {
            Refused("a uniform predicate to hold in a function that was not given a register for it".into())
        })
    }

    /// Restore predicate `p` from the nonzero value in general register `holder`.
    pub(super) fn read_back(&self, holder: u32, p: u32, i: usize) -> Result<Item, Refused> {
        let word = assemble::assemble(
            "isetp__RRR_RR",
            &[
                ("Pu", assemble::OpVal::Int(p)),
                ("Ra", assemble::OpVal::Int(holder)),
                ("Rb", assemble::OpVal::Int(255)),
                ("icmp", assemble::OpVal::Str("NE")),
                ("fmt", assemble::OpVal::Str("U32")),
                ("bop", assemble::OpVal::Str("AND")),
            ],
        )
        .ok_or_else(|| Refused("cannot assemble ISETP for held predicate".into()))?;
        Ok(Item::new(word, i, "held: back", true))
    }

    /// Save the bit for predicate `p` in general register `holder`.
    pub(super) fn written_down(&self, holder: u32, p: u32, i: usize) -> Result<Item, Refused> {
        let word = assemble::assemble(
            "p2r__RsIR_RIR",
            &[
                ("Rd", assemble::OpVal::Int(holder)),
                ("Ra", assemble::OpVal::Int(255)),
                ("Sb", assemble::OpVal::Int(1 << p)),
            ],
        )
        .ok_or_else(|| Refused("cannot assemble P2R for held predicate".into()))?;
        let mut item = Item::new(word, i, "held: kept", true);
        item.more_reads.push((LETTER_P, p, 0));
        Ok(item)
    }

    // ---- Pairs of products on whole rows ----

    /// The first of the two registers that hold half `half` of k of the row of fragment or
    /// pair `q` of the whole-row loads with destination `rd`.
    fn half_row(&self, rd: u32, q: u32, half: u32) -> Result<u32, Refused> {
        let rows_at_rd = self
            .rows_at
            .get(&rd)
            .ok_or_else(|| Refused("rows of a load in a function that was not given registers for them".into()))?;
        let n = self.rows_n.get(&rd).copied().unwrap_or(4);
        if n == 4 {
            if half == 0 {
                Ok(rd + 2 * q)
            } else {
                Ok(rows_at_rd + 2 * q)
            }
        } else {
            Ok(rows_at_rd + 2 * q + 4 * half)
        }
    }

    /// What a function with pairs computes of the lane when it is entered.
    pub(super) fn kept_for_rows(&mut self) -> Result<Vec<Item>, Refused> {
        use assemble::OpVal::Int;
        let asm = assemble::assemble;
        let lane = self.lane()?;
        let t = self.scratch()?;
        let q = self.q.ok_or_else(|| Refused("kept_for_rows without a plan".into()))?;
        let plan_kept = q.kept();

        let shf = |rd: u32, ra: u32, sb: u32, rc: u32, dir: &str, hilo: &str| {
            asm(
                "shf__RuIR_RIR",
                &[
                    ("Rd", Int(rd)),
                    ("Ra", Int(ra)),
                    ("Sb", Int(sb)),
                    ("Rc", Int(rc)),
                    ("dir", assemble::OpVal::Str(dir)),
                    ("hilo", assemble::OpVal::Str(hilo)),
                    ("fmt", assemble::OpVal::Str("U32")),
                ],
            )
        };
        let right = |to: u32, x: u32, n: u32| shf(to, 255, n, x, "R", "HI");
        let left = |to: u32, x: u32, n: u32| shf(to, x, n, 255, "L", "LO");
        let masked = |to: u32, x: u32, m: u32| {
            asm(
                "lop3_lut__RuIR_RIR",
                &[
                    ("Pu", Int(7)),
                    ("Pp", Int(7)),
                    ("Pp_not", Int(1)),
                    ("Rd", Int(to)),
                    ("Ra", Int(x)),
                    ("Sb", Int(m)),
                    ("Rc", Int(255)),
                    ("imm8", Int(0xc0)),
                ],
            )
        };
        let both = |to: u32, a: u32, b: u32| {
            asm(
                "lop3_lut__RRR_RRR",
                &[
                    ("Pu", Int(7)),
                    ("Pp", Int(7)),
                    ("Pp_not", Int(1)),
                    ("imm8", Int(0xfc)),
                    ("Rd", Int(to)),
                    ("Ra", Int(a)),
                    ("Rb", Int(b)),
                    ("Rc", Int(255)),
                ],
            )
        };
        let with_low = |to: u32, x: u32, m: u32| {
            asm(
                "lop3_lut__RuIR_RIR",
                &[
                    ("Pu", Int(7)),
                    ("Pp", Int(7)),
                    ("Pp_not", Int(1)),
                    ("Rd", Int(to)),
                    ("Ra", Int(x)),
                    ("Sb", Int(m)),
                    ("Rc", Int(lane)),
                    ("imm8", Int(0xf8)),
                ],
            )
        };

        let err = || Refused("cannot assemble what is kept of the lane for rows".into());
        let r_ = |w: Option<u128>| w.ok_or_else(err);
        let mut words: Vec<u128> = Vec::new();
        for &name in crate::pairing::ROW_KEPT {
            let Some(&r) = self.kept_rows.get(name) else {
                continue;
            };
            if !plan_kept.contains(name) {
                continue;
            }
            match name {
                "kA" => {
                    for w in [
                        left(r, lane, 2),
                        masked(r, r, 8),
                        right(t, lane, 1),
                        masked(t, t, 4),
                        both(r, r, t),
                        right(t, lane, 3),
                        masked(t, t, 2),
                        both(r, r, t),
                        with_low(r, r, 1),
                    ] {
                        words.push(r_(w)?);
                    }
                }
                "kB" | "kBT" => {
                    let mask = if name == "kB" { 7 } else { 3 };
                    for w in [right(r, lane, 1), masked(r, r, 8), with_low(r, r, mask)] {
                        words.push(r_(w)?);
                    }
                }
                "kA2" | "kB2" => {
                    let base_name = &name[..2];
                    let base = *self
                        .kept_rows
                        .get(base_name)
                        .ok_or_else(|| Refused(format!("kept value {base_name} not found for {name}")))?;
                    words.push(r_(asm(
                        "lop3_lut__RuIR_RIR",
                        &[
                            ("Pu", Int(7)),
                            ("Pp", Int(7)),
                            ("Pp_not", Int(1)),
                            ("Rd", Int(r)),
                            ("Ra", Int(base)),
                            ("Sb", Int(16)),
                            ("Rc", Int(255)),
                            ("imm8", Int(0xfc)),
                        ],
                    ))?);
                }
                "cB" => {
                    for w in [left(r, lane, 1), masked(r, r, 8)] {
                        words.push(r_(w)?);
                    }
                }
                "pi" => {
                    for w in [
                        right(r, lane, 1),
                        masked(r, r, 0xb),
                        left(t, lane, 1),
                        masked(t, t, 16),
                        both(r, r, t),
                        left(t, lane, 2),
                        masked(t, t, 4),
                        both(r, r, t),
                    ] {
                        words.push(r_(w)?);
                    }
                }
                "inv" => {
                    for w in [
                        left(r, lane, 1),
                        masked(r, r, 0x16),
                        right(t, lane, 1),
                        masked(t, t, 8),
                        both(r, r, t),
                        right(t, lane, 2),
                        masked(t, t, 1),
                        both(r, r, t),
                    ] {
                        words.push(r_(w)?);
                    }
                }
                _ => {}
            }
        }
        Ok(words.into_iter().map(|w| Item::new(w, 0, "kept", true)).collect())
    }

    /// A matrix load whose results the products of pairs read: whole rows.
    pub(super) fn whole_rows(&mut self, i: usize, d: &Decoded, wait: u128) -> Result<Vec<Item>, Refused> {
        use assemble::OpVal::{Int, Str};
        let asm = assemble::assemble;
        let q = self.q.ok_or_else(|| Refused("whole_rows without a plan".into()))?;
        let load = q
            .loads
            .get(&i)
            .ok_or_else(|| Refused("whole_rows for a non-load".into()))?;
        let role = load.role;
        let rd = load.rd;
        let also = load.also;
        let ra = d.op_or(OP_RA, 255);
        let offset = d.op_or(OP_RA_OFFSET, 0);
        let result = d.op_or(OP_DST_WR_SB, 7);
        let w = d.word;

        if d.op_or(OP_PG, 7) != 7 || d.op_or(OP_SZ, 0) != 0 || d.op_or(OP_NUM, 0) != 2 {
            return Err(Refused(format!("this form of LDSM for whole rows at 0x{:x}", 16 * i)));
        }

        let uniform = if d.name == "ldsm__URsIR" {
            let u = self.r(d.op_or(OP_RA_URB, 63))?;
            if u == 255 { None } else { Some(u) }
        } else {
            None
        };

        let s = self.scratch()?;
        // The scoreboard for shuffles and loads: not the one the products use.
        let board = if result == ARRIVES { SEQUENCE } else { ARRIVES };
        let mut new: Vec<Item> = Vec::new();
        let mut add = |word: u128, what: &'static str, waits: bool| {
            // The program's waits go on the first instruction only.
            let w = if new.is_empty() { wait } else { 0 };
            let wait_bits = if waits { 1u128 << (WAIT + board) } else { 0 };
            new.push(Item::new(word | w | wait_bits, i, what, true));
        };

        // The rows: (scratch, key register, further offset, word pairs).
        struct Row {
            at: u32,
            key: u32,
            further: u32,
            words: Vec<(u32, u32)>,
        }
        let rows = match role {
            crate::pairing::LoadRole::A | crate::pairing::LoadRole::B => {
                let (k0, k1) = if role == crate::pairing::LoadRole::A {
                    (
                        *self.kept_rows.get("kA").ok_or_else(|| Refused("missing kA".into()))?,
                        *self.kept_rows.get("kA2").ok_or_else(|| Refused("missing kA2".into()))?,
                    )
                } else {
                    (
                        *self.kept_rows.get("kB").ok_or_else(|| Refused("missing kB".into()))?,
                        *self.kept_rows.get("kB2").ok_or_else(|| Refused("missing kB2".into()))?,
                    )
                };
                let keys = [k0, k1];
                let mut rows = Vec::new();
                for q_idx in 0..2u32 {
                    rows.push(Row {
                        at: s + q_idx,
                        key: keys[q_idx as usize],
                        further: 0,
                        words: vec![(self.half_row(rd, q_idx, 0)?, 0), (self.half_row(rd, q_idx, 1)?, 8)],
                    });
                }
                rows
            }
            crate::pairing::LoadRole::BT => {
                let kbt = *self.kept_rows.get("kBT").ok_or_else(|| Refused("missing kBT".into()))?;
                let mut rows = Vec::new();
                for q_idx in 0..2u32 {
                    for half in 0..2u32 {
                        rows.push(Row {
                            at: s + 2 * q_idx + half,
                            key: kbt,
                            further: 16 * q_idx + 4 * half,
                            words: vec![(self.half_row(rd, q_idx, half)?, 0)],
                        });
                    }
                }
                rows
            }
        };

        // Lane indices: for rows with further>0, compute key+further into spare registers.
        let mut spare = s + 4;
        let mut lanes: Vec<u32> = Vec::new();
        for row in &rows {
            if row.further != 0 {
                add(
                    asm(
                        "iadd3_imm__RsIR_RIR",
                        &[
                            ("Rd", Int(spare)),
                            ("Ra", Int(row.key)),
                            ("Sb", Int(row.further)),
                            ("Rc", Int(255)),
                        ],
                    )
                    .ok_or_else(|| Refused("cannot assemble IADD3 for lane".into()))?,
                    "whole row: lane",
                    false,
                );
                lanes.push(spare);
                spare += 1;
            } else {
                lanes.push(row.key);
            }
        }

        // For BT: compute the "more" value (cB, or uniform+cB).
        let more: Option<u32> = if role == crate::pairing::LoadRole::BT {
            let cb = *self.kept_rows.get("cB").ok_or_else(|| Refused("missing cB".into()))?;
            if let Some(uniform_r) = uniform {
                let m = s + 7;
                add(
                    asm(
                        "iadd3_noimm__RRR_RRR",
                        &[
                            ("Rd", Int(m)),
                            ("Ra", Int(uniform_r)),
                            ("Rb", Int(cb)),
                            ("Rc", Int(255)),
                        ],
                    )
                    .ok_or_else(|| Refused("cannot assemble IADD3 for cB+uniform".into()))?,
                    "whole row: address",
                    false,
                );
                Some(m)
            } else {
                Some(cb)
            }
        } else {
            uniform
        };

        // The shuffles: each row's address from the lane that holds it.
        for (row, &held) in rows.iter().zip(lanes.iter()) {
            add(
                asm(
                    "shfl__RRI",
                    &[
                        ("shflmd", Int(0)),
                        ("Pu", Int(7)),
                        ("Rd", Int(row.at)),
                        ("Ra", Int(ra)),
                        ("Rb", Int(held)),
                        ("Sc", Int(31)),
                        ("dst_wr_sb", Int(board)),
                    ],
                )
                .ok_or_else(|| Refused("cannot assemble SHFL for row address".into()))?,
                "whole row: address",
                false,
            );
        }

        // Add the uniform part / cB to each address.
        let mut first = true;
        if let Some(more_reg) = more {
            for row in &rows {
                add(
                    asm(
                        "iadd3_noimm__RRR_RRR",
                        &[
                            ("Rd", Int(row.at)),
                            ("Ra", Int(row.at)),
                            ("Rb", Int(more_reg)),
                            ("Rc", Int(255)),
                        ],
                    )
                    .ok_or_else(|| Refused("cannot assemble IADD3 for address".into()))?,
                    "whole row: address",
                    first,
                );
                first = false;
            }
        }

        // The loads.
        let aligned = self.aligned_rows;
        for row in &rows {
            for &(to, byte) in &row.words {
                for k in 0..if aligned { 1u32 } else { 2 } {
                    let sz = if aligned { "U.64" } else { "U.32" };
                    let rd_load = to + k;
                    let off = offset + byte + 4 * k;
                    let sb = if result == 7 { SEQUENCE } else { result };
                    let word = asm(
                        "lds__sImmOffset",
                        &[
                            ("sz", Str(sz)),
                            ("Rd", Int(rd_load)),
                            ("Ra", Int(row.at)),
                            ("Ra_offset", Int(off)),
                            ("dst_wr_sb", Int(sb)),
                            ("src_rel_sb", Int(board)),
                        ],
                    )
                    .ok_or_else(|| Refused("cannot assemble LDS for whole rows".into()))?;
                    add(word & !REUSE, "whole row", first);
                    first = false;
                }
            }
        }
        new.last_mut().unwrap().late = true;

        if also {
            // What the program's registers get, as before.
            let guard = if d.has_op(OP_PG) {
                vec![
                    ("Pg", assemble::OpVal::Int(d.op_or(OP_PG, 7))),
                    ("Pg_not", assemble::OpVal::Int(d.op_or(OP_PG_AT_NOT, 0))),
                ]
            } else {
                Vec::new()
            };
            let mut old = translate_matrix_load(self, i, d, &guard, wait)?;
            old[0].word |= 1u128 << (WAIT + ((new.last().unwrap().word >> READ_BARRIER) & 7) as u32);
            new.last_mut().unwrap().late = false;
            new.extend(old);
            return Ok(new);
        }

        // As late after the load as in the original: what its instructions have not already taken.
        let taken: u32 = new[..new.len() - 1].iter().map(|item| stall_of(item.word)).sum();
        let last_stall = stall_of(new.last().unwrap().word);
        let orig_stall = stall_of(w);
        let final_stall = last_stall.max(orig_stall.saturating_sub(taken)).max(1);
        {
            let last = new.last_mut().unwrap();
            last.word = with_stall(last.word, final_stall, false);
        }
        Ok(new)
    }

    /// A product that is half of a pair.
    pub(super) fn half_of_pair(&self, i: usize, d: &Decoded, wait: u128) -> Result<Vec<Item>, Refused> {
        use assemble::OpVal::Int;
        let asm = assemble::assemble;
        let w = d.word;
        let q = self.q.ok_or_else(|| Refused("half_of_pair without a plan".into()))?;
        let site = q
            .sites
            .get(&i)
            .ok_or_else(|| Refused("half_of_pair for a non-site".into()))?;

        let a = self.half_row(site.ra, site.a, site.half)?;
        let b = self.half_row(site.rb, site.p, site.half)?;

        let mut new: Vec<Item> = Vec::new();
        let to = [site.x, site.x + 2, site.y, site.y + 2];
        // The first product of a pair starts from the accumulators the two products of the
        // program read (which may be other registers, or zero); the second goes on in place.
        let onto: [u32; 4] = if site.half != 0 {
            to
        } else {
            [
                site.cx.unwrap_or(255),
                site.cx.map_or(255, |c| c + 2),
                site.cy.unwrap_or(255),
                site.cy.map_or(255, |c| c + 2),
            ]
        };
        for n in 0..4u32 {
            let mut ops = vec![
                ("size", Int(0)),
                ("dstfmt", Int(1)),
                ("srcfmt", Int(1)),
                ("step", Int(n)),
                ("satfinite", Int(0)),
                ("Rd", Int(to[n as usize])),
                ("Ra", Int(a)),
                ("tnspA", Int(0)),
                ("Rb", Int(b)),
                ("tnspB", Int(if site.transposed { 0 } else { 1 })),
                ("Rc", Int(onto[n as usize])),
            ];
            if n == 3 {
                ops.push(("dst_wr_sb", Int(SEQUENCE)));
            }
            let word = asm("hmma__884_f32", &ops).ok_or_else(|| Refused("cannot assemble HMMA for pair".into()))?;
            // The program's waits and the wait for the products before: on the first step only.
            let waits = if n == 0 { wait | (1u128 << (WAIT + SEQUENCE)) } else { 0 };
            let mut item = Item::new(with_stall(word, 2, false) | waits, i, "half of a pair", true);
            item.covered = n < 3;
            new.push(item);
        }
        // What follows an instruction is as late after it as in the original: the four steps
        // take eight cycles, and the last one more when the original's stall was longer than that.
        {
            let last = new.last_mut().unwrap();
            last.word = with_stall(last.word, stall_of(w).saturating_sub(6).max(2), false);
        }
        Ok(new)
    }

    /// Accumulator registers shuffled to change state (before or after instruction i).
    pub(super) fn converted(
        &self,
        i: usize,
        plan: &std::collections::BTreeMap<usize, Vec<crate::pairing::Conversion>>,
    ) -> Result<Vec<Item>, Refused> {
        use assemble::OpVal::Int;
        let asm = assemble::assemble;
        let wanted = match plan.get(&i) {
            Some(cs) if !cs.is_empty() => cs,
            _ => return Ok(Vec::new()),
        };
        let mut out = vec![Item::new(
            with_stall(NOP, 15, false) | (63u128 << WAIT),
            i,
            "settled",
            true,
        )];
        for (k, &(r, to)) in wanted.iter().enumerate() {
            let key_name = if to == crate::pairing::State::Q { "pi" } else { "inv" };
            let key = *self
                .kept_rows
                .get(key_name)
                .ok_or_else(|| Refused("missing pi/inv for conversion".into()))?;
            let word = asm(
                "shfl__RRI",
                &[
                    ("shflmd", Int(0)),
                    ("Pu", Int(7)),
                    ("Rd", Int(r)),
                    ("Ra", Int(r)),
                    ("Rb", Int(key)),
                    ("Sc", Int(31)),
                    ("dst_wr_sb", Int(SEQUENCE)),
                ],
            )
            .ok_or_else(|| Refused("cannot assemble SHFL for conversion".into()))?;
            let wait_bits = if k > 0 && k % 8 == 0 {
                1u128 << (WAIT + SEQUENCE)
            } else {
                0
            };
            let what = if to == crate::pairing::State::Q {
                "to pairs"
            } else {
                "from pairs"
            };
            out.push(Item::new(word | wait_bits, i, what, true));
        }
        out.last_mut().unwrap().then = Some(SEQUENCE);
        Ok(out)
    }
}

fn no_save() -> Refused {
    Refused("a predicate to save in a function that was not given a register for it".into())
}

fn no_register(u: u32) -> Refused {
    Refused(format!("uniform register UR{u} is used in a way that was not foreseen"))
}

fn no_predicate(q: u32) -> Refused {
    Refused(format!(
        "uniform predicate UP{q} is used in a way that was not foreseen"
    ))
}

/// The sm_70 word that a rule gives for source instruction `i`, or why it gives none.
pub(super) fn apply_rule(ctx: &Ctx, rule_idx: usize, i: usize, d: &Decoded) -> Result<u128, Refused> {
    match rule::apply(rule_idx, d.word, &|u| ctx.r(u).ok(), &|q| ctx.p(q).ok()) {
        rule::Applied::Word(word) => Ok(word),
        rule::Applied::NotAllowed => Err(Refused(format!(
            "{} at 0x{:x} has a value that sm_70 does not have",
            d.name,
            16 * i
        ))),
        rule::Applied::NoRegister(u) => Err(no_register(u)),
        rule::Applied::NoPredicate(q) => Err(no_predicate(q)),
    }
}

/// `place`, `pool`, `free`: where the uniform registers go, the two registers to compute into and
/// the first register left free, for this function and those it calls or that call it
/// (`placement`).
/// What `stand_ins` works out of the predicates of a function.
pub(super) struct Predicates {
    /// (uniform predicate, instruction) -> the ordinary predicate that stands for it there.
    pub(super) stand: HashMap<(u32, usize), u32>,
    /// The predicate each minimum or maximum computes into, and the register it is saved in when
    /// it holds something.
    pub(super) scratch_at: HashMap<usize, (u32, Option<u32>)>,
    /// The predicates to save before an instruction, and to give back after it: (register, predicate).
    pub(super) saves: HashMap<usize, Vec<(u32, u32)>>,
    pub(super) restores: HashMap<usize, Vec<(u32, u32)>>,
    /// How many registers that needs to save predicates in.
    pub(super) save_slots: u32,
    /// The uniform predicates held in a register: for each instruction that names one,
    /// (uniform predicate, ordinary predicate, read back before, written after).
    pub(super) held: HashMap<usize, Vec<(u32, u32, bool, bool)>>,
    /// The lives held in a register: (uniform predicate, the instructions of the life).
    pub(super) held_lives: Vec<(u32, Vec<usize>)>,
    /// Each life and the ordinary predicate that stands for the uniform one along it, in the order
    /// they were given (a held life: one for each instruction that names it): (uniform predicate,
    /// ordinary predicate, saved around the life, its instructions). For the validator (`MAP.lives`).
    pub(super) lives: Vec<(u32, u32, bool, Vec<usize>)>,
    /// The uniform predicates that some life holds in a register, sorted.
    pub(super) held_predicates: Vec<u32>,
    /// A predicate that a late instruction writes: at this instruction, wait this many cycles
    /// of nothing before the translation writes or saves it (`stand_ins`).
    pub(super) delayed: HashMap<usize, u32>,
    /// The predicates that some instruction of the function writes late, as a bit mask.
    #[allow(dead_code)]
    pub(super) slow_predicates: u32,
}

/// Pick a predicate: first one not in `slow`, then one not in `never`, each in the given order.
pub(super) fn pick(taken: u32, order: &[u32], slow: u32, never: u32) -> Option<u32> {
    for &avoid in &[slow, never] {
        if let Some(&p) = order.iter().find(|&&p| (taken | avoid) >> p & 1 == 0) {
            return Some(p);
        }
    }
    None
}

/// The ordinary predicate that stands for each uniform predicate at each instruction where the
/// uniform one holds something, and a predicate to compute into for each minimum and maximum.
///
/// A uniform predicate has separate lives: a carry is born and dies within two instructions, many
/// times in a function. Each life gets an ordinary predicate that holds nothing of the program
/// all along it and stands for no other uniform predicate there.
///
/// Where there is none (all seven hold something), the program has one that it neither reads nor
/// writes during that life: its value is saved in a register before the life (P2R) and given back
/// after it (R2P), as native code does when it runs out of predicates, and it stands for the
/// uniform one in between. That takes a life that is a run of instructions in a straight line:
/// entered at its first, left after its last.
pub(super) fn stand_ins(fd: &FuncData) -> Result<Predicates, Refused> {
    let live = compute_liveness(fd);
    let n = fd.src.len();
    let busy: Vec<u64> = (0..n)
        .map(|i| live.live_in[i] | live.live_out[i] | live.defs[i] | live.uses[i])
        .collect();
    // Which predicates are written late, and what delay each needs (`docs/translator.md`, "Timing rules").
    let mut slow: u32 = 0;
    let mut never: u32 = 0;
    let mut pred_wait = [0u32; 7];
    for d in &fd.src {
        if d.mnemonic == "DSETP" || LATE_WRITER_KINDS.contains(&d.kind) {
            for &(letter, register) in &d.writes {
                if letter != LETTER_P {
                    continue;
                }
                slow |= 1 << register;
                if ((d.word >> WRITE_BARRIER) & 7) != 7 {
                    pred_wait[register as usize] = pred_wait[register as usize].max(1); // on a scoreboard: waited for
                } else if d.kind == Kind::MioSlow {
                    never |= 1 << register;
                } else {
                    let dist = crate::distance::least(LETTER_P, 0, d, Kind::FXU);
                    pred_wait[register as usize] = pred_wait[register as usize].max(dist);
                }
            }
        }
    }
    let mut delayed: HashMap<usize, u32> = HashMap::new();
    let delay = |delayed: &mut HashMap<usize, u32>, i: usize, p: u32| {
        if slow >> p & 1 != 0 {
            let d = delayed.entry(i).or_insert(0);
            *d = (*d).max(pred_wait[p as usize]);
        }
    };
    let order_up: Vec<u32> = (0..7u32).collect();
    let order_down: Vec<u32> = (0..7u32).rev().collect();

    let mut out = Predicates {
        stand: HashMap::new(),
        scratch_at: HashMap::new(),
        saves: HashMap::new(),
        restores: HashMap::new(),
        save_slots: 0,
        held: HashMap::new(),
        held_lives: Vec::new(),
        lives: Vec::new(),
        held_predicates: Vec::new(),
        delayed: HashMap::new(),
        slow_predicates: slow,
    };
    let mut taken_at = vec![0u32; n];
    // (first, last, register) of what is saved.
    let mut runs: Vec<(usize, usize, u32)> = Vec::new();
    let mut slot = |i0: usize, i1: usize| {
        let k = (0..)
            .find(|k| !runs.iter().any(|&(a, b, used)| used == *k && a <= i1 && i0 <= b))
            .unwrap();
        runs.push((i0, i1, k));
        k
    };

    // The lives: instructions where a uniform predicate holds something, joined where the value
    // goes from one to the next. (first instruction, predicate, instructions in order.)
    let mut lives: Vec<(usize, u32, Vec<usize>)> = Vec::new();
    for &q in &fd.uniform_predicates {
        let bit = 7 + q;
        let alive: Vec<usize> = (0..n).filter(|&i| busy[i] >> bit & 1 != 0).collect();
        let mut parent: HashMap<usize, usize> = alive.iter().map(|&i| (i, i)).collect();
        fn find(parent: &mut HashMap<usize, usize>, mut x: usize) -> usize {
            while parent[&x] != x {
                let up = parent[&parent[&x]];
                parent.insert(x, up);
                x = up;
            }
            x
        }
        for &i in &alive {
            for &j in &live.successors[i] {
                if parent.contains_key(&j) && live.live_out[i] >> bit & 1 != 0 && live.live_in[j] >> bit & 1 != 0 {
                    let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                    parent.insert(a, b);
                }
            }
        }
        let mut parts: Vec<(usize, Vec<usize>)> = Vec::new();
        for &i in &alive {
            let root = find(&mut parent, i);
            match parts.iter_mut().find(|(r, _)| *r == root) {
                Some((_, part)) => part.push(i),
                None => parts.push((root, vec![i])),
            }
        }
        lives.extend(parts.into_iter().map(|(_, part)| (part[0], q, part)));
    }
    lives.sort();

    let give = |out: &mut Predicates, taken_at: &mut Vec<u32>, q: u32, part: &[usize], p: u32, saved: bool| {
        for &i in part {
            out.stand.insert((q, i), p);
            taken_at[i] |= 1 << p;
        }
        out.lives.push((q, p, saved, part.to_vec()));
    };
    let mut hard: Vec<(u32, Vec<usize>)> = Vec::new();
    for (_, q, part) in lives {
        if part.iter().any(|&i| live.external[i]) {
            return Err(Refused(
                "a uniform predicate is alive across a call to another function".into(),
            ));
        }
        let taken = part
            .iter()
            .fold(0u32, |taken, &i| taken | (busy[i] as u32 & 0x7f) | taken_at[i]);
        match pick(taken, &order_up, slow, never) {
            Some(p) => {
                give(&mut out, &mut taken_at, q, &part, p, false);
                for &i in &part {
                    if live.defs[i] >> (7 + q) & 1 != 0 {
                        delay(&mut delayed, i, p);
                    }
                }
            }
            None => hard.push((q, part)),
        }
    }
    let mut targets = vec![false; n + 1];
    for (i, leads) in live.successors.iter().enumerate() {
        for &j in leads {
            if j != i + 1 {
                targets[j] = true;
            }
        }
    }
    let straight = |i: usize| {
        let d = &fd.src[i];
        live.successors[i] == [i + 1]
            && !live.external[i]
            && (d.kind != Kind::BRU || matches!(d.mnemonic, "BSSY" | "BMOV"))
    };
    let mut later: Vec<(u32, Vec<usize>)> = Vec::new();
    for (q, part) in hard {
        let (i0, i1) = (part[0], part[part.len() - 1]);
        let touched = (i0..=i1).fold(0u32, |touched, i| {
            touched | ((live.defs[i] | live.uses[i]) as u32 & 0x7f) | taken_at[i]
        });
        let p = pick(touched, &order_up, slow, never);
        let run = part.len() == i1 - i0 + 1 && !(i0 + 1..=i1).any(|i| targets[i]) && (i0..=i1).all(straight);
        if p.is_none() || !run {
            later.push((q, part));
            continue;
        }
        let p = p.unwrap();
        let k = slot(i0, i1);
        give(&mut out, &mut taken_at, q, &part, p, true);
        out.saves.entry(i0).or_default().push((k, p));
        out.restores.entry(i1).or_default().push((k, p));
        delay(&mut delayed, i0, p);
    }
    // The lives that are held in a register: a predicate for each instruction that names the
    // uniform one, there only. (After the lives above: what stands for those all along them
    // is taken.)
    for (q, part) in &later {
        let q = *q;
        for &i in part {
            let reads = live.uses[i] >> (7 + q) & 1 != 0;
            let writes = live.defs[i] >> (7 + q) & 1 != 0;
            if !reads && !writes {
                continue;
            }
            let taken = (busy[i] as u32 & 0x7f) | taken_at[i];
            let mut p = pick(taken, &order_down, slow, never);
            let saved = p.is_none();
            if saved {
                let touched = ((live.defs[i] | live.uses[i]) as u32 & 0x7f) | taken_at[i];
                p = pick(touched, &order_down, slow, never);
                // (Given back after the instruction: which must then go on to the next one, always.)
                let p = p.ok_or_else(|| Refused("no predicate is free while a uniform predicate is alive".into()))?;
                if !straight(i) {
                    return Err(Refused(
                        "no predicate is free while a uniform predicate is alive".into(),
                    ));
                }
                let k = slot(i, i);
                out.saves.entry(i).or_default().push((k, p));
                out.restores.entry(i).or_default().push((k, p));
            }
            let p = p.unwrap();
            delay(&mut delayed, i, p);
            out.stand.insert((q, i), p);
            taken_at[i] |= 1 << p;
            // (An instruction that may leave the uniform predicate as it was, under a guard
            // that does not hold, finds it in the ordinary one: read back before it too.)
            let read_back = reads || (writes && live.kills[i] >> (7 + q) & 1 == 0);
            let write_down = writes;
            out.held.entry(i).or_default().push((q, p, read_back, write_down));
            out.lives.push((q, p, saved, vec![i]));
        }
        out.held_lives.push((q, part.clone()));
    }
    out.held_predicates = {
        let mut s: Vec<u32> = out
            .held_lives
            .iter()
            .map(|(q, _)| *q)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        s.sort();
        s
    };
    for (i, d) in fd.src.iter().enumerate() {
        if d.name.starts_with("imnmx__") {
            // One that holds nothing alive after it, is not one of its own, and does not stand for
            // a uniform predicate alive there; or else one it does not touch, saved.
            let taken = (busy[i] as u32 & 0x7f) | taken_at[i];
            let scratch = match pick(taken, &order_down, slow, never) {
                Some(p) => (p, None),
                None => {
                    let mut touched = (live.defs[i] | live.uses[i]) as u32 & 0x7f;
                    for &q in &fd.uniform_predicates {
                        if (live.defs[i] | live.uses[i]) >> (7 + q) & 1 != 0 {
                            touched |= 1 << out.stand[&(q, i)];
                        }
                    }
                    let p = pick(touched, &order_down, slow, never)
                        .ok_or_else(|| Refused(format!("no predicate to compute a minimum into at 0x{:x}", 16 * i)))?;
                    (p, Some(slot(i, i)))
                }
            };
            delay(&mut delayed, i, scratch.0);
            out.scratch_at.insert(i, scratch);
        }
    }
    out.save_slots = runs.iter().map(|&(_, _, k)| k + 1).max().unwrap_or(0);
    out.delayed = delayed;
    Ok(out)
}

/// `place`, `pool`, `save`, `lane`, `scratch`, `free`: where the uniform registers go, the two
/// registers to compute into, the first of the registers to save predicates in, the register of
/// the lane, the first of the scratch registers and the first register left free, for this
/// function and those it calls or that call it (`placement`).
/// The instructions of a function that are not only entered from the one before them: where
/// the relative operand of an instruction leads (a branch, a call within the section, the place
/// where the threads of a warp are to meet again), where the table of a branch through a
/// register leads, and what follows a call within the section when the section has a return
/// for it. None when that cannot be known: a branch through a register that has no table, a
/// jump to an address.
pub(super) fn entries(fd: &FuncData) -> Option<HashSet<usize>> {
    let n = fd.src.len();
    let within = |target: i64| target % 16 == 0 && 0 <= target && target < 16 * n as i64;
    let returns = fd.src.iter().any(|d| d.name == "ret__REL");
    let mut out = HashSet::new();
    for (i, d) in fd.src.iter().enumerate() {
        if d.name.starts_with("brx") || d.name.starts_with("jmx") {
            // Through a register: where its table says, or anywhere.
            let table = fd.tables.iter().find(|(branch, _)| *branch == 16 * i as u64);
            match table {
                Some((_, places)) if d.name == "brx_" => {
                    out.extend(places.iter().filter(|&&t| within(t as i64)).map(|&t| (t / 16) as usize))
                }
                _ => return None,
            }
            continue;
        }
        if let Some((rel_fid, scale)) = find_relative(d, 75) {
            let target = 16 * i as i64 + 16 + field::signed(rel_fid, d.word) * scale as i64;
            if within(target) {
                out.insert((target / 16) as usize);
            }
        } else if d.name.starts_with("jmp") {
            return None;
        }
        if returns && d.name.starts_with("call_rel") && i + 1 < n {
            out.insert(i + 1);
        }
    }
    Some(out)
}
