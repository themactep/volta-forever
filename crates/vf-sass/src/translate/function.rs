//! Single-function translation: build items for each instruction, handle borrowed registers
//! and their loads and stores, then resolve relocations (`follow`).

use std::collections::{HashMap, HashSet};

use crate::assemble;
use crate::decode::{Decoder, LETTER_R, LETTER_U};
use crate::field;
use crate::generated::special::*;

use super::borrowed::Borrowed;
use super::context::{Ctx, Predicates, entries};
use super::finish::finish;
use super::instruction::translate_one;
use super::parse::FuncData;
use super::return_addr::return_addresses;
use super::schedule::schedule;
use super::util::{find_relative, read_announced, unmarked};
use super::{
    AROUND, ARRIVES, HALVES, Item, Refused, Relocation, SEQUENCE, TransResult, hex, regset_empty, regset_test,
    stack_top, unannounced, with_stall,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn translate_function(
    dec70: &mut Decoder,
    fd: &FuncData,
    predicates: &Predicates,
    place: &HashMap<u32, u32>,
    pool: Option<u32>,
    save: Option<u32>,
    lane: Option<u32>,
    kept: Option<u32>,
    gathered: Option<u32>,
    scratch: Option<u32>,
    free: u32,
    q: Option<&crate::pairing::Plan>,
    rows_at: HashMap<u32, u32>,
    kept_rows: HashMap<&'static str, u32>,
    aligned_rows: bool,
    has_product_non_pair: bool,
    hold_at: &HashMap<u32, u32>,
    borrow: Option<&mut Borrowed>,
) -> Result<TransResult, Refused> {
    if fd.needs_lane && fd.entered_inside {
        return Err(Refused(
            "matrix instructions in a function that can be entered elsewhere than at its start".into(),
        ));
    }
    let rows_n = q.map_or_else(Default::default, |p| p.rows());
    let mut ctx = Ctx {
        dec70,
        place,
        pool_at: pool,
        save_at: save,
        lane_at: lane,
        kept_at: kept,
        gathered_at: gathered,
        scratch_at_reg: scratch,
        src: &fd.src,
        entered: if has_product_non_pair || q.is_some() {
            entries(fd)
        } else {
            None
        },
        holds: None,
        next_register: free,
        predicates,
        at: 0,
        q,
        rows_at,
        kept_rows,
        rows_n,
        aligned_rows,
        hold_at,
        borrow,
        pool_taken: HashMap::new(),
        pool_width: HashMap::new(),
        given: regset_empty(),
        given_at: HashMap::new(),
        lane_taken: HashMap::new(),
        scratch_taken: HashMap::new(),
        mem_regs: HashMap::new(),
        base_reg: None,
        spilled: HashMap::new(),
        lower_item_idx: None,
        top_again: None,
        entry_sets_r1: false,
    };

    let frame = ctx.borrow.as_ref().map_or(0, |b| b.frame_bytes);
    let has_room = ctx.borrow.as_ref().is_some_and(|b| b.room);
    // A frame addressed from the stack pointer (it is moved under the frame at the start); or
    // from the top of the thread's local memory, loaded at each instruction that reads it.
    let from_r1 = ctx.borrow.as_ref().is_some_and(|b| b.from_r1);
    let mut items: Vec<Item> = Vec::new();
    let mut targets: Vec<u64> = Vec::new();
    for (i, d) in fd.src.iter().enumerate() {
        ctx.at = i;
        ctx.given = regset_empty();
        ctx.mem_regs.clear();
        ctx.base_reg = None;
        if i == 0 {
            ctx.entry_sets_r1 = stack_top(d) && d.writes == [(LETTER_R, 1)] && crate::pairing::unguarded(d);
        } else if stack_top(d) && ctx.top_again.is_none() {
            ctx.top_again = Some(i);
        }
        // Level 3: check the stack pointer, where the frame is addressed from it.
        if frame > 0 && from_r1 {
            if i == 0 {
                // The first instruction must set R1 from c[0x0][0x28] unguarded.
                if !ctx.entry_sets_r1 {
                    return Err(Refused(
                        "uniform registers in memory, and the stack pointer is not set once, by the first instruction"
                            .into(),
                    ));
                }
            } else if stack_top(d) {
                return Err(Refused(format!(
                    "uniform registers in memory, and the top of the stack is read again at 0x{:x}",
                    16 * i
                )));
            }
        }
        // Level 3: load from memory before the instruction.
        let before_mem = ctx.loads_before(i, d)?;
        let mut new_items = if let Some(op) = fd.ampere_ops.get(&i) {
            super::ampere::emit(&mut ctx, i, d, op)?
        } else {
            translate_one(&mut ctx, i, d)?
        };
        // The source compiler scheduled even shared opcodes for Ampere. Volta's
        // fixed-pipeline latency differs, including LEA.HI.X -> LDG and HADD2 -> FADD.
        if fd.ampere {
            for item in &mut new_items {
                item.changed = true;
                item.anew = true;
            }
        }
        // Level 3: insert lane S2R where a sequence goes by it.
        if let Some(&lane_r) = ctx.lane_taken.get(&i) {
            let word = assemble::assemble(
                "s2r_",
                &[
                    ("Rd", assemble::OpVal::Int(lane_r)),
                    ("SRa", assemble::OpVal::Str("SR_LANEID")),
                    ("dst_wr_sb", assemble::OpVal::Int(SEQUENCE)),
                ],
            )
            .ok_or_else(|| Refused("cannot assemble S2R for lane".into()))?;
            let mut lane_item = Item::new(word, i, "lane", true);
            lane_item.then = Some(SEQUENCE);
            new_items.insert(0, lane_item);
        }
        // A held predicate in the frame only needs a register while this instruction uses
        // its value. Keep the conversion inside the saves/restores of any borrowed register.
        let mut frame_before = Vec::new();
        let mut frame_after = Vec::new();
        for &(q, p, reads, writes) in predicates.held.get(&i).into_iter().flatten() {
            if let Some(slot) = ctx.borrow.as_ref().and_then(|b| b.predicate_slots.get(&q)).copied() {
                let r = ctx.take(1, None)?;
                if reads {
                    ctx.load_frame_word(slot, r, i, format!("mem: load UP{q}"), &mut frame_before)?;
                    frame_before.last_mut().unwrap().then = Some(ARRIVES);
                    frame_before.push(ctx.read_back(r, p, i)?);
                }
                if writes {
                    frame_after.push(ctx.written_down(r, p, i)?);
                    ctx.store_frame_word(slot, r, i, format!("mem: store UP{q}"), &mut frame_after)?;
                }
            }
        }
        // Level 3: store to memory after the instruction.
        let after_mem = ctx.stores_after(i, d)?;
        {
            let mut combined = before_mem;
            combined.extend(frame_before);
            combined.extend(new_items);
            combined.extend(frame_after);
            combined.extend(after_mem);
            new_items = combined;
        }
        // Room made: the registers taken from the program are put in the frame before
        // everything the instruction became, and brought back after it. A store reads its
        // operands late: it says when, and what follows waits; what is brought back arrives
        // on a scoreboard, and what follows the last waits.
        if let Some(spill_regs) = ctx.spilled.get(&i).cloned() {
            let borrow = ctx.borrow.as_ref().unwrap();
            let depth_at_i = borrow.depth[i].unwrap();
            let n_slots = borrow.slot_count() as i32;
            let mut saves: Vec<Item> = Vec::new();
            let mut backs: Vec<Item> = Vec::new();
            for (k, &r) in spill_regs.iter().enumerate() {
                let offset = (depth_at_i + 4 * (n_slots + k as i32)) as u32;
                // STL: save the register into the frame.
                let stl_word = assemble::assemble(
                    "stl__sImmOffset",
                    &[
                        ("Ra", assemble::OpVal::Int(1)),
                        ("Rb", assemble::OpVal::Int(r)),
                        ("Ra_offset", assemble::OpVal::Int(offset)),
                    ],
                )
                .ok_or_else(|| Refused("cannot assemble STL for room save".into()))?;
                saves.push(ctx.late_item(
                    "stl__sImmOffset",
                    stl_word,
                    i,
                    "room: away",
                    Some(format!("room: R{} away", r)),
                )?);
                // LDL: bring the register back.
                let ldl_word = assemble::assemble(
                    "ldl__sImmOffset",
                    &[
                        ("Rd", assemble::OpVal::Int(r)),
                        ("Ra", assemble::OpVal::Int(1)),
                        ("Ra_offset", assemble::OpVal::Int(offset)),
                        ("dst_wr_sb", assemble::OpVal::Int(ARRIVES)),
                    ],
                )
                .ok_or_else(|| Refused("cannot assemble LDL for room restore".into()))?;
                let mut back_item = Item::new(ldl_word, i, "room: back", true);
                back_item.what_detail = Some(format!("room: R{} back", r));
                backs.push(back_item);
            }
            backs.last_mut().unwrap().then = Some(ARRIVES);
            let borrow = ctx.borrow.as_mut().unwrap();
            borrow.spill = borrow.spill.max(spill_regs.len() as u32);
            let mut combined = saves;
            combined.extend(new_items);
            combined.extend(backs);
            new_items = combined;
        }
        if let Some((rel_fid, scale)) = find_relative(d, 75) {
            let v = field::signed(rel_fid, d.word);
            let target = 16 * i as i64 + 16 + v * scale as i64;
            if target % 16 != 0 || target < 0 || target > 16 * fd.src.len() as i64 {
                return Err(Refused(format!(
                    "a branch at 0x{:x} to {}, outside the function",
                    16 * i,
                    hex(target)
                )));
            }
            let target = target as u64;
            new_items.last_mut().unwrap().target = Some(target);
            targets.push(target);
        }
        // The uniform predicates held in a register: read back for the instruction, written after it.
        {
            let mut held_before: Vec<Item> = Vec::new();
            let mut held_after: Vec<Item> = Vec::new();
            for &(q, p, reads, writes) in predicates.held.get(&i).into_iter().flatten() {
                if ctx.borrow.as_ref().is_some_and(|b| b.predicate_slots.contains_key(&q)) {
                    continue;
                }
                let holder = ctx.holder(q)?;
                if reads {
                    held_before.push(ctx.read_back(holder, p, i)?);
                }
                if writes {
                    held_after.push(ctx.written_down(holder, p, i)?);
                }
            }
            new_items.splice(0..0, held_before);
            new_items.extend(held_after);
        }
        // The predicates saved for a life that starts here, and given back after one that ends here.
        let mut around: Vec<Item> = Vec::new();
        for &(k, p) in predicates.saves.get(&i).into_iter().flatten() {
            around.push(ctx.saved(k, p, i)?);
        }
        around.extend(new_items);
        for &(k, p) in predicates.restores.get(&i).into_iter().flatten() {
            around.push(ctx.given_back(k, p, i)?);
        }
        // The accumulators of pairs brought to the state the instruction, or what follows it, takes them in.
        if let Some(plan) = ctx.q {
            let before = ctx.converted(i, &plan.before)?;
            let after = ctx.converted(i, &plan.after)?;
            around.splice(0..0, before);
            around.extend(after);
        }
        // The lane read is the first item of instruction 0: offset 0 stays where it was (a return
        // within the section counts from there), and a branch to offset 0 reads the lane again.
        // Level 3: the stack pointer starts under the frame. With room to be made, how large
        // the frame is is known at the end: `lower_item_idx`.
        if i == 0 && from_r1 && (frame > 0 || (has_room && ctx.entry_sets_r1)) {
            let lower_frame = frame.max(16);
            let iadd = assemble::assemble(
                "iadd3_imm__RsIR_RIR",
                &[
                    ("Rd", assemble::OpVal::Int(1)),
                    ("Ra", assemble::OpVal::Int(1)),
                    ("Sb", assemble::OpVal::Int(-(lower_frame as i32) as u32)),
                    ("Rc", assemble::OpVal::Int(255)),
                ],
            )
            .ok_or_else(|| Refused("cannot assemble IADD3 for R1".into()))?;
            let mut r1_item = Item::new(iadd, 0, "mem: R1", true);
            r1_item.what_detail = Some(format!("mem: R1 -= {}", lower_frame));
            around.push(r1_item);
            ctx.lower_item_idx = Some(items.len() + around.len() - 1);
        }
        if i == 0 && fd.needs_lane && ctx.lane_at.is_some() {
            let lane_reg = ctx.lane()?;
            let word = assemble::assemble(
                "s2r_",
                &[
                    ("Rd", assemble::OpVal::Int(lane_reg)),
                    ("SRa", assemble::OpVal::Str("SR_LANEID")),
                    ("dst_wr_sb", assemble::OpVal::Int(SEQUENCE)),
                ],
            )
            .ok_or_else(|| Refused("cannot assemble S2R for lane".into()))?;
            let mut item = Item::new(word, 0, "lane", true);
            item.then = Some(SEQUENCE);
            let mut first = vec![item];
            if has_product_non_pair {
                first.extend(ctx.kept()?);
            }
            if ctx.q.is_some() {
                first.extend(ctx.kept_for_rows()?);
            }
            around.splice(0..0, first);
        }
        // Its translation writes or saves a predicate that a late instruction of the program
        // writes (`docs/translator.md`, "Timing rules"): first an instruction that does nothing, waits for every
        // scoreboard, and lasts until whatever was issued before has written.
        if let Some(&cycles) = predicates.delayed.get(&i)
            && cycles > 0
        {
            let nop_word = assemble::assemble("nop_", &[]).ok_or_else(|| Refused("cannot assemble NOP".into()))?;
            let mut nothing = vec![Item::new(
                with_stall(nop_word, cycles.min(15), false) | (63u128 << WAIT),
                i,
                "late predicate",
                true,
            )];
            let mut remaining = cycles.saturating_sub(15);
            while remaining > 0 {
                nothing.push(Item::new(
                    with_stall(nop_word, remaining.min(15), false),
                    i,
                    "late predicate",
                    true,
                ));
                remaining = remaining.saturating_sub(15);
            }
            around.splice(0..0, nothing);
        }
        // What the instruction waited for, the first instruction of what it became waits for: a
        // wait is for everything after it, and the compiler may have put on this one the wait
        // for what a later one reads. (The constants of ULDC lost theirs until 2026-11-17: the test
        // `translated-waits-moved` of tests/gpu/run.sh is for that.)
        around[0].word |= d.word & (63u128 << WAIT);
        // Borrowed registers: what the program's instructions wrote there last may still be on
        // its way. The first instruction of the translation that writes a borrowed register
        // waits for all six scoreboards (`docs/translator.md`, "Timing rules").
        // What the instruction was given (take): the pool, and at level 3 the rest; and at
        // level 3 the register that holds a uniform predicate, where it is written.
        let mut at_instr: HashSet<u32> = HashSet::new();
        for r in 2..255u32 {
            if regset_test(&ctx.given, r) {
                at_instr.insert(r);
            }
        }
        if let Some(ref borrow) = ctx.borrow {
            // A held uniform predicate's register, where it is written and that register is
            // one of the program's (not above).
            for &(q, _p, _reads, writes) in predicates.held.get(&i).into_iter().flatten() {
                if writes
                    && let Some(&hr) = hold_at.get(&q)
                    && hr <= borrow.top
                {
                    at_instr.insert(hr);
                }
            }
        }
        if let Some(ref borrow) = ctx.borrow {
            let mut mine: HashSet<u32> = HashSet::new();
            for &(l, u) in &d.writes {
                if l == LETTER_U
                    && let Some(&r) = borrow.homes.get(&u)
                {
                    mine.insert(r);
                }
            }
            if let Some(&pool_r) = ctx.pool_taken.get(&i) {
                mine.insert(pool_r);
                mine.insert(pool_r + 1);
            }
            mine.extend(&at_instr);
            if !mine.is_empty() {
                around[0].word |= 63u128 << WAIT;
                for item in &mut around {
                    item.borrows.clone_from(&mine);
                }
            }
        }
        around[0].first = true;
        items.extend(around);
    }
    // After the loop: finish the frame for room made.
    if has_room {
        let borrow = ctx.borrow.as_mut().unwrap();
        // The frame with the room made: the uniform registers, then as many words as an
        // instruction put away at once.
        borrow.frame_bytes = (4 * (borrow.slot_count() + borrow.spill) + 15) & !15;
        if from_r1
            && borrow.frame_bytes > 0
            && let Some(ta) = ctx.top_again
        {
            return Err(Refused(format!(
                "a frame for the translation, and the top of the stack is read again at 0x{:x}",
                16 * ta
            )));
        }
        if let Some(idx) = ctx.lower_item_idx {
            let final_frame = borrow.frame_bytes;
            if final_frame > 0 {
                // Rewrite the R1 lowering instruction with the final frame size.
                let iadd = assemble::assemble(
                    "iadd3_imm__RsIR_RIR",
                    &[
                        ("Rd", assemble::OpVal::Int(1)),
                        ("Ra", assemble::OpVal::Int(1)),
                        ("Sb", assemble::OpVal::Int(-(final_frame as i32) as u32)),
                        ("Rc", assemble::OpVal::Int(255)),
                    ],
                )
                .ok_or_else(|| Refused("cannot assemble IADD3 for R1 (final)".into()))?;
                // Preserve the wait mask from the original item.
                let old_wait = items[idx].word & (63u128 << WAIT);
                items[idx].word = iadd | old_wait;
                items[idx].what_detail = Some(format!("mem: R1 -= {}", final_frame));
            } else {
                // No frame needed: remove the R1 lowering instruction.
                items.remove(idx);
            }
        }
    }
    for (branch, places) in &fd.tables {
        let within = |offset: u64| offset.is_multiple_of(16) && offset < 16 * fd.src.len() as u64;
        if !within(*branch) || fd.src[(*branch / 16) as usize].name != "brx_" {
            return Err(Refused(format!(
                "a jump table for 0x{:x}, which is not a branch through a register",
                branch
            )));
        }
        if !places.iter().all(|&t| within(t)) {
            return Err(Refused(format!(
                "the jump table of the branch at 0x{:x} leads outside the function",
                branch
            )));
        }
        targets.extend(places.iter().copied());
    }

    if let Some(ref borrow) = ctx.borrow
        && !borrow.used.is_empty()
    {
        let used = &borrow.used;
        for item in &mut items {
            let d70 = ctx.dec70.decode(item.word);
            let Some(ref d70) = d70 else { continue };
            let board = ((item.word >> WRITE_BARRIER) & 7) as u32;
            let written: HashSet<u32> = d70
                .writes
                .iter()
                .filter(|&&(l, _)| l == LETTER_R)
                .map(|&(_, r)| r)
                .collect();
            if !written.is_disjoint(used) && unannounced(d70) && !item.covered {
                // Borrowed leaves out the registers the program writes so; this is for
                // what the translation made of an instruction. A covered instruction is
                // part of a sequence whose later instruction says when the result is there,
                // and what follows waits for it: the late arrival is handled.
                return Err(Refused(format!(
                    "a borrowed register receives at 0x{:x} a result that nothing announces",
                    16 * item.origin
                )));
            }
            if board != 7 && !written.is_disjoint(&item.borrows) {
                // A result of the translation that is written when it arrives, in a register
                // the program may write next: the next instruction waits until it is there.
                item.also.push(board);
            }
            // An instruction that reads a borrowed register some time after it is issued,
            // for the program: it says on a scoreboard when it has read, so that what the
            // translation writes there next, which waits for every scoreboard, comes after.
            if !item.late
                && d70.kind.is_late()
                && ((item.word >> READ_BARRIER) & 7) == 7
                && d70.reads.iter().any(|&(l, r, _)| l == LETTER_R && used.contains(&r))
                && let Ok(w) = read_announced(d70.name, unmarked(d70.name, item.word, item.what), item.origin)
            {
                item.word = w;
                item.says = true;
            }
        }
    }

    for k in 0..items.len() {
        if items[k].late {
            // (`late`.) The instruction after it waits until it has read.
            items[k].then = Some(((items[k].word >> READ_BARRIER) & 7) as u32);
        }
        if items[k].then.is_some() || !items[k].also.is_empty() {
            if k + 1 == items.len() {
                return Err(Refused("a function that ends on a memory instruction".into()));
            }
            if let Some(board) = items[k].then {
                items[k + 1].word |= 1u128 << (WAIT + board);
            }
            for &board in &items[k].also.clone() {
                items[k + 1].word |= 1u128 << (WAIT + board);
            }
        }
    }

    // Decode all items as sm_70
    for item in &mut items {
        item.d = Some(match ctx.dec70.decode(item.word) {
            Some(d) => d,
            None => {
                return Err(Refused(format!(
                    "the translation of the instruction at 0x{:x} ({}) is not an sm_70 instruction: {:032x}",
                    16 * item.origin,
                    fd.src[item.origin].name,
                    item.word
                )));
            }
        });
    }

    let (relocate, lepc_pairs) = return_addresses(&mut items, &targets, fd)?;
    schedule(&mut items, &targets)?;
    super::return_addr::lepc_pairs_adjacent(&items, &lepc_pairs)?;
    let (code, new_of, count) = finish(&mut items, &fd.src, ctx.next_register, ctx.dec70)?;
    let frame_bytes = ctx.borrow.as_ref().map_or(0, |b| b.frame_bytes);
    let lives = lives(predicates, hold_at, ctx.borrow.as_deref(), &ctx.given_at, &ctx.spilled);
    Ok(TransResult {
        code,
        new_of,
        count,
        items,
        relocate,
        moved: Vec::new(),
        frame_bytes,
        lives,
        paired: q.is_some_and(|plan| !plan.sites.is_empty()),
    })
}

/// What the validator is told of a function (`MAP.lives`), in the order in which
/// `reference/translate.py` writes it: the lives of the uniform predicates, those held, and with
/// borrowed registers the homes, the pools and the room taken at each instruction, the words of
/// the frame and its size.
fn lives(
    predicates: &Predicates,
    hold_at: &HashMap<u32, u32>,
    borrow: Option<&Borrowed>,
    given_at: &HashMap<usize, Vec<(u32, u32)>>,
    spilled: &HashMap<usize, Vec<u32>>,
) -> Vec<(i64, i64, &'static str, Vec<u64>)> {
    let offsets = |part: &[usize]| part.iter().map(|&at| 16 * at as u64).collect::<Vec<_>>();
    let mut lives: Vec<_> = predicates
        .lives
        .iter()
        .map(|(q, p, saved, part)| {
            let how = if *saved { "saved" } else { "free" };
            (i64::from(*q), i64::from(*p), how, offsets(part))
        })
        .collect();
    for (q, part) in &predicates.held_lives {
        // (In a register, or else in the frame: the translation of the function has one or the
        // other for each held predicate.)
        let held = match hold_at.get(q) {
            Some(&r) => Some((i64::from(r), "held")),
            None => borrow.and_then(|b| Some((-b.slot_offset(*b.predicate_slots.get(q)?), "held-slot"))),
        };
        if let Some((location, kind)) = held {
            lives.push((i64::from(*q), location, kind, offsets(part)));
        }
    }
    if let Some(b) = borrow {
        let mut homes: Vec<(u32, u32)> = b.homes.iter().map(|(&u, &r)| (u, r)).collect();
        homes.sort_unstable();
        for (u, r) in homes {
            let span = b.spans.get(&u).map_or_else(Vec::new, |span| offsets(span));
            lives.push((i64::from(u), i64::from(r), "home", span));
        }
        let mut given: Vec<_> = given_at.iter().collect();
        given.sort_unstable();
        for (&at, runs) in given {
            for &(first, width) in runs {
                lives.push((i64::from(width), i64::from(first), "pool", vec![16 * at as u64]));
            }
        }
        let mut room: Vec<_> = spilled.iter().collect();
        room.sort_unstable();
        for (&at, registers) in room {
            for &r in registers {
                lives.push((1, i64::from(r), "room", vec![16 * at as u64]));
            }
        }
        let mut slots: Vec<(u32, u32)> = b.slots.iter().map(|(&u, &word)| (u, word)).collect();
        slots.sort_unstable();
        for (u, word) in slots {
            lives.push((i64::from(u), -b.slot_offset(word), "slot", Vec::new()));
        }
        if b.frame_bytes != 0 {
            lives.push((i64::from(b.frame_bytes), 0, "frame", Vec::new()));
        }
    }
    lives
}

/// Where each relocation on the code of a function applies once the function is translated: on
/// its instruction, which is there as it was, wherever it is now.
///
/// One rewritten instruction also keeps its relocation: a number moved into a uniform register
/// (`UMOV UR4, 32@lo(table)`), which becomes the same number moved into an ordinary register, with
/// the number in the same 32 bits of the word, for the two kinds of relocation that write half of
/// an address there. Any other instruction that the translation rewrites makes the function one
/// it refuses: what is written at a place the instruction no longer has would be written into
/// another one. (Until 2026-11-17 the relocations stayed where they were; PyTorch has none on
/// code, cuFFT has them in 145 kernels.)
pub(super) fn follow(fd: &FuncData, tr: &TransResult, list: &[Relocation]) -> Result<Vec<u64>, Refused> {
    let mut became: HashMap<usize, Vec<usize>> = HashMap::new();
    for (k, item) in tr.items.iter().enumerate() {
        if !AROUND.contains(&item.what) {
            became.entry(item.origin).or_default().push(k);
        }
    }
    let mut moved = Vec::with_capacity(list.len());
    for r in list {
        let origin = (r.offset / 16) as usize;
        if origin >= fd.src.len() {
            return Err(Refused("a relocation outside its code".into()));
        }
        let kept = match became.get(&origin).map(Vec::as_slice) {
            Some(&[k]) => {
                let (item, source) = (&tr.items[k], &fd.src[origin]);
                let as_it_was = !item.changed && item.fix.is_none() && item.target.is_none();
                let half_of_an_address = HALVES.contains(&r.kind)
                    && item.what == "ordinary registers"
                    && source.name == "umov__UI"
                    && item.d.as_ref().is_some_and(|d| d.name == "mov__RuIR_RI")
                    && (item.word >> 32) as u32 == (source.word >> 32) as u32;
                (as_it_was || half_of_an_address).then_some(16 * k as u64 + r.offset % 16)
            }
            _ => None,
        };
        match kept {
            Some(now) => moved.push(now),
            None => {
                return Err(Refused(
                    "a relocation on an instruction that the translation rewrites".into(),
                ));
            }
        }
    }
    Ok(moved)
}
