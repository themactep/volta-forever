//! Cubin-level translation: parse the ELF, group functions by call edges, translate
//! each group, rewrite attributes and relocations, and re-tag for sm_70.

use std::collections::{HashMap, HashSet};

use crate::assemble;
use crate::decode::Decoder;
use crate::elf::Elf;
use crate::generated::special::*;

use super::attributes::{check_attributes, find_exits, rewrite_attributes, strip_attributes, strip_attrs_by_kind};
use super::borrowed::Borrowed;
use super::context::{Predicates, entries, stand_ins};
use super::function::{follow, translate_function};
use super::parse::{FuncData, parse_function, placement, tables_of};
use super::{
    FunctionReport, LifeEntry, MapEntry, Options, Refused, Relocation, STAY, TRAP, TransResult, TranslateResult,
    with_stall,
};

/// Translate an sm_75 or supported sm_80 cubin to sm_70.
pub fn translate(data: &[u8], want_map: bool) -> Result<TranslateResult, String> {
    translate_with_options(data, want_map, &Options::default())
}

/// Translate an sm_75 or supported sm_80 cubin to sm_70 with the given options.
pub fn translate_with_options(data: &[u8], want_map: bool, opts: &Options) -> Result<TranslateResult, String> {
    let mut dec75 = Decoder::new(75);
    let mut dec70 = Decoder::new(70);
    let mut e = Elf::parse(data)?;
    let arch = super::ampere::source(&e.header)?;
    let symbols = e.symbols()?;

    let texts: Vec<usize> = e
        .sections
        .iter()
        .enumerate()
        .filter(|(_, s)| s.name.starts_with(".text.") && s.stype == 1)
        .map(|(i, _)| i)
        .collect();
    for (k, &si) in texts.iter().enumerate() {
        if texts[..k]
            .iter()
            .any(|&other| e.sections[other].name == e.sections[si].name)
        {
            return Err("two sections of code with the same name".into());
        }
    }

    // Union-find for call groups
    let mut group: HashMap<usize, usize> = texts.iter().map(|&i| (i, i)).collect();
    fn find(group: &mut HashMap<usize, usize>, mut x: usize) -> usize {
        while group[&x] != x {
            x = group[&x];
        }
        x
    }
    let by_symbol: HashMap<usize, usize> = symbols
        .iter()
        .enumerate()
        .filter(|(_, sym)| group.contains_key(&(sym.shndx as usize)))
        .map(|(i, sym)| (i, sym.shndx as usize))
        .collect();

    // Relocations: the driver, or a linker, writes addresses into the instructions and the data
    // they name once it knows them. Those on code are kept by the function they are on (`follow`);
    // the functions whose code one names, and the functions that hold such a one, are noted: the
    // place it names moves with the translation, and nothing here carries it over. (Those of the
    // debug sections are for a debugger.)
    let mut relocations: HashMap<usize, Vec<Relocation>> = HashMap::new();
    let (mut holders, mut named) = (HashSet::new(), HashSet::new());
    for (section, s) in e.sections.iter().enumerate() {
        let on = s.info as usize;
        if (s.stype != 4 && s.stype != 9) || on >= e.sections.len() {
            continue;
        }
        if e.sections[on].name.starts_with(".debug") || e.sections[on].name.starts_with(".nv_debug") {
            continue;
        }
        let step = if s.stype == 4 { 24 } else { 16 };
        for at in (0..s.data.len().saturating_sub(step - 1)).step_by(step) {
            let offset = u64::from_le_bytes(s.data[at..at + 8].try_into().unwrap());
            let info = u64::from_le_bytes(s.data[at + 8..at + 16].try_into().unwrap());
            let names = symbols
                .get((info >> 32) as usize)
                .map(|sym| sym.shndx as usize)
                .filter(|place| group.contains_key(place));
            if group.contains_key(&on) {
                relocations.entry(on).or_default().push(Relocation {
                    section,
                    at,
                    offset,
                    kind: info as u32,
                });
                if names.is_some() {
                    holders.insert(on);
                }
            }
            if let Some(place) = names {
                named.insert(place);
            }
        }
    }

    // Edges from .nv.callgraph
    let mut edges: Vec<(usize, usize)> = Vec::new();
    if let Some(gi) = e.section(".nv.callgraph") {
        let gdata = &e.sections[gi].data;
        let mut at = 0;
        while at + 8 <= gdata.len() {
            let a = u32::from_le_bytes(gdata[at..at + 4].try_into().unwrap()) as usize;
            let b = i32::from_le_bytes(gdata[at + 4..at + 8].try_into().unwrap());
            if b >= 0
                && let (Some(&sa), Some(&sb)) = (by_symbol.get(&a), by_symbol.get(&(b as usize)))
            {
                edges.push((sa, sb));
            }
            at += 8;
        }
    }
    // Edges from .nv.info.* attribute 0x0f
    for si in 0..e.sections.len() {
        if !e.sections[si].name.starts_with(".nv.info.") {
            continue;
        }
        let fn_name = e.sections[si].name[".nv.info.".len()..].to_string();
        let text_idx = match e.section(&format!(".text.{}", fn_name)) {
            Some(i) => i,
            None => continue,
        };
        let sdata = e.sections[si].data.clone();
        let mut pos = 0;
        while pos + 4 <= sdata.len() {
            let fmt = sdata[pos];
            let attr = sdata[pos + 1];
            let val = u16::from_le_bytes(sdata[pos + 2..pos + 4].try_into().unwrap()) as usize;
            if fmt == 4 && attr == 0x0f {
                let mut at = pos + 4;
                while at + 4 <= (pos + 4 + val).min(sdata.len()) {
                    let callee = u32::from_le_bytes(sdata[at..at + 4].try_into().unwrap()) as usize;
                    if let Some(&cs) = by_symbol.get(&callee) {
                        edges.push((text_idx, cs));
                    }
                    at += 4;
                }
            }
            pos += if fmt == 4 { 4 + val } else { 4 };
        }
    }
    for &(a, b) in &edges {
        let fa = find(&mut group, a);
        let fb = find(&mut group, b);
        if fa != fb {
            group.insert(fa, fb);
        }
    }

    // Parse each function
    let before: Vec<usize> = texts.iter().map(|&si| e.sections[si].data.len() / 16).collect();
    let mut report: Vec<(String, Option<String>)> = Vec::new();
    let mut functions: HashMap<usize, FuncData> = HashMap::new();
    for &si in &texts {
        let name = e.sections[si].name[".text.".len()..].to_string();
        let registers = e.sections[si].info >> 24;
        let tables = match e.section(&format!(".nv.info.{}", name)) {
            Some(ii) => tables_of(&e.sections[ii].data),
            None => Ok(Vec::new()),
        };
        let parsed = tables.and_then(|tables| {
            let code = &e.sections[si].data;
            if arch != 80 {
                return parse_function(&mut dec75, &name, code, registers, tables);
            }
            let mut helpers = symbols
                .iter()
                .enumerate()
                .filter(|(_, s)| s.shndx as usize == si && s.info & 15 == 2 && s.value != 0);
            let mut helper = None;
            if let Some((index, s)) = helpers.next() {
                if helpers.next().is_some()
                    || (s.info >> 4 != 0 && !vf_fatbin::cubin_has_assembler_helper(data, index))
                    || s.other & 0x10 != 0
                    || s.size == 0
                    || !s.value.is_multiple_of(16)
                    || s.value.checked_add(s.size) != Some(code.len() as u64)
                {
                    return Err(Refused("unsupported secondary entry in sm_80 code".into()));
                }
                helper = Some(s.value as usize / 16);
            }
            if relocations.contains_key(&si) {
                return Err(Refused("a relocation in sm_80 code".into()));
            }
            let normalized = super::ampere::normalize(code, &mut dec75, helper)?;
            let mut fd = parse_function(&mut dec75, &name, &normalized.words, registers, tables)?;
            fd.ampere_ops = normalized.operations;
            for (i, op) in &fd.ampere_ops {
                match op {
                    super::ampere::Operation::GlobalToShared { .. } => fd.scratch_need = fd.scratch_need.max(6),
                    super::ampere::Operation::IncrementShared { .. } => fd.scratch_need = fd.scratch_need.max(2),
                    super::ampere::Operation::Bf16Add | super::ampere::Operation::Bf16Pack => {
                        fd.scratch_need = fd.scratch_need.max(3)
                    }
                    super::ampere::Operation::Pack => {
                        fd.needs_pool |= fd.src[*i].op_or(crate::generated::fields::OP_RC, 255) != 255
                    }
                }
            }
            fd.ampere = true;
            Ok(fd)
        });
        match parsed {
            Ok(fd) => {
                functions.insert(si, fd);
                report.push((name, None));
            }
            Err(why) => {
                report.push((name, Some(why.0)));
            }
        }
    }

    // The sections that are kernels: a function symbol with the mark of an entry point.
    let entries: HashSet<usize> = symbols
        .iter()
        .filter(|sym| sym.info & 0xf == 2 && sym.other & 0x10 != 0)
        .map(|sym| sym.shndx as usize)
        .collect();

    // A function of its own within the section of another (the helpers that the code of the
    // section calls are weak symbols): it is entered there, not at the start of the section.
    for sym in &symbols {
        if sym.info & 0xf == 2 && sym.info >> 4 == 1 && sym.value != 0 {
            let si = sym.shndx as usize;
            if let Some(fd) = functions.get_mut(&si) {
                fd.entered_inside = true;
            }
        }
    }

    // A relocation that names a place in code: the function it names and the function it is in.
    for &si in &texts {
        if (holders.contains(&si) || named.contains(&si))
            && let Some(fd) = functions.remove(&si)
        {
            let why = if holders.contains(&si) {
                "a relocation in its code names a place in code"
            } else {
                "a relocation names a place in its code"
            };
            set_report(&mut report, &fd.name, why);
        }
    }

    // Group and translate
    let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
    for &si in &texts {
        let root = find(&mut group, si);
        members.entry(root).or_default().push(si);
    }

    let mut results: HashMap<usize, TransResult> = HashMap::new();

    for indices in members.values() {
        if indices.iter().any(|i| !functions.contains_key(i)) {
            for i in indices {
                if functions.contains_key(i) {
                    let name = e.sections[*i].name[".text.".len()..].to_string();
                    set_report(&mut report, &name, "a function it calls, or that calls it, is refused");
                }
            }
            continue;
        }
        let fs: Vec<&FuncData> = indices.iter().map(|i| &functions[i]).collect();

        // Where the registers of the translation go, every function of the group, then the
        // attributes of each: the first refusal is the group's.
        let mut done: Vec<(usize, TransResult)> = Vec::new();
        let mut why: Option<String> = None;
        let declared = fs.iter().map(|f| f.registers).max().unwrap();
        // The fewest registers that the block a kernel of the group requires allows, and that block.
        let required = fs
            .iter()
            .filter_map(|f| {
                let info = e
                    .section(&format!(".nv.info.{}", f.name))
                    .map_or(&[][..], |i| e.sections[i].data.as_slice());
                let threads = super::block_registers::required_threads(info)?;
                Some((super::block_registers::registers_for(threads), threads))
            })
            .min();
        if fs.len() > 1 && fs.iter().any(|f| !f.uniform_predicates.is_empty()) {
            why = Some("uniform predicates in functions that call one another".into());
        } else if let Some(f) = fs.iter().find(|f| f.highest >= declared) {
            why = Some(format!(
                "register R{} is used and the function declares {}",
                f.highest, declared
            ));
        } else {
            // The predicates of each, then the registers of the translation, which start after
            // the last one any of them uses.
            let predicates: Result<Vec<Predicates>, Refused> = fs.iter().map(|f| stand_ins(f)).collect();
            let attempt_result = predicates.and_then(|predicates| {
                let saves = predicates.iter().map(|p| p.save_slots).max().unwrap_or(0);
                // A function's pairing plan depends on the original instructions and control
                // flow. Register-placement retries and budget alternatives can reuse it.
                let paired = std::cell::LazyCell::new(|| {
                    fs.iter()
                        .map(|f| {
                            if opts.pairs && f.needs_lane && !f.entered_inside {
                                crate::pairing::plan(&f.src, &f.tables, &self::entries(f))
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                });
                let unpaired: Vec<_> = fs.iter().map(|_| None).collect();
                // `borrow_level`: 2, the uniform registers that find a home in the function's
                // own registers have it and the pool is taken there at each instruction; 1, the
                // homes only, the pool above; 0, everything above, as it was.
                let mut attempt = |do_pairs: bool, borrow_level: u32, top_override: Option<u32>, gave_up: &[String]| -> Result<Vec<(usize, TransResult)>, Refused> {
                    let plans = if do_pairs { &*paired } else { &unpaired };
                    // has_product when not in a pair site.
                    let has_product_flags: Vec<bool> = fs
                        .iter()
                        .zip(plans.iter())
                        .map(|(f, q)| {
                            let sites: std::collections::BTreeSet<usize> = q
                                .as_ref()
                                .map_or_else(Default::default, |p| p.sites.keys().copied().collect());
                            f.src
                                .iter()
                                .enumerate()
                                .any(|(i, d)| d.name == "hmma_x8_" && !sites.contains(&i))
                        })
                        .collect();
                    // The rows and their sharing.
                    let mut rows: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
                    let mut shares: HashMap<u32, u32> = HashMap::new();
                    for plan in plans.iter().flatten() {
                        for (&rd, &count) in &plan.rows() {
                            let to = if fs.len() == 1 {
                                plan.share.get(&rd).copied().unwrap_or(rd)
                            } else {
                                rd
                            };
                            shares.insert(rd, to);
                            let entry = rows.entry(to).or_insert(0);
                            *entry = (*entry).max(count);
                        }
                    }
                    let row_kept: Vec<&'static str> = crate::pairing::ROW_KEPT
                        .iter()
                        .copied()
                        .filter(|&name| {
                            plans
                                .iter()
                                .any(|q| q.as_ref().is_some_and(|p| p.kept().contains(name)))
                        })
                        .collect();
                    let held_preds = {
                        let mut s: Vec<u32> = predicates
                            .iter()
                            .flat_map(|p| p.held_predicates.iter().copied())
                            .collect::<HashSet<_>>()
                            .into_iter()
                            .collect();
                        s.sort();
                        s
                    };
                    // When borrowing: compute Borrowed for the single kernel.
                    let mut borrowed = if borrow_level > 0 {
                        let all_in_memory = borrow_level == 3 && (opts.borrow == 3 || opts.borrow == 8);
                        Some(Borrowed::new(fs[0], all_in_memory, top_override)?)
                    } else {
                        None
                    };
                    if borrow_level == 3 {
                        // Level 3: everything at the instruction. Unhomed uniform registers in
                        // local memory; pool, scratch, lane taken at each instruction.
                        // At VF_BORROW >= 7 a function with a matrix product is translated at
                        // level 3: the lane is read at each product, the four values computed
                        // there, B gathered every time, and every register taken at the instruction
                        // (`docs/translator.md`, "Registers"). At VF_BORROW < 7: refused as before.
                        let f = &fs[0];
                        if f.product_count > 0 && opts.borrow < 7 {
                            return Err(Refused("a matrix product in a function at level 3".into()));
                        }
                        let b = borrowed.as_mut().unwrap();
                        b.everything = true;
                        let pairs_set: HashSet<u32> = f.pairs.iter().copied().collect();
                        let mut sorted_uniform = f.uniform.clone();
                        sorted_uniform.sort();
                        b.assign_slots(&sorted_uniform, &pairs_set);
                        // VF_BORROW >= 5: room is made where there is none, and the frame
                        // is addressed from the stack pointer; in a function that never moves it.
                        // A kernel that never uses R1 needs no stack pointer: its frame is
                        // addressed from the top of the thread's local memory at each instruction,
                        // and no room is made (spilling victims is addressed from the pointer).
                        if opts.borrow >= 5 && b.r1_fixed {
                            let first = &fs[0].src[0];
                            let entry_sets_r1 = super::stack_top(first)
                                && first.writes == [(crate::decode::LETTER_R, 1)]
                                && crate::pairing::unguarded(first);
                            let uses_r1 = fs[0].src.iter().any(|d| {
                                d.reads
                                    .iter()
                                    .any(|&(letter, register, _)| letter == crate::decode::LETTER_R && register == 1)
                                    || d.writes
                                        .iter()
                                        .any(|&(letter, register)| letter == crate::decode::LETTER_R && register == 1)
                            });
                            if entry_sets_r1 || uses_r1 {
                                b.room = true;
                                b.from_r1 = true;
                            }
                        }
                        // A uniform predicate held in a register: one register free through all its lives.
                        let mut spans_by_q: std::collections::BTreeMap<u32, std::collections::BTreeSet<usize>> =
                            std::collections::BTreeMap::new();
                        for &(q, ref part) in &predicates[0].held_lives {
                            spans_by_q.entry(q).or_default().extend(part.iter().copied());
                        }
                        let mut held_at: HashMap<u32, u32> = HashMap::new();
                        for (&q, span_set) in &spans_by_q {
                            let span_sorted: Vec<usize> = span_set.iter().copied().collect();
                            if let Some(r) = b.kept(&span_sorted) {
                                held_at.insert(q, r);
                            } else if b.from_r1 {
                                let slot = b.slot_count();
                                b.predicate_slots.insert(q, slot);
                                b.frame_bytes = (4 * b.slot_count()).next_multiple_of(16);
                            } else {
                                return Err(Refused(format!("no register holds nothing through the lives of uniform predicate UP{q}")));
                            }
                        }
                        // Above the registers that may be taken: saved predicates only.
                        let placed = placement(
                            b.top + 1,
                            &[],
                            &[],
                            false,
                            saves,
                            false,
                            0,
                            false,
                            None,
                            &[],
                            &[],
                        )?;
                        let si = indices[0];
                        let place_map: HashMap<u32, u32> = b.homes.clone();
                        match translate_function(
                            &mut dec70,
                            &functions[&si],
                            &predicates[0],
                            &place_map,
                            None,
                            placed.2, // save
                            None,     // no lane (taken at instruction)
                            None,     // no kept
                            None,     // no gathered
                            None,     // no scratch (taken at instruction)
                            placed.7, // free
                            None,     // no pairs plan
                            HashMap::new(),
                            HashMap::new(),
                            opts.aligned_rows,
                            false, // no has_product_non_pair
                            &held_at,
                            borrowed.as_mut(),
                        ) {
                            Ok(tr) => {
                                if std::env::var("VF_BORROW_SAY").is_ok() {
                                    let f = &fs[0];
                                    let b = borrowed.as_ref().unwrap();
                                    let gu = if gave_up.is_empty() { "-".to_string() } else { gave_up.join("; ") };
                                    eprintln!(
                                        "  registers\t{}\t{}\t{}\tlevel 3\tuniform {}\thomed {}\tin memory {}\tframe {}\tsaves {}\theld {}\tproducts {}\tgave up: {}",
                                        f.name, f.registers, tr.count,
                                        f.uniform.len(), b.homes.len(),
                                        b.slots.len(), b.frame_bytes,
                                        predicates[0].save_slots, held_at.len(),
                                        f.product_count, gu
                                    );
                                }
                                return Ok(vec![(si, tr)]);
                            }
                            Err(refused) => return Err(refused),
                        }
                    }
                    let homes: HashMap<u32, u32> = borrowed.as_ref().map_or_else(HashMap::new, |b| b.homes.clone());
                    let homed_set: HashSet<u32> = homes.keys().copied().collect();
                    // Uniform registers not homed: those that still need placement above.
                    let uniform_for_placement: Vec<u32> = {
                        let mut all: Vec<u32> = fs.iter().flat_map(|f| f.uniform.iter().copied()).collect();
                        all.sort_unstable();
                        all.dedup();
                        all.into_iter().filter(|u| !homed_set.contains(u)).collect()
                    };
                    let pairs_for_placement: Vec<u32> = {
                        let mut all: Vec<u32> = fs.iter().flat_map(|f| f.pairs.iter().copied()).collect();
                        all.sort_unstable();
                        all.dedup();
                        all.into_iter().filter(|u| !homed_set.contains(u)).collect()
                    };
                    let placed = placement(
                        fs.iter().map(|f| f.highest).max().unwrap() + 1,
                        &uniform_for_placement,
                        &pairs_for_placement,
                        fs.iter().any(|f| f.needs_pool) && borrow_level != 2,
                        saves,
                        fs.iter().any(|f| f.needs_lane),
                        fs.iter().map(|f| f.scratch_need).max().unwrap_or(0),
                        fs.iter().zip(has_product_flags.iter()).any(|(_, &hp)| hp),
                        if rows.is_empty() { None } else { Some(&rows) },
                        &row_kept,
                        &held_preds,
                    )?;
                    let (
                        mut place,
                        pool,
                        save,
                        lane,
                        kept,
                        gathered,
                        scratch,
                        free,
                        rows_at_base,
                        kept_rows_base,
                        hold_at,
                    ) = placed;
                    // Merge the borrowed homes into the placement map.
                    for (&u, &r) in &homes {
                        place.insert(u, r);
                    }
                    // Apply sharing: each destination gets the rows_at of the destination it shares with.
                    let mut result = Vec::new();
                    for (k, (&si, predicates)) in indices.iter().zip(&predicates).enumerate() {
                        let plan = &plans[k];
                        let mut func_rows_at = HashMap::new();
                        if let Some(plan) = plan {
                            for &rd in plan.rows().keys() {
                                let to = shares.get(&rd).copied().unwrap_or(rd);
                                if let Some(&at) = rows_at_base.get(&to) {
                                    func_rows_at.insert(rd, at);
                                }
                            }
                        }
                        match translate_function(
                            &mut dec70,
                            &functions[&si],
                            predicates,
                            &place,
                            pool,
                            save,
                            lane,
                            kept,
                            gathered,
                            scratch,
                            free,
                            plan.as_ref(),
                            func_rows_at,
                            kept_rows_base.clone(),
                            opts.aligned_rows,
                            has_product_flags[k],
                            &hold_at,
                            borrowed.as_mut(),
                        ) {
                            Ok(tr) => result.push((si, tr)),
                            Err(refused) => return Err(refused),
                        }
                    }
                    if std::env::var("VF_BORROW_SAY").is_ok() && fs.len() == 1 {
                        let f = &fs[0];
                        let count = result.iter().map(|(_, tr)| tr.count).max().unwrap_or(0);
                        let n_homes = borrowed.as_ref().map_or(0, |b| b.homes.len());
                        let pool_count = if f.needs_pool && borrow_level != 2 { 2 } else { 0 };
                        let gu = if gave_up.is_empty() { "-".to_string() } else { gave_up.join("; ") };
                        eprintln!(
                            "  registers\t{}\t{}\t{}\tlevel {}\tuniform {}\thomed {}\tpool {}\tsaves {}\theld {}\tproducts {}\tgave up: {}",
                            f.name, f.registers, count, borrow_level,
                            f.uniform.len(), n_homes, pool_count,
                            saves, held_preds.len(), f.product_count,
                            gu
                        );
                    }
                    Ok(result)
                };
                let alone = fs.len() == 1 && indices.iter().any(|si| entries.contains(si));
                let has_pairs_to_drop = || paired.iter().any(Option::is_some);
                type Tried = Result<Vec<(usize, TransResult)>, Refused>;
                type AttemptFn<'a> = dyn FnMut(bool, u32, Option<u32>, &[String]) -> Tried + 'a;
                let empty_gave_up: Vec<String> = Vec::new();
                // usual(): the chain of attempts (levels 2, 1, 0, pairs fallback).
                let usual = |attempt: &mut AttemptFn<'_>| {
                    let r = if opts.borrow > 0 && alone {
                        attempt(true, 2, None, &[])
                            .or_else(|_| attempt(true, 1, None, &[]))
                            .or_else(|_| attempt(true, 0, None, &[]))
                    } else {
                        attempt(true, 0, None, &[])
                    };
                    r.or_else(|refused| {
                        if has_pairs_to_drop() {
                            attempt(false, 0, None, &[])
                        } else {
                            Err(refused)
                        }
                    })
                };
                // At level 3, when the registers of the program are not enough, a kernel that is
                // entered from outside may take those above the ones the program names: no
                // caller keeps anything there, and the V100 has them (up to R252).
                let extend_level3 = |attempt: &mut AttemptFn<'_>, gave_up: &[String]| -> Tried {
                    if required.is_none() && fs.len() == 1 && entries.contains(&indices[0]) {
                        attempt(false, 3, Some(252u32.saturating_sub(saves)), gave_up)
                    } else {
                        Err(Refused(String::new()))
                    }
                };
                // plain(): level 0 with pairs fallback but no borrow (for the budget).
                let plain = |attempt: &mut AttemptFn<'_>, gave_up: &[String]| {
                    attempt(true, 0, None, gave_up).or_else(|refused| {
                        if has_pairs_to_drop() {
                            attempt(false, 0, None, gave_up)
                        } else {
                            Err(refused)
                        }
                    })
                };
                let chosen: Tried = 'chain: { if opts.borrow == 8 && alone {
                    // For tests: level 3 first for EVERY kernel, products included (as 3 is
                    // without products). With room made: a small function that has not enough
                    // registers for the product's scratch at once gets them above its own (top
                    // extended by 16 for the product, as it would at level 0). Falls back to
                    // the usual chain when level 3 is refused.
                    let f = &fs[0];
                    let extra: u32 = if f.product_count > 0 { 16 } else { 0 };
                    let top_for_8 = if extra > 0 { Some(f.highest + extra) } else { None };
                    match attempt(false, 3, top_for_8, &empty_gave_up) {
                        Ok(d) => Ok(d),
                        Err(ref e) => {
                            eprintln!("level 3 failed for {}: {}", fs[0].name, e.0);
                            usual(&mut attempt)
                        }
                    }
                } else if matches!(opts.borrow, 7 | 9) && alone {
                    // What 6 does, and a function with products may be translated at level 3.
                    // The chain: plain; levels 1, 2; level 3 with room and with products.
                    let mut gave_up: Vec<String> = Vec::new();
                    let plain_result = plain(&mut attempt, &gave_up);
                    let normal = match plain_result {
                        Ok(d) => Ok(d),
                        Err(original_why) if original_why.0 == "more registers than there are" => {
                            gave_up.push(original_why.0.clone());
                            let original_msg = original_why.0.clone();
                            // Try levels 1, 2, 3 in order.
                            let mut done = None;
                            for (do_p, bl) in [(true, 1u32), (true, 2), (false, 3)] {
                                let mut result = attempt(do_p, bl, None, &gave_up);
                                if bl == 3
                                    && result.is_err()
                                    && let Ok(d) = extend_level3(&mut attempt, &gave_up)
                                {
                                    result = Ok(d);
                                }
                                match result {
                                    Ok(d) => { done = Some(d); break; }
                                    Err(e) => { gave_up.push(e.0); }
                                }
                            }
                            match done {
                                Some(d) => Ok(d),
                                None => {
                                    if std::env::var("VF_BORROW_SAY").is_ok() {
                                        eprintln!(
                                            "  refused\t{}\t{}\tand with borrowed registers: {}",
                                            fs[0].name, original_msg,
                                            gave_up[1..].join("; ")
                                        );
                                    }
                                    Err(Refused(original_msg))
                                }
                            }
                        }
                        Err(e) => Err(e),
                    }?;
                    if opts.borrow == 9 {
                        let f = fs[0];
                        let info = e.section(&format!(".nv.info.{}", f.name))
                            .map_or(&[][..], |i| e.sections[i].data.as_slice());
                        let limit = super::block_registers::budget(f.registers, info);
                        // Relocations and jump tables are checked after placement. Keep the working
                        // layout when a second placement could change how those sites are rewritten.
                        if normal[0].1.count > limit && !relocations.contains_key(&indices[0]) && f.tables.is_empty() {
                            let top = limit.saturating_sub(3).saturating_sub(saves);
                            for (pairs, level, bound) in [(true, 1, None), (true, 2, None), (false, 3, Some(top))] {
                                if let Ok(candidate) = attempt(pairs, level, bound, &gave_up)
                                    && candidate[0].1.count <= limit
                                {
                                    // Attributes and symbols may point at any original instruction. Preserve
                                    // every address that the ordinary translation could carry over.
                                    let old = &normal[0].1.new_of;
                                    let new = &candidate[0].1.new_of;
                                    if old.len() == new.len() && old.keys().all(|offset| new.contains_key(offset)) {
                                        break 'chain Ok(candidate);
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    Ok(normal)
                } else if opts.borrow == 6 && alone {
                    // What the layer does by itself. Wherever the translation of the
                    // switch off exists it is the one taken, byte for byte. A kernel that is
                    // refused for want of registers gets, in this order: homes (level 1),
                    // the pool at the instruction too (level 2), then everything at the
                    // instruction, with room made where there is none (level 3). And when
                    // none translates, it is refused for the reason it was refused before.
                    let mut gave_up: Vec<String> = Vec::new();
                    let plain_result = plain(&mut attempt, &gave_up);
                    match plain_result {
                        Ok(d) => Ok(d),
                        Err(original_why) if original_why.0 == "more registers than there are" => {
                            gave_up.push(original_why.0.clone());
                            let original_msg = original_why.0.clone();
                            // Try levels 1, 2, 3 in order.
                            match attempt(true, 1, None, &gave_up) {
                                Ok(d) => Ok(d),
                                Err(e1) => {
                                    gave_up.push(e1.0);
                                    match attempt(true, 2, None, &gave_up) {
                                        Ok(d) => Ok(d),
                                        Err(e2) => {
                                            gave_up.push(e2.0);
                                            match attempt(false, 3, None, &gave_up) {
                                                Ok(d) => Ok(d),
                                                Err(e3) => {
                                                    gave_up.push(e3.0);
                                                    match extend_level3(&mut attempt, &gave_up) {
                                                        Ok(d) => Ok(d),
                                                        Err(_) => {
                                                            if std::env::var("VF_BORROW_SAY").is_ok() {
                                                                eprintln!(
                                                                    "  refused\t{}\t{}\tand with borrowed registers: {}",
                                                                    fs[0].name, original_msg,
                                                                    gave_up[1..].join("; ")
                                                                );
                                                            }
                                                            Err(Refused(original_msg))
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => Err(e),
                    }
                } else if opts.borrow >= 4 && alone {
                    // By the budget: a V100 gives a thread its registers by eights, and has
                    // 65,536 of them a multiprocessor for at most 2,048 threads: a kernel
                    // loses no thread as long as its count stays in the eight it was in, or
                    // under 32. The levels in the order of the speed of what they make
                    // (0: everything above, nothing to wait for; 1: homes; 2: and the pool
                    // at the instruction; 3: everything at the instruction, with the
                    // registers left under the budget, and memory): the first that stays
                    // within the budget, or else the one that takes the fewest.
                    let f = &fs[0];
                    let budget = (f.registers.div_ceil(8) * 8).clamp(32, 255);
                    let top_for_l3 = Some(budget.saturating_sub(3).saturating_sub(saves));
                    // Try each level in order: 0 (plain), 1, 2, 3.
                    struct Best {
                        count: u32,
                        level: u32,
                        result: Vec<(usize, TransResult)>,
                        within: bool,
                    }
                    let mut best: Option<Best> = None;
                    let mut first_why: Option<Refused> = None;
                    let mut gave_up: Vec<String> = Vec::new();
                    // Level 0: plain.
                    match plain(&mut attempt, &gave_up) {
                        Ok(d) => {
                            let count = d.iter().map(|(_, tr)| tr.count).max().unwrap_or(0);
                            if count <= budget {
                                best = Some(Best { count, level: 0, result: d, within: true });
                            } else {
                                best = Some(Best { count, level: 0, result: d, within: false });
                            }
                        }
                        Err(refused) => {
                            gave_up.push(refused.0.clone());
                            first_why = Some(refused);
                        }
                    }
                    if !best.as_ref().is_some_and(|b| b.within) {
                        // Level 1.
                        match attempt(true, 1, None, &gave_up) {
                            Ok(d) => {
                                let count = d.iter().map(|(_, tr)| tr.count).max().unwrap_or(0);
                                if count <= budget {
                                    best = Some(Best { count, level: 1, result: d, within: true });
                                } else if best.is_none() || count < best.as_ref().unwrap().count {
                                    best = Some(Best { count, level: 1, result: d, within: false });
                                }
                            }
                            Err(refused) => {
                                gave_up.push(refused.0.clone());
                                if first_why.is_none() { first_why = Some(refused); }
                            }
                        }
                    }
                    if !best.as_ref().is_some_and(|b| b.within) {
                        // Level 2.
                        match attempt(true, 2, None, &gave_up) {
                            Ok(d) => {
                                let count = d.iter().map(|(_, tr)| tr.count).max().unwrap_or(0);
                                if count <= budget {
                                    best = Some(Best { count, level: 2, result: d, within: true });
                                } else if best.is_none() || count < best.as_ref().unwrap().count {
                                    best = Some(Best { count, level: 2, result: d, within: false });
                                }
                            }
                            Err(refused) => {
                                gave_up.push(refused.0.clone());
                                if first_why.is_none() { first_why = Some(refused); }
                            }
                        }
                    }
                    if !best.as_ref().is_some_and(|b| b.within) {
                        // Level 3.
                        match attempt(false, 3, top_for_l3, &gave_up) {
                            Ok(d) => {
                                let count = d.iter().map(|(_, tr)| tr.count).max().unwrap_or(0);
                                if count <= budget {
                                    best = Some(Best { count, level: 3, result: d, within: true });
                                } else if best.is_none() || count < best.as_ref().unwrap().count {
                                    best = Some(Best { count, level: 3, result: d, within: false });
                                }
                            }
                            Err(refused) => {
                                gave_up.push(refused.0.clone());
                                if first_why.is_none() { first_why = Some(refused); }
                            }
                        }
                    }
                    match best {
                        None => Err(first_why.unwrap()),
                        Some(mut b) => {
                            if !b.within {
                                // Run the best level again (the function is left as its last
                                // attempt made it).
                                b.result = match b.level {
                                    0 => plain(&mut attempt, &gave_up)?,
                                    1 => attempt(true, 1, None, &gave_up)?,
                                    2 => attempt(true, 2, None, &gave_up)?,
                                    3 => attempt(false, 3, top_for_l3, &gave_up)?,
                                    _ => unreachable!(),
                                };
                            }
                            if std::env::var("VF_BORROW_SAY").is_ok() {
                                eprintln!(
                                    "  budget\t{}\t{}\t{}\t{}\tlevel {}\t{}",
                                    f.name, f.registers, b.count, budget, b.level,
                                    if b.within { "within" } else { "over" }
                                );
                            }
                            Ok(b.result)
                        }
                    }
                } else if opts.borrow >= 3 && alone {
                    // For tests: everything at the instruction first, every uniform register
                    // in memory.
                    match attempt(false, 3, None, &empty_gave_up) {
                        Ok(d) => Ok(d),
                        Err(ref e) => {
                            eprintln!("level 3 failed for {}: {}", fs[0].name, e.0);
                            usual(&mut attempt)
                        }
                    }
                } else if opts.borrow >= 2 && alone {
                    // Everything at the instruction for a kernel that is refused otherwise;
                    // and when that is refused too, for the reason it was refused before.
                    match usual(&mut attempt) {
                        Ok(d) => Ok(d),
                        Err(original_why) => match attempt(false, 3, None, &empty_gave_up) {
                            Ok(d) => Ok(d),
                            Err(also) => {
                                if std::env::var("VF_BORROW_SAY").is_ok() {
                                    eprintln!(
                                        "  refused\t{}\t{}\tat level 3: {}",
                                        fs[0].name, original_why.0, also.0
                                    );
                                }
                                Err(original_why)
                            }
                        },
                    }
                } else {
                    usual(&mut attempt)
                } };
                // A kernel that requires its block size (`block_registers::required_threads`) is
                // launched with that block and no other: a translation with more registers than
                // such a block may have on a V100 never runs, and a program that reads the limit
                // from the loaded kernel (Triton) stops before any launch, so that the second
                // translation at launch never comes. The first of levels 1, 2 and 3 that keeps
                // within it is taken instead; when none does, the kernel is refused for its
                // registers, and the layer runs the PTX that the cubin keeps, when it keeps some.
                let Some((limit, threads)) = required else {
                    return chosen;
                };
                let count = |d: &[(usize, TransResult)]| d.iter().map(|(_, tr)| tr.count).max().unwrap_or(0);
                let chosen = chosen?;
                if count(&chosen) <= limit {
                    return Ok(chosen);
                }
                if opts.borrow > 0 && alone {
                    let top = limit.saturating_sub(3).saturating_sub(saves);
                    for (pairs, level, bound) in [(true, 1, None), (true, 2, None), (false, 3, Some(top))] {
                        if let Ok(candidate) = attempt(pairs, level, bound, &empty_gave_up)
                            && count(&candidate) <= limit
                        {
                            return Ok(candidate);
                        }
                    }
                }
                Err(Refused(format!(
                    "more registers than its required block of {threads} threads allows ({limit})"
                )))
            });
            match attempt_result {
                Ok(d) => done = d,
                Err(refused) => why = Some(refused.0),
            }
        }
        if why.is_none() {
            // No kernel of the file calls them: their callers are elsewhere (a library of
            // functions, linked later) and may keep something in the registers above those these
            // functions declare, which is where the translation would write.
            let count = done.iter().map(|(_, tr)| tr.count).max().unwrap_or(0);
            if count > declared && !indices.iter().any(|si| entries.contains(si)) {
                why = Some(
                    "called from code that is not in the file, and the translation needs registers above its own"
                        .into(),
                );
            }
        }
        if why.is_none() {
            for (si, tr) in &mut done {
                if let Some(list) = relocations.get(si) {
                    match follow(&functions[si], tr, list) {
                        Ok(moved) => tr.moved = moved,
                        Err(refused) => {
                            why = Some(refused.0);
                            break;
                        }
                    }
                }
            }
        }
        if why.is_none() {
            for (si, tr) in &done {
                let info_name = format!(".nv.info.{}", functions[si].name);
                if let Some(ii) = e.section(&info_name)
                    && let Err(refused) = check_attributes(&e.sections[ii].data, &tr.new_of)
                {
                    why = Some(refused.0);
                    break;
                }
                let bank = e.section(&format!(".nv.constant2.{}", functions[si].name));
                let found = |base: usize, places: &[u64]| {
                    bank.is_some_and(|bi| {
                        let data = &e.sections[bi].data;
                        base + 4 * places.len() <= data.len()
                            && places.iter().enumerate().all(|(j, &place)| {
                                let at = base + 4 * j;
                                u32::from_le_bytes(data[at..at + 4].try_into().unwrap()) as u64 == place
                            })
                    })
                };
                if tr.relocate.iter().any(|(base, places)| !found(*base as usize, places)) {
                    why = Some("a jump table is not in the constants where its load reads it".into());
                    break;
                }
            }
        }
        match why {
            Some(why) => {
                for i in indices {
                    set_report(&mut report, &functions[i].name, &why);
                }
            }
            None => {
                let count = done.iter().map(|(_, tr)| tr.count).max().unwrap_or(0);
                for (si, mut tr) in done {
                    tr.count = count;
                    results.insert(si, tr);
                }
            }
        }
    }

    // Build the listing, and what the validator is told beside it
    let mut listing: Option<Vec<MapEntry>> = if want_map { Some(Vec::new()) } else { None };
    let mut lives: Option<Vec<LifeEntry>> = if want_map { Some(Vec::new()) } else { None };

    // Update the container
    let mut counts: HashMap<u32, u32> = HashMap::new();
    let mut frames: HashMap<u32, u32> = HashMap::new();
    let mut symbols = symbols;

    for &si in &texts {
        let name = e.sections[si].name[".text.".len()..].to_string();
        if let Some(tr) = results.get(&si) {
            if let Some(ref mut map) = listing {
                for (k, item) in tr.items.iter().enumerate() {
                    map.push((
                        name.clone(),
                        (16 * k) as u64,
                        (16 * item.origin) as u64,
                        item.what_str().to_string(),
                    ));
                }
            }
            if let Some(ref mut lives) = lives {
                for (first, second, how, offsets) in &tr.lives {
                    lives.push((name.clone(), *first, *second, how, offsets.clone()));
                }
            }
            e.sections[si].data = tr.code.clone();
            e.sections[si].info = (tr.count << 24) | (e.sections[si].info & 0xffffff);
            counts.insert(e.sections[si].info & 0xffffff, tr.count);
            if tr.frame_bytes > 0 {
                frames.insert(e.sections[si].info & 0xffffff, tr.frame_bytes);
            }
            let info_name = format!(".nv.info.{}", name);
            if let Some(ii) = e.section(&info_name) {
                e.sections[ii].data = rewrite_attributes(&e.sections[ii].data, &tr.new_of);
            }
            if let Some(list) = relocations.get(&si) {
                // Relocations: each one to where its instruction is now.
                for (r, now) in list.iter().zip(&tr.moved) {
                    e.sections[r.section].data[r.at..r.at + 8].copy_from_slice(&now.to_le_bytes());
                }
            }
            if let Some(bi) = e.section(&format!(".nv.constant2.{}", name)) {
                // Jump tables: their places as they were, to where they are now.
                for (base, places) in &tr.relocate {
                    for (j, place) in places.iter().enumerate() {
                        let at = *base as usize + 4 * j;
                        e.sections[bi].data[at..at + 4].copy_from_slice(&(tr.new_of[place] as u32).to_le_bytes());
                    }
                }
            }
            for sym in &mut symbols {
                if sym.shndx as usize == si {
                    if let Some(&new_off) = tr.new_of.get(&sym.value) {
                        sym.value = new_off;
                    } else if sym.value != 0 {
                        set_report(
                            &mut report,
                            &name,
                            &format!("a symbol at 0x{:x}, which is not an instruction", sym.value),
                        );
                    }
                    if sym.info & 0xf == 2 {
                        // An accepted sm_80 local helper fills the tail of its section.
                        sym.size = tr.code.len() as u64 - if arch == 80 { sym.value } else { 0 };
                    }
                }
            }
        } else {
            // Refused function. Never to run: the proxy refuses its kernels. What reads the file
            // must still find valid sm_70 code that agrees with its attributes (the driver visits
            // the instructions they list): instructions that do nothing, an EXIT wherever one is
            // recorded, and no other record of an instruction. Should it be launched all the same,
            // it must not pass for a kernel that ran: it starts with a trap, as the compiler
            // writes one.
            let info_name = format!(".nv.info.{}", name);
            let mut exits = Vec::new();
            if let Some(ii) = e.section(&info_name) {
                exits = find_exits(&e.sections[ii].data);
                e.sections[ii].data = strip_attributes(&e.sections[ii].data);
            }
            let nop_bytes = NOP.to_le_bytes();
            let n_insns = e.sections[si].data.len() / 16;
            let mut code = vec![0u8; n_insns * 16];
            for i in 0..n_insns {
                code[i * 16..(i + 1) * 16].copy_from_slice(&nop_bytes[..16]);
            }
            if let Some(leave) = assemble::assemble("exit_", &[]) {
                let leave = with_stall(leave, 5, false) | (16u128 << USCHED);
                let lb = leave.to_le_bytes();
                for &at in if exits.is_empty() {
                    &[0u32][..]
                } else {
                    exits.as_slice()
                } {
                    let off = at as usize;
                    if off + 16 <= code.len() {
                        code[off..off + 16].copy_from_slice(&lb[..16]);
                    }
                }
            }
            for (at, word) in [(0usize, TRAP), (16, STAY)] {
                if at + 16 <= code.len() && !exits.contains(&(at as u32)) {
                    code[at..at + 16].copy_from_slice(&word.to_le_bytes());
                }
            }
            e.sections[si].data = code;
        }
    }

    e.set_symbols(&symbols);

    // Update global info
    if let Some(ii) = e.section(".nv.info") {
        let d = &mut e.sections[ii].data;
        let mut raised: HashSet<(u32, u8)> = HashSet::new();
        let mut pos = 0;
        while pos + 4 <= d.len() {
            let fmt = d[pos];
            let attr = d[pos + 1];
            let val = u16::from_le_bytes(d[pos + 2..pos + 4].try_into().unwrap()) as usize;
            if fmt == 4 && attr == 0x2f && pos + 12 <= d.len() {
                let symbol = u32::from_le_bytes(d[pos + 4..pos + 8].try_into().unwrap());
                if let Some(&count) = counts.get(&symbol) {
                    d[pos + 8..pos + 12].copy_from_slice(&count.to_le_bytes());
                }
            }
            // The frame of a function: more by the frame the translation added (level 3).
            if fmt == 4 && (attr == 0x11 || attr == 0x12 || attr == 0x23) && pos + 12 <= d.len() {
                let symbol = u32::from_le_bytes(d[pos + 4..pos + 8].try_into().unwrap());
                if let Some(&frame) = frames.get(&symbol) {
                    let old = u32::from_le_bytes(d[pos + 8..pos + 12].try_into().unwrap());
                    d[pos + 8..pos + 12].copy_from_slice(&(old + frame).to_le_bytes());
                    raised.insert((symbol, attr));
                }
            }
            pos += if fmt == 4 { 4 + val } else { 4 };
        }
        // Check that every function with a frame has both 0x11 and 0x12 raised.
        for &symbol in frames.keys() {
            if !raised.contains(&(symbol, 0x11)) || !raised.contains(&(symbol, 0x12)) {
                return Err("a frame was added to a function that has no frame attributes to raise".into());
            }
        }
    }

    // Retag as sm_70
    if e.header[7] == 0x41 {
        e.header[7] = 0x33;
        e.header[8] = 7;
        e.header[0x14..0x18].copy_from_slice(&0x80u32.to_le_bytes());
        // Machine code for sm_70 that still says it was compiled for the architecture 7.5 (what
        // `cudaFuncGetAttributes` gives as `ptxVersion`): a program chooses by it among what it
        // was built with (the algorithms of Thrust and CUB answer "invalid device function" to an
        // architecture below the lowest of theirs), and it was built for nothing below.
        let elf_flags = (u32::from(arch) << 16) | 0x0500 | 70;
        e.header[0x30..0x34].copy_from_slice(&elf_flags.to_le_bytes());
        for s in &mut e.sections {
            if s.name == ".note.nv.tkinfo" || s.name == ".note.nv.cuinfo" {
                s.stype = 0;
                s.data.clear();
                s.size = 0;
            } else if s.name.starts_with(".nv.info") {
                s.data = strip_attrs_by_kind(&s.data, &[0x5f, 0x66]);
            }
        }
    } else {
        e.header[0x30] = 70;
    }

    let ordered_report: Vec<FunctionReport> = texts
        .iter()
        .zip(before)
        .map(|(&si, before)| {
            let name = e.sections[si].name[".text.".len()..].to_string();
            let refused = report.iter().find(|(n, _)| n == &name).and_then(|(_, r)| r.clone());
            FunctionReport {
                name,
                paired: refused.is_none() && results.get(&si).is_some_and(|tr| tr.paired),
                refused,
                before,
                after: e.sections[si].data.len() / 16,
            }
        })
        .collect();

    Ok(TranslateResult {
        bytes: e.write(),
        report: ordered_report,
        map: listing,
        lives,
    })
}

fn set_report(report: &mut [(String, Option<String>)], name: &str, reason: &str) {
    if let Some(entry) = report.iter_mut().find(|(n, _)| n == name) {
        entry.1 = Some(reason.to_string());
    }
}
