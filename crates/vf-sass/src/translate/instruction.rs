//! Per-instruction translation: `translate_one` decides what each sm_75 instruction
//! becomes (a plain rule, a sequence for a uniform operand, a matrix instruction,
//! or nothing at all), and the specific translators for IMAD_HI, IMNMX, strided
//! constants, ULDC, and ULDC at an index.

use crate::assemble;
use crate::decode::{Decoded, LETTER_Q, LETTER_U};
use crate::generated::fields::*;
use crate::generated::special::*;
use crate::rule;

use super::context::{Ctx, apply_rule};
use super::matrix::{translate_matrix_load, translate_matrix_product};
use super::matrix::{translate_rule, translate_uniform_address};
use super::util::{late, put_operand, put_operand_by_class, put_operand_on_word};
use super::{ARRIVES, Item, Refused, with_stall};

// ---- translate_one ----

pub(super) fn translate_one(ctx: &mut Ctx, i: usize, d: &Decoded) -> Result<Vec<Item>, Refused> {
    let w = d.word;
    let name = d.name;

    // 1. Already an sm_70 instruction with no uniform operands
    if ctx.dec70.decode(w).is_some()
        && !d.reads.iter().any(|&(l, _, _)| l == LETTER_U || l == LETTER_Q)
        && !d.writes.iter().any(|&(l, _)| l == LETTER_U || l == LETTER_Q)
    {
        return Ok(vec![Item::new(w, i, "same", false)]);
    }

    let guard = if d.has_op(OP_PG) {
        vec![
            ("Pg", assemble::OpVal::Int(d.op_or(OP_PG, 7))),
            ("Pg_not", assemble::OpVal::Int(d.op_or(OP_PG_AT_NOT, 0))),
        ]
    } else {
        Vec::new()
    };
    let wait = w & (63u128 << WAIT);

    // Matrix instructions: Turing's LDSM and HMMA.1688.
    if matches!(name, "ldsm__sImmOffset" | "ldsm__URsIR") {
        if ctx.q.is_some() && ctx.q.unwrap().loads.contains_key(&i) {
            return ctx.whole_rows(i, d, wait);
        }
        return translate_matrix_load(ctx, i, d, &guard, wait);
    }
    if name == "hmma_x8_" {
        if ctx.q.is_some() && ctx.q.unwrap().sites.contains_key(&i) {
            return ctx.half_of_pair(i, d, wait);
        }
        return translate_matrix_product(ctx, i, d, wait);
    }

    // 2. IMAD.HI
    if name.starts_with("imad_hi__") {
        return translate_imad_hi(ctx, i, d, &guard, wait);
    }

    // 3. IMNMX
    if name.starts_with("imnmx__") {
        return translate_imnmx(ctx, i, d, &guard, wait);
    }

    // 4. Memory with stride
    if STRIDED.contains(&name) && d.op_or(OP_STRIDE, 0) != 0 {
        return translate_stride(ctx, i, d, wait);
    }

    // 5. ULDC
    if name == "uldc_const__RCR" {
        return translate_uldc(ctx, i, d, w);
    }

    // A constant at a place that a uniform register gives
    if name == "uldc_ur_offset_" {
        return translate_uldc_at_index(ctx, i, d, wait);
    }

    // 6. Uniform address
    for &(prefix, plain) in UNIFORM_ADDRESS {
        if name.starts_with(prefix) {
            return translate_uniform_address(ctx, i, d, plain);
        }
    }

    // 7. R2UR
    if name == "r2ur__noOR" && d.op_or(OP_PU, 7) == 7 {
        let r_dest = ctx.r(d.op_or(OP_URD, 63))?;
        let mut ops = guard.clone();
        ops.push(("Rd", assemble::OpVal::Int(r_dest)));
        ops.push(("Rb", assemble::OpVal::Int(d.op_or(OP_RA, 255))));
        let mov =
            assemble::assemble("mov__RRR_RR", &ops).ok_or_else(|| Refused("cannot assemble MOV for R2UR".into()))?;
        let mov = (mov & !(31u128 << USCHED)) | (w & (31u128 << USCHED)) | wait;
        return Ok(vec![Item::new(mov, i, "move", true)]);
    }

    // 8. Cache hint
    if matches!(name, "ldg__sImmOffset" | "ld__sImmOffset") && d.op_or(OP_SP2, 0) != 0 {
        let cleared = put_operand(d, OP_SP2, 0, 75)?;
        if ctx.dec70.decode(cleared).is_some() {
            let cleared = without_batch(name, d, cleared)?;
            return Ok(vec![Item::new(cleared, i, "without its cache hint", true)]);
        }
    }
    // 8b. A Turing table with hints that sm_70's does not have (LTC128B and the like), and the
    // zero-fill predicate a Turing load may carry. Rebuild with sm_70's table, keeping its data,
    // guard and scoreboards: what changes is the cache hint.
    if name == "ldg__sImmOffset" && ctx.dec70.decode(w).is_none() {
        return translate_ldg_without_hints(ctx, i, d);
    }

    // 8c. F2FP (Turing): two f32 to two f16 in one register.
    if name == "f2fp__RRR" {
        return translate_f2fp_f16(ctx, i, d);
    }

    // 9. Rule
    if let Some(target) = rule::rule_of_target(d.class_id) {
        return translate_rule(ctx, i, d, target);
    }

    Err(Refused(format!(
        "no translation for {} ({}) at 0x{:x}",
        d.name,
        d.mnemonic,
        16 * i
    )))
}

fn translate_imad_hi(
    ctx: &mut Ctx,
    i: usize,
    d: &Decoded,
    guard: &[(&str, assemble::OpVal)],
    _wait: u128,
) -> Result<Vec<Item>, Refused> {
    let name = d.name;
    // Its carry into the predicate that guards the instruction: the move after the product
    // would go by the carry. No code seen does that.
    let carry = d.op_or(OP_PU, 7);
    if carry != 7 && carry == d.op_or(OP_PG, 7) {
        return Err(Refused(format!(
            "IMAD.HI with its carry into the predicate that guards it at 0x{:x}",
            16 * i
        )));
    }
    let suffix = &name["imad_hi__".len()..];
    let wide_target = format!("imad_wide__{}", suffix.replace("RUR_RUR", "RRR_RRR"));
    let rule_idx = rule::find_rule(name, &wide_target, "")
        .ok_or_else(|| Refused(format!("no rule from {} to {}", name, wide_target)))?;
    let wide = apply_rule(ctx, rule_idx, i, d)?;
    let wide_name70 = rule::rule_sm70_name(rule_idx);
    let pair = ctx.pool(2)?;
    let wide = put_operand_by_class(wide_name70, wide, "Rd", pair as u64, false)?;
    let mut ops = guard.to_vec();
    ops.push(("Rd", assemble::OpVal::Int(d.op_or(OP_RD, 255))));
    ops.push(("Rb", assemble::OpVal::Int(pair + 1)));
    let mov =
        assemble::assemble("mov__RRR_RR", &ops).ok_or_else(|| Refused("cannot assemble MOV for IMAD.HI".into()))?;
    let mov = (mov & !(31u128 << USCHED)) | (d.word & (31u128 << USCHED));
    Ok(vec![
        Item::new(wide & !REUSE, i, "product", true),
        Item::new(mov, i, "upper half", true),
    ])
}

fn translate_imnmx(
    ctx: &mut Ctx,
    i: usize,
    d: &Decoded,
    guard: &[(&str, assemble::OpVal)],
    wait: u128,
) -> Result<Vec<Item>, Refused> {
    if d.op_or(OP_PP, 7) != 7 {
        return Err(Refused(format!(
            "IMNMX under a predicate that is not constant at 0x{:x}",
            16 * i
        )));
    }
    let (p, saved) = ctx.predicates.scratch_at[&i];
    let larger = d.op_or(OP_PP_AT_NOT, 0);
    let fmt_val = d.op_or(OP_FMT, 0);
    // FMT enum for IMNMX: 0=U32, 1=S32
    let fmt_name = match fmt_val {
        0 => "U32".to_string(),
        1 => "S32".to_string(),
        _ => fmt_val.to_string(),
    };

    let name = d.name;
    let (compare, select) = if name.ends_with("__RRR_RRR") || name.ends_with("__RUR_RUR") {
        let b = if d.has_op(OP_URB) {
            ctx.r(d.op_or(OP_URB, 63))?
        } else {
            d.op_or(OP_RB, 255)
        };
        let cmp = assemble::assemble(
            "isetp__RRR_RR",
            &[
                ("Pu", assemble::OpVal::Int(p)),
                ("Ra", assemble::OpVal::Int(d.op_or(OP_RA, 255))),
                ("Rb", assemble::OpVal::Int(b)),
                ("icmp", assemble::OpVal::Str("LT")),
                ("fmt", assemble::OpVal::Str(&fmt_name)),
                ("bop", assemble::OpVal::Str("AND")),
            ],
        )
        .ok_or_else(|| Refused("cannot assemble ISETP for IMNMX".into()))?;
        let mut ops = guard.to_vec();
        ops.extend([
            ("Rd", assemble::OpVal::Int(d.op_or(OP_RD, 255))),
            ("Ra", assemble::OpVal::Int(d.op_or(OP_RA, 255))),
            ("Rb", assemble::OpVal::Int(b)),
            ("Pp", assemble::OpVal::Int(p)),
            ("Pp_not", assemble::OpVal::Int(larger)),
        ]);
        let sel =
            assemble::assemble("sel__RRR_RRR", &ops).ok_or_else(|| Refused("cannot assemble SEL for IMNMX".into()))?;
        (cmp, sel)
    } else if name.ends_with("__RsIR_RIR") {
        let cmp = assemble::assemble(
            "isetp__RsIR_RI",
            &[
                ("Pu", assemble::OpVal::Int(p)),
                ("Ra", assemble::OpVal::Int(d.op_or(OP_RA, 255))),
                ("Sb", assemble::OpVal::Int(d.op_or(OP_SB, 0))),
                ("icmp", assemble::OpVal::Str("LT")),
                ("fmt", assemble::OpVal::Str(&fmt_name)),
                ("bop", assemble::OpVal::Str("AND")),
            ],
        )
        .ok_or_else(|| Refused("cannot assemble ISETP for IMNMX".into()))?;
        let mut ops = guard.to_vec();
        ops.extend([
            ("Rd", assemble::OpVal::Int(d.op_or(OP_RD, 255))),
            ("Ra", assemble::OpVal::Int(d.op_or(OP_RA, 255))),
            ("Sb", assemble::OpVal::Int(d.op_or(OP_SB, 0))),
            ("Pp", assemble::OpVal::Int(p)),
            ("Pp_not", assemble::OpVal::Int(larger)),
        ]);
        let sel =
            assemble::assemble("sel__RuIR_RIR", &ops).ok_or_else(|| Refused("cannot assemble SEL for IMNMX".into()))?;
        (cmp, sel)
    } else if name.ends_with("__RCR_RCR") {
        let cmp_ops = vec![
            ("Pu", assemble::OpVal::Int(p)),
            ("Ra", assemble::OpVal::Int(d.op_or(OP_RA, 255))),
            ("icmp", assemble::OpVal::Str("LT")),
            ("fmt", assemble::OpVal::Str(&fmt_name)),
            ("bop", assemble::OpVal::Str("AND")),
            ("Sb_bank", assemble::OpVal::Int(d.op_or(OP_SB_BANK, 0))),
            ("Sb_addr", assemble::OpVal::Int(d.op_or(OP_SB_ADDR, 0))),
        ];
        let cmp = assemble::assemble("isetp__RCR_RC", &cmp_ops)
            .ok_or_else(|| Refused("cannot assemble ISETP for IMNMX".into()))?;
        let mut ops = guard.to_vec();
        ops.extend([
            ("Rd", assemble::OpVal::Int(d.op_or(OP_RD, 255))),
            ("Ra", assemble::OpVal::Int(d.op_or(OP_RA, 255))),
            ("Pp", assemble::OpVal::Int(p)),
            ("Pp_not", assemble::OpVal::Int(larger)),
            ("Sb_bank", assemble::OpVal::Int(d.op_or(OP_SB_BANK, 0))),
            ("Sb_addr", assemble::OpVal::Int(d.op_or(OP_SB_ADDR, 0))),
        ]);
        let sel =
            assemble::assemble("sel__RCR_RCR", &ops).ok_or_else(|| Refused("cannot assemble SEL for IMNMX".into()))?;
        (cmp, sel)
    } else {
        return Err(Refused(format!("this form of IMNMX: {}", name)));
    };

    let select = (select & !(31u128 << USCHED))
        | (d.word & (31u128 << USCHED))
        | if !name.ends_with("RUR") { d.word & REUSE } else { 0 };
    let mut new = vec![
        Item::new(compare | wait, i, "comparison", true),
        Item::new(select | wait, i, "selection", true),
    ];
    if let Some(k) = saved {
        // (Every predicate holds something there: the one it computes into is saved around it.)
        new.insert(0, ctx.saved(k, p, i)?);
        new.push(ctx.given_back(k, p, i)?);
    }
    Ok(new)
}

fn translate_stride(ctx: &mut Ctx, i: usize, d: &Decoded, wait: u128) -> Result<Vec<Item>, Refused> {
    let stride_val = d.op_or(OP_STRIDE, 0);
    let shift = match stride_val {
        1 => 2,
        2 => 3,
        3 => 4,
        _ => return Err(Refused(format!("unknown stride {}", stride_val))),
    };
    let ra = d.op_or(OP_RA, 255);
    // The register times the stride in the pool, then the plain instruction on it.
    let shadow = ctx.pool(1)?;
    let scale = assemble::assemble(
        "shf__RuIR_RIR",
        &[
            ("Rd", assemble::OpVal::Int(shadow)),
            ("Ra", assemble::OpVal::Int(ra)),
            ("Sb", assemble::OpVal::Int(shift)),
            ("Rc", assemble::OpVal::Int(255)),
            ("dir", assemble::OpVal::Str("L")),
            ("fmt", assemble::OpVal::Str("U32")),
        ],
    )
    .ok_or_else(|| Refused("cannot assemble SHF for stride".into()))?;
    let mut plain = put_operand(d, OP_STRIDE, 0, 75)?;
    plain = put_operand_on_word(d, plain, OP_RA, shadow as u64, 75)?;
    Ok(vec![
        Item::new(scale | wait, i, "stride", true),
        late(d.name, plain, i, "on the product")?,
    ])
}

fn translate_uldc(ctx: &mut Ctx, i: usize, d: &Decoded, w: u128) -> Result<Vec<Item>, Refused> {
    let sz_val = d.op_or(OP_SZ, 0);
    let urd = d.op_or(OP_URD, 63);
    let sa_bank = d.op_or(OP_SA_BANK, 0);
    let sa_addr = d.op_or(OP_SA_ADDR, 0);

    let upg = d.op_or(OP_UPG, 7);
    let upg_not = d.op_or(OP_UPG_AT_NOT, 0);
    let guard = vec![
        ("Pg", assemble::OpVal::Int(ctx.p(upg)?)),
        ("Pg_not", assemble::OpVal::Int(upg_not)),
    ];

    // Determine size from enum value (SZ_U8_S8_U16_S16_32_64 enum of uldc_const__RCR)
    let sizes: Vec<&str> = match sz_val {
        0 => vec!["U8"],
        1 => vec!["S8"],
        2 => vec!["U16"],
        3 => vec!["S16"],
        4 => vec!["32"],
        5 => vec!["64"],
        _ => vec![],
    };
    let small = sizes.iter().find(|&&s| matches!(s, "U8" | "S8" | "U16" | "S16"));
    if let Some(kind) = small {
        let bits: u32 = kind[1..].parse().unwrap_or(8);
        let at = sa_addr % 4;
        if at * 8 + bits > 32 {
            return Err(Refused(format!(
                "ULDC of bytes that straddle two words at 0x{:x}",
                16 * i
            )));
        }
        let r = ctx.r(urd)?;
        let mut out = vec![];
        let mut ops = guard.clone();
        ops.push(("Rd", assemble::OpVal::Int(r)));
        ops.push(("Sb_bank", assemble::OpVal::Int(sa_bank)));
        ops.push(("Sb_addr", assemble::OpVal::Int(sa_addr / 4)));
        out.push(Item::new(
            assemble::assemble("mov__RCR_RC", &ops).ok_or_else(|| Refused("cannot assemble MOV for ULDC".into()))?,
            i,
            "constant",
            true,
        ));
        let up = 32 - bits - 8 * at;
        if up > 0 {
            let mut ops2 = guard.clone();
            ops2.extend([
                ("Rd", assemble::OpVal::Int(r)),
                ("Ra", assemble::OpVal::Int(r)),
                ("Sb", assemble::OpVal::Int(up)),
                ("Rc", assemble::OpVal::Int(255)),
                ("dir", assemble::OpVal::Str("L")),
                ("fmt", assemble::OpVal::Str("U32")),
            ]);
            out.push(Item::new(
                assemble::assemble("shf__RuIR_RIR", &ops2)
                    .ok_or_else(|| Refused("cannot assemble SHF for ULDC".into()))?,
                i,
                "bytes",
                true,
            ));
        }
        let fmt_str = if kind.starts_with('S') { "S32" } else { "U32" };
        let mut ops3 = guard.clone();
        ops3.extend([
            ("Rd", assemble::OpVal::Int(r)),
            ("Ra", assemble::OpVal::Int(255)),
            ("Sb", assemble::OpVal::Int(32 - bits)),
            ("Rc", assemble::OpVal::Int(r)),
            ("dir", assemble::OpVal::Str("R")),
            ("fmt", assemble::OpVal::Str(fmt_str)),
            ("hilo", assemble::OpVal::Str("HI")),
        ]);
        let last =
            assemble::assemble("shf__RuIR_RIR", &ops3).ok_or_else(|| Refused("cannot assemble SHF for ULDC".into()))?;
        let last = (last & !(31u128 << USCHED)) | (w & (31u128 << USCHED));
        out.push(Item::new(last, i, "bytes", true));
        return Ok(out);
    }

    if !sizes.iter().any(|&s| s == "32" || s == "64") {
        return Err(Refused(format!("ULDC of {} bits at 0x{:x}", sizes.join("/"), 16 * i)));
    }
    if !sa_addr.is_multiple_of(4) {
        return Err(Refused(format!(
            "ULDC of a constant that is not at a multiple of 4 at 0x{:x}",
            16 * i
        )));
    }
    let is_64 = sizes.contains(&"64");
    let n_halves = if is_64 { 2 } else { 1 };
    let mut out = vec![];
    for half in 0..n_halves {
        let mut ops = guard.clone();
        ops.push(("Rd", assemble::OpVal::Int(ctx.r(urd + half)?)));
        ops.push(("Sb_bank", assemble::OpVal::Int(sa_bank)));
        ops.push(("Sb_addr", assemble::OpVal::Int(sa_addr / 4 + half)));
        out.push(Item::new(
            assemble::assemble("mov__RCR_RC", &ops).ok_or_else(|| Refused("cannot assemble MOV for ULDC".into()))?,
            i,
            "constant",
            true,
        ));
    }
    let last = out.last_mut().unwrap();
    last.word = (last.word & !(31u128 << USCHED)) | (w & (31u128 << USCHED));
    Ok(out)
}

/// A constant at a place that a uniform register gives. Turing has it at once. On Volta a load of
/// constants brings it some cycles later and says so on a scoreboard, which what follows was not
/// written to wait for: an instruction that does nothing waits for it there. (By then the load has
/// long read the register: it needs no copy.)
fn translate_uldc_at_index(ctx: &mut Ctx, i: usize, d: &Decoded, wait: u128) -> Result<Vec<Item>, Refused> {
    use assemble::OpVal::Int;
    if d.op_or(OP_URA, 63) == 63 {
        return Err(Refused(format!(
            "ULDC at an index that is the zero register at 0x{:x}",
            16 * i
        )));
    }
    let guard = ctx.p(d.op_or(OP_UPG, 7))?;
    let to = ctx.r(d.op_or(OP_URD, 63))?;
    let index = ctx.r(d.op_or(OP_URA, 63))?;
    let load = assemble::assemble(
        "ldc__RaNonRZ",
        &[
            ("Pg", Int(guard)),
            ("Pg_not", Int(d.op_or(OP_UPG_AT_NOT, 0))),
            ("sz", Int(d.op_or(OP_SZ, 0))),
            ("Rd", Int(to)),
            ("Ra", Int(index)),
            ("Sa_bank", Int(d.op_or(OP_SA_BANK, 0))),
            ("Ra_offset", Int(d.op_or(OP_SA_OFFSET, 0))),
            ("dst_wr_sb", Int(ARRIVES)),
        ],
    )
    .ok_or_else(|| Refused("cannot assemble LDC for ULDC".into()))?;
    let arrived =
        (with_stall(NOP, 1, false) & !(31u128 << USCHED)) | (d.word & (31u128 << USCHED)) | (1u128 << (WAIT + ARRIVES));
    Ok(vec![
        Item::new(load | wait, i, "constant at an index", true),
        Item::new(arrived, i, "until it is there", true),
    ])
}

/// `word`, a load the translation changed, with the batch of its original scheduling table
/// cleared: what the translation changed names no batch (its schedule is the translation's).
/// The wait hint of the table (its low five bits) stays.
pub(super) fn without_batch(class_name: &str, d: &Decoded, word: u128) -> Result<u128, Refused> {
    let table = "TABLES_opex_0(batch_t,usched_info)";
    let value = d.op_or_name(table, 0);
    if value >> 5 == 0 {
        return Ok(word);
    }
    put_operand_by_class(class_name, word, table, u64::from(value & 31), false)
}

/// The same global load as `d`, with the sm_70 cache and scope table. The e, size, weak or
/// constant state, scope and cop fields are the sm_70 ones; the data, guard and scoreboards are
/// the instruction's. None for a size or state that this does not know.
pub(super) fn sm70_ldg_without_hints(d: &Decoded) -> Option<u128> {
    sm70_ldg(d, d.op_or(OP_PG, 7), d.op_or(OP_PG_AT_NOT, 0))
}

/// The same global load as `d` with the given guard, and with the sm_70 cache and scope table.
fn sm70_ldg(d: &Decoded, pg: u32, pg_not: u32) -> Option<u128> {
    // The source's e, sem, sco and cop are fields 2, 6, 7 and 4 of the sm_75 class (their names
    // in `OP_NAMES`); the values are those of the sm_70 enumerations too.
    let e = match d.op_or(2, 255) {
        0 => "noe",
        1 => "E",
        _ => return None,
    };
    let sz = match d.op_or(OP_SZ, 255) {
        4 => "_32",
        5 => "_64",
        6 => "_128",
        _ => return None,
    };
    let sem = match d.op_or(6, 255) {
        0 => "CONSTANT",
        1 => "WEAK",
        2 => "STRONG",
        3 => "MMIO",
        _ => return None,
    };
    let sco = match d.op_or(7, 255) {
        0 => "CTA",
        1 => "SM",
        2 => "GPU",
        3 => "SYS",
        _ => return None,
    };
    let cop = match d.op_or(4, 255) {
        0 => "EF",
        1 => "EN",
        2 => "EL",
        3 => "LU",
        4 => "EU",
        5 => "NA",
        _ => return None,
    };
    assemble::assemble(
        "ldg__sImmOffset",
        &[
            ("e", assemble::OpVal::Str(e)),
            ("sz", assemble::OpVal::Str(sz)),
            ("sem", assemble::OpVal::Str(sem)),
            ("sco", assemble::OpVal::Str(sco)),
            ("cop", assemble::OpVal::Str(cop)),
            ("Pu", assemble::OpVal::Int(d.op_or(OP_PU, 7))),
            ("Rd", assemble::OpVal::Int(d.op_or(OP_RD, 255))),
            ("Ra", assemble::OpVal::Int(d.op_or(OP_RA, 255))),
            ("Ra_offset", assemble::OpVal::Int(d.op_or(OP_RA_OFFSET, 0))),
            ("Pg", assemble::OpVal::Int(pg)),
            ("Pg_not", assemble::OpVal::Int(pg_not)),
            ("src_rel_sb", assemble::OpVal::Int(d.op_or(OP_SRC_REL_SB, 7))),
            ("dst_wr_sb", assemble::OpVal::Int(d.op_or(OP_DST_WR_SB, 7))),
        ],
    )
}

/// A Turing global load whose sm_70 form is not the same word: the cache/scope table it carries
/// is not sm_70's, and it may name a zero-fill predicate (Pnz). The word is rebuilt with sm_70's
/// table. When Pnz is named, the load is done where the predicate holds and the destinations
/// are set to zero elsewhere, as ptxas writes such a load for Volta itself.
fn translate_ldg_without_hints(ctx: &mut Ctx<'_>, i: usize, d: &Decoded) -> Result<Vec<Item>, Refused> {
    let fill = d.op_or_name("TABLES_Pnz_0(Pnz@not,Pnz)", 0);
    let no_translation = || {
        Refused(format!(
            "no translation for the Turing global load's cache or scope at 0x{:x}",
            16 * i
        ))
    };
    if fill == 0 {
        let word = sm70_ldg_without_hints(d).ok_or_else(no_translation)?;
        if ctx.dec70.decode(word).is_none() {
            return Err(no_translation());
        }
        return Ok(vec![Item::new(word, i, "without its cache hint", true)]);
    }
    if d.op_or(OP_PG, 7) != 7 || d.op_or(OP_PG_AT_NOT, 0) != 0 {
        return Err(Refused(
            "a Turing global load with a guard and a zero-fill predicate".into(),
        ));
    }
    // The table of Pnz: 0 is PT, 1 to 7 are P6 down to P0 (the table of an operand's predicate
    // and that of the instruction's guard number them differently), bit 3 negates.
    let index = fill & 7;
    let not = (fill >> 3) & 1;
    let predicate = if index == 0 { 7 } else { 7 - index };
    let word = sm70_ldg(d, predicate, not).ok_or_else(no_translation)?;
    if ctx.dec70.decode(word).is_none() {
        return Err(no_translation());
    }
    let mut items = vec![Item::new(word, i, "zero fill: load where the predicate holds", true)];
    for &(letter, register) in &d.writes {
        if letter != crate::decode::LETTER_R {
            continue;
        }
        let zero = assemble::assemble(
            "mov__RRR_RR",
            &[
                ("Pg", assemble::OpVal::Int(predicate)),
                ("Pg_not", assemble::OpVal::Int(1 - not)),
                ("Rd", assemble::OpVal::Int(register)),
                ("Rb", assemble::OpVal::Int(255)),
            ],
        )
        .ok_or_else(|| Refused("cannot assemble MOV for a zero-filled load".into()))?;
        items.push(Item::new(zero, i, "zero fill: zeros elsewhere", true));
    }
    Ok(items)
}

/// Turing's `F2FP.F16.F32` packs two f32 values into two f16 halves of one register. Volta's
/// `F2F.F16.F32` converts one value; two of them and a PRMT pack the pair. Only the plain
/// rounding to f16 (the form the half kernels use) is translated.
fn translate_f2fp_f16(ctx: &mut Ctx<'_>, i: usize, d: &Decoded) -> Result<Vec<Item>, Refused> {
    // dstfmt 73, satfinite 107, relu 116, merge 117, rndMode 118 (names in `OP_NAMES`).
    if d.op_or(73, 0) != 0
        || d.op_or(107, 0) != 0
        || d.op_or(116, 0) != 0
        || d.op_or(117, 0) != 0
        || d.op_or(118, 0) != 0
    {
        return Err(Refused(
            "f2fp with a destination format or a rounding that is not translated".into(),
        ));
    }
    let (rd, high, low) = (d.op_or(OP_RD, 255), d.op_or(OP_RA, 255), d.op_or(13, 255));
    if rd == 255 || low == 255 {
        return Err(Refused("f2fp without a destination or a low source".into()));
    }
    let (pg, pg_not) = (d.op_or(OP_PG, 7), d.op_or(OP_PG_AT_NOT, 0));
    let mut items = Vec::new();
    let upper = if high == 255 {
        255
    } else if let Ok(register) = ctx.pool(1) {
        register
    } else {
        // No pool: the high half is converted in the scratch register that `parse` placed for
        // it (one per function, not one for each conversion: they do not overlap).
        ctx.scratch()?
    };
    let mut convert = |dest: u32, source: u32, guarded: bool, what: &'static str| -> Result<(), Refused> {
        let mut ops = vec![
            ("dstfmt.srcfmt", assemble::OpVal::Str("F16.F32")),
            ("Rd", assemble::OpVal::Int(dest)),
            ("Rb", assemble::OpVal::Int(source)),
        ];
        if guarded {
            ops.push(("Pg", assemble::OpVal::Int(pg)));
            ops.push(("Pg_not", assemble::OpVal::Int(pg_not)));
        }
        let word = assemble::assemble("f2f_f32_downconvert__RRR_RRR", &ops)
            .ok_or_else(|| Refused("cannot assemble F2F for f2fp".into()))?;
        let word = (word & !(7u128 << WRITE_BARRIER)) | (u128::from(ARRIVES) << WRITE_BARRIER) | (63u128 << WAIT);
        let mut item = Item::new(word, i, what, true);
        item.then = Some(ARRIVES);
        items.push(item);
        Ok(())
    };
    if high != 255 {
        convert(upper, high, false, "f2fp: convert high half")?;
    }
    convert(rd, low, true, "f2fp: convert low half")?;
    let permute = assemble::assemble(
        "prmt__RuIR_RIR",
        &[
            ("Rd", assemble::OpVal::Int(rd)),
            ("Ra", assemble::OpVal::Int(rd)),
            ("Sb", assemble::OpVal::Int(0x5410)),
            ("Rc", assemble::OpVal::Int(upper)),
            ("Pg", assemble::OpVal::Int(pg)),
            ("Pg_not", assemble::OpVal::Int(pg_not)),
        ],
    )
    .ok_or_else(|| Refused("cannot assemble PRMT for f2fp".into()))?;
    // The source's write barrier is the translation's last instruction's, or none.
    let permute = (permute & !(7u128 << WRITE_BARRIER)) | ((d.word >> WRITE_BARRIER) & 7) << WRITE_BARRIER;
    items.push(Item::new(permute, i, "f2fp: pack halves", true));
    Ok(items)
}
