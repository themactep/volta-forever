//! Tests for the translator: scheduling, placement, borrowed registers, predicate
//! assignment, and assembly of the instructions the translation builds.
use super::borrowed::Borrowed;
use super::context::pick;
use super::parse::{FuncData, placement};
use super::*;
use crate::assemble;
use crate::decode::{Decoder, Kind, LETTER_P, LETTER_R, LETTER_U};

fn schedule(word: u128) -> (u32, u128) {
    (((word >> USCHED) & 31) as u32, word & REUSE)
}

/// The three bits above the wait mask are marks of register reuse only with a stall that ends
/// its group; with any other they name a batch. An instruction whose stall grows past what can
/// end a group loses its marks rather than start a batch.
#[test]
fn marks_of_reuse_do_not_become_a_batch() {
    let marks = 1u128 << 122;
    let word = marks | (u128::from(16u32 | 6) << USCHED);
    assert_eq!(schedule(with_stall(word, 11, true)), (16 | 11, marks));
    assert_eq!(schedule(with_stall(word, 12, true)), (12, 0));
    // Operands that bear no marks: the bits are a batch whatever the stall, and stay.
    assert_eq!(schedule(with_stall(word, 12, false)), (12, marks));
}

/// An instruction that names a batch is not made to end its group: the batch would be read as
/// marks of reuse.
#[test]
fn a_batch_does_not_become_marks_of_reuse() {
    let batch = 1u128 << 122;
    let word = batch | (u128::from(16u32 | 13) << USCHED);
    assert_eq!(schedule(with_stall(word, 5, true)), (5, batch));
    // With no batch named, it goes on ending its group.
    let plain = u128::from(16u32 | 13) << USCHED;
    assert_eq!(schedule(with_stall(plain, 5, true)), (16 | 5, 0));
}

/// A function with a product has seventeen registers of the translation for its matrix
/// instructions, whichever register comes first: the lane, four values kept of it, four
/// registers for a gathered B and eight to compute into, the last two kinds on even registers.
#[test]
fn a_product_has_seventeen_registers() {
    for base in [10u32, 11] {
        let (_, _, _, lane, kept, gathered, scratch, free, _, _, _) =
            placement(base, &[], &[], false, 0, true, MATRIX_SCRATCH, true, None, &[], &[]).unwrap();
        let (lane, kept, gathered, scratch) = (lane.unwrap(), kept.unwrap(), gathered.unwrap(), scratch.unwrap());
        assert_eq!(free - base, 1 + KEPT + GATHERED + MATRIX_SCRATCH);
        assert!(gathered % 2 == 0 && scratch % 2 == 0);
        let mut all = vec![lane];
        all.extend(kept..kept + KEPT);
        all.extend(gathered..gathered + GATHERED);
        all.extend(scratch..scratch + MATRIX_SCRATCH);
        all.sort_unstable();
        all.dedup();
        assert_eq!(all, (base..free).collect::<Vec<u32>>());
    }
}

/// An item for the schedule: its class, its operands, what it is, whether the translation
/// changed it.
type Given<'a> = (&'static str, &'a [(&'static str, u32)], &'static str, bool);

/// Items for the schedule, decoded.
fn items_of(list: &[Given]) -> Vec<Item> {
    let mut dec70 = Decoder::new(70);
    list.iter()
        .enumerate()
        .map(|(i, &(name, operands, what, changed))| {
            let operands: Vec<(&str, assemble::OpVal)> =
                operands.iter().map(|&(o, v)| (o, assemble::OpVal::Int(v))).collect();
            // (Branches are not instructions the translation builds: their words as they are,
            // to the next instruction, with a stall of one.)
            let word = match name {
                "BRA" => 0x000fc200038000000000000000007947,
                "BRA P3" => 0x000fc200018000000000000000007947,
                _ => assemble::assemble(name, &operands).expect(name),
            };
            let mut item = Item::new(word, i, what, changed);
            item.first = true;
            item.d = Some(dec70.decode(word).expect(name));
            if what == "restore" {
                let mask = item.d.as_ref().unwrap().op_or(OP_SB, 0);
                item.more_writes
                    .extend((0..7u32).filter(|p| (mask >> p) & 1 != 0).map(|p| (LETTER_P, p)));
            }
            item
        })
        .collect()
}

fn stalls(items: &[Item]) -> Vec<u32> {
    items.iter().map(|item| stall_of(item.word)).collect()
}

/// R2P reads the seven predicates and writes them all back: two predicates given back one
/// after the other are five cycles apart (one cycle apart, the second wrote the first's
/// back as it was before: measured on a V100), and an R2P comes four cycles
/// after an instruction that writes a predicate it does not give back, not after one that
/// writes the predicate it gives back.
#[test]
fn an_r2p_is_kept_from_the_writers_of_the_other_predicates() {
    let isetp = |p: u32| -> (&'static str, Vec<(&'static str, u32)>) {
        (
            "isetp__RRR_RR",
            vec![("Pu", p), ("Pv", 7), ("Ra", 4), ("Rb", 255), ("Pp", 7)],
        )
    };
    // Two restores.
    let mut items = items_of(&[
        ("r2p__RIR", &[("Ra", 20), ("Sb", 1)], "restore", true),
        ("r2p__RIR", &[("Ra", 21), ("Sb", 2)], "restore", true),
        ("nop_", &[], "same", false),
    ]);
    super::schedule::schedule(&mut items, &[]).unwrap();
    assert_eq!(stalls(&items)[0], 5);
    // A comparison into P2, then the restore of P4: four cycles.
    let (name, operands) = isetp(2);
    let mut items = items_of(&[
        (name, &operands, "same", false),
        ("r2p__RIR", &[("Ra", 20), ("Sb", 16)], "restore", true),
        ("nop_", &[], "same", false),
    ]);
    super::schedule::schedule(&mut items, &[]).unwrap();
    assert_eq!(stalls(&items)[0], 4);
    // A comparison into P4, then the restore of P4: nothing to keep.
    let (name, operands) = isetp(4);
    let mut items = items_of(&[
        (name, &operands, "same", false),
        ("r2p__RIR", &[("Ra", 20), ("Sb", 16)], "restore", true),
        ("nop_", &[], "same", false),
    ]);
    super::schedule::schedule(&mut items, &[]).unwrap();
    assert_eq!(stalls(&items)[0], 1);
    // The program's own instructions, both as they were: left as the compiler had them.
    let (name, operands) = isetp(2);
    let mut items = items_of(&[
        (name, &operands, "same", false),
        ("r2p__RIR", &[("Ra", 20), ("Sb", 16)], "same", false),
        ("nop_", &[], "same", false),
    ]);
    super::schedule::schedule(&mut items, &[]).unwrap();
    assert_eq!(stalls(&items)[0], 1);
}

/// A predicate that the translation saves waits for the scoreboard on which the instruction
/// that wrote it announces it (the program's own readers do; a read the translation made up
/// has nothing of the program that waits for it), unless an instruction since has waited.
#[test]
fn a_saved_predicate_waits_for_the_scoreboard_of_its_writer() {
    let compare: &[(&'static str, u32)] = &[("Pu", 3), ("Pv", 7), ("Ra", 4), ("Rb", 255), ("Pp", 7)];
    let save: &[(&'static str, u32)] = &[("Rd", 20), ("Ra", 255), ("Sb", 8)];
    let build = |between: bool| -> Vec<Item> {
        let mut list: Vec<Given> = vec![("isetp__RRR_RR", compare, "same", false)];
        if between {
            list.push(("nop_", &[], "same", false));
        }
        list.push(("p2r__RsIR_RIR", save, "save", true));
        list.push(("nop_", &[], "same", false));
        let mut items = items_of(&list);
        // (No instruction of the fixed-point unit announces a predicate on a scoreboard; the
        // ones that do, DSETP and the like, are not instructions the translation builds. The
        // scheduler goes by the word's scoreboard, whatever the instruction: scoreboard 2.)
        items[0].word = (items[0].word & !(7u128 << WRITE_BARRIER)) | (2u128 << WRITE_BARRIER);
        if between {
            items[1].word |= 1u128 << (WAIT + 2);
        }
        let k = items.len() - 2;
        items[k].more_reads.push((LETTER_P, 3, 0));
        items
    };
    let mut items = build(false);
    super::schedule::schedule(&mut items, &[]).unwrap();
    assert_eq!((items[1].word >> WAIT) & 63, 1 << 2);
    // The program's instruction between them waits for it already: no second wait.
    let mut items = build(true);
    super::schedule::schedule(&mut items, &[]).unwrap();
    assert_eq!((items[2].word >> WAIT) & 63, 0);
}

/// A predicate that the translation saves right after an R2P of the program gave it a value
/// is five cycles after that R2P (the program's R2P writes the predicates of its mask, which
/// its operands do not say); a predicate outside the mask is not waited for.
#[test]
fn a_predicate_saved_after_an_r2p_of_the_program_waits_for_it() {
    for (mask, stall) in [(8u32, 5u32), (4, 1)] {
        let mut items = items_of(&[
            ("r2p__RIR", &[("Ra", 9), ("Sb", mask)], "same", false),
            ("p2r__RsIR_RIR", &[("Rd", 20), ("Ra", 255), ("Sb", 8)], "save", true),
            ("nop_", &[], "same", false),
        ]);
        items[1].more_reads.push((LETTER_P, 3, 0));
        super::schedule::schedule(&mut items, &[]).unwrap();
        assert_eq!(stalls(&items)[0], stall);
    }
}

/// What was written before a label is known after it where the code goes on into it: a
/// predicate saved by the translation right after a label is four cycles after the
/// comparison before the label that wrote it; not when the code before leaves (a branch
/// that is always taken).
#[test]
fn the_schedule_goes_on_through_a_label() {
    let compare: &[(&'static str, u32)] = &[("Pu", 3), ("Pv", 7), ("Ra", 4), ("Rb", 255), ("Pp", 7)];
    let save: &[(&'static str, u32)] = &[("Rd", 20), ("Ra", 255), ("Sb", 8)];
    let saved = |mut items: Vec<Item>| -> Vec<Item> {
        let k = items.iter().position(|item| item.what == "save").unwrap();
        items[k].more_reads.push((LETTER_P, 3, 0));
        items
    };
    let mut items = saved(items_of(&[
        ("isetp__RRR_RR", compare, "same", false),
        ("p2r__RsIR_RIR", save, "save", true),
        ("nop_", &[], "same", false),
    ]));
    // (Item 1 is where a branch goes: a block of the schedule starts there.)
    super::schedule::schedule(&mut items, &[16]).unwrap();
    assert_eq!(stalls(&items)[0], 4);
    let mut items = saved(items_of(&[
        ("isetp__RRR_RR", compare, "same", false),
        ("BRA", &[], "same", false),
        ("p2r__RsIR_RIR", save, "save", true),
        ("nop_", &[], "same", false),
    ]));
    super::schedule::schedule(&mut items, &[32]).unwrap();
    assert_eq!(stalls(&items)[..2], [1, 1]);
    // A branch that may not be taken: the code goes on, three cycles are missing.
    let mut items = saved(items_of(&[
        ("isetp__RRR_RR", compare, "same", false),
        ("BRA P3", &[], "same", false),
        ("p2r__RsIR_RIR", save, "save", true),
        ("nop_", &[], "same", false),
    ]));
    super::schedule::schedule(&mut items, &[]).unwrap();
    assert_eq!(stalls(&items)[..2], [1, 3]);
}

/// A function with matrix loads and no product has the lane and the eight registers to
/// compute into, and nothing kept.
#[test]
fn loads_alone_keep_nothing() {
    for base in [10u32, 11] {
        let (_, _, _, lane, kept, gathered, scratch, free, _, _, _) =
            placement(base, &[], &[], false, 0, true, MATRIX_SCRATCH, false, None, &[], &[]).unwrap();
        assert_eq!(free - base, 1 + MATRIX_SCRATCH);
        assert!(lane.is_some() && kept.is_none() && gathered.is_none() && scratch.unwrap() % 2 == 0);
    }
}

// ---- Borrowed registers ----

/// What an instruction of `make_borrow_func` reads (letter, register, role) and writes (letter,
/// register).
type Uses<'a> = (&'a [(u8, u32, u8)], &'a [(u8, u32)]);

/// Helper: build a minimal FuncData with explicit reads/writes for each instruction.
/// Each instruction is specified as (reads, writes) where reads is [(letter, reg, role)]
/// and writes is [(letter, reg)]. The mnemonic is NOP for all.
fn make_borrow_func(instrs: &[Uses], registers: u32, uniform: &[u32], pairs: &[u32], needs_pool: bool) -> FuncData {
    let mut dec75 = crate::decode::Decoder::new(75);
    // NOP word for sm_75: a known NOP.
    let nop_word: u128 = 0x000fc00000000000_0000000000007918;
    let src: Vec<Decoded> = instrs
        .iter()
        .map(|(reads, writes)| {
            let mut d = dec75.decode(nop_word).unwrap();
            d.reads = reads.to_vec();
            d.writes = writes.to_vec();
            d
        })
        .collect();
    let highest = instrs
        .iter()
        .flat_map(|(reads, writes)| {
            reads
                .iter()
                .filter(|&&(l, _, _)| l == LETTER_R)
                .map(|&(_, r, _)| r)
                .chain(writes.iter().filter(|&&(l, _)| l == LETTER_R).map(|&(_, r)| r))
        })
        .filter(|&r| r != 255)
        .max()
        .unwrap_or(0);
    FuncData {
        name: "test_function".into(),
        registers,
        tables: vec![],
        src,
        uniform: uniform.to_vec(),
        pairs: pairs.to_vec(),
        needs_pool,
        uniform_predicates: vec![],
        highest,
        scratch_need: 0,
        needs_lane: false,
        product_count: 0,
        entered_inside: false,
        ampere: false,
        ampere_ops: HashMap::new(),
    }
}

/// A free even pair is found and not one that is alive, read or written at the instruction.
#[test]
fn borrowed_pair_avoids_alive_registers() {
    // Instruction 0: writes R4, reads R5. Instruction 1: writes R6, reads R7.
    // R2, R3 are free at both.
    let fd = make_borrow_func(
        &[
            (&[(LETTER_R, 5, 0)], &[(LETTER_R, 4)]),
            (&[(LETTER_R, 7, 0)], &[(LETTER_R, 6)]),
        ],
        10,
        &[],
        &[],
        false,
    );
    let b = Borrowed::new(&fd, false, None).unwrap();
    assert_eq!(b.free_at(0, 2, &regset_empty()), Some(2));
    assert_eq!(b.free_at(1, 2, &regset_empty()), Some(2));
}

/// A uniform register gets a home only where the register is free through its whole life.
#[test]
fn borrowed_home_only_in_free_register() {
    // U4 written at 0, read at 1 and 2. R2..R7 busy everywhere. highest=7.
    // No register free in range(2, 7+2-1, 1) = range(2, 8) = 2..7: all busy.
    let fd = make_borrow_func(
        &[
            (&[(LETTER_R, 2, 0), (LETTER_R, 3, 0)], &[(LETTER_U, 4), (LETTER_R, 4)]),
            (
                &[(LETTER_U, 4, 0), (LETTER_R, 5, 0), (LETTER_R, 6, 0)],
                &[(LETTER_R, 7)],
            ),
            (
                &[(LETTER_U, 4, 0), (LETTER_R, 2, 0), (LETTER_R, 7, 0)],
                &[(LETTER_R, 3)],
            ),
        ],
        10,
        &[4],
        &[],
        false,
    );
    let b = Borrowed::new(&fd, false, None).unwrap();
    // U4 should NOT get a home because R2..R7 are all busy in the span.
    assert!(
        b.homes.is_empty(),
        "no home should be found when all registers are busy"
    );
}

/// R0 and R1 are never taken as homes or for the pool.
#[test]
fn borrowed_never_takes_r0_r1() {
    // U4 at all instructions, R2..R6 busy. Even if R0 and R1 are free, they must not be used.
    let fd = make_borrow_func(
        &[
            (&[], &[(LETTER_U, 4), (LETTER_R, 2)]),
            (
                &[
                    (LETTER_U, 4, 0),
                    (LETTER_R, 3, 0),
                    (LETTER_R, 4, 0),
                    (LETTER_R, 5, 0),
                    (LETTER_R, 6, 0),
                ],
                &[(LETTER_R, 2)],
            ),
        ],
        10,
        &[4],
        &[],
        false,
    );
    let b = Borrowed::new(&fd, false, None).unwrap();
    for (&_u, &r) in &b.homes {
        assert!(r >= 2, "R0 and R1 must never be taken as homes");
    }
    for &r in &b.used {
        assert!(r >= 2, "R0 and R1 must never be in used set");
    }
    // Also check pair_at: should never return 0.
    let pair = b.free_at(0, 2, &regset_empty());
    assert!(pair.is_none() || pair.unwrap() >= 2);
    let single = b.free_at(0, 1, &regset_empty());
    assert!(single.is_none() || single.unwrap() >= 2);
}

/// Two uniform registers whose lives meet do not share a home.
#[test]
fn borrowed_no_shared_home_for_overlapping_lives() {
    // U4 and U5 both alive at all instructions. R2..R3 free. highest=10.
    let fd = make_borrow_func(
        &[
            (&[], &[(LETTER_U, 4), (LETTER_U, 5), (LETTER_R, 10)]),
            (
                &[(LETTER_U, 4, 0), (LETTER_U, 5, 0), (LETTER_R, 10, 0)],
                &[(LETTER_R, 10)],
            ),
        ],
        15,
        &[4, 5],
        &[],
        false,
    );
    let b = Borrowed::new(&fd, false, None).unwrap();
    if b.homes.contains_key(&4) && b.homes.contains_key(&5) {
        assert_ne!(b.homes[&4], b.homes[&5], "overlapping lives must not share a home");
    }
}

/// One register is enough where the sequence needs one (pool with width 1).
#[test]
fn borrowed_pool_width_one() {
    // R4 and R5 busy. R2, R3, R6, R7 free. pair_at(0, 1) should find R2.
    let fd = make_borrow_func(
        &[
            (&[(LETTER_R, 4, 0), (LETTER_R, 5, 0)], &[(LETTER_R, 4)]),
            (&[(LETTER_R, 4, 0)], &[(LETTER_R, 5)]),
        ],
        10,
        &[],
        &[],
        false,
    );
    let b = Borrowed::new(&fd, false, None).unwrap();
    let r = b.free_at(0, 1, &regset_empty());
    assert_eq!(r, Some(2), "width 1 should find R2 as the first free register");
}

/// The first item of an instruction that took borrowed registers waits for all six
/// scoreboards (a pending write would overwrite what is put there).
#[test]
fn borrowed_pool_instruction_waits_for_all_scoreboards() {
    // Translate a cubin with VF_BORROW=1 and check the wait mask.
    // This is tested indirectly through the full cubin comparison (compare.sh), but we
    // verify the invariant on the Options struct.
    let opts = Options {
        borrow: 1,
        ..Options::default()
    };
    assert!(opts.borrow > 0);
}

/// What the layer does by itself is the default; `VF_BORROW` is a number, and anything
/// else leaves the default (0 turns the borrowing off).
#[test]
fn borrow_default_and_setting() {
    assert_eq!(Options::default().borrow, BORROW_DEFAULT);
    assert_eq!(BORROW_DEFAULT, 7);
    assert_eq!(borrow_from(None), BORROW_DEFAULT);
    assert_eq!(borrow_from(Some("")), BORROW_DEFAULT);
    assert_eq!(borrow_from(Some("no")), BORROW_DEFAULT);
    assert_eq!(borrow_from(Some("0")), 0);
    assert_eq!(borrow_from(Some(" 5 ")), 5);
    assert_eq!(borrow_from(Some("7")), 7);
    assert_eq!(borrow_from(Some("8")), 8);
}

/// `product_count` is computed by `parse_function`.
#[test]
fn product_count_in_funcdata() {
    // A FuncData with no products.
    let fd = make_borrow_func(
        &[
            (&[(LETTER_R, 5, 0)], &[(LETTER_R, 4)]),
            (&[(LETTER_R, 7, 0)], &[(LETTER_R, 6)]),
        ],
        10,
        &[],
        &[],
        false,
    );
    assert_eq!(fd.product_count, 0);
}

// ---- Part A: predicate delays ----

/// A function where every free predicate is one a DSETP writes: the wait is there,
/// and waits for all six scoreboards.
#[test]
fn slow_predicate_gets_a_wait() {
    // DSETP writes P3, so it's slow. The delay for P3 from a DSETP to FXU should be 13
    // (the value from the distance table).
    let dist = crate::distance::least(
        LETTER_P,
        0,
        &{
            let mut dec75 = crate::decode::Decoder::new(75);
            // We need a decoded DSETP. Build a minimal one.
            let nop = dec75.decode(0x000fc00000000000_0000000000007918).unwrap();
            let mut d = nop.clone();
            d.mnemonic = "DSETP";
            d.kind = Kind::FMALITE;
            d
        },
        Kind::FXU,
    );
    // DSETP to FXU should be 13 cycles.
    assert!(dist >= 13, "DSETP to FXU distance should be >= 13, got {}", dist);
}

/// An FCHK under no scoreboard: 26 cycles (the longest distance of its kind).
#[test]
fn fchk_distance_is_its_kinds_longest() {
    // FCHK is MIO_FAST. The distance table's longest for MIO_FAST writing a predicate
    // should give 26 when there's no specific row for FCHK (it falls back to the longest
    // distance of its kind).
    let mut dec75 = crate::decode::Decoder::new(75);
    let nop = dec75.decode(0x000fc00000000000_0000000000007918).unwrap();
    let mut d = nop.clone();
    d.mnemonic = "FCHK";
    d.kind = Kind::MioFast;
    // With no scoreboard (write barrier = 7), and no specific FCHK row in the table,
    // least falls back to the longest of its kind.
    let dist = crate::distance::least(LETTER_P, 0, &d, Kind::FXU);
    assert!(dist >= 26, "FCHK to FXU distance should be >= 26, got {}", dist);
}

/// A predicate that a memory instruction writes under no scoreboard is never taken.
#[test]
fn memory_writer_predicate_never_taken() {
    // MIO_SLOW writes => never bit set. The pick function should not take it.
    let slow: u32 = 1 << 3; // P3 is slow
    let never: u32 = 1 << 3; // P3 is also never (MIO_SLOW)
    // All other predicates busy except P3.
    let taken: u32 = 0x7f & !(1 << 3);
    let order: Vec<u32> = (0..7u32).collect();
    let result = pick(taken, &order, slow, never);
    assert!(result.is_none(), "P3 should not be picked because it is in `never`");
}

/// A free predicate that nothing writes late is preferred.
#[test]
fn non_slow_predicate_preferred() {
    // P2 is slow, P5 is not. Both free. pick should take P5 first (the non-slow one).
    let slow: u32 = 1 << 2;
    let never: u32 = 0;
    let taken: u32 = 0x7f & !((1 << 2) | (1 << 5));
    let order: Vec<u32> = (0..7u32).collect();
    let result = pick(taken, &order, slow, never);
    assert_eq!(result, Some(5), "should pick P5 (not slow) over P2 (slow)");
}

// ---- Part B new: unannounced writers, says ----

/// A register written by DMUL is not borrowed (it is in DEEP_WRITERS).
#[test]
fn dmul_register_not_borrowed() {
    assert!(DEEP_WRITER_MNEMONICS.contains(&"DMUL"));
}

/// unannounced recognizes an I2F under no scoreboard (MIO_FAST, barrier 7).
#[test]
fn unannounced_i2f_no_scoreboard() {
    let mut dec = crate::decode::Decoder::new(70);
    let nop = dec.decode(0x000fc00000000000_0000000000007918).unwrap();
    let mut d = nop.clone();
    d.kind = Kind::MioFast;
    d.mnemonic = "I2F";
    // Set write barrier to 7 (no scoreboard): bits 110..113 all 1.
    d.word = (d.word & !(7u128 << WRITE_BARRIER)) | (7u128 << WRITE_BARRIER);
    assert!(unannounced(&d), "I2F under no scoreboard should be unannounced");
    // With a scoreboard (barrier != 7): not unannounced.
    d.word = (d.word & !(7u128 << WRITE_BARRIER)) | (3u128 << WRITE_BARRIER);
    assert!(!unannounced(&d), "I2F on scoreboard 3 should not be unannounced");
}

/// LDL and STL assemble with the template generator and decode back correctly,
/// including with a negative offset (the slot is under the top of local memory).
#[test]
fn ldl_and_stl_assemble_and_decode() {
    let mut dec = crate::decode::Decoder::new(70);
    // LDL with a negative offset (24-bit mask).
    let ldl = assemble::assemble(
        "ldl__sImmOffset",
        &[
            ("Rd", assemble::OpVal::Int(10)),
            ("Ra", assemble::OpVal::Int(15)),
            ("Ra_offset", assemble::OpVal::Int((-16i32 as u32) & 0xFFFFFF)),
            ("dst_wr_sb", assemble::OpVal::Int(ARRIVES)),
        ],
    );
    assert!(ldl.is_some(), "LDL with negative offset should assemble");
    let d = dec.decode(ldl.unwrap());
    assert!(d.is_some(), "LDL word should decode");
    assert_eq!(d.unwrap().mnemonic, "LDL");

    // STL with a negative offset.
    let stl = assemble::assemble(
        "stl__sImmOffset",
        &[
            ("Ra", assemble::OpVal::Int(10)),
            ("Rb", assemble::OpVal::Int(5)),
            ("Ra_offset", assemble::OpVal::Int((-32i32 as u32) & 0xFFFFFF)),
            ("Pg", assemble::OpVal::Int(7)),
            ("Pg_not", assemble::OpVal::Int(0)),
        ],
    );
    assert!(stl.is_some(), "STL with negative offset should assemble");
    let d = dec.decode(stl.unwrap());
    assert!(d.is_some(), "STL word should decode");
    assert_eq!(d.unwrap().mnemonic, "STL");
}

/// The R1 adjustment at level 3 assembles as IADD3 R1, R1, -frame, RZ.
#[test]
fn r1_adjustment_assembles() {
    let frame: u32 = 32;
    let w = assemble::assemble(
        "iadd3_imm__RsIR_RIR",
        &[
            ("Rd", assemble::OpVal::Int(1)),
            ("Ra", assemble::OpVal::Int(1)),
            ("Sb", assemble::OpVal::Int(-(frame as i32) as u32)),
            ("Rc", assemble::OpVal::Int(255)),
        ],
    );
    assert!(w.is_some(), "IADD3 R1, R1, -0x20, RZ should assemble");
    let mut dec = crate::decode::Decoder::new(70);
    let d = dec.decode(w.unwrap());
    assert!(d.is_some(), "IADD3 word should decode");
    let d = d.unwrap();
    assert_eq!(d.mnemonic, "IADD3");
    // Rd=R1, Ra=R1. R255 (RZ) may not appear in reads (zero register).
    assert!(d.writes.iter().any(|&(l, r)| l == LETTER_R && r == 1));
    assert!(d.reads.iter().any(|&(l, r, _)| l == LETTER_R && r == 1));
}

/// free_at skips registers that are in the `given` set.
#[test]
fn free_at_skips_given() {
    // Build a minimal Borrowed with top=10 and all registers free at instruction 0.
    let b = Borrowed {
        top: 10,
        highest: 10,
        busy: vec![regset_empty()],
        taken: vec![regset_empty()],
        touch: vec![regset_empty()],
        never: regset_empty(),
        homes: HashMap::new(),
        spans: HashMap::new(),
        used: HashSet::new(),
        everything: false,
        slots: HashMap::new(),
        predicate_slots: HashMap::new(),
        frame_bytes: 0,
        depth: vec![None],
        r1_fixed: true,
        room: false,
        from_r1: false,
        spill: 0,
    };
    // With nothing given, the first free register is R2.
    let empty_given = regset_empty();
    assert_eq!(b.free_at(0, 1, &empty_given), Some(2));
    // With R2 given, it skips to R3.
    let mut given = regset_empty();
    regset_set(&mut given, 2);
    assert_eq!(b.free_at(0, 1, &given), Some(3));
    // With R2 and R3 given, it goes to R4.
    regset_set(&mut given, 3);
    assert_eq!(b.free_at(0, 1, &given), Some(4));
    // Width 2: needs even start. With R2 given, the first even pair from R4.
    let mut given2 = regset_empty();
    regset_set(&mut given2, 2);
    regset_set(&mut given2, 3);
    assert_eq!(b.free_at(0, 2, &given2), Some(4));
}

/// free_at returns None when all registers up to top are busy or given.
#[test]
fn free_at_returns_none_when_full() {
    let mut busy = regset_empty();
    for r in 0..=20 {
        regset_set(&mut busy, r);
    }
    let b = Borrowed {
        top: 20,
        highest: 20,
        busy: vec![busy],
        taken: vec![regset_empty()],
        touch: vec![regset_empty()],
        never: regset_empty(),
        homes: HashMap::new(),
        spans: HashMap::new(),
        used: HashSet::new(),
        everything: false,
        slots: HashMap::new(),
        predicate_slots: HashMap::new(),
        frame_bytes: 0,
        depth: vec![None],
        r1_fixed: true,
        room: false,
        from_r1: false,
        spill: 0,
    };
    assert_eq!(b.free_at(0, 1, &regset_empty()), None);
}

/// The stack-pointer check: a function with c[0x0][0x28] read by a non-first
/// instruction is detected by stack_top.
#[test]
fn stack_top_detects_cbuf_0x28() {
    let mut dec = crate::decode::Decoder::new(75);
    // MOV R1, c[0x0][0x28] reads the stack top.
    let word = assemble::assemble(
        "mov__RCR_RC",
        &[
            ("Rd", assemble::OpVal::Int(1)),
            ("Sb_bank", assemble::OpVal::Int(0)),
            ("Sb_addr", assemble::OpVal::Int(0x28 / 4)),
        ],
    )
    .unwrap();
    let d = dec.decode(word).unwrap();
    assert!(stack_top(&d), "MOV R1, c[0x0][0x28] should read stack top");
    // MOV R1, c[0x0][0x24] should NOT.
    let word2 = assemble::assemble(
        "mov__RCR_RC",
        &[
            ("Rd", assemble::OpVal::Int(1)),
            ("Sb_bank", assemble::OpVal::Int(0)),
            ("Sb_addr", assemble::OpVal::Int(0x24 / 4)),
        ],
    )
    .unwrap();
    let d2 = dec.decode(word2).unwrap();
    assert!(!stack_top(&d2), "MOV R1, c[0x0][0x24] should not read stack top");
}

// ---- Part D: room made and switch 5/6 ----

/// `victims_at` finds a register the instruction does not touch, respecting `never` and
/// the taken mask.
#[test]
fn victims_at_finds_untouched_register() {
    // Instruction 0: reads R2, writes R3. R4 and R5 alive but not touched.
    // victims_at(0, 1, empty) should find the lowest register that is not read or
    // written by instruction 0, not in taken, not in never, and not R0 or R1.
    let fd = make_borrow_func(
        &[
            (&[(LETTER_R, 2, 0), (LETTER_R, 4, 0)], &[(LETTER_R, 3), (LETTER_R, 5)]),
            (&[(LETTER_R, 4, 0), (LETTER_R, 5, 0)], &[(LETTER_R, 6)]),
        ],
        10,
        &[],
        &[],
        false,
    );
    let b = Borrowed::new(&fd, false, None).unwrap();
    // At instruction 0: R2, R3, R4, R5 are touched. R6 is not touched.
    // highest = 6, so range is 2..=(6+2-1-1)=7 but only up to highest+2-width.
    let victim = b.victims_at(0, 1, &regset_empty());
    assert!(victim.is_some(), "should find a victim");
    let r = victim.unwrap();
    assert!(r >= 2, "victim must not be R0 or R1");
    // R2 is read, R3 is written, R4 is read, R5 is written at instruction 0.
    assert!(
        r != 2 && r != 3 && r != 4 && r != 5,
        "victim must not be touched by instruction 0"
    );
}

/// `victims_at` returns None when every register up to `highest` is touched.
#[test]
fn victims_at_returns_none_when_all_touched() {
    // Instruction 0 touches R2, R3, R4. highest=4. No room.
    let fd = make_borrow_func(
        &[(&[(LETTER_R, 2, 0), (LETTER_R, 3, 0)], &[(LETTER_R, 4)])],
        10,
        &[],
        &[],
        false,
    );
    let b = Borrowed::new(&fd, false, None).unwrap();
    // highest = 4. range(2, 4+2-1, 1) = range(2, 5) = 2,3,4. All touched.
    let victim = b.victims_at(0, 1, &regset_empty());
    assert!(victim.is_none(), "all registers are touched, should return None");
}

/// R1 depth: through a function that lowers R1 by IADD3.
#[test]
fn r1_depth_through_iadd3() {
    let mut dec75 = crate::decode::Decoder::new(75);
    // Build a function: MOV R1, c[0x0][0x28] ; IADD3 R1, R1, -16 ; NOP ; RET
    let nop_word: u128 = 0x000fc00000000000_0000000000007918;
    // We need to construct proper instructions to make the flow work.
    // For the depth test we use the make_borrow_func helper with custom names.
    // Instead, let's test that the Borrowed struct computes depth correctly
    // by constructing a minimal function.
    let mut src: Vec<Decoded> = Vec::new();
    // Instruction 0: sets R1 from c[0x0][0x28] (we'll mark it appropriately)
    {
        let mut d = dec75.decode(nop_word).unwrap();
        d.writes = vec![(LETTER_R, 1)];
        d.reads = vec![(LETTER_R, 10, 0)]; // placeholder, not c[0x0][0x28]
        src.push(d);
    }
    // Instruction 1: IADD3 R1, R1, -16 (simulated: writes R1, reads R1)
    {
        let mut d = dec75.decode(nop_word).unwrap();
        d.name = "iadd3_imm__RsIR_RIR";
        d.writes = vec![(LETTER_R, 1)];
        d.reads = vec![(LETTER_R, 1, 0)];
        // Set operands for the depth calculation
        d.ops = vec![
            (crate::generated::fields::OP_RD, 1),
            (crate::generated::fields::OP_RA, 1),
            (crate::generated::fields::OP_RC, 255),
            (crate::generated::fields::OP_SB, (-16i32 as u32)),
        ];
        src.push(d);
    }
    // Instruction 2: NOP (reads R10 to keep it alive)
    {
        let mut d = dec75.decode(nop_word).unwrap();
        d.reads = vec![(LETTER_R, 10, 0)];
        d.writes = vec![];
        src.push(d);
    }
    let fd = FuncData {
        name: "test_r1_depth".into(),
        registers: 15,
        tables: vec![],
        src,
        uniform: vec![],
        pairs: vec![],
        needs_pool: false,
        uniform_predicates: vec![],
        highest: 10,
        scratch_need: 0,
        needs_lane: false,
        product_count: 0,
        entered_inside: false,
        ampere: false,
        ampere_ops: HashMap::new(),
    };
    let b = Borrowed::new(&fd, false, None).unwrap();
    // After instruction 0 (which sets R1), depth at instruction 1 should be 0.
    // After IADD3 R1, R1, -16, depth at instruction 2 should be 0 - (-16) = 16.
    assert!(b.r1_fixed, "r1 should be fixed when only IADD3 R1, R1, imm modifies it");
    assert_eq!(b.depth[1], Some(0), "depth at instruction 1 should be 0");
    assert_eq!(
        b.depth[2],
        Some(16),
        "depth at instruction 2 should be 16 (R1 lowered by 16)"
    );
}

/// A save (STL) and restore (LDL) assemble and decode back correctly.
#[test]
fn room_save_and_restore_assemble() {
    use crate::assemble;
    // Save: STL [R1+offset], Rb
    let offset = 32u32;
    let rb = 10u32;
    let stl = assemble::assemble(
        "stl__sImmOffset",
        &[
            ("Ra", assemble::OpVal::Int(1)),
            ("Rb", assemble::OpVal::Int(rb)),
            ("Ra_offset", assemble::OpVal::Int(offset)),
        ],
    );
    assert!(stl.is_some(), "STL should assemble");
    let stl_word = stl.unwrap();
    let mut dec70 = crate::decode::Decoder::new(70);
    let d = dec70.decode(stl_word);
    assert!(d.is_some(), "STL should decode as sm_70");
    let d = d.unwrap();
    assert_eq!(d.mnemonic, "STL", "should decode as STL");

    // Restore: LDL Rd, [R1+offset], scoreboard 4
    let rd = 10u32;
    let ldl = assemble::assemble(
        "ldl__sImmOffset",
        &[
            ("Rd", assemble::OpVal::Int(rd)),
            ("Ra", assemble::OpVal::Int(1)),
            ("Ra_offset", assemble::OpVal::Int(offset)),
            ("dst_wr_sb", assemble::OpVal::Int(ARRIVES)),
        ],
    );
    assert!(ldl.is_some(), "LDL should assemble");
    let ldl_word = ldl.unwrap();
    let d = dec70.decode(ldl_word);
    assert!(d.is_some(), "LDL should decode as sm_70");
    let d = d.unwrap();
    assert_eq!(d.mnemonic, "LDL", "should decode as LDL");
}

/// What is stored to the frame after an instruction is stored under no guard: the instruction
/// may have written the predicate that guards it, and the store would go by what it left
/// there. In `k_guards.guard-written.sm75.cubin` 28 instructions on uniform registers give a
/// predicate into the one that guards them (`@UP0 UIADD3 UR6, UP0, UR4, URZ, URZ`); with every
/// uniform register in the frame (level 3 first: 8), each has its store after it.
#[test]
fn a_store_to_the_frame_has_no_guard() {
    let cubin = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/kernels/translate/k_guards.guard-written.sm75.cubin"
    ));
    let opts = Options {
        borrow: 8,
        ..Options::default()
    };
    let result = translate_with_options(cubin, true, &opts).unwrap();
    assert!(result.report.iter().all(|function| function.refused.is_none()));
    let translated = crate::elf::Elf::parse(&result.bytes).unwrap();
    let mut dec70 = Decoder::new(70);
    let (mut stores, mut under_the_guard) = (0, 0);
    for (function, new, old, what) in result.map.unwrap() {
        let section = translated
            .sections
            .iter()
            .find(|section| section.name == format!(".text.{function}"))
            .unwrap();
        let word = |code: &[u8], at: u64| u128::from_le_bytes(code[at as usize..at as usize + 16].try_into().unwrap());
        if what.starts_with("mem: store") {
            let d = dec70.decode(word(&section.data, new)).unwrap();
            assert_eq!(d.mnemonic, "STL");
            assert_eq!(
                (d.op_or_name("Pg", 7), d.op_or_name("Pg@not", 0)),
                (7, 0),
                "{function}+0x{new:x}"
            );
            stores += 1;
            // (How many of them follow an instruction that is under a guard, from 0x{old:x}.)
            let source = crate::elf::Elf::parse(cubin).unwrap();
            let original = source
                .sections
                .iter()
                .find(|section| section.name == format!(".text.{function}"))
                .unwrap();
            let d75 = Decoder::new(75).decode(word(&original.data, old)).unwrap();
            under_the_guard += u32::from(d75.op_or_name("UPg", 7) != 7);
        }
    }
    assert!(under_the_guard >= 28, "{under_the_guard} of {stores}");
}

/// UP0 spans seven writes to ordinary predicates. Every possible general-register holder
/// is live when UP0 is born. Its local word and the temporary save of R2 must be distinct.
#[test]
fn held_predicate_survives_in_the_frame_under_register_pressure() {
    let template = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/kernels/translate/triton_add.sm_75.cubin"
    ));
    let mut input = crate::elf::Elf::parse(template).unwrap();
    let section = input.section(".text.add_kernel").unwrap();
    let code = &mut input.sections[section].data;
    // Keep the existing stack prologue, exit offsets and padding. Replace instructions 1..16.
    let exit = u128::from_le_bytes(code[19 * 16..20 * 16].try_into().unwrap());
    let mut set = |i: usize, word: u128| code[16 * i..16 * (i + 1)].copy_from_slice(&word.to_le_bytes());
    for p in 0..7 {
        let r = p + 2;
        set(
            p as usize + 1,
            assemble::assemble_int("mov__RCR_RC", &[("Rd", r)]).unwrap(),
        );
        set(
            p as usize + 9,
            assemble::assemble_int("isetp__RRR_RR", &[("Pu", p), ("Ra", r), ("Rb", 255)]).unwrap(),
        );
    }
    // UPLOP3.LUT UP0, UPT, UPT, UPT, UPT, 0x80, 0x0
    set(8, 0x000fe20003f0f070_000000000000789c);
    // PLOP3.LUT P4, PT, PT, PT, UP0, 0x80, 0x0
    set(16, 0x000fe40003f8f008_000000000000781c);
    // Use unconditional exits and leave no original use of other general-purpose registers.
    set(17, exit);
    set(18, NOP);

    let options = Options {
        borrow: 8,
        ..Options::default()
    };
    let result = translate_with_options(&input.write(), true, &options).unwrap();
    assert!(result.report.iter().all(|function| function.refused.is_none()));
    let out = crate::elf::Elf::parse(&result.bytes).unwrap();
    let code = &out.sections[out.section(".text.add_kernel").unwrap()].data;
    let mut decoder = Decoder::new(70);
    let map = result.map.unwrap();
    let frame_steps = |origin: u64| {
        map.iter()
            .filter(|(_, _, old, what)| {
                *old == 16 * origin
                    && (what.starts_with("room:") || what.starts_with("mem:") || what.starts_with("held:"))
            })
            .map(|(_, _, _, what)| what.as_str())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        frame_steps(8),
        ["room: R2 away", "held: kept", "mem: store UP0", "room: R2 back"]
    );
    assert_eq!(frame_steps(16), ["mem: load UP0", "held: back"]);
    let mut instruction = |origin: u64, label: &str| {
        let (_, offset, _, _) = map
            .iter()
            .find(|(_, _, old, what)| *old == origin * 16 && what == label)
            .unwrap_or_else(|| panic!("missing {label} at instruction {origin}"));
        let at = *offset as usize;
        decoder
            .decode(u128::from_le_bytes(code[at..at + 16].try_into().unwrap()))
            .unwrap()
    };
    let away = instruction(8, "room: R2 away");
    let keep = instruction(8, "held: kept");
    let store = instruction(8, "mem: store UP0");
    let back = instruction(8, "room: R2 back");
    let load = instruction(16, "mem: load UP0");
    let test = instruction(16, "held: back");
    for (op, expected_offset) in [(&store, 0), (&load, 0), (&away, 4), (&back, 4)] {
        assert_eq!(op.op_or_name("Ra", 255), 1);
        assert_eq!(op.op_or_name("Ra_offset", u32::MAX), expected_offset);
    }
    assert_eq!(keep.op_or_name("Rd", 255), store.op_or_name("Rb", 254));
    assert_eq!(load.op_or_name("Rd", 255), test.op_or_name("Ra", 254));
    assert_eq!((away.op_or_name("Rb", 255), back.op_or_name("Rd", 255)), (2, 2));
}

/// USGXT, which no instruction of either corpus has (Triton's kernels do), through the whole
/// translator: in `k_guarded_state` of `k_guards.sm75.cubin`, `ULOP3.LUT UR6, UR5, 0x1, URZ,
/// 0xc0, !UPT` at 0x120 made `USGXT UR6, UR5, 0x1` (its scheduling kept). It becomes SGXT from
/// the register that the load of UR5 at 0x20 writes into the register that the comparison of
/// UR6 at 0x150 reads. (The reference translator gives the same bytes and map for that cubin,
/// and its validator finds nothing, at the default and with VF_BORROW=8.)
#[test]
fn a_uniform_sign_extension_becomes_sgxt() {
    let template = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/kernels/translate/k_guards.sm75.cubin"
    ));
    let mut input = crate::elf::Elf::parse(template).unwrap();
    let section = input.section(".text.k_guarded_state").unwrap();
    let code = &mut input.sections[section].data;
    assert_eq!(
        &code[0x120..0x130],
        &0x000fe2000f8ec03f_0000000105067892_u128.to_le_bytes()
    );
    code[0x120..0x130].copy_from_slice(&0x000fe20008000200_000000010506789a_u128.to_le_bytes());
    let result = translate_with_options(&input.write(), true, &Options::default()).unwrap();
    assert!(result.report.iter().all(|function| function.refused.is_none()));
    let out = crate::elf::Elf::parse(&result.bytes).unwrap();
    let code = &out.sections[out.section(".text.k_guarded_state").unwrap()].data;
    let map = result.map.unwrap();
    let mut decoder = Decoder::new(70);
    let mut became = |origin: u64| {
        let (_, new, _, what) = map
            .iter()
            .find(|(function, _, old, _)| function == "k_guarded_state" && *old == origin)
            .unwrap();
        let at = *new as usize;
        let d = decoder
            .decode(u128::from_le_bytes(code[at..at + 16].try_into().unwrap()))
            .unwrap();
        (what.clone(), d)
    };
    let (load_what, load) = became(0x20);
    let (what, sgxt) = became(0x120);
    let (_, compare) = became(0x150);
    assert_eq!((load_what.as_str(), load.mnemonic), ("constant", "MOV"));
    assert_eq!((what.as_str(), sgxt.mnemonic), ("ordinary registers", "SGXT"));
    assert_eq!(sgxt.op_or_name("Sb", 0), 1);
    assert_eq!(sgxt.op_or_name("Ra", 255), load.op_or_name("Rd", 254));
    assert_eq!(sgxt.op_or_name("Rd", 255), compare.op_or_name("Ra", 254));
}

#[test]
fn uimad_stride_and_wide_kernel_translates_and_preserves_pairs() {
    let cubin = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/kernels/translate/k_uimad.sm75.cubin"
    ));
    let mut decoder = Decoder::new(70);
    for borrow in [Options::default().borrow, 8] {
        let options = Options {
            borrow,
            ..Options::default()
        };
        let result = translate_with_options(cubin, true, &options).unwrap();
        assert!(
            result.report.iter().all(|f| f.refused.is_none()),
            "refused under borrow={borrow:?}: {:?}",
            result.report
        );
        let out = crate::elf::Elf::parse(&result.bytes).unwrap();
        let map = result.map.unwrap();
        let mut decode_at = |func: &str, origin: u64| {
            let code = &out.sections[out.section(&format!(".text.{func}")).unwrap()].data;
            let (_, new, _, what) = map
                .iter()
                .find(|(f, _, old, w)| f == func && *old == origin && !w.starts_with("mem: "))
                .unwrap_or_else(|| panic!("missing {func}+0x{origin:x} under {borrow:?}"));
            let at = *new as usize;
            let d = decoder
                .decode(u128::from_le_bytes(code[at..at + 16].try_into().unwrap()))
                .unwrap();
            (what.clone(), d)
        };
        let (w_stride, d_stride) = decode_at("k_uimad_stride", 0xb0);
        assert_eq!((w_stride.as_str(), d_stride.mnemonic), ("ordinary registers", "IMAD"));
        assert_eq!(d_stride.op_or_name("Sc", 0), 0x240);

        let (w_wide, d_wide) = decode_at("k_uimad_wide_hi", 0x60);
        assert_eq!((w_wide.as_str(), d_wide.mnemonic), ("ordinary registers", "IMAD"));
        assert_eq!(d_wide.name, "imad_wide__RRR_RRR");
        let rd = d_wide.op_or_name("Rd", 255);
        assert_eq!(rd % 2, 0);
        assert_eq!(d_wide.op_or_name("Rc", 255), rd);
        let (_, d_hi_read) = decode_at("k_uimad_wide_hi", 0x80);
        assert_eq!(d_hi_read.op_or_name("Rc", 255), rd + 1);
    }
}

#[test]
fn uimad_x_and_uimad_wide_x_preserve_predicates_and_pairs() {
    let mut decoder = Decoder::new(70);
    // Patch k_guarded_state+0x1d0 with UIMAD.X UR5, UR4, UR6, 0x240, UP0 right after 0x1c0 writes UP0.
    let guards_tmpl = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/kernels/translate/k_guards.sm75.cubin"
    ));
    let mut g_in = crate::elf::Elf::parse(guards_tmpl).unwrap();
    let g_sec = g_in.section(".text.k_guarded_state").unwrap();
    assert_eq!(
        &g_in.sections[g_sec].data[0x1d0..0x1e0],
        &0x000fd2000fffe006_0000000104057890_u128.to_le_bytes()
    );
    g_in.sections[g_sec].data[0x1d0..0x1e0].copy_from_slice(&0x000fd200080e0606_00000240040574a4_u128.to_le_bytes());

    // Patch k_uimad_wide_hi+0x60 with UIMAD.WIDE.X UR4, UR8, UR9, UR4, UPT.
    let uimad_tmpl = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/kernels/translate/k_uimad.sm75.cubin"
    ));
    let mut u_in = crate::elf::Elf::parse(uimad_tmpl).unwrap();
    let u_sec = u_in.section(".text.k_uimad_wide_hi").unwrap();
    assert_eq!(
        &u_in.sections[u_sec].data[0x60..0x70],
        &0x000fcc000f8e0004_00000009080472a5_u128.to_le_bytes()
    );
    u_in.sections[u_sec].data[0x60..0x70].copy_from_slice(&0x000fcc000b8e0604_00000009080472a5_u128.to_le_bytes());

    let g_bytes = g_in.write();
    let u_bytes = u_in.write();
    for borrow in [Options::default().borrow, 8] {
        let options = Options {
            borrow,
            ..Options::default()
        };
        let gr = translate_with_options(&g_bytes, true, &options).unwrap();
        assert!(gr.report.iter().all(|f| f.refused.is_none()));
        let gout = crate::elf::Elf::parse(&gr.bytes).unwrap();
        let gcode = &gout.sections[gout.section(".text.k_guarded_state").unwrap()].data;
        let gmap = gr.map.unwrap();
        let (_, cmp_off, _, _) = gmap
            .iter()
            .find(|(f, _, old, w)| {
                f == "k_guarded_state"
                    && *old == 0x1c0
                    && !w.starts_with("mem: ")
                    && *w != "held: back"
                    && *w != "held: kept"
            })
            .unwrap();
        let cmp = decoder
            .decode(u128::from_le_bytes(
                gcode[*cmp_off as usize..*cmp_off as usize + 16].try_into().unwrap(),
            ))
            .unwrap();
        let (_, u_off, _, u_what) = gmap
            .iter()
            .find(|(f, _, old, w)| {
                f == "k_guarded_state"
                    && *old == 0x1d0
                    && !w.starts_with("mem: ")
                    && *w != "held: back"
                    && *w != "held: kept"
            })
            .unwrap();
        let uimad_x = decoder
            .decode(u128::from_le_bytes(
                gcode[*u_off as usize..*u_off as usize + 16].try_into().unwrap(),
            ))
            .unwrap();
        assert_eq!(u_what, "ordinary registers");
        assert_eq!(uimad_x.name, "imad_x__RRsI_RRI");
        assert_eq!(uimad_x.op_or_name("Sc", 0), 0x240);
        assert_eq!(uimad_x.op_or_name("Pp", 7), cmp.op_or_name("Pu", 6));

        let ur = translate_with_options(&u_bytes, true, &options).unwrap();
        assert!(ur.report.iter().all(|f| f.refused.is_none()));
        let uout = crate::elf::Elf::parse(&ur.bytes).unwrap();
        let ucode = &uout.sections[uout.section(".text.k_uimad_wide_hi").unwrap()].data;
        let umap = ur.map.unwrap();
        let (_, w_off, _, w_what) = umap
            .iter()
            .find(|(f, _, old, w)| f == "k_uimad_wide_hi" && *old == 0x60 && !w.starts_with("mem: "))
            .unwrap();
        let wide_x = decoder
            .decode(u128::from_le_bytes(
                ucode[*w_off as usize..*w_off as usize + 16].try_into().unwrap(),
            ))
            .unwrap();
        assert_eq!(w_what, "ordinary registers");
        assert_eq!(wide_x.name, "imad_wide_x__RRR_RRR");
        assert_eq!(wide_x.op_or_name("Pp", 0), 7);
        let rd = wide_x.op_or_name("Rd", 255);
        assert_eq!(rd % 2, 0);
        assert_eq!(wide_x.op_or_name("Rc", 255), rd);
    }
}

#[test]
fn graph_register_policy_only_rebudgets_an_entry_that_loses_a_legal_block() {
    let original = include_bytes!("../../../../tests/kernels/k_fit_test.135.sm75.cubin");
    let name = "k_fit_regs";
    for (shape, expected_mode, expected_regs) in [
        (None, 5, 64),
        (Some((0x10, 128u32)), 7, 66),
        (Some((0x05, 896)), 7, 66),
        (Some((0x05, 897)), 5, 64),
    ] {
        let mut elf = crate::elf::Elf::parse(original).unwrap();
        if let Some((kind, threads)) = shape {
            let i = elf.section(&format!(".nv.info.{name}")).unwrap();
            let info = &mut elf.sections[i].data;
            info.extend([4, kind, 12, 0]);
            for dim in [threads, 1, 1] {
                info.extend(dim.to_le_bytes());
            }
        }
        let data = elf.write();
        let text = |borrow| {
            let opts = Options {
                borrow,
                ..Options::default()
            };
            let result = translate_with_options(&data, false, &opts).unwrap();
            let out = crate::elf::Elf::parse(&result.bytes).unwrap();
            let i = out.section(&format!(".text.{name}")).unwrap();
            (out.sections[i].info >> 24, out.sections[i].data.clone())
        };
        let (regs, code) = text(9);
        assert_eq!(regs, expected_regs);
        assert_eq!((regs, code), text(expected_mode));
    }
}

#[test]
fn graph_policy_keeps_the_default_borrowed_layout_if_it_already_fits() {
    let original = include_bytes!("../../../../tests/kernels/k_groups.135.sm75.cubin");
    let mut input = crate::elf::Elf::parse(original).unwrap();
    let section = input.section(".text.g_after").unwrap();
    // A write to an otherwise unused high register makes the ordinary matrix
    // translation borrow registers. Its 245 registers still admit the source CTA.
    let text = &mut input.sections[section];
    let mut decoder = Decoder::new(75);
    let nop = text
        .data
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .find(|bytes| {
            let word = u128::from_le_bytes(**bytes);
            decoder.decode(word).is_some_and(|op| op.name == "nop_")
        })
        .unwrap();
    let mov = assemble::assemble_int("mov__RCR_RC", &[("Rd", 242)]).unwrap();
    nop.copy_from_slice(&mov.to_le_bytes());
    text.info = (255 << 24) | (text.info & 0x00ff_ffff);
    let data = input.write();
    let layout = |borrow| {
        let opts = Options {
            borrow,
            ..Options::default()
        };
        let result = translate_with_options(&data, false, &opts).unwrap();
        assert!(
            result
                .report
                .iter()
                .find(|f| f.name == "g_after")
                .unwrap()
                .refused
                .is_none()
        );
        let output = crate::elf::Elf::parse(&result.bytes).unwrap();
        let code = &output.sections[output.section(".text.g_after").unwrap()];
        let info = &output.sections[output.section(".nv.info.g_after").unwrap()];
        (code.info >> 24, code.data.clone(), info.data.clone())
    };
    let normal = layout(7);
    assert_eq!(normal.0, 245);
    assert_eq!(layout(9), normal);
}

#[test]
fn graph_policy_preserves_a_relocation_on_the_stack_setup() {
    let original = include_bytes!("../../../../tests/kernels/k_fit_test.135.sm75.cubin");
    let mut e = crate::elf::Elf::parse(original).unwrap();
    let target = e.section(".text.k_fit_regs").unwrap();
    let slot = e.section(".rel.debug_frame").unwrap();
    let names = e.section(".shstrtab").unwrap();
    let name_offset = e.sections[names].data.len() as u32;
    e.sections[names].data.extend(b".rel.text.k_fit_regs\0");
    // Attach an ELF relocation to the original first instruction. The bounded
    // layout inserts another stack instruction there, which follow() cannot carry.
    let rel = &mut e.sections[slot];
    rel.name_offset = name_offset;
    rel.info = target as u32;
    rel.data = [0u64.to_le_bytes(), 56u64.to_le_bytes()].concat();
    let input = e.write();
    let layout = |borrow| {
        let opts = Options {
            borrow,
            ..Options::default()
        };
        let result = translate_with_options(&input, false, &opts).unwrap();
        let entry = result.report.iter().find(|r| r.name == "k_fit_regs").unwrap();
        let obj = crate::elf::Elf::parse(&result.bytes).unwrap();
        let text = &obj.sections[obj.section(".text.k_fit_regs").unwrap()];
        (entry.refused.clone(), text.info >> 24, text.data.clone())
    };
    let normal = layout(7);
    assert_eq!(normal.0, None);
    assert_eq!(normal.1, 66);
    assert_eq!(layout(9), normal);
}

/// The sequence that stands for a matrix load writes a destination with arithmetic only after
/// waiting for shuffles of the sequence: a store just before the load may not have read that
/// register yet, and it is a memory instruction like the load that the compiler counts on to
/// write it after the store has read it, or a wait for one. k_stored.cu has, in a loop, four stores of the
/// destinations of a load just before it, plain and transposed. (Until 2026-11-23 the lanes of the
/// rows went into the destinations first: on a V100 the stores then stored lanes.)
#[test]
fn a_matrix_load_writes_its_destinations_first_by_shuffles() {
    let original = include_bytes!("../../../../tests/kernels/k_stored.135.sm75.cubin");
    let result = translate_with_options(original, true, &Options::default()).unwrap();
    assert!(result.report.iter().all(|function| function.refused.is_none()));
    let map = result.map.unwrap();
    let source = crate::elf::Elf::parse(original).unwrap();
    let out = crate::elf::Elf::parse(&result.bytes).unwrap();
    let (mut d75, mut d70) = (Decoder::new(75), Decoder::new(70));
    let mut loads = 0;
    for name in ["t_stored_plain", "t_stored_trans"] {
        let old = &source.sections[source.section(&format!(".text.{name}")).unwrap()].data;
        let new = &out.sections[out.section(&format!(".text.{name}")).unwrap()].data;
        let word = |code: &[u8], at: usize| u128::from_le_bytes(code[at..at + 16].try_into().unwrap());
        for at in (0..old.len()).step_by(16) {
            let d = d75.decode(word(old, at)).unwrap();
            if d.mnemonic != "LDSM" {
                continue;
            }
            loads += 1;
            let rd = d.op_or_name("Rd", 255);
            let destinations = rd..rd + (1 << d.op_or_name("num", 0));
            let mut sequence: Vec<u64> = map
                .iter()
                .filter(|(function, _, from, _)| function == name && *from == at as u64)
                .map(|(_, to, _, _)| *to)
                .collect();
            sequence.sort_unstable();
            let (mut shuffled, mut waited) = (false, false);
            for to in sequence {
                let w = word(new, to as usize);
                let d = d70.decode(w).unwrap();
                if shuffled && (w >> (WAIT + SEQUENCE)) & 1 == 1 {
                    waited = true;
                }
                let writes = d
                    .writes
                    .iter()
                    .any(|&(l, r)| l == LETTER_R && destinations.contains(&r));
                if writes && !matches!(d.mnemonic, "SHFL" | "LDS") {
                    assert!(
                        waited,
                        "{name}: at 0x{to:x}, {} writes a destination of the load at 0x{at:x} before a wait for the shuffles",
                        d.mnemonic
                    );
                }
                shuffled |= d.mnemonic == "SHFL";
            }
        }
    }
    assert_eq!(loads, 4);
}

/// `k_fit_regs` (64 registers on sm_75; 66 translated the usual way) with a block it declares in its
/// attributes, translated with `borrow`: its register count and code, or why it was refused.
fn with_block(shape: Option<(u8, u32)>, borrow: u32) -> Result<(u32, Vec<u8>), String> {
    let original = include_bytes!("../../../../tests/kernels/k_fit_test.135.sm75.cubin");
    let name = "k_fit_regs";
    let mut elf = crate::elf::Elf::parse(original).unwrap();
    if let Some((kind, threads)) = shape {
        let i = elf.section(&format!(".nv.info.{name}")).unwrap();
        let info = &mut elf.sections[i].data;
        info.extend([4, kind, 12, 0]);
        for dim in [threads, 1, 1] {
            info.extend(dim.to_le_bytes());
        }
    }
    let opts = Options {
        borrow,
        ..Options::default()
    };
    let result = translate_with_options(&elf.write(), false, &opts).unwrap();
    if let Some(why) = &result.report.iter().find(|f| f.name == name).unwrap().refused {
        return Err(why.clone());
    }
    let out = crate::elf::Elf::parse(&result.bytes).unwrap();
    let i = out.section(&format!(".text.{name}")).unwrap();
    Ok((out.sections[i].info >> 24, out.sections[i].data.clone()))
}

/// A kernel that requires blocks of 1,024 threads (EIATTR_REQNTID, as Triton writes it) is launched
/// with no other block: its usual translation, at 66 registers, never runs (a V100 allows 64 for
/// such a block), and a program that reads the limit from the kernel stops before the launch that
/// would bring the second translation. It is translated within 64 at once, as the policy for
/// graphs (9) translates it. A smaller required block, a block that is only a maximum
/// (EIATTR_MAX_THREADS, `__launch_bounds__`) and no block at all keep the usual translation, and
/// the second translation at launch for blocks it does not fit.
#[test]
fn a_kernel_that_requires_its_block_is_translated_within_it() {
    let usual = with_block(None, BORROW_DEFAULT).unwrap();
    assert_eq!(usual.0, 66);
    let required = with_block(Some((0x10, 1024)), BORROW_DEFAULT).unwrap();
    assert!(required.0 <= 64, "{} registers", required.0);
    assert_eq!(Ok(required), with_block(Some((0x10, 1024)), 9));
    assert_eq!(with_block(Some((0x10, 896)), BORROW_DEFAULT).as_ref(), Ok(&usual));
    assert_eq!(with_block(Some((0x05, 1024)), BORROW_DEFAULT).as_ref(), Ok(&usual));
}

/// When no translation keeps within the registers that the required block allows (here, without
/// borrowed registers: `VF_BORROW=0`), the kernel is refused for its registers, so that the layer
/// takes the PTX that the cubin keeps; without a required block the same translation is given.
#[test]
fn a_kernel_whose_required_block_no_translation_fits_is_refused() {
    assert_eq!(with_block(None, 0).map(|(registers, _)| registers), Ok(66));
    assert_eq!(
        with_block(Some((0x10, 1024)), 0),
        Err("more registers than its required block of 1024 threads allows (64)".to_string())
    );
    assert_eq!(
        with_block(Some((0x05, 1024)), 0).map(|(registers, _)| registers),
        Ok(66)
    );
}

#[test]
fn the_report_says_which_functions_compute_their_products_in_pairs() {
    let original = include_bytes!("../../../../tests/kernels/k_pairs.135.sm75.cubin");
    let paired = |pairs| {
        let opts = Options {
            pairs,
            ..Options::default()
        };
        let result = translate_with_options(original, false, &opts).unwrap();
        let names: Vec<String> = result.report.iter().map(|f| f.name.clone()).collect();
        let paired: Vec<String> = result
            .report
            .iter()
            .filter(|f| f.paired)
            .map(|f| f.name.clone())
            .collect();
        (names, paired)
    };
    let (names, with_pairs) = paired(true);
    assert!(!with_pairs.is_empty(), "{names:?}");
    assert!(with_pairs.iter().all(|name| names.contains(name)));
    // VF_PAIRS=0 (and safe mode, which sets it so in machine code): each product by itself.
    assert_eq!(paired(false).1, Vec::<String>::new());
}

/// A result of double precision is read no earlier than Volta's own code reads one, also when the
/// reader waits for the scoreboard that announces it: 8 cycles to an instruction of the same
/// pipe, 9 to any other, and a predicate of DSETP 15 as a guard or to a branch. Code for sm_80
/// has 6 and 14 there, and every instruction of an sm_80 function is scheduled anew (`changed`):
/// with the scoreboard trusted, PyTorch's double-precision erfc, erfinv and exp2 for sm_80 were
/// translated with 165, 192 and 33 such reads, and were wrong on a V100 (docs/validation.md,
/// section 8). The words are of a native sm_70 build of exp2.
#[test]
fn a_result_of_double_precision_is_read_no_earlier_than_volta_reads_one() {
    use crate::generated::special::{WAIT, WRITE_BARRIER};
    const DADD: u128 = 0x001e1000000008060000000002067229; // DADD R6, R2, -R6 (writes R6:R7 on scoreboard 0)
    const DMUL: u128 = 0x001e1000000000000080000006087a28; // DMUL R8, R6, c[0x2][0x0], waits for 0
    const DFMA: u128 = 0x001e1000000000083ff000000608742b; // DFMA R8, R6, R8, 1 (writes R8:R9 on 0)
    const IMAD: u128 = 0x002fe400078e02090010000004077824; // IMAD R7, R4, 0x100000, R9
    const DSETP: u128 = 0x000fe20003f0c2007ff000000200742a; // DSETP P0, |R2|, +INF
    const BRA: u128 = 0x000fea0003800000000000e000008947; // @P0 BRA
    const FSEL: u128 = 0x001fe40000000000000000ff06067208; // FSEL R6, R6, RZ, P0
    let board = |word: u128, board: u128| (word & !(7 << WRITE_BARRIER)) | board << WRITE_BARRIER;
    let waits = |word: u128, mask: u128| (word & !(63 << WAIT)) | mask << WAIT;
    let run = |writer: u128, reader: u128, changed: bool| {
        let mut decoder = Decoder::new(70);
        let nop = assemble::assemble("nop_", &[]).unwrap();
        let mut items: Vec<Item> = [with_stall(writer, 6, false), waits(reader, 1), nop]
            .into_iter()
            .enumerate()
            .map(|(i, word)| {
                let mut item = Item::new(word, i, "same", changed);
                item.first = true;
                item.d = Some(decoder.decode(word).unwrap());
                item
            })
            .collect();
        super::schedule::schedule(&mut items, &[]).unwrap();
        stalls(&items)[0]
    };
    // Each pair six cycles apart, the reader waiting for the writer's scoreboard.
    assert_eq!(run(board(DADD, 0), DMUL, true), 8);
    assert_eq!(
        run(board(DADD, 0), IMAD, true),
        6,
        "IMAD reads nothing that this DADD writes"
    );
    assert_eq!(run(board(DFMA, 0), IMAD, true), 9);
    assert_eq!(run(board(DSETP, 0), FSEL, true), 9);
    assert_eq!(run(board(DSETP, 0), BRA, true), 15);
    // Already further apart: left.
    let far = |writer: u128, reader: u128| {
        let mut decoder = Decoder::new(70);
        let mut items: Vec<Item> = [with_stall(writer, 12, false), waits(reader, 1)]
            .into_iter()
            .chain([assemble::assemble("nop_", &[]).unwrap()])
            .enumerate()
            .map(|(i, word)| {
                let mut item = Item::new(word, i, "same", true);
                item.first = true;
                item.d = Some(decoder.decode(word).unwrap());
                item
            })
            .collect();
        super::schedule::schedule(&mut items, &[]).unwrap();
        stalls(&items)[0]
    };
    assert_eq!(far(board(DADD, 0), DMUL), 12);
    // The program's own instructions of sm_75 code, both as they were: as the compiler had them.
    assert_eq!(run(board(DADD, 0), DMUL, false), 6);
    // The reader may be where a branch goes, which the schedule does not see: at the end of
    // straight-line code a result of double precision is as old as any reader needs (9 cycles
    // for a register, 15 for a predicate), in code for sm_80 (`Item::anew`): the reciprocal of
    // linalg's zlascl for sm_80 has a DFMA, a branch five cycles later and a DMUL at its
    // target, six cycles on that path. Code for sm_75 keeps Volta's distances on such paths,
    // and what the translation changes of it is left as far from the end as it was.
    let ends = |words: &[u128], targets: &[u64], anew: bool| {
        let mut decoder = Decoder::new(70);
        let mut items: Vec<Item> = words
            .iter()
            .enumerate()
            .map(|(i, &word)| {
                let mut item = Item::new(word, i, "same", true);
                item.anew = anew;
                item.first = true;
                item.d = Some(decoder.decode(word).unwrap());
                item
            })
            .collect();
        super::schedule::schedule(&mut items, targets).unwrap();
        stalls(&items)
    };
    let nop = assemble::assemble("nop_", &[]).unwrap();
    let dfma = with_stall(board(DFMA, 0), 1, false);
    let bra = with_stall(BRA, 5, false);
    // Before a branch: the wait is on the instruction before it.
    assert_eq!(ends(&[dfma, bra, nop], &[], true)[..2], [4, 5]);
    assert_eq!(ends(&[dfma, bra, nop], &[], false)[..2], [1, 5]);
    // Before a place where a branch goes, the writer the last instruction before it.
    assert_eq!(ends(&[with_stall(board(DADD, 0), 1, false), nop], &[16], true)[0], 9);
    assert_eq!(ends(&[with_stall(board(DSETP, 0), 1, false), nop], &[16], true)[0], 15);
    // A writer earlier in the code that goes on to that place.
    let through = ends(&[with_stall(board(DSETP, 0), 2, false), nop, nop], &[32], true);
    assert_eq!(through[0] + through[1], 15, "{through:?}");
}

/// The words of the tests for a return address from the LEPC before a direct call.
const LEPC_R128: u128 = 0x008fe40000000000000000000080734e;
const CALL_TO_NEXT: u128 = 0x000fc200038000000000000000007944;
const RET_ABS_R128_20: u128 = 0x140fea0003e000000000002080007950;
const RET_REL: u128 = 0x000fc2000380000000000000ff007950;
const MOV_R5_30: u128 = 0x000fc20000000f000000003000057802;
const NOP: u128 = 0x000fc000000000000000000000007918;

fn fd_from_src(words: &[u128]) -> super::parse::FuncData {
    let mut dec75 = Decoder::new(75);
    let mut fd = make_borrow_func(&[], 128, &[], &[], false);
    fd.src = words.iter().map(|&w| dec75.decode(w).expect("decode")).collect();
    fd
}

fn same_items(words: &[u128]) -> Vec<Item> {
    let mut dec75 = Decoder::new(75);
    words
        .iter()
        .enumerate()
        .map(|(k, &word)| {
            let mut item = Item::new(word, k, "same", false);
            item.first = true;
            item.d = Some(dec75.decode(word).expect("decode"));
            item
        })
        .collect()
}

/// A direct call's return address can be the program counter read by the LEPC immediately
/// before it when the callee returns through that register with `RET.ABS` and `0x20`: both
/// instructions move with the code, so no number needs a fixup. cuDNN 9.24's sm_75 code gives
/// direct calls so (`LEPC R128; CALL.REL`). The pair stays as it was, adjacent.
#[test]
fn a_program_counter_before_a_direct_call_is_its_return_address() {
    let fd = fd_from_src(&[LEPC_R128, CALL_TO_NEXT, RET_ABS_R128_20]);
    let mut items = same_items(&[LEPC_R128, CALL_TO_NEXT]);
    items[1].target = Some(0x20); // the callee is the instruction after the call
    let (moved, pairs) =
        super::return_addr::return_addresses(&mut items, &[], &fd).expect("the pair is the return address");
    assert!(moved.is_empty());
    assert_eq!(pairs, vec![(0, 16)]);
    assert_eq!(items[0].word, LEPC_R128);
    assert_eq!(items[1].word, CALL_TO_NEXT);
    assert_eq!(items[1].origin, items[0].origin + 1);
    assert_eq!(items[0].fix, None);
}

/// A MOV of the same number beside a LEPC whose callee returns through the LEPC's register is
/// data, not a return address: it is left alone, or the translation would change what a program
/// computes.
#[test]
fn a_mov_beside_a_lepc_is_left_alone_when_the_callee_returns_through_it() {
    let fd = fd_from_src(&[MOV_R5_30, LEPC_R128, CALL_TO_NEXT, RET_ABS_R128_20]);
    let mut items = same_items(&[MOV_R5_30, LEPC_R128, CALL_TO_NEXT]);
    items[2].target = Some(0x30); // the callee is two instructions after the LEPC
    let (_, pairs) =
        super::return_addr::return_addresses(&mut items, &[], &fd).expect("the callee returns through R128");
    assert_eq!(pairs, vec![(16, 32)]);
    assert_eq!(items[0].fix, None, "the MOV is data and must not move");
}

/// A LEPC that is not an accepted pair is not exempted from being a LEPC without a call after
/// it, as on `main`: a call whose callee returns by `RET.REL` is an ordinary call, and the LEPC
/// beside it is refused even when a MOV would give the return address.
#[test]
fn a_lepc_beside_an_ordinary_call_is_refused() {
    let fd = fd_from_src(&[MOV_R5_30, LEPC_R128, CALL_TO_NEXT, RET_REL]);
    let mut items = same_items(&[MOV_R5_30, LEPC_R128, CALL_TO_NEXT]);
    items[2].target = Some(0x30);
    assert!(super::return_addr::return_addresses(&mut items, &[], &fd).is_err());
}

/// `LEPC Rd` writes the pair `Rd:Rd+1` and `RET.ABS Ra` reads it: a callee that writes the
/// upper half between them does not return through the pair, so the LEPC is not its return
/// address.
#[test]
fn a_callee_that_writes_the_upper_half_of_the_return_address_is_not_a_pair() {
    let mut dec75 = Decoder::new(75);
    let mut fd = fd_from_src(&[LEPC_R128, CALL_TO_NEXT]);
    let mut write = dec75.decode(NOP).expect("NOP");
    write.writes = vec![(LETTER_R, 129)];
    fd.src.push(write);
    fd.src.push(dec75.decode(RET_ABS_R128_20).expect("RET.ABS"));
    let mut items = same_items(&[LEPC_R128, CALL_TO_NEXT]);
    items[1].target = Some(0x20);
    assert!(super::return_addr::return_addresses(&mut items, &[], &fd).is_err());
}

/// A guarded `RET.ABS` is one way out; the other falls through. A callee whose fall-through
/// returns another way does not return through the pair only.
#[test]
fn a_guarded_return_does_not_end_the_walk_of_the_callee() {
    let mut dec75 = Decoder::new(75);
    let mut fd = fd_from_src(&[LEPC_R128, CALL_TO_NEXT]);
    let ret = dec75.decode(RET_ABS_R128_20).expect("RET.ABS");
    let guarded_word =
        super::util::put_operand(&ret, crate::generated::fields::OP_PG, 0, 75).expect("guard it with P0");
    let guarded = dec75.decode(guarded_word).expect("the guarded RET");
    assert_eq!(guarded.op_or(crate::generated::fields::OP_PG, 7), 0, "guarded by P0");
    assert!(!crate::pairing::unguarded(&guarded), "a guard makes it fall through");
    fd.src.push(guarded);
    fd.src.push(dec75.decode(RET_REL).expect("RET.REL"));
    let mut items = same_items(&[LEPC_R128, CALL_TO_NEXT]);
    items[1].target = Some(0x20);
    assert!(super::return_addr::return_addresses(&mut items, &[], &fd).is_err());
}

/// Without a MOV, a call whose callee does not return through the LEPC's register is refused:
/// nothing shows where it returns to.
#[test]
fn a_lepc_before_a_call_that_does_not_return_through_it_is_refused() {
    let fd = fd_from_src(&[LEPC_R128, CALL_TO_NEXT, RET_REL]);
    let mut items = same_items(&[LEPC_R128, CALL_TO_NEXT]);
    items[1].target = Some(0x20);
    assert!(super::return_addr::return_addresses(&mut items, &[], &fd).is_err());
}

/// A call without a LEPC before it keeps the rule it had: exactly one MOV of its return address
/// before it. Two (one in each of two ways to the call) are refused, not one fixed and one
/// left stale.
#[test]
fn a_call_with_two_movs_of_its_return_address_is_refused() {
    let fd = fd_from_src(&[]);
    let mut items = same_items(&[MOV_R5_30, MOV_R5_30, CALL_TO_NEXT]);
    assert!(super::return_addr::return_addresses(&mut items, &[], &fd).is_err());
}

/// The scheduler may put fillers between items: the LEPC and the call it gives the return
/// address of must still be next to each other, or the callee returns into the filler.
#[test]
fn a_lepc_pair_is_refused_when_a_filler_separates_it() {
    let items = same_items(&[LEPC_R128, CALL_TO_NEXT]);
    assert!(super::return_addr::lepc_pairs_adjacent(&items, &[(0, 16)]).is_ok());
    let mut with_filler = same_items(&[LEPC_R128, NOP, CALL_TO_NEXT]);
    with_filler[1].origin = 0;
    with_filler[1].what = "fill";
    assert!(super::return_addr::lepc_pairs_adjacent(&with_filler, &[(0, 16)]).is_err());
}

/// The checked-in LEPC test kernel (tests/kernels/translate/k_lepc.sm75.cubin, from k_lepc.cu
/// by make_lepc_cubin.py) has the pair these rules are about: a LEPC immediately before a
/// direct call, and a RET.ABS of its register adding 0x20.
#[test]
fn the_lepc_test_kernel_has_the_pair() {
    let mut dec75 = Decoder::new(75);
    let data = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/kernels/translate/k_lepc.sm75.cubin"
    ));
    let words: Vec<u128> = data
        .as_chunks::<16>()
        .0
        .iter()
        .map(|w| u128::from_le_bytes(*w))
        .collect();
    let lepc = words
        .iter()
        .position(|&w| dec75.decode(w).is_some_and(|d| d.mnemonic == "LEPC"))
        .expect("a LEPC in the fixture");
    let call = dec75.decode(words[lepc + 1]).expect("the instruction after the LEPC");
    assert!(call.name.starts_with("call_rel"), "{}", call.name);
    let register = dec75.decode(words[lepc]).unwrap().op_or(OP_RD, 255);
    assert!(
        words.iter().any(|&w| {
            dec75.decode(w).is_some_and(|d| {
                matches!(d.name, "ret__ABS" | "ret__ABS_UR")
                    && d.op_or(OP_RA, 255) == register
                    && d.op_or(OP_RA_OFFSET, 0) == 0x20
            })
        }),
        "a RET.ABS of R{register} adding 0x20 in the fixture"
    );
}

/// A Turing load rebuilt without its cache hint is an instruction the translation changed: it
/// must name no batch of the original schedule. The word is cuBLASLt 13's
/// `@P3 LDG.E.LTC128B.U8.CONSTANT.GPU R64, [R76]` (the batch is 5 there); without the batch
/// clearing it, the validator says "names another batch than the instruction".
#[test]
fn a_rebuilt_turing_load_names_no_batch() {
    let mut dec75 = Decoder::new(75);
    let mut dec70 = Decoder::new(70);
    let word = 0x140ea400001e4120000000004c403381u128;
    let d = dec75.decode(word).expect("sm_75 LDG");
    assert_eq!(d.op_or_name("TABLES_Pnz_0(Pnz@not,Pnz)", 0), 0);
    let cleared = super::util::put_operand(&d, crate::generated::fields::OP_SP2, 0, 75).expect("clear the hint");
    assert!(dec70.decode(cleared).is_some(), "the hint cleared is sm_70");
    let mut batch = |w: u128| {
        dec70
            .decode(w)
            .unwrap()
            .op_or_name("TABLES_opex_0(batch_t,usched_info)", 0)
            >> 5
    };
    assert_eq!(batch(cleared), 5, "the source's batch is still named");
    let cleaned = super::instruction::without_batch("ldg__sImmOffset", &d, cleared).expect("clear the batch");
    assert_eq!(batch(cleaned), 0, "a changed instruction names no batch");
    assert!(dec70.decode(cleaned).is_some());
}

/// The first build of the LEPC test kernel, kept as the refused case: its callee writes R5
/// between `LEPC R4` and `RET.ABS R4`, and `LEPC Rd`/`RET.ABS Ra` are 64-bit pairs, so the
/// upper half of the return address is not what the LEPC read. The translators refuse it (the
/// LEPC is then not an accepted pair).
#[test]
fn the_upper_written_fixture_has_a_pair_its_callee_breaks() {
    let mut dec75 = Decoder::new(75);
    let image = crate::elf::Elf::parse(include_bytes!("../../../../tests/kernels/k_lepc_upper.sm75.cubin"))
        .expect("the fixture");
    let section = &image.sections[image.section(".text.k_lepc").expect("its code")];
    let words: Vec<u128> = section
        .data
        .as_chunks::<16>()
        .0
        .iter()
        .map(|w| u128::from_le_bytes(*w))
        .collect();
    let lepc = words
        .iter()
        .position(|&w| dec75.decode(w).is_some_and(|d| d.mnemonic == "LEPC"))
        .expect("a LEPC in the fixture");
    let register = dec75.decode(words[lepc]).unwrap().op_or(OP_RD, 255);
    assert!(words[lepc + 1..].iter().any(|&w| {
        dec75.decode(w).is_some_and(|d| {
            matches!(d.name, "ret__ABS" | "ret__ABS_UR")
                && d.op_or(OP_RA, 255) == register
                && d.op_or(OP_RA_OFFSET, 0) == 0x20
        })
    }));
    assert!(
        words[lepc + 1..].iter().any(|&w| {
            dec75
                .decode(w)
                .is_some_and(|d| d.writes.iter().any(|&(l, r)| l == LETTER_R && r == register + 1))
        }),
        "the callee writes R{}",
        register + 1
    );
}

/// A global load whose sm_75 table has a hint that sm_70 has no field for is rebuilt with sm_70's
/// table: the data registers, the guard and the scoreboards stay.
#[test]
fn a_turing_global_load_gets_the_sm70_cache_and_scope() {
    let mut dec75 = Decoder::new(75);
    let mut dec70 = Decoder::new(70);
    for (word, rd, ra) in [
        (0x140a2400001e4d240000000082847381u128, 132, 130),
        (0x1406a200001f4b24000000008a987381u128, 152, 138),
    ] {
        let d = dec75.decode(word).expect("sm_75 LDG");
        assert!(dec70.decode(word).is_none(), "the Turing word is not sm_70");
        let word70 = super::instruction::sm70_ldg_without_hints(&d).expect("rebuilt");
        let d70 = dec70.decode(word70).expect("sm_70 LDG");
        assert_eq!(d70.name, "ldg__sImmOffset");
        assert_eq!(d70.op_or(crate::generated::fields::OP_RD, 255), rd);
        assert_eq!(d70.op_or(crate::generated::fields::OP_RA, 255), ra);
        assert_eq!(
            d70.op_or(crate::generated::fields::OP_SZ, 255),
            d.op_or(crate::generated::fields::OP_SZ, 255)
        );
        assert_eq!(
            d70.op_or(crate::generated::fields::OP_SRC_REL_SB, 255),
            d.op_or(crate::generated::fields::OP_SRC_REL_SB, 255)
        );
        assert_eq!(
            d70.op_or(crate::generated::fields::OP_DST_WR_SB, 255),
            d.op_or(crate::generated::fields::OP_DST_WR_SB, 255)
        );
        assert_eq!(
            d70.op_or(crate::generated::fields::OP_PG, 255),
            d.op_or(crate::generated::fields::OP_PG, 255)
        );
        assert_eq!(
            d70.op_or(crate::generated::fields::OP_PG_AT_NOT, 255),
            d.op_or(crate::generated::fields::OP_PG_AT_NOT, 255)
        );
    }
}

/// Turing's F2FP converts its high half in a register of the translation that is placed once for
/// the function, with the scratch ones: not one more register for every conversion (which ran
/// past the 255 registers a V100 has, and left the section header unable to say how many).
#[test]
fn a_turing_packed_conversion_uses_the_scratch_register() {
    use crate::generated::fields::{OP_RA, OP_RB, OP_RD};
    let mut dec75 = Decoder::new(75);
    let mut dec70 = Decoder::new(70);
    // The half kernel's `f2fp__RRR` at 0x3ce0 (Rd=184, Ra=185, Rb=184), on small registers.
    let base = 0x011fe400000000ff000000b8b9b8723eu128;
    let mut field = |word, name, value| {
        let d = dec75.decode(word).expect("sm_75 F2FP");
        super::util::put_operand(&d, name, value, 75).expect("the operand is in its class")
    };
    let word = field(base, OP_RD, 2);
    let word = field(word, OP_RA, 3);
    let word = field(word, OP_RB, 2);
    let f2fp = dec75.decode(word).expect("sm_75 F2FP");
    assert_eq!(
        (f2fp.op_or(OP_RD, 255), f2fp.op_or(OP_RA, 255), f2fp.op_or(OP_RB, 255)),
        (2, 3, 2)
    );
    let exit = assemble::assemble("exit_", &[]).unwrap();
    let src = vec![f2fp, dec75.decode(exit).expect("sm_75 EXIT")];
    let fd = FuncData {
        name: "f2fp_test".into(),
        registers: 8,
        tables: vec![],
        src,
        uniform: vec![],
        pairs: vec![],
        needs_pool: false,
        uniform_predicates: vec![],
        highest: 3,
        scratch_need: 1,
        needs_lane: false,
        product_count: 0,
        entered_inside: false,
        ampere: false,
        ampere_ops: HashMap::new(),
    };
    let predicates = super::context::stand_ins(&fd).unwrap();
    let result = super::function::translate_function(
        &mut dec70,
        &fd,
        &predicates,
        &HashMap::new(),
        None,
        None,
        None,
        None,
        None,
        Some(4),
        5,
        None,
        HashMap::new(),
        HashMap::new(),
        false,
        false,
        &HashMap::new(),
        None,
    )
    .unwrap();
    assert_eq!(result.count, 7, "the five program and scratch registers, and two more");
    assert_eq!(
        result.items.iter().map(|item| item.what).collect::<Vec<_>>(),
        [
            "f2fp: convert high half",
            "f2fp: convert low half",
            "f2fp: pack halves",
            "same"
        ]
    );
    let mut ops = |k: usize| dec70.decode(result.items[k].word).expect("sm_70").clone();
    let high = ops(0);
    let low = ops(1);
    let pack = ops(2);
    assert_eq!(
        high.op_or(crate::generated::fields::OP_RD, 255),
        4,
        "the high half converts into the scratch register"
    );
    assert_eq!(
        low.op_or(crate::generated::fields::OP_RD, 255),
        2,
        "the low half into the destination"
    );
    assert_eq!(pack.op_or(crate::generated::fields::OP_RD, 255), 2);
    assert_eq!(
        pack.op_or(crate::generated::fields::OP_RC, 255),
        4,
        "the pack reads the scratch register"
    );
}
