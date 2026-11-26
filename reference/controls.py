#!/usr/bin/env python3
"""controls.py [-u] [-c CONTROL]... CUBIN...: translations that are wrong on purpose, which
validate.py must object to.

validate.py says "0 problems" of the translation of all of PyTorch. That is worth what the check
can see: each control here takes one thing away from the prototype (translate.py), translates the
sm_75 cubins named with it, and counts what validate.py finds. A control that passes unnoticed
(0 problems in every cubin) is a hole in the check; so is a problem in the translation as it is.

    as it is    nothing taken away: no problem expected
    unsaved     a predicate that stands for a uniform one while it holds something of the program
                is not saved before nor given back after
    blind       the stand-ins of uniform predicates are chosen as if no ordinary predicate held
                anything
    cut         the lives of a uniform predicate are cut at every instruction: what one
                instruction writes, the next reads in what may be another predicate
    masks       the P2R and R2P of the program are taken to read and write no predicate (the
                predicates are a mask, a number: the tables do not say which)
    nowait      the instruction after one that reads late a register of the translation does not
                wait for it
    close       a wait comes one cycle after the instruction that names its scoreboard
    doublesoon  the instruction after an instruction of double precision, which reads its
                result, comes 6 cycles after it, as in code for sm_80 (Volta's own code has 8 or
                more: with that, three kernels of PyTorch were wrong on a V100)
    doubleacross
                the first instruction where a branch goes, which reads a result of double
                precision written before the branch, comes 6 cycles after it on the path through
                the branch (as the reciprocal of a double-precision division is in code for
                sm_80: the schedule sees straight-line code, and so did the validator's rule
                until 2026-11-25)
    returns     a number that makes a return address is off by one instruction
    waitless    what an instruction waited for, the first instruction of what it became does not
                wait for (ULDC lost its wait that way until a review found it)
    unlisted    the translation as it is, and a map that leaves out one instruction of each
                function (what the map does not have, the check did not look at)
    batched     an instruction with marks of register reuse that gets a stall which does not end
                its group keeps the marks, which then name a batch (as until 2026-11-17)
    unmoved     the relocations on code stay at the offsets they had, where other instructions
                are now (as until 2026-11-17; cuFFT has such kernels)
    named       the first relocation on code is made to name the function it is on, and the
                function is translated all the same: the place it names would have moved
    swapped     two instructions of the result that follow each other are exchanged, and the map
                says so: each is what the map says, in another order
    unguarded   the loads that stand for a matrix load under a guard have no guard
    lanes       in the sequence that stands for a matrix load or product, a value is taken from
                another lane, or another part of the lane's number
    unwaited    in such a sequence, what reads the result of shuffles does not wait for them
    unordered   where a store before a matrix load reads, as nothing orders it, a destination
                that the sequence writes with arithmetic first: the lanes of the rows of a plain
                load go into its destinations, and the arithmetic of a transposed one does not
                wait for the shuffles before it (as until 2026-11-23: the store may store the lane)
    complement_rewritten
                in the ORIGINAL, where an instruction under the complement of the guard of a store
                writes a register that the store reads as data, with nothing waited for between
                (`ptxas` does so: the store reads nothing in the lanes where that instruction
                runs), an instruction between them that can write a predicate besides its result
                (a carry) writes the store's predicate, and that program is translated as it is:
                the write then runs in lanes where the store may not have read the register yet
    complement_same
                in the ORIGINAL, such a write is put under the store's own guard, and that program
                is translated as it is
    soon        the selections after the last step of a product come two cycles after it, with a
                wait for its scoreboard, and not eight (on a V100 they then read registers that
                are not written yet)
    unheld      a uniform predicate held in a register is not read back before the instructions
                that read it: they go by whatever their predicate held
    unwritten   it is not written down in the register after the instructions that write it
    partial     R2P is taken to read its register only, not the predicates: two predicates given
                back one after the other are one cycle apart (as until 2026-11-18: the second R2P
                writes the first's predicate back as it was before), and a predicate that an
                instruction just wrote is not waited for
    boardless   a predicate saved, or all of them for R2P, is read without waiting for the
                scoreboard that the instruction that wrote it announces it on
    unmasked    an R2P of the program is taken to write what its operands say, no predicate (as
                until 2026-11-18): a predicate that the translation saves right after it is
                saved as it was before
    early       a predicate saved in a register is given back as soon as the instruction before
                allows, and not nine cycles after the last one that read what stood in it (on a
                V100 that reader then goes by the value given back: PyTorch's f32 attention
                faulted, 2026-11-18)
    stale       every product after the first that gathers takes the elements of B where that
                one left them, whatever happened since: the register of B written, a way in
                from elsewhere, a call, another register of B gathered
    unkept      of what a function with a product keeps of the lane, the lane that holds what
                a thread gathers for the columns 4 to 7 is 8 further and not 16
    unpaired    the upper half of a 64-bit uniform operand of an instruction that goes to
                ordinary registers (UIMAD.WIDE, UIMAD.WIDE.X) gets another register than the
                lower half's + 1, even when the lower half is even: that of a single uniform
                register, or else the first register that the translation leaves free (only
                where the upper half is also named alone: elsewhere its register is named nowhere)
    misruled    the immediate of the first instruction translated by a rule to ordinary registers
                is altered by one

With pairs of products on whole rows (the translation as it is has them, unless VF_PAIRS=0:
these change nothing then):

    rowlane     one of the values kept of the lane for the rows is computed with another shift
    otherrow    a whole-row load takes its second half of k where the first is
    rowwait     the loads of rows do not wait for the shuffles that bring their addresses
    rowread     the loads of rows do not say when they have read their address
    halfswap    a half of a pair writes the accumulators of its two products the other way round
    halfk       the first product of a pair is done on the second half of k and the second on the first
    inplace     the first product of a pair adds to the destinations, whatever the two products
                of the program add to (changes something only where a product does not add in place)
    stepfast    the steps of a half of a pair are one cycle apart
    unsettled   the conversions of accumulators start two cycles after the instruction that waits, not fifteen
    unsynced    the instruction before conversions waits for the products' scoreboard only (as until
                2026-11-18: a register that a load of the program was still filling was converted)
    unpermuted  the conversions use the permutation of the other way
    unconverted the conversions of the first place that has some are left out: every sequence is
                as the design has it, and the plan is wrong (the checker of plans, qcheck.py,
                must object: nothing else looks at the states of the accumulators)
    notalso     a load whose result in Turing's layout something else reads does not give it
                (the same: only the checker of plans knows who reads what)

With a predicate that the program writes late (a comparison of doubles, an instruction of a late
kind: docs/translator.md, "Predicates written late"; the translator takes such a predicate only
behind a wait):

    slowno      the instructions that do nothing, before what writes or saves a slow predicate,
                are left out
    slowshort   they last 3 cycles where 13 are needed and 15 where 26 are
    slownowait  the first of them does not wait for the six scoreboards
    slownever   slow predicates are taken like the others, with no wait: among them one that is
                written late under no scoreboard, which may never stand for a uniform predicate
    slowfree    no wait before the instructions that write a uniform predicate whose stand-in, in
                a life where it is free, is slow
    slowsaved   no wait before the first instruction of a life whose stand-in is slow and saved
    alive_scratch
                the predicate that a minimum or maximum computes into is one that holds something
                of the program there

With registers of the program borrowed (docs/translator.md, "Registers"). Few functions borrow
at the default; with VF_BORROW=1 every function does:

    alive_pair  the pair of registers given at an instruction is the lowest one of which a
                register holds something there
    alive_home  the home of each uniform register is the register after the one chosen
    no_borrow_wait
                where the first instruction of what an instruction became waits for the six
                scoreboards, it waits for what the original waited for only
    no_late_borrow
                an instruction of the program that reads a borrowed register late does not say
                when it has read
    written_pair
                the pair given at an instruction is one that the original instruction writes
    shared_home what the homes take is forgotten once they are chosen: a register given at an
                instruction may be the home of a uniform register that is alive there
    overlap_pool
                the same for the pairs only
    short_home  the declarations say less than the life of a uniform register in its home: an
                instruction of the life that neither reads nor writes it is left out (the
                translation is as it was: what the validator is told is wrong)
    pool_in_home_unsaid
                as overlap_pool, and no line of the declarations says that the pair is taken
                where it is the home of a uniform register that is alive

With a frame in the thread's local memory (levels 2 and 3; VF_BORROW=8 puts every function at
level 3):

    no_lane_l3  the S2R that reads the lane before a matrix sequence is replaced by a NOP
    odd_pair_l3 two uniform registers that make a pair are loaded from the frame into registers
                that begin on an odd one
    no_r1_lower the instruction that lowers the stack pointer by the frame is replaced by a NOP
    wrong_r1_lower
                it lowers the stack pointer by twice the frame
    wrong_slot_load
                the first load from the frame takes the word after its own
    no_store_l3 nothing is stored to the frame
    held_busy_l3
                the register that holds a uniform predicate between the instructions that name
                it is one where the program holds something at one of them
    unkept_holder_l3
                that register is not kept from the sequences: one of them may compute into it
                while it holds the predicate
    shared_holder_l3
                the uniform predicates held in registers are all held in the first one's,
                whether or not they are held at the same instructions
    home_unsaid the declarations leave out the home of one of two uniform registers that have
                one register for home at different instructions (the translation is as it was:
                the register then stands for that uniform register in the whole function, for
                all the validator is told)
    guarded_store_l3
                the stores to the frame after an instruction under a guard are under that guard
    no_old_load_l3
                what an instruction under a guard writes is not loaded from the frame before it
    no_load_wait
                what follows the last load from the frame does not wait for it
    store_no_read_sb
                the stores to the frame do not say when they have read
    wrong_base_const
                the base of the frame is taken from another constant
    base_after_use_l3
                the top of the frame is loaded after the first load or store that takes its
                address from it, with the waits that this order asks for: that access is
                addressed from what the register held before
    missing_pool_line
                the last register of the program that the translator says it took at an
                instruction is not said
    room_lane_unsaid
                where the lane is read into a register of the program that the translator says
                it took at the instruction, the declarations say that it puts that register away
                there instead, and nothing puts it away (the translation is as it was: what the
                validator is told is wrong)
    no_attr_raise
                the attributes of the function are the original's, not raised by the frame
    no_borrow_wait_l3
                no instruction of a sequence that reads or writes the frame waits for the six
                scoreboards
    no_c3_wait  where the translation takes a register that the program writes late, the first
                instruction of the sequence does not wait for the six scoreboards
    store_before_l3
                the stores to the frame come before what the instruction became, with the waits
                that this order asks for: they store what was there, or what the register held
    store_other_l3
                where an instruction only writes a uniform register of the frame, the register
                next to the one it computed into is stored
    stray_store_l3
                after an instruction that writes a uniform register of the frame, the same
                value is stored to another uniform register too, which the instruction does
                not write
    standin_home_l3
                a uniform register of the frame is loaded into the home of another uniform
                register that the instruction reads or writes, and read from there
    standin_operand_l3
                a uniform register of the frame is loaded into an ordinary register that the
                instruction itself reads, and read from there
    standin_shared_l3
                two uniform registers of the frame that an instruction reads are loaded into
                one register
    up_standin_l3
                a uniform predicate kept in the frame is loaded, after the uniform registers of
                the frame that the instruction reads, into the register that stands for the
                first of them, and read back from there
    up_wrong_word_l3
                a uniform predicate kept in the frame is loaded from the word after its own
    up_no_store_l3
                what an instruction writes of a uniform predicate kept in the frame is written
                down in its temporary register and not stored
    up_unwaited_l3
                what reads back a uniform predicate loaded from the frame does not wait for the
                load

With room made at an instruction (VF_BORROW=5 makes it where a kernel would otherwise need more
registers than it has):

    room_reads_orig
                the first register put away is one that the original instruction reads
    room_home_alive
                the first register put away is also declared the home of a uniform register
    room_no_back
                the first load that brings a register back is replaced by a NOP
    room_guarded_away
                the first store that puts a register away is under a guard (P0): where P0 does
                not hold, the register is not put away, and it is brought back all the same
    room_lane_first
                where the lane is read into a register put away at the instruction, it is read
                before the stores that put registers away, which wait for it: the register is put
                away with the lane in it, and brought back so
    room_back_wrong_word
                it brings the register back from the word after its own
    room_away_after_write
                the first store that puts a register away comes after the first instruction of
                the sequence
    room_same_word
                two registers are put away in one word
    room_at_branch
                R2 is declared put away at the first branch or exit of the function, where
                nothing puts it away
    room_depth_wrong
                the first store that puts a register away is where it would be if the program
                had not moved the stack pointer
    room_frame_short
                the frame is declared with the size of the uniform values only
    room_missing_lives
                the registers put away are not declared
    room_no_back_wait
                what follows the last load that brings a register back does not wait for it
    room_scratch_busy
                at an instruction with room, the scratch register of a matrix load is R14, which
                the program uses and which is not put away
    room_back_first
                the registers put away are brought back right after they were put away, before
                the sequence that uses them, with the waits that this order asks for
    room_back_before_store
                a register put away that stands for a uniform register of the frame which the
                instruction writes is brought back before that register is stored
    room_away_after_load
                a register put away that a load from the frame writes is put away after the
                load: what is put away, and brought back, is the uniform register
    copy_after_write
                in the ORIGINAL, the copy of a return address and the write that follows it are
                exchanged, and that program is translated as it is: the validator can no longer
                work out what the call reads and writes, and must not take registers for free
                where the translator did (no objection where the translator did not)

With tensor-core products that take everything at the instruction (level 3):

    product_odd_before_l3
                `odd` is computed before the gathering, which still reads `half` in that register
    product_busy_take_l3
                the register of the A operand is declared taken at the product, where it holds
                the operand
    product_odd_pair_l3
                the first pair taken for a product begins on an odd register
    product_no_s2r_wait_l3
                the first instruction that reads the lane does not wait for the S2R that reads it
    product_soon_l3
                the selections come two cycles after the last step of a product, not eight
                (thirteen for halves)
    load_ur_wrong_slot_l3
                the first load of a uniform register before a matrix sequence takes the word
                after its own
    load_ur_no_wait_l3
                the last such load names no scoreboard, and what follows does not wait for it

Printed for each control and cubin: the number of problems; `=` when the control changed
nothing in that cubin's translation, its map or its lives (it has nothing of what the control
takes away, and says nothing); or `refused` when all that the control changed is functions that
the translator then refused: they are traps, and nothing of what the control did reaches the
validator. The exit status is 0 when the translation as it is has no problem and every control is
objected to in every cubin whose translation it changes, but the cases of RIGHT_THERE (below).
(Until 2026-11-23 a control objected to in one cubin passed here, whatever the others.) Which cubins: ones that have what the controls take away
(predicates saved, a P2R of the program next to a uniform predicate, a call that is told where
to return). With -c, only the controls named (and the translation as it is). With -u, the
translation as it is is made and not validated: for a run of many processes, one of which
validates it (`-c 'as it is'`).

A control needs a translation that has what it takes away, and so the setting of the translator
that makes it (above). tests/sass/controls.sh runs every control in a process of its own, with
each setting, on the test kernels and on cubins of PyTorch that have what the controls need, and
says which are objected to; docs/validation.md, section 3, has what comes of it. (Until
2026-11-20 a control of borrowed registers left its change behind for the controls run after it
in the same process.)

A control may change a translation and find no objection: `blind` takes the lowest predicate the
instructions of a life do not touch, and where the translator's own lives of predicates are more
careful than nvdisasm's (they are, across calls within a function) that predicate may hold
nothing after all. Such a 0 is worth a look, and is not a hole by itself: this program and
tests/sass/controls.sh fail on each but those of RIGHT_THERE, each looked at.
"""
import collections, inspect, os, re, struct, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import elf, machine, translate
from machine import READ_BARRIER, WRITE_BARRIER, WAIT, stall_of, with_stall

HERE = os.path.dirname(os.path.abspath(__file__))
Function = translate.Function


def stored_before(f, i):
    """The registers that a memory instruction before the instruction `i` of the original reads
    as data (what a store stores, what a shuffle exchanges; validate.py, `data_overwritten`), in
    straight-line code, where nothing up to `i` orders it: no barrier, no wait for a scoreboard
    that it or a later memory instruction names, no memory instruction that writes the register,
    and no matrix load (what stands for one waits for its shuffles, which come after)."""
    found = set()
    for j in range(i - 1, -1, -1):
        dj = f.src[j]
        if dj.kind == 'BRU' and dj.mnemonic not in ('BSSY', 'BMOV') or dj.mnemonic == 'BAR':
            break
        address = re.match(r'(LD|ST|ATOM|RED)', dj.mnemonic)
        if not dj.kind.startswith('MIO') or not (address and dj.mnemonic[:2] != 'LD' or dj.mnemonic == 'SHFL'):
            continue
        uses = f.mc._uses[(75, dj.c.name)]
        data = {dj.ops[o] + n for letter, o, span in uses.reads if letter == 'R' and o in dj.ops and dj.ops[o] != 255
                and not (address and o == 'Ra') for n in range(machine.registers(span, dj.ops))}
        boards = {(f.words[j] >> WRITE_BARRIER) & 7, (f.words[j] >> READ_BARRIER) & 7} - {7}
        for k in range(j + 1, i + 1):
            if ((f.words[k] >> WAIT) & 63) & sum(1 << b for b in boards) or k < i and f.src[k].c.name in ('ldsm__sImmOffset', 'ldsm__URsIR'):
                data = set()
                break
            if k < i and f.src[k].kind.startswith('MIO'):
                boards |= {(f.words[k] >> WRITE_BARRIER) & 7, (f.words[k] >> READ_BARRIER) & 7} - {7}
                data -= {r for letter, r in f.src[k].writes if letter == 'R'}
        found |= data
    return found


def complemented(mc, data, same, listing):
    """The cubin `data` changed at the first place where an instruction under the complement of the
    guard of a store (an atomic operation, a reduction) writes a register that the store reads as
    data, in straight-line code, with no barrier and no wait for a scoreboard that the store or a
    later memory instruction names between them, and nothing between them writing the store's
    predicate (validate.py, `data_overwritten`, leaves such a write out): with `same`, that write
    is put under the store's own guard; otherwise the first instruction between them that can
    write a predicate besides its result (a carry, the outcome of a comparison) and writes none is
    made to write the store's predicate. Only where the translation as it is (its map, `listing`)
    has each instruction from the store to the write unchanged and alone: what the translation
    puts there may wait for the store (the arithmetic of an address register that it reuses).
    None when the cubin has no such place."""
    unchanged = collections.defaultdict(collections.Counter)
    for function, new, old, what in listing:
        unchanged[function][old] += 1 if what == 'same' else 2
    def field(c, word, wanted, value):
        for kind, fields, what in mc.d75.plan(c):
            if kind in ('operand', 'attribute') and what == wanted:
                return fields[0].put(word, value)
        return None
    container = elf.Elf(data)
    for s in container.sections:
        if not s.name.startswith('.text.') or s.type != 1:
            continue
        words = [int.from_bytes(s.data[i:i + 16], 'little') for i in range(0, len(s.data) - 15, 16)]
        decoded = [mc.decode(75, w) for w in words]
        alone = unchanged[s.name[len('.text.'):]]
        for k, d in enumerate(decoded):
            if d is None or not d.kind.startswith('MIO') or not re.match(r'(ST|ATOM|RED)', d.mnemonic):
                continue
            pg, negated = d.ops.get('Pg', 7), bool(d.ops.get('Pg@not'))
            uses = mc._uses.get((75, d.c.name))
            if pg == 7 or uses is None:
                continue
            read = {d.ops[o] + n for letter, o, span in uses.reads if letter == 'R' and o in d.ops and d.ops[o] != 255
                    and o != 'Ra' for n in range(machine.registers(span, d.ops))}
            boards = {(words[k] >> WRITE_BARRIER) & 7, (words[k] >> READ_BARRIER) & 7} - {7}
            between = None
            for j in range(k + 1, len(words)):
                y = decoded[j]
                if not read or y is None or ((words[j] >> WAIT) & 63) & sum(1 << b for b in boards) or y.mnemonic == 'BAR':
                    break
                written = read & {r for letter, r in y.writes if letter == 'R'}
                complement = written and not y.kind.startswith('MIO') and y.ops.get('Pg', 7) == pg and bool(y.ops.get('Pg@not')) != negated
                if complement and all(alone[16 * x] == 1 for x in range(k, j + 1)):
                    if same:
                        word = field(y.c, words[j], ('Pg', 'not'), int(negated))
                        at = j
                    elif between is not None:
                        word, at = field(decoded[between].c, words[between], 'Pu', pg), between
                    else:
                        word = None
                    if word is not None:
                        s.data = s.data[:16 * at] + word.to_bytes(16, 'little') + s.data[16 * at + 16:]
                        return container.write()
                if not complement:
                    read -= written
                if y.kind.startswith('MIO'):
                    boards |= {(words[j] >> WRITE_BARRIER) & 7, (words[j] >> READ_BARRIER) & 7} - {7}
                predicates = {r for letter, r in y.writes if letter == 'P'}
                if y.mnemonic == 'R2P':
                    predicates |= {p for p in range(7) if 'Sb' not in y.ops or y.ops['Sb'] >> p & 1}
                if pg in predicates or y.kind == 'BRU' and y.mnemonic not in ('BSSY', 'BMOV'):
                    break
                if between is None and not y.kind.startswith('MIO') and y.kind != 'BRU' and 'Pu' in y.c.operands and y.ops.get('Pu', 7) == 7:
                    between = j
    return None


def patched(control):
    """{method name: replacement} for `control`."""
    stand_ins, liveness, late, schedule, return_addresses = (
        Function.stand_ins, Function.liveness, Function.late, Function.schedule, Function.return_addresses)
    matrix_load, matrix_product, kept = Function.matrix_load, Function.matrix_product, Function.kept
    kept_for_rows, whole_rows, half_of_pair, converted = Function.kept_for_rows, Function.whole_rows, Function.half_of_pair, Function.converted
    given_back = Function.given_back
    f2fp_f16, ldg_without_hints = Function.f2fp_f16, Function.ldg_without_hints

    def f2fp_without_pack(self, i, d, guard):
        """The packed conversion without the instruction that packs the halves: the high half
        is converted into a register of the translation and never used. The validator's rule
        for the sequence must object."""
        return [item for item in f2fp_f16(self, i, d, guard) if item.what != 'f2fp: pack halves']

    def ldg_missing_zeros(self, i, d):
        """A zero-filled load without the MOVs that zero it where its predicate does not hold:
        what is out of range is loaded. The validator's rule for the sequence must object."""
        return [item for item in ldg_without_hints(self, i, d) if item.what != 'zero fill: zeros elsewhere']

    NOP = 0x000fc000000000000000000000007918

    def lepc_not_adjacent(self):
        """The LEPC of a direct call and the call, with a NOP between them: the callee returns
        into the NOP, not to it. The validator's adjacency rule for the pair must object."""
        out = return_addresses(self)
        for k in range(len(self.items) - 1):
            one, two = self.items[k], self.items[k + 1]
            if one.d is not None and two.d is not None and one.d.mnemonic == 'LEPC' and two.d.c.name.startswith('call_rel'):
                nop = translate.Item(NOP, one.origin, 'same', changed=False)
                nop.d = self.mc.decode(75, NOP)
                self.items.insert(k + 1, nop)
                break
        # So that the translation with the filler reaches the validator: the translator's own
        # check after scheduling would refuse it, and the point here is what the validator sees.
        self.check_lepc_pairs = lambda: None
        return out

    def lepc_register_changed(self):
        """The LEPC of a direct call writes another register than the one the callee returns
        through. The validator's rule for the pair must object."""
        out = return_addresses(self)
        for item in self.items:
            if item.d is not None and item.d.mnemonic == 'LEPC':
                register = item.d.ops['Rd']
                item.word = self.put(item.d.c, item.word, 'Rd', (register + 1) & 0xff)
                break
        return out

    def alive_scratch_pred(self):
        """For each minimum (scratch_at): replace its predicate with the HIGHEST one that
        the instruction reads or writes (one that is alive there). The validator's check
        for 'a predicate of the function, taken as dead, for a minimum or maximum' must
        catch it."""
        stand_ins(self)
        for i in list(self.scratch_at):
            d = self.src[i]
            # Pick a predicate that IS alive: one the instruction reads or writes.
            busy = (self.live_in[i] | self.live_out[i]) & 0x7f
            if not busy:
                continue
            # Pick the highest busy predicate (most likely to be alive and not in scratch_at already).
            for p in range(6, -1, -1):
                if busy >> p & 1:
                    self.scratch_at[i] = (p, self.scratch_at[i][1])
                    break

    def slowfree(self):
        """Drop the delay at instructions that write the uniform predicate of a FREE life
        whose stand-in is slow. The delay was set by delay(i, p) in stand_ins for each
        instruction of a free life that writes the uniform predicate."""
        stand_ins(self)
        # Find free lives: (q, part, p, saved=False) in self.lives.
        free_writes = set()   # {instruction index}
        for q, p, saved, part in self.lives:
            if saved or isinstance(saved, str):
                continue
            for i in part:
                if self.defs[i] >> (7 + q) & 1 and self.slow_predicates >> p & 1:
                    free_writes.add(i)
        for i in list(self.delayed):
            if i in free_writes:
                del self.delayed[i]

    def unsaved(self):
        stand_ins(self)
        self.saves.clear()
        self.restores.clear()
        self.scratch_at = {i: (p, None) for i, (p, k) in self.scratch_at.items()}

    def blind(self):
        liveness(self)
        self.live_in = [x & ~0x7f for x in self.live_in]
        self.live_out = [x & ~0x7f for x in self.live_out]

    def cut(self):
        liveness(self)
        self.live_out = [x & 0x7f for x in self.live_out]

    def masks(self):
        kept = []
        for d in self.src:
            if d.c.name in ('p2r__RsIR_RIR', 'r2p__RIR'):
                kept.append((d, d.ops['Sb']))
                d.ops['Sb'] = 0
        try:
            liveness(self)
        finally:
            for d, value in kept:
                d.ops['Sb'] = value

    def nowait(self, c, word, i, what):
        item = late(self, c, word, i, what)
        item.late = False
        return item

    def close(self):
        schedule(self)
        items = self.items
        for k in range(1, len(items)):
            mask = (items[k].word >> WAIT) & 63
            before = items[k - 1]
            named = {(before.word >> WRITE_BARRIER) & 7, (before.word >> READ_BARRIER) & 7} - {7}
            if (before.changed or items[k].changed) and any(mask >> s & 1 for s in named) and before.d.kind != 'BRU':
                before.word = with_stall(before.word, 1)

    def doublesoon(self):
        schedule(self)
        items = self.items
        for k in range(len(items) - 1):
            if items[k].d.mnemonic in ('DADD', 'DMUL', 'DFMA', 'DSETP') and stall_of(items[k].word) > 6 and any(
                    (letter, register) in items[k].d.writes for letter, register, role in items[k + 1].d.reads):
                items[k].word = with_stall(items[k].word, 6)

    def doubleacross(self):
        schedule(self)
        items = self.items
        place = {16 * item.origin: k for k, item in reversed(list(enumerate(items))) if item.first}
        for k, item in enumerate(items):
            if item.d.mnemonic not in ('DADD', 'DMUL', 'DFMA', 'DSETP'):
                continue
            written = set(item.d.writes)
            for b in range(k + 1, min(k + 6, len(items))):
                y = items[b]
                reads = {(letter, register) for letter, register, role in y.d.reads}
                if y.d.mnemonic == 'BRA' and y.target in place and not reads & written and any(
                        (letter, register) in written for letter, register, role in items[place[y.target]].d.reads):
                    # One cycle for each instruction up to the branch, the rest on the writer.
                    for j in range(k + 1, b + 1):
                        items[j].word = with_stall(items[j].word, 1)
                    item.word = with_stall(item.word, 6 - (b - k))
                    break
                # (Nothing in straight-line code reads the result before the branch: the reader
                # where the branch goes is the only one that the control brings closer.)
                if reads & written or y.d.kind == 'BRU':
                    break
                written -= set(y.d.writes)
                if not written:
                    break

    def returns(self):
        return_addresses(self)
        for item in self.items:
            if item.fix is not None and item.fix + 16 <= 16 * len(self.src):
                item.fix += 16
                break

    def waitless(self):
        for item in self.items:
            if item.first and item.changed:
                item.word &= ~(self.src[item.origin].word & (63 << WAIT))
        schedule(self)

    def other_lane(self, *args):
        new = (matrix_load if len(args) == 4 else matrix_product)(self, *args)
        by_name = self.mc.md70.by_name
        for item in new:
            if item.what == 'neighbour':
                item.word = self.put(by_name['shfl__RII'], item.word, 'Sb', 2)
                break
            if item.what == 'row':
                c = by_name['iadd3_imm__RsIR_RIR']
                item.word = self.put(c, item.word, 'Sb', self.mc.d70.operands(c, item.word)['Sb'] ^ 2)
                break
        else:
            for item in new:
                if item.what == 'thread':
                    c = by_name['lop3_lut__RuIR_RIR']
                    item.word = self.put(c, item.word, 'Sb', 1)
                    break
        return new

    def unguarded(self, *args):
        new = matrix_load(self, *args)
        c = self.mc.md70.by_name['lds__sImmOffset']
        for item in new:
            if item.what in ('half of the matrix', 'word of the matrix'):
                item.word = self.put(c, item.word, 'Pg', 7)
                for kind, fields, data in self.mc.d70.plan(c):
                    if kind == 'attribute' and data == ('Pg', 'not'):
                        item.word = fields[0].put(item.word, 0)
        return new

    def unwaited(self, *args):
        new = (matrix_load if len(args) == 4 else matrix_product)(self, *args)
        for item in new:
            if item.what not in ('product',):
                item.word &= ~(1 << (WAIT + translate.SEQUENCE))
        return new

    def unordered(self, i, d, guard, wait):
        new = matrix_load(self, i, d, guard, wait)
        ops, by_name = d.ops, self.mc.md70.by_name
        count, rd, ra = 1 << ops['num'], ops['Rd'], ops['Ra']
        if not {rd + m for m in range(1, count) if rd + m != ra} & stored_before(self, i):
            return new
        add, shfl = by_name['iadd3_imm__RsIR_RIR'], by_name['shfl__RRI']
        rows = {}
        for item in new:
            if item.what != 'row':
                continue
            if ops['mode'] == 1:
                item.word &= ~(1 << (WAIT + translate.SEQUENCE))
            else:
                o = self.mc.d70.operands(add, item.word)
                m = o['Sb'] // 8
                if rd + m != ra:
                    rows[o['Rd']] = rd + m
                    item.word = self.put(add, item.word, 'Rd', rd + m)
        for item in new:
            if item.what == 'address of the row' and ops['mode'] == 0:
                o = self.mc.d70.operands(shfl, item.word)
                if o['Rb'] in rows:
                    item.word = self.put(shfl, item.word, 'Rb', rows[o['Rb']])
        return new

    def soon(self, *args):
        new = matrix_product(self, *args)
        last = [item for item in new if item.what == 'product'][-1]
        last.word = with_stall(last.word, 2)
        return new

    def early(self, k, p, i):
        item = given_back(self, k, p, i)
        item.more_writes = []
        return item

    def partial(self, item):
        return []

    def boardless(self, item, board):
        pass

    def unmasked(self, item):
        return []

    def unheld(self):
        stand_ins(self)
        for i in self.held:
            self.held[i] = [(q, p, False, writes) for q, p, reads, writes in self.held[i]]

    def unwritten(self):
        stand_ins(self)
        for i in self.held:
            self.held[i] = [(q, p, reads, False) for q, p, reads, writes in self.held[i]]

    def stale(self, i, rb):
        return self.holds is not None

    def unkept(self):
        new = kept(self)
        c = self.mc.md70.by_name['lop3_lut__RuIR_RIR']
        new[3].word = self.put(c, new[3].word, 'Sb', 8)
        return new

    def rowlane(self):
        # (The first instruction is the first shift of the row of A: by 2, here by 3.)
        new = kept_for_rows(self)
        if new:
            new[0].word = self.put(self.mc.md70.by_name['shf__RuIR_RIR'], new[0].word, 'Sb', 3)
        return new

    def otherrow(self, *args):
        new = whole_rows(self, *args)
        c = self.mc.md70.by_name['lds__sImmOffset']
        loads = [item for item in new if item.what == 'whole row']
        first = self.mc.decode(70, loads[0].word).ops['Ra_offset']
        loads[1].word = self.put(c, loads[1].word, 'Ra_offset', first)
        return new

    def rowwait(self, *args):
        new = whole_rows(self, *args)
        for item in new:
            if item.what.startswith('whole row'):
                item.word &= ~((1 << (WAIT + translate.SEQUENCE)) | (1 << (WAIT + translate.ARRIVES)))
        return new

    def rowread(self, *args):
        new = whole_rows(self, *args)
        for item in new:
            if item.what == 'whole row':
                item.word |= 7 << READ_BARRIER
                item.late = False
        return new

    def halfswap(self, i, *args):
        site = self.q.sites[i]
        site['x'], site['y'] = site['y'], site['x']
        try:
            return half_of_pair(self, i, *args)
        finally:
            site['x'], site['y'] = site['y'], site['x']

    def halfk(self, i, *args):
        site = self.q.sites[i]
        site['half'] ^= 1
        try:
            return half_of_pair(self, i, *args)
        finally:
            site['half'] ^= 1

    def inplace(self, i, *args):
        site = self.q.sites[i]
        kept = site['cx'], site['cy']
        site['cx'], site['cy'] = site['x'], site['y']
        try:
            return half_of_pair(self, i, *args)
        finally:
            site['cx'], site['cy'] = kept

    def stepfast(self, *args):
        new = half_of_pair(self, *args)
        for item in new[:3]:
            item.word = with_stall(item.word, 1)
        return new

    def unsettled(self, *args):
        new = converted(self, *args)
        if new:
            new[0].word = with_stall(new[0].word, 2)
        return new

    def unsynced(self, *args):
        new = converted(self, *args)
        if new:
            new[0].word = (new[0].word & ~(63 << WAIT)) | (1 << (WAIT + translate.SEQUENCE))
        return new

    def unconverted(self, i, plan):
        if plan and i == min(plan):
            return []
        return converted(self, i, plan)

    def notalso(self, i, *args):
        load = self.q.loads[i]
        was, load['also'] = load['also'], False
        try:
            return whole_rows(self, i, *args)
        finally:
            load['also'] = was

    def unpermuted(self, i, plan):
        if not {'pi', 'inv'} <= set(self.kept_rows):
            return converted(self, i, plan)
        other = {i: [(r, translate.pairing.T if to == translate.pairing.Q else translate.pairing.Q) for r, to in plan.get(i, [])]}
        new = converted(self, i, other)
        for item in new:
            item.what = {'to pairs': 'from pairs', 'from pairs': 'to pairs'}.get(item.what, item.what)
        return new

    translate_orig = Function.translate
    one_orig = Function.one

    # (The arguments of `Function.translate` are read and given back by their names. Until
    # 2026-11-26 the sixth after `place` was taken by its place for the register to look from
    # where no single uniform register is there to exchange with: it is `scratch`, the first of
    # the registers of matrix sequences, which `placement` gives as None to a function without
    # them, as level 3 does (it takes them at each instruction). The control stopped there on
    # such a function, and in a function with them gave the upper half that first scratch
    # register.)
    translate_parameters = inspect.signature(translate_orig)

    def unpaired_translate(*args, **kwargs):
        given = translate_parameters.bind(*args, **kwargs)
        self, place, free = given.arguments['self'], dict(given.arguments['place']), given.arguments['free']
        paired64 = {
            d.ops[op]
            for d in self.src
            if d.c.name in self.mc.rule_of
            for l, op, span in self.mc._uses[(75, d.c.name)].reads + self.mc._uses[(75, d.c.name)].writes
            if l == 'U' and op in d.ops and d.ops[op] != 63 and machine.registers(span, d.ops) == 2
        }
        indiv = {
            d.ops[op] + k
            for d in self.src
            for l, op, span in self.mc._uses[(75, d.c.name)].reads + self.mc._uses[(75, d.c.name)].writes
            if l == 'U' and op in d.ops and d.ops[op] != 63
            for k in ((0, 1) if d.c.name.startswith('uldc') and machine.registers(span, d.ops) == 2 else ((0,) if machine.registers(span, d.ops) == 1 else ()))
        }
        for u in sorted(paired64):
            if u in place and u + 1 in place and u + 1 in indiv and place[u] % 2 == 0 and place[u + 1] == place[u] + 1:
                singleton = next((v for v in sorted(place) if v != u and v != u + 1 and v not in self.pairs and (v - 1) not in self.pairs and place[v] != place[u] + 1), None)
                if singleton is not None:
                    place[u + 1], place[singleton] = place[singleton], place[u + 1]
                    break
                if free < 253:
                    # None to exchange with: the first register that the translation leaves
                    # free, which holds nothing else, and the function counts it.
                    place[u + 1], free = free, free + 1
                    break
        given.arguments['place'], given.arguments['free'] = place, free
        return translate_orig(*given.args, **given.kwargs)

    def unpaired_from_memory(self, i, d):
        items = from_memory(self, i, d)
        if d.c.name in self.mc.rule_of and self.borrow and self.borrow.slots:
            _, _, together = self.in_frame(d)
            for u in sorted(together):
                if u in self._mem_regs and u + 1 in self._mem_regs and self._mem_regs[u] % 2 == 0 and self._mem_regs[u + 1] == self._mem_regs[u] + 1:
                    old_r1 = self._mem_regs[u + 1]
                    try:
                        alt = self.take(1)
                    except translate.Refused:
                        alt = None
                    if alt is not None and alt != old_r1:
                        self._mem_regs[u + 1] = alt
                        for item in items:
                            d70 = self.mc.decode(70, item.word)
                            if d70 and d70.mnemonic == 'LDL' and d70.ops.get('Rd') == old_r1:
                                item.word = self.mc.assemble('ldl__sImmOffset', Rd=alt, Ra=d70.ops['Ra'], Ra_offset=d70.ops.get('Ra_offset', 0), dst_wr_sb=d70.ops.get('dst_wr_sb', 7))
                        break
        return items

    def misruled_one(self, i, d):
        # (The instruction that was altered is kept, not that one was: the translator may
        # translate a function again, after an attempt that it gives up, and the next attempt
        # alters the same instruction. Until 2026-11-26 it altered none, and the function was
        # translated as it is: one function of cuSOLVER's 0015, of 198 cubins looked at.)
        items = one_orig(self, i, d)
        if getattr(self, '_misruled_at', i) == i:
            for item in items:
                if item.what == 'ordinary registers':
                    d70 = self.mc.decode(70, item.word)
                    if d70 is not None:
                        for kind, fields, data in self.mc.d70.plan(d70.c):
                            if kind == 'operand' and data in ('Sc', 'Sb') and any('Imm' in f.name or 'imm' in f.name or d70.c.operands[data].type == 'SImm' for f in fields if data in d70.c.operands):
                                val = fields[0].get(item.word)
                                item.word = fields[0].put(item.word, val ^ 1)
                                self._misruled_at = i
                                return items
        return items

    all_controls = {
        'as it is': {},
        'lepc_not_adjacent': {'return_addresses': lepc_not_adjacent},
        'lepc_register_changed': {'return_addresses': lepc_register_changed},
        'f2fp_without_pack': {'f2fp_f16': f2fp_without_pack},
        'ldg_missing_zeros': {'ldg_without_hints': ldg_missing_zeros},
        'unsaved': {'stand_ins': unsaved},
        'blind': {'liveness': blind},
        'cut': {'liveness': cut},
        'masks': {'liveness': masks},
        'nowait': {'late': nowait},
        'close': {'schedule': close},
        'doublesoon': {'schedule': doublesoon},
        'doubleacross': {'schedule': doubleacross},
        'returns': {'return_addresses': returns},
        'waitless': {'schedule': waitless},
        'unlisted': {},
        'swapped': {},
        'batched': {},
        'unmoved': {},
        'named': {},
        'unguarded': {'matrix_load': unguarded},
        'lanes': {'matrix_load': other_lane, 'matrix_product': other_lane},
        'unwaited': {'matrix_load': unwaited, 'matrix_product': unwaited},
        'unordered': {'matrix_load': unordered},
        'complement_rewritten': {},
        'complement_same': {},
        'soon': {'matrix_product': soon},
        'partial': {'all_predicates': partial},
        'boardless': {'waits_for': boardless},
        'unmasked': {'masked': unmasked},
        'early': {'given_back': early},
        'unheld': {'stand_ins': unheld},
        'unwritten': {'stand_ins': unwritten},
        'stale': {'kept_gathering': stale},
        'unkept': {'kept': unkept},
        'rowlane': {'kept_for_rows': rowlane},
        'otherrow': {'whole_rows': otherrow},
        'rowwait': {'whole_rows': rowwait},
        'rowread': {'whole_rows': rowread},
        'halfswap': {'half_of_pair': halfswap},
        'halfk': {'half_of_pair': halfk},
        'inplace': {'half_of_pair': inplace},
        'stepfast': {'half_of_pair': stepfast},
        'unsettled': {'converted': unsettled},
        'unsynced': {'converted': unsynced},
        'unpermuted': {'converted': unpermuted},
        'unconverted': {'converted': unconverted},
        'notalso': {'whole_rows': notalso},
        'slowno': {},
        'slowshort': {},
        'slownowait': {},
        'slownever': {},
        'slowfree': {'stand_ins': slowfree},
        'slowsaved': {},
        'alive_scratch': {'stand_ins': alive_scratch_pred},
        'alive_pair': {},
        'alive_home': {},
        'no_borrow_wait': {},
        'no_late_borrow': {},
        'written_pair': {},
        'shared_home': {},
        'overlap_pool': {},
        'short_home': {},
        'pool_in_home_unsaid': {},
        'unpaired': {'translate': unpaired_translate, 'from_memory': unpaired_from_memory},
        'misruled': {'one': misruled_one},
    }
    # ---- level 3 controls (the frame) ----
    from_memory, to_memory = Function.from_memory, Function.to_memory
    frame_top_orig = Function.frame_top

    def from_memory_no_lane(self, i, d):
        """Unused: the lane is not created by from_memory."""
        return from_memory(self, i, d)

    def from_memory_odd_pair(self, i, d):
        """A pair loaded into two single registers (odd first register)."""
        items = from_memory(self, i, d)
        if self.borrow and self.borrow.slots:
            _, _, together = self.in_frame(d)
            for u in together:
                if u in self._mem_regs and u + 1 in self._mem_regs:
                    r = self._mem_regs[u]
                    if r % 2 == 0:
                        # Shift to odd: use r+1 and r+2 instead of r and r+1
                        old_r, old_r1 = self._mem_regs[u], self._mem_regs[u + 1]
                        new_r = r + 1
                        self._mem_regs[u] = new_r
                        self._mem_regs[u + 1] = new_r + 1
                        for item in items:
                            # Replace register references in the LDL items
                            mc = self.mc
                            d70 = mc.decode(70, item.word)
                            if d70 and d70.mnemonic == 'LDL':
                                if d70.ops.get('Rd') == old_r:
                                    item.word = mc.assemble('ldl__sImmOffset', Rd=new_r, Ra=d70.ops['Ra'], Ra_offset=d70.ops.get('Ra_offset', 0), dst_wr_sb=d70.ops.get('dst_wr_sb', 7))
                                elif d70.ops.get('Rd') == old_r1:
                                    item.word = mc.assemble('ldl__sImmOffset', Rd=new_r + 1, Ra=d70.ops['Ra'], Ra_offset=d70.ops.get('Ra_offset', 0), dst_wr_sb=d70.ops.get('dst_wr_sb', 7))
                        break
        return items

    def from_memory_wrong_slot(self, i, d):
        """A load from the wrong slot (offset +4)."""
        items = from_memory(self, i, d)
        mc = self.mc
        for item in items:
            d70 = mc.decode(70, item.word)
            if d70 and d70.mnemonic == 'LDL':
                offset = d70.ops.get('Ra_offset', 0)
                # Shift offset by 4 (wrong slot)
                item.word = mc.assemble('ldl__sImmOffset', Rd=d70.ops['Rd'], Ra=d70.ops['Ra'],
                                        Ra_offset=(offset + 4) & 0xFFFFFF, dst_wr_sb=d70.ops.get('dst_wr_sb', 7))
                break
        return items

    def from_memory_no_wait(self, i, d):
        """No wait after the last load: clear the 'then' attribute on the last load."""
        items = from_memory(self, i, d)
        # Clear the wait-for on loads.
        for item in reversed(items):
            if item.what.startswith('mem: load'):
                item.then = None
                break
        return items

    def to_memory_none(self, i, d):
        """A store left out (no stores at all)."""
        return []

    def to_memory_guarded(self, i, d):
        """The stores under the instruction's guard: they go by what the predicate holds after
        the instruction."""
        items = to_memory(self, i, d)
        mc, ops, guard = self.mc, d.ops, None
        if ops.get('UPg', 7) != 7 or ops.get('UPg@not'):
            guard = {'Pg': self.P(ops['UPg']), 'Pg_not': ops.get('UPg@not', 0)}
        elif ops.get('Pg', 7) != 7 or ops.get('Pg@not'):
            guard = {'Pg': ops['Pg'], 'Pg_not': ops.get('Pg@not', 0)}
        if guard is not None:
            plain = mc.assemble('stl__sImmOffset', Ra=1, Rb=2, Ra_offset=0)
            under = mc.assemble('stl__sImmOffset', Ra=1, Rb=2, Ra_offset=0, **guard)
            for item in items:
                if item.what.startswith('mem: store '):
                    item.word ^= plain ^ under
        return items

    def from_memory_no_old(self, i, d):
        """What an instruction under a guard writes is not loaded before it: the store after it
        writes what the register happened to hold when the instruction did not run."""
        saved = translate.pairing.unguarded
        translate.pairing.unguarded = lambda d: True
        try:
            return from_memory(self, i, d)
        finally:
            translate.pairing.unguarded = saved

    def to_memory_no_read_sb(self, i, d):
        """A store that does not say when it has read."""
        items = to_memory(self, i, d)
        for item in items:
            d70 = self.mc.decode(70, item.word)
            if d70 and d70.mnemonic == 'STL':
                item.word |= 7 << READ_BARRIER
                item.late = False
        return items



    def frame_top_wrong_const(self, i, items):
        """The base loaded from another constant (c[0x0][0x20] instead of c[0x0][0x28])."""
        if self._base is None:
            self._base = self.take(1)
            items.append(translate.Item(self.mc.assemble('mov__RCR_RC', Rd=self._base, Sb_bank=0, Sb_addr=0x20 // 4), i, 'mem: base'))
        return self._base

    # ---- level 3 controls: the values that the moves of the frame carry ----
    AROUND = ('wait', 'save', 'restore', 'lane', 'kept', 'settled', 'to pairs', 'from pairs', 'held: back', 'held: kept', 'late predicate')

    def is_core(what):
        return what not in AROUND and not what.startswith(('mem: ', 'room: '))

    def reordered(self, pick):
        """The items of each instruction put in the order that `pick` gives for them (None: as
        they are), before the schedule is made: the first of them waits for what the first
        waited for, and what follows an item that asks to be waited for waits for it. So the
        order is the only thing wrong."""
        items, out, k, changed = self.items, [], 0, False
        while k < len(items):
            j = k
            while j < len(items) and items[j].origin == items[k].origin:
                j += 1
            group = items[k:j]
            new = pick(group) if all(item.target is None for item in group) else None
            if new is not None and new != group:
                new[0].word |= group[0].word & (63 << WAIT)
                for n, item in enumerate(new):
                    item.first = n == 0
                group, changed = new, True
            out += group
            k = j
        if changed:
            for k, item in enumerate(out[:-1]):
                for board in ([item.then] if item.then is not None else []) + item.also:
                    out[k + 1].word |= 1 << (WAIT + board)
            self.items = out

    def register_of(self, item, operand):
        d70 = self.mc.decode(70, item.word)
        return d70.ops.get(operand) if d70 is not None else None

    def schedule_store_before(self):
        def pick(group):
            whats = [item.what for item in group]
            core = [k for k, w in enumerate(whats) if is_core(w)]
            # (With the base, when it is loaded for the stores.)
            moved = [k for k, w in enumerate(whats) if core and k > core[-1] and (w.startswith('mem: store ') or w == 'mem: base')]
            if not any(whats[k].startswith('mem: store ') for k in moved):
                return None
            return group[:core[0]] + [group[k] for k in moved] + [item for k, item in enumerate(group) if k >= core[0] and k not in moved]
        reordered(self, pick)
        schedule(self)

    def schedule_base_late(self):
        def pick(group):
            bases = [k for k, item in enumerate(group) if item.what == 'mem: base']
            if not bases:
                return None
            users = [k for k, item in enumerate(group)
                     if k > bases[0] and item.what.startswith(('mem: load ', 'mem: store '))
                     and register_of(self, item, 'Ra') == register_of(self, group[bases[0]], 'Rd')]
            if not users:
                return None
            b, u = bases[0], users[0]
            return group[:b] + group[b + 1:u + 1] + [group[b]] + group[u + 1:]
        reordered(self, pick)
        schedule(self)

    def schedule_back_first(self):
        def pick(group):
            aways = [k for k, item in enumerate(group) if item.what.startswith('room: ') and item.what.endswith(' away')]
            backs = [item for item in group if item.what.startswith('room: ') and item.what.endswith(' back')]
            if not aways or not backs:
                return None
            rest = [item for item in group if item not in backs]
            at = aways[-1] + 1
            return rest[:at] + backs + rest[at:]
        reordered(self, pick)
        schedule(self)

    def schedule_back_before_store(self):
        def pick(group):
            backs = [item for item in group if item.what.startswith('room: ') and item.what.endswith(' back')]
            stores = [k for k, item in enumerate(group) if item.what.startswith('mem: store ')]
            away = {register_of(self, item, 'Rd') for item in backs}
            if not backs or not stores or not any(register_of(self, group[k], 'Rb') in away for k in stores):
                return None
            rest = [item for item in group if item not in backs]
            at = rest.index(group[stores[0]])
            return rest[:at] + backs + rest[at:]
        reordered(self, pick)
        schedule(self)

    def schedule_away_after_load(self):
        def pick(group):
            aways = [item for item in group if item.what.startswith('room: ') and item.what.endswith(' away')]
            loads = [k for k, item in enumerate(group) if item.what.startswith('mem: load ')]
            away = {register_of(self, item, 'Rb') for item in aways}
            if not aways or not loads or not any(register_of(self, group[k], 'Rd') in away for k in loads):
                return None
            last = group[loads[-1]]
            rest = [item for item in group if item not in aways]
            at = rest.index(last) + 1
            return rest[:at] + aways + rest[at:]
        reordered(self, pick)
        schedule(self)

    def load_into(self, items, u, r):
        """The load of uniform register `u` among `items` made to write register `r`, which
        then stands for `u` in what the instruction becomes."""
        mc = self.mc
        self._mem_regs[u] = r
        for item in items:
            if item.what == 'mem: load UR%d' % u:
                d70 = mc.decode(70, item.word)
                item.word = mc.assemble('ldl__sImmOffset', Rd=r, Ra=d70.ops['Ra'], Ra_offset=d70.ops.get('Ra_offset', 0),
                                        dst_wr_sb=d70.ops.get('dst_wr_sb', 7))

    def alone(self, d):
        """The uniform registers of the frame that instruction `d` reads, one register each."""
        read, written, together = self.in_frame(d)
        return sorted(u for u in read if u not in together and u - 1 not in together)

    def from_memory_into_home(self, i, d):
        items = from_memory(self, i, d)
        if items and self.borrow is not None and self.borrow.homes:
            homes = self.borrow.homes
            there = [homes[u] for l, u, *_ in d.reads if l == 'U' and u in homes] + [homes[u] for l, u in d.writes if l == 'U' and u in homes]
            mine = alone(self, d)
            if there and mine:
                load_into(self, items, mine[0], there[0])
        return items

    def from_memory_into_operand(self, i, d):
        items = from_memory(self, i, d)
        if items and self.borrow is not None:
            reads = [r for l, r, *_ in d.reads if l == 'R' and 2 <= r < 255]
            writes = {r for l, r in d.writes if l == 'R'}
            # (One that it writes too, where there is one: a register of the program that the
            # translation may write at that instruction.)
            both = [r for r in reads if r in writes]
            mine = alone(self, d)
            if reads and mine:
                load_into(self, items, mine[0], (both or reads)[0])
        return items

    def from_memory_shared(self, i, d):
        items = from_memory(self, i, d)
        if items and self.borrow is not None:
            mine = alone(self, d)
            if len(mine) >= 2:
                load_into(self, items, mine[1], self._mem_regs[mine[0]])
        return items

    def to_memory_other(self, i, d):
        items = to_memory(self, i, d)
        if not items or not translate.pairing.unguarded(d):
            return items        # (What an instruction under a guard writes was loaded: rule B3 has that.)
        read, written, together = self.in_frame(d)
        mc = self.mc
        c = mc.md70.by_name['stl__sImmOffset']
        for k, item in enumerate(items):
            m = re.match(r'mem: store UR(\d+)$', item.what)
            if m and int(m.group(1)) not in read and int(m.group(1)) not in together and int(m.group(1)) - 1 not in together:
                d70 = mc.decode(70, item.word)
                word = mc.assemble('stl__sImmOffset', Ra=d70.ops['Ra'], Rb=d70.ops['Rb'] ^ 1, Ra_offset=d70.ops.get('Ra_offset', 0))
                items[k] = self.late(c, word, i, item.what)
        return items

    def to_memory_stray(self, i, d):
        items = to_memory(self, i, d)
        if items and self.borrow is not None:
            read, written, together = self.in_frame(d)
            other = [u for u in sorted(self.borrow.slots) if u not in read and u not in written]
            if written and other:
                mc = self.mc
                top, offset = self.slot(other[0], i, items)
                word = mc.assemble('stl__sImmOffset', Ra=top, Rb=self._mem_regs[min(written)], Ra_offset=offset)
                items.append(self.late(mc.md70.by_name['stl__sImmOffset'], word, i, 'mem: store UR%d' % other[0]))
        return items

    # ---- level 3 controls: the uniform predicates kept in the frame (`mem: load UP<n>`,
    # `mem: store UP<n>`), which `translate` moves with `load_word` and `store_word` and reads
    # back with `read_back` ----
    load_word, store_word, read_back = Function.load_word, Function.store_word, Function.read_back

    def load_word_up_standin(self, word, rd, i, items, what):
        if what.startswith('mem: load UP'):
            read, written, together = self.in_frame(self.src[i])
            there = [self._mem_regs[u] for u in sorted(read) if u in self._mem_regs]
            if there:
                self._up_read_back = (i, rd, there[0])
                rd = there[0]
        return load_word(self, word, rd, i, items, what)

    def read_back_up_standin(self, holder, p, i):
        moved = getattr(self, '_up_read_back', None)
        if moved is not None and moved[:2] == (i, holder):
            holder = moved[2]
        return read_back(self, holder, p, i)

    def load_word_up_wrong_word(self, word, rd, i, items, what):
        return load_word(self, word + 1 if what.startswith('mem: load UP') else word, rd, i, items, what)

    def store_word_up_none(self, word, rb, i, items, what):
        if not what.startswith('mem: store UP'):
            store_word(self, word, rb, i, items, what)

    # ---- level 3 product controls ----
    def product_odd_before_l3(self, i, d, wait):
        """Odd computed before the gathering that still reads half: move the two 'kept: odd'
        instructions to right after the 'kept: half' instructions and before 'neighbour'."""
        new = matrix_product(self, i, d, wait)
        if not (self.borrow is not None and self.borrow.everything):
            return new
        odd_items = [item for item in new if item.what == 'kept: odd']
        rest_items = [item for item in new if item.what != 'kept: odd']
        # Insert odd items right after the last 'kept: half' and before 'neighbour'.
        insert_at = next((k for k, item in enumerate(rest_items) if item.what == 'neighbour'), len(rest_items))
        return rest_items[:insert_at] + odd_items + rest_items[insert_at:]

    def product_odd_pair_l3(self, i, d, wait):
        """A pair from an odd register: make take(2) return an odd register for the first pair."""
        if not (self.borrow is not None and self.borrow.everything):
            return matrix_product(self, i, d, wait)
        orig_free = self.borrow.__class__.free_at
        pair_count = [0]
        def odd_free(borrow_self, inst, width=1, given=0):
            r = orig_free(borrow_self, inst, width, given)
            if width == 2 and r is not None and r % 2 == 0:
                pair_count[0] += 1
                if pair_count[0] == 1:
                    # Return an odd register instead.
                    return r + 1
            return r
        self.borrow.__class__.free_at = odd_free
        try:
            return matrix_product(self, i, d, wait)
        finally:
            self.borrow.__class__.free_at = orig_free

    def product_soon_l3(self, i, d, wait):
        """Selections too soon after the last step: set the stall on the last HMMA step to 2
        instead of 8 (floats) or 13 (halves)."""
        new = matrix_product(self, i, d, wait)
        if not (self.borrow is not None and self.borrow.everything):
            return new
        products = [item for item in new if item.what == 'product']
        if products:
            products[-1].word = with_stall(products[-1].word, 2)
        return new

    all_controls.update({
        'no_lane_l3': {},
        'odd_pair_l3': {'from_memory': from_memory_odd_pair},
        'no_r1_lower': {},
        'wrong_r1_lower': {},
        'no_attr_raise': {},
        'wrong_slot_load': {'from_memory': from_memory_wrong_slot},
        'no_store_l3': {'to_memory': to_memory_none},
        'held_busy_l3': {},
        'unkept_holder_l3': {},
        'shared_holder_l3': {},
        'home_unsaid': {},
        'guarded_store_l3': {'to_memory': to_memory_guarded},
        'no_old_load_l3': {'from_memory': from_memory_no_old},
        'no_load_wait': {},
        'store_no_read_sb': {'to_memory': to_memory_no_read_sb},
        'no_borrow_wait_l3': {},
        'wrong_base_const': {'frame_top': frame_top_wrong_const},
        'base_after_use_l3': {'schedule': schedule_base_late},
        'missing_pool_line': {},
        'room_lane_unsaid': {},
        'no_c3_wait': {},
        'store_before_l3': {'schedule': schedule_store_before},
        'store_other_l3': {'to_memory': to_memory_other},
        'stray_store_l3': {'to_memory': to_memory_stray},
        'standin_home_l3': {'from_memory': from_memory_into_home},
        'standin_operand_l3': {'from_memory': from_memory_into_operand},
        'standin_shared_l3': {'from_memory': from_memory_shared},
        'up_standin_l3': {'load_word': load_word_up_standin, 'read_back': read_back_up_standin},
        'up_wrong_word_l3': {'load_word': load_word_up_wrong_word},
        'up_no_store_l3': {'store_word': store_word_up_none},
        'up_unwaited_l3': {},
        # ---- room controls ----
        'room_reads_orig': {},
        'room_home_alive': {},
        'room_no_back': {},
        'room_guarded_away': {},
        'room_lane_first': {},
        'room_back_wrong_word': {},
        'room_away_after_write': {},
        'room_same_word': {},
        'room_at_branch': {},
        'room_depth_wrong': {},
        'room_frame_short': {},
        'room_missing_lives': {},
        'room_no_back_wait': {},
        'room_scratch_busy': {},
        'room_back_first': {'schedule': schedule_back_first},
        'room_back_before_store': {'schedule': schedule_back_before_store},
        'room_away_after_load': {'schedule': schedule_away_after_load},
        'copy_after_write': {},
        # ---- level 3 product controls ----
        'product_odd_before_l3': {'matrix_product': product_odd_before_l3},
        'product_busy_take_l3': {},
        'product_odd_pair_l3': {'matrix_product': product_odd_pair_l3},
        'product_no_s2r_wait_l3': {},
        'product_soon_l3': {'matrix_product': product_soon_l3},
        # ---- level 3 load with UR controls ----
        'load_ur_wrong_slot_l3': {},
        'load_ur_no_wait_l3': {},
    })
    return all_controls[control]


CONTROLS = ['as it is', 'unsaved', 'blind', 'cut', 'masks', 'nowait', 'close', 'doublesoon', 'doubleacross', 'returns', 'waitless', 'unlisted', 'swapped', 'batched', 'unmoved', 'named', 'unguarded', 'lanes', 'unwaited', 'unordered', 'complement_rewritten', 'complement_same', 'soon', 'partial', 'boardless', 'unmasked', 'early', 'unheld', 'unwritten', 'stale', 'unkept', 'unpaired', 'misruled', 'slowno', 'slowshort', 'slownowait', 'slownever', 'slowfree', 'slowsaved', 'alive_scratch']

# Encodings of sm_75 that no cubin of the repository has: a Turing zero-fill load (Pnz) and
# F2FP, which the CUDA 13.5 cuBLAS half kernels use. Run them by name on a cubin that has one
# (`controls.py -c ...`); tests/sass/controls.sh does not run them.
HALF_CONTROLS = ['f2fp_without_pack', 'ldg_missing_zeros']
# A direct call whose return address is the program counter of the LEPC immediately before it
# (cuDNN 9.24's sm_75 kernels): tests/kernels/translate/k_lepc.sm75.cubin carries the pair, so
# these run in the passes of tests/sass/controls.sh.
LEPC_CONTROLS = ['lepc_not_adjacent', 'lepc_register_changed']
CONTROLS += LEPC_CONTROLS
# (Those of pairs of products on whole rows: only when the translator makes them, not with VF_PAIRS=0.)
PAIR_CONTROLS = ['rowlane', 'otherrow', 'rowwait', 'rowread', 'halfswap', 'halfk', 'inplace', 'stepfast', 'unsettled', 'unsynced', 'unpermuted', 'unconverted', 'notalso']
CONTROLS += PAIR_CONTROLS
# (Those of borrowed registers: only when VF_BORROW=1.)
BORROW_CONTROLS = ['alive_pair', 'alive_home', 'no_borrow_wait', 'no_late_borrow', 'written_pair', 'shared_home', 'overlap_pool', 'short_home', 'pool_in_home_unsaid']
CONTROLS += BORROW_CONTROLS
# (Those of level 3: only when VF_BORROW>=2 and a function uses the frame.)
FRAME_CONTROLS = ['no_lane_l3', 'odd_pair_l3', 'no_r1_lower', 'wrong_r1_lower', 'wrong_slot_load', 'no_store_l3', 'held_busy_l3', 'unkept_holder_l3', 'shared_holder_l3', 'home_unsaid', 'guarded_store_l3', 'no_old_load_l3', 'no_load_wait', 'store_no_read_sb', 'wrong_base_const', 'missing_pool_line', 'room_lane_unsaid', 'no_attr_raise', 'no_borrow_wait_l3', 'no_c3_wait',
                  'store_before_l3', 'store_other_l3', 'stray_store_l3', 'standin_home_l3', 'standin_operand_l3', 'standin_shared_l3',
                  'base_after_use_l3', 'up_standin_l3', 'up_wrong_word_l3', 'up_no_store_l3', 'up_unwaited_l3']
CONTROLS += FRAME_CONTROLS
# (Those of a frame addressed from its top, which a register is loaded with: at VF_BORROW=3,
# and wherever the program moves its stack pointer in a way that is not followed.)
BASE_CONTROLS = ['wrong_base_const', 'base_after_use_l3']
# (Those of room made: only when VF_BORROW>=5 and a function has room.)
ROOM_CONTROLS = ['room_reads_orig', 'room_home_alive', 'room_no_back', 'room_guarded_away', 'room_lane_first', 'room_back_wrong_word', 'room_away_after_write', 'room_same_word', 'room_at_branch', 'room_depth_wrong', 'room_frame_short', 'room_missing_lives', 'room_no_back_wait', 'room_scratch_busy', 'copy_after_write',
                 'room_back_first', 'room_back_before_store', 'room_away_after_load']
CONTROLS += ROOM_CONTROLS
# (Those of level-3 products: only when VF_BORROW>=7 and a function has products at level 3.)
PRODUCT_L3_CONTROLS = ['product_odd_before_l3', 'product_busy_take_l3', 'product_odd_pair_l3', 'product_no_s2r_wait_l3', 'product_soon_l3', 'load_ur_wrong_slot_l3', 'load_ur_no_wait_l3']
CONTROLS += PRODUCT_L3_CONTROLS
# (Of another kind: the PROGRAM is changed, and what is validated is the translator's own
# translation of it. The validator objects where it is the more careful of the two; no objection
# is what a translator as careful as the validator gets, and is no hole.)
CHANGES_THE_PROGRAM = ['copy_after_write']

# Where a control changes the translation of a cubin and the validator rightly finds nothing, each
# looked at (2026-11-23): (setting, control, cubin): why. The setting is `default` or the value of
# VF_BORROW (setting()), the cubin the name of its file. Any other such case fails, here (main)
# and in tests/sass/controls.sh, which both decide by unobjected().
HOLDS_NOTHING = ('the instruction left out of the declared life is one where, by the validator\'s own lives, '
                 'the uniform register holds nothing: the translator declares the life longer than it is')
FREE_TOO = 'the next register, where every home is moved, is free too'
NOTHING_IN_FLIGHT = ('no load of the program in flight writes the borrowed register: the wait for the six '
                     'scoreboards that the control takes away is not needed there')
ROUTINE_ONLY = ('the registers taken (R231 at the default, R229 to R231 with VF_BORROW=8: the 64-bit addresses '
                'that two IMAD.WIDE write in R228 and R230 and two loads read, all before the first call) are '
                'busy by the translator\'s lives only at the three instructions of the routine that the kernel '
                'calls (WARPSYNC, MOV, RET.REL), where it takes every register of the function to be alive at the '
                'return; by the validator\'s own lives, which follow each of the four calls, they hold nothing '
                'there, nor anywhere along the lives of the predicates held in them')
RIGHT_THERE = {
    ('default', 'short_home', '0161.sm_75.cubin'): HOLDS_NOTHING,
    ('1', 'short_home', '0142.sm_75.cubin'): HOLDS_NOTHING,
    ('1', 'alive_home', 'k_convert.sm75.cubin'): FREE_TOO,
    ('1', 'alive_home', '0097.sm_75.cubin'): FREE_TOO,
    ('1', 'alive_home', '0142.sm_75.cubin'): FREE_TOO,
    ('1', 'no_borrow_wait', 'k_params.sm75.cubin'): NOTHING_IN_FLIGHT,
    ('1', 'no_borrow_wait', '0142.sm_75.cubin'): NOTHING_IN_FLIGHT,
    ('1', 'written_pair', '0086.sm_75.cubin'): ('the translator gives level 2 up for the pair (a borrowed register '
                                                'receives a result that nothing announces) and translates the four '
                                                'functions at level 1, with pairs that their instructions do not write'),
    ('default', 'held_busy_l3', '0854.sm_75.cubin'): ROUTINE_ONLY,
    ('8', 'held_busy_l3', '0854.sm_75.cubin'): ROUTINE_ONLY,
}


def setting():
    """The setting of the translator as RIGHT_THERE names it: `default`, or the value of VF_BORROW."""
    return os.environ.get('VF_BORROW') or 'default'


def unobjected(control, setting, cubins, counts):
    """Of the cubins (`cubins`, the names of their files) with what the validator found in the
    translation that `control` changed at `setting` (`counts`, as printed: '0' when it found
    nothing there, '=' when the control changed nothing, `refused` or `failed`), those where it
    found nothing: ([the others], [the cases of RIGHT_THERE]). A control is objected to where
    it changes a translation, but in the first list's cubins (a hole in the validator, or a
    control that tests nothing there)."""
    assert len(cubins) == len(counts)
    holes, right = [], []
    for cubin, count in zip(cubins, counts):
        if str(count) == '0':
            (right if (setting, control, cubin) in RIGHT_THERE else holes).append(cubin)
    return holes, right


def naming_code(data):
    """The cubin `data` with its first relocation on code made to name the function it is on;
    None when it has no relocation on code."""
    container = elf.Elf(data)
    symbols = container.symbols()
    for s in container.sections:
        if s.type in (elf.SHT_REL, elf.SHT_RELA) and s.info < len(container.sections) and container.sections[s.info].name.startswith('.text.') and len(s.data) >= 16:
            own = next((k for k, symbol in enumerate(symbols) if symbol[3] == s.info and symbol[1] & 15 == 2), None)
            if own is not None:
                offset, info = struct.unpack_from('<QQ', s.data, 0)
                s.data = struct.pack('<QQ', offset, (own << 32) | (info & 0xffffffff)) + s.data[16:]
                return container.write()
    return None


def frozen(row):
    """A row of the map or of the lives, as a tuple of what it has (a list of offsets as a tuple)."""
    return tuple(tuple(x) if isinstance(x, list) else x for x in row)


def refused_only(out, listing, lives, report, before, rest):
    """The functions that the translation `out` refuses and the translation as it is (`before`, and
    `rest`: its map, lives and report) does not, when that is all the control changed: the code,
    the attributes, the map and the lives of every other function are as they are. Empty
    otherwise. (Nothing that the control did reaches the validator then: the translator refused it,
    and those functions are traps.)"""
    newly = {f for f, why in report.items() if why and not rest['report'].get(f)}
    if not newly:
        return set()
    a, b = elf.Elf(before), elf.Elf(out)
    for f in report:
        if f not in newly:
            for name in ('.text.' + f, '.nv.info.' + f):
                x, y = a.section(name), b.section(name)
                if (x.data if x else None) != (y.data if y else None):
                    return set()
    def others(rows):
        return collections.Counter(frozen(row) for row in rows if row[0] not in newly)
    if others(listing) != others(rest['listing']) or others(lives) != others(rest['lives']):
        return set()
    return newly


def run(mc, control, path, out_dir, untouched, validated=True):
    """(problems validate.py finds, its first objection) for one cubin; (None, '') when the control
    changed nothing in its translation, which `untouched` holds by path. (`validated` false: the
    translation as it is is kept there and not validated.)"""
    originals = {name: getattr(Function, name) for name in ('stand_ins', 'liveness', 'late', 'schedule', 'return_addresses', 'matrix_load', 'matrix_product', 'kept', 'kept_gathering', 'given_back', 'all_predicates', 'waits_for', 'masked', 'translate', 'one', 'f2fp_f16', 'ldg_without_hints',
                                                             'kept_for_rows', 'whole_rows', 'half_of_pair', 'converted',
                                                             'from_memory', 'to_memory', 'frame_top', 'load_word', 'store_word', 'read_back')}
    for name, replacement in patched(control).items():
        setattr(Function, name, replacement)
    stall, relocations_of = translate.with_stall, translate.relocations_of
    # Slow-predicate controls.
    if control == 'slowno':
        # The wait is left out: delayed cleared.
        orig_stand_ins_sn = Function.stand_ins
        def no_delay_stand_ins(self):
            orig_stand_ins_sn(self)
            self.delayed = {}
        originals['stand_ins'] = Function.stand_ins
        Function.stand_ins = no_delay_stand_ins
    elif control == 'slowshort':
        # The wait is too short: 13 made 3, 26 made 15.
        orig_stand_ins_ss = Function.stand_ins
        def short_delay_stand_ins(self):
            orig_stand_ins_ss(self)
            for i in list(self.delayed):
                if self.delayed[i] >= 26:
                    self.delayed[i] = 15
                elif self.delayed[i] >= 13:
                    self.delayed[i] = 3
        originals['stand_ins'] = Function.stand_ins
        Function.stand_ins = short_delay_stand_ins
    elif control == 'slownowait':
        # The first NOP does not wait for all scoreboards. (The words are changed when the
        # function is assembled, `Function.finish`: `Function.translate` returns the bytes.)
        orig_finish_snw = Function.finish
        def nowait_finish(self):
            for item in self.items:
                if item.what == 'late predicate' and (item.word >> WAIT) & 63 == 63:
                    item.word &= ~(63 << WAIT)
                    break
            return orig_finish_snw(self)
        originals['finish'] = Function.finish
        Function.finish = nowait_finish
    elif control == 'slownever':
        # `never` not respected: force slow_predicates to 0 so the translator picks never-predicates.
        orig_stand_ins_snv = Function.stand_ins
        def no_never_stand_ins(self):
            orig_stand_ins_snv(self)
            self.slow_predicates = 0
            self.delayed = {}
        originals['stand_ins'] = Function.stand_ins
        Function.stand_ins = no_never_stand_ins
    elif control == 'slowsaved':
        # Drop the delay at the first instruction of a SAVED life whose stand-in is slow:
        # the writes within the run then have no coverage.
        orig_stand_ins_ssv = Function.stand_ins
        def no_saved_delay_stand_ins(self):
            orig_stand_ins_ssv(self)
            saved_first = set()
            for q, p, saved, part in self.lives:
                if saved and not isinstance(saved, str) and len(part) >= 2:
                    if self.slow_predicates >> p & 1:
                        saved_first.add(part[0])
            for i in list(self.delayed):
                if i in saved_first:
                    del self.delayed[i]
        originals['stand_ins'] = Function.stand_ins
        Function.stand_ins = no_saved_delay_stand_ins
    # Borrow controls: patch Borrowed methods.
    Borrowed = translate.Borrowed
    borrow_saved = {}
    # (The pair is what `Borrowed.free_at` gives for a width of two: the controls on it change that
    # answer and leave the registers taken one by one alone.)
    if control == 'alive_pair':
        borrow_saved['free_at'] = Borrowed.free_at
        orig = Borrowed.free_at
        def bad_pair(self, i, width=1, given=0):
            good = orig(self, i, width, given)
            if good is None or width != 2:
                return good
            held = self.busy[i] | self.taken[i] | given
            for r in range(2, self.top + 2 - width, 2):
                if any(held >> (r + k) & 1 for k in range(width)):
                    return r
            return good
        Borrowed.free_at = bad_pair
    elif control == 'alive_home':
        borrow_saved['__init__'] = Borrowed.__init__
        orig_init = Borrowed.__init__
        def bad_init(self, f, *args, **kwargs):
            orig_init(self, f, *args, **kwargs)
            for u, r in list(self.homes.items()):
                if r + 2 <= self.top + 1:
                    self.homes[u] = r + 1
                    self.used.discard(r)
                    self.used.add(r + 1)
        Borrowed.__init__ = bad_init
    elif control == 'no_borrow_wait':
        # Where the first instruction of what an instruction became waits for all six scoreboards
        # (it writes a borrowed register), it waits for what the original waited for only.
        orig_finish_nbw = Function.finish
        def no_wait_finish(self):
            if self.borrow is not None:
                for item in self.items:
                    if item.first and item.changed and (item.word >> WAIT) & 63 == 63:
                        src_wait = (self.src[item.origin].word >> WAIT) & 63
                        item.word = (item.word & ~(63 << WAIT)) | (src_wait << WAIT)
            return orig_finish_nbw(self)
        originals['finish'] = Function.finish
        Function.finish = no_wait_finish
    elif control == 'no_late_borrow':
        # An instruction of the program that was made to say when it has read (it reads a
        # borrowed register late) does not say it.
        orig_finish_nlb = Function.finish
        def no_late_finish(self):
            if self.borrow is not None and self.borrow.used:
                for item in self.items:
                    if item.says:
                        item.word |= 7 << READ_BARRIER
                        item.says = False
            return orig_finish_nlb(self)
        originals['finish'] = Function.finish
        Function.finish = no_late_finish
    elif control == 'shared_home':
        borrow_saved['__init__'] = Borrowed.__init__
        orig_init_sh = Borrowed.__init__
        def no_taken_init(self, f, *args, **kwargs):
            orig_init_sh(self, f, *args, **kwargs)
            # Clear `taken` so that two URs may share a home.
            self.taken = [0] * len(self.taken)
        Borrowed.__init__ = no_taken_init
    elif control == 'held_busy_l3':
        borrow_saved['kept'] = Borrowed.kept
        orig_kept_busy = Borrowed.kept
        def busy_kept(self, span):
            # A register where the program holds something at an instruction of the span, and
            # that the translation has not taken there.
            union = taken = 0
            for i in span:
                union |= self.busy[i]
                taken |= self.taken[i]
            for r in range(self.top, 1, -1):
                if union >> r & 1 and not taken >> r & 1:
                    for i in span:
                        self.taken[i] |= 1 << r
                    self.used.add(r)
                    return r
            return orig_kept_busy(self, span)
        Borrowed.kept = busy_kept
    elif control == 'shared_holder_l3':
        borrow_saved['kept'] = Borrowed.kept
        orig_kept_shared = Borrowed.kept
        def shared_kept(self, span):
            # The register of the first predicate held, for every one after it where the
            # program has nothing in it: whatever else of the translation is there.
            first = getattr(self, '_shared_holder', None)
            if first is not None and not any(self.busy[i] >> first & 1 for i in span):
                for i in span:
                    self.taken[i] |= 1 << first
                return first
            r = orig_kept_shared(self, span)
            if first is None:
                self._shared_holder = r
            return r
        Borrowed.kept = shared_kept
    elif control == 'unkept_holder_l3':
        borrow_saved['kept'] = Borrowed.kept
        orig_kept_free = Borrowed.kept
        def free_kept(self, span):
            r = orig_kept_free(self, span)
            if r is not None:
                for i in span:
                    self.taken[i] &= ~(1 << r)
            return r
        Borrowed.kept = free_kept
    elif control in ('overlap_pool', 'pool_in_home_unsaid'):
        borrow_saved['free_at'] = Borrowed.free_at
        orig_pair_ov = Borrowed.free_at
        def ignore_taken_pair(self, i, width=1, given=0):
            if width != 2:
                return orig_pair_ov(self, i, width, given)
            # Ignore `taken` so the pool may overlap an alive home.
            held = self.busy[i] | given
            for r in range(2, self.top + 2 - width, 2):
                if not held >> r & ((1 << width) - 1):
                    return r
            return None
        Borrowed.free_at = ignore_taken_pair
    elif control == 'written_pair':
        borrow_saved['free_at'] = Borrowed.free_at
        borrow_saved['__init__'] = Borrowed.__init__
        orig_pair = Borrowed.free_at
        orig_init_wp = Borrowed.__init__
        def init_save_writes(self, f, *args, **kwargs):
            orig_init_wp(self, f, *args, **kwargs)
            self._src_writes = [0] * len(f.src)
            for i, d in enumerate(f.src):
                for l, x in d.writes:
                    if l == 'R' and x != 255:
                        self._src_writes[i] |= 1 << x
        Borrowed.__init__ = init_save_writes
        def pick_written(self, i, width=1, given=0):
            good = orig_pair(self, i, width, given)
            if good is None or width != 2:
                return good
            # (`busy` has what the instruction writes: a pair that it writes is looked for among
            # those that no home and nothing given already has.)
            held = self.taken[i] | given
            writes = self._src_writes[i] if hasattr(self, '_src_writes') else 0
            mask = (1 << width) - 1
            for r in range(2, self.top + 2 - width, 2):
                if not held >> r & mask and any(writes >> (r + k) & 1 for k in range(width)):
                    return r
            return good
        Borrowed.free_at = pick_written
    given, source = path, open(path, 'rb').read()
    if control == 'batched':
        translate.with_stall = lambda word, cycles, marks=False: stall(word, cycles)
    if control == 'named':
        source = naming_code(source)
        if source is None:
            return None, ''
        translate.relocations_of = lambda *args: (relocations_of(*args)[0], set(), set())
    if control in ('complement_rewritten', 'complement_same'):
        source = complemented(mc, source, control == 'complement_same', untouched[(path, 'rest')]['listing'])
        if source is None:
            return None, ''
    try:
        listing, lives = [], []
        out, report = translate.translate(source, mc, listing, lives)
    finally:
        translate.with_stall, translate.relocations_of = stall, relocations_of
        for name, original in originals.items():
            setattr(Function, name, original)
        for name, original in borrow_saved.items():
            setattr(Borrowed, name, original)
    if control == 'as it is':
        untouched[path] = out
        untouched[(path, 'rest')] = {'listing': list(listing), 'lives': [frozen(row) for row in lives], 'report': dict(report)}
        if not validated:
            return None, ''
    elif control == 'unlisted':
        # The same translation; the map without the last instruction of each function.
        last = {}
        for k, row in enumerate(listing):
            last[row[0]] = k
        listing = [row for k, row in enumerate(listing) if k not in set(last.values())]
    elif control == 'swapped':
        # In each function, the first two instructions that follow each other, come from two
        # instructions, are unchanged and are no branches: exchanged, in the code and in the map.
        container = elf.Elf(out)
        rows_of = collections.defaultdict(list)
        for k, row in enumerate(listing):
            rows_of[row[0]].append(k)
        for name, ks in rows_of.items():
            section = container.section('.text.' + name)
            data = bytearray(section.data)
            for x, y in zip(ks, ks[1:]):
                (_, nx, ox, wx), (_, ny, oy, wy) = listing[x], listing[y]
                dx, dy = (mc.decode(70, int.from_bytes(data[n:n + 16], 'little')) for n in (nx, ny))
                if wx == wy == 'same' and ox != oy and ny == nx + 16 and None not in (dx, dy) and 'BRU' not in (dx.kind, dy.kind) and data[nx:nx + 16] != data[ny:ny + 16]:
                    data[nx:nx + 16], data[ny:ny + 16] = data[ny:ny + 16], data[nx:nx + 16]
                    listing[x], listing[y] = (name, nx, oy, 'same'), (name, ny, ox, 'same')
                    break
            section.data = bytes(data)
        out = container.write()
    elif control == 'unmoved':
        # The translation as it is; the relocations on code where they were in the original.
        container, before = elf.Elf(out), elf.Elf(source)
        for s in container.sections:
            if s.type in (elf.SHT_REL, elf.SHT_RELA) and s.info < len(container.sections) and container.sections[s.info].name.startswith('.text.'):
                s.data = before.section(s.name).data
        out = container.write()
        if out == untouched[path]:
            return None, ''
    elif control in ('named', 'complement_rewritten', 'complement_same'):
        # The check is given the original that was translated: the one with the relocation, or the
        # instruction, changed.
        given = os.path.join(out_dir, 'original.' + os.path.basename(path))
        open(given, 'wb').write(source)
    elif control == 'no_c3_wait':
        # C3 control: find a pool register that the function writes late somewhere, at an
        # original instruction that has NO mem: store after it. Clear the 63-wait. (Until
        # 2026-11-23 the registers written late, the pool lines and the parts of the instructions
        # were those of every function at once: it cleared a wait in one function at the offset of
        # another's.)
        container = elf.Elf(out)
        # Build: program registers written late, by function (original sm_75 code).
        orig_data = open(path, 'rb').read()
        orig_e = elf.Elf(orig_data)
        late_written = collections.defaultdict(set)
        for s in orig_e.sections:
            if s.name.startswith('.text.') and s.type == 1:
                for at in range(0, len(s.data), 16):
                    w75 = int.from_bytes(s.data[at:at+16], 'little')
                    d75 = mc.decode(75, w75)
                    if d75 is None:
                        continue
                    wb = (w75 >> WRITE_BARRIER) & 7
                    if wb != 7 and d75.kind in machine.LATE:
                        for l, r in d75.writes:
                            if l == 'R':
                                late_written[s.name[len('.text.'):]].add(r)
        # Find pool declarations at instructions without stores.
        by_old = collections.defaultdict(list)
        for row in listing:
            by_old[(row[0], row[2])].append((row[1], row[3]))
        pool_at = {}  # {(function, old_offset): set of registers}
        for entry in lives:
            if len(entry) >= 4 and entry[3] == 'pool':
                for off in entry[4]:
                    pool_at.setdefault((entry[0], off), set()).update(range(entry[2], entry[2] + entry[1]))
        changed = False
        for func_name in sorted({row[0] for row in listing}):
            section = container.section('.text.' + func_name)
            if section is None:
                continue
            data = bytearray(section.data)
            for function, old_off in sorted(pool_at):
                regs = pool_at[(function, old_off)]
                if function != func_name or not regs & late_written[func_name]:
                    continue
                parts = by_old.get((func_name, old_off), [])
                # No store AND no mem: at all (so B3 and the existing wait check cannot report).
                has_mem = any(w.startswith('mem:') for _, w in parts)
                if has_mem:
                    continue
                # Clear the 63-wait from every instruction of this sequence.
                for new_off, what in sorted(parts):
                    word = int.from_bytes(data[new_off:new_off+16], 'little')
                    if (word >> WAIT) & 63 == 63:
                        word &= ~(63 << WAIT)
                        data[new_off:new_off+16] = word.to_bytes(16, 'little')
                        changed = True
                        break
                if changed:
                    break
            if changed:
                section.data = bytes(data)
                break
        if changed:
            out = container.write()
        if out == untouched[path]:
            return None, ''
    elif control == 'no_lane_l3':
        # Drop the S2R for the per-instruction lane: replace with NOP in the binary,
        # keep the map entry (so the validator sees the kind 'lane' but the instruction is a NOP).
        container = elf.Elf(out)
        nop = machine.NOP.to_bytes(16, 'little')
        changed = False
        for row in listing:
            name, new_off, old_off, what = row
            if what == 'lane' and old_off != 0:
                section = container.section('.text.' + name)
                data = bytearray(section.data)
                data[new_off:new_off + 16] = nop
                section.data = bytes(data)
                changed = True
        if not changed:
            return None, ''
        out = container.write()
        if out == untouched[path]:
            return None, ''
    elif control == 'no_r1_lower':
        # Remove the IADD3 R1, R1, -frame, RZ instruction from the binary and the map.
        container = elf.Elf(out)
        new_listing = []
        for row in listing:
            name, new_off, old_off, what = row
            if what.startswith('mem: R1 -='):
                # Remove this instruction from code and shift everything after it.
                section = container.section('.text.' + name)
                data = bytearray(section.data)
                # Replace the IADD3 with a NOP.
                nop = machine.NOP.to_bytes(16, 'little')
                data[new_off:new_off + 16] = nop
                section.data = bytes(data)
                # Keep the map entry but it's now a NOP.
            new_listing.append(row)
        listing[:] = new_listing
        out = container.write()
        if out == untouched[path]:
            return None, ''
    elif control == 'wrong_r1_lower':
        # R1 lowered by the wrong amount: double the frame size in the IADD3.
        container = elf.Elf(out)
        for row in listing:
            name, new_off, old_off, what = row
            if what.startswith('mem: R1 -='):
                section = container.section('.text.' + name)
                data = bytearray(section.data)
                word = int.from_bytes(data[new_off:new_off + 16], 'little')
                d70 = mc.decode(70, word)
                if d70 and d70.mnemonic == 'IADD3':
                    sb = d70.ops.get('Sb', 0)
                    new_word = mc.assemble('iadd3_imm__RsIR_RIR', Rd=1, Ra=1, Sb=sb * 2, Rc=255)
                    # Preserve the scheduling bits.
                    sched_mask = 0x3FFFF << machine.USCHED
                    new_word = (new_word & ~sched_mask) | (word & sched_mask)
                    data[new_off:new_off + 16] = new_word.to_bytes(16, 'little')
                    section.data = bytes(data)
                    break
        out = container.write()
        if out == untouched[path]:
            return None, ''
    elif control == 'no_load_wait':
        # Clear the wait bit on the instruction after the last mem: load of each instruction.
        # (Until 2026-11-23 it patched the section of the first function at the offsets of the
        # others, and tested bit 0 of the wait mask, not that of the scoreboard it clears.)
        container = elf.Elf(out)
        by_old = collections.defaultdict(list)
        for row in listing:
            by_old[(row[0], row[2])].append((row[1], row[3]))
        changed = False
        for (name, old_off), parts in by_old.items():
            loads = [(n, w) for n, w in sorted(parts) if w.startswith('mem: load ')]
            if not loads:
                continue
            last_load_new = loads[-1][0]
            # Find the instruction right after the last load.
            all_news = sorted(n for n, w in parts)
            after_idx = all_news.index(last_load_new) + 1 if last_load_new in all_news else -1
            if after_idx < len(all_news):
                next_new = all_news[after_idx]
            else:
                next_new = last_load_new + 16
            section = container.section('.text.' + name)
            if section and next_new + 16 <= len(section.data):
                data = bytearray(section.data)
                # Find the write scoreboard of the last load.
                lw = int.from_bytes(data[last_load_new:last_load_new + 16], 'little')
                wb = (lw >> WRITE_BARRIER) & 7
                if wb != 7:
                    nw = int.from_bytes(data[next_new:next_new + 16], 'little')
                    if nw >> (WAIT + wb) & 1:
                        nw &= ~(1 << (WAIT + wb))
                        data[next_new:next_new + 16] = nw.to_bytes(16, 'little')
                        section.data = bytes(data)
                        changed = True
        if changed:
            out = container.write()
        if out == untouched[path]:
            return None, ''
    elif control == 'up_unwaited_l3':
        # The instruction after each load of a uniform predicate from the frame, which reads it
        # back, does not wait for the load's scoreboard.
        container = elf.Elf(out)
        for name, new_off, old_off, what in listing:
            if what.startswith('mem: load UP'):
                section = container.section('.text.' + name)
                data = bytearray(section.data)
                board = (int.from_bytes(data[new_off:new_off + 16], 'little') >> WRITE_BARRIER) & 7
                after = int.from_bytes(data[new_off + 16:new_off + 32], 'little')
                if board != 7 and after >> (WAIT + board) & 1:
                    data[new_off + 16:new_off + 32] = (after & ~(1 << (WAIT + board))).to_bytes(16, 'little')
                    section.data = bytes(data)
        out = container.write()
        if out == untouched[path]:
            return None, ''
    elif control == 'no_attr_raise':
        # Undo the attribute raise: set 0x11, 0x12, 0x23 back to the original's values.
        container, before = elf.Elf(out), elf.Elf(source)
        for section_name in ['.nv.info'] + [s.name for s in container.sections if s.name.startswith('.nv.info.')]:
            s = container.section(section_name)
            orig_s = before.section(section_name)
            if s is not None and orig_s is not None:
                # Replace the attribute values for 0x11, 0x12, 0x23 with the original's.
                d = bytearray(s.data)
                orig_d = orig_s.data
                pos, opos = 0, 0
                while pos + 4 <= len(d) and opos + 4 <= len(orig_d):
                    fmt, attr, val = d[pos], d[pos+1], struct.unpack_from('<H', d, pos+2)[0]
                    ofmt, oattr, oval = orig_d[opos], orig_d[opos+1], struct.unpack_from('<H', orig_d, opos+2)[0]
                    if fmt == 4 and attr in (0x11, 0x12, 0x23) and pos + 12 <= len(d) and ofmt == 4 and oattr == attr and opos + 12 <= len(orig_d):
                        struct.pack_into('<I', d, pos + 8, struct.unpack_from('<I', orig_d, opos + 8)[0])
                    pos += 4 + val if fmt == 4 else 4
                    opos += 4 + oval if ofmt == 4 else 4
                s.data = bytes(d)
        out = container.write()
        if out == untouched[path]:
            return None, ''
    elif control == 'no_borrow_wait_l3':
        # At level 3: the first instruction of a sequence that writes a borrowed register
        # does NOT wait for all six scoreboards. Remove the 63 << WAIT from the first
        # instruction of each sequence that has mem: kinds.
        container = elf.Elf(out)
        funcs = {row[0] for row in listing}
        by_func = collections.defaultdict(list)
        for row in listing:
            by_func[row[0]].append(row)
        for name, rows in by_func.items():
            section = container.section('.text.' + name)
            if section is None:
                continue
            data = bytearray(section.data)
            by_old = collections.defaultdict(list)
            for _, new, old, what in rows:
                by_old[old].append((new, what))
            for old, parts in by_old.items():
                has_mem = any(w.startswith('mem: ') for _, w in parts)
                if not has_mem:
                    continue
                # Clear the 63 << WAIT from EVERY instruction of the sequence.
                for new, w in sorted(parts):
                    word = int.from_bytes(data[new:new + 16], 'little')
                    if (word >> WAIT) & 63 == 63:
                        word &= ~(63 << WAIT)
                        data[new:new + 16] = word.to_bytes(16, 'little')
            section.data = bytes(data)
        out = container.write()
        if out == untouched[path]:
            return None, ''
    elif control == 'short_home':
        # The first home whose declared life has an instruction, not its first nor its last,
        # that names no uniform register: that instruction is left out of the life.
        source = elf.Elf(open(path, 'rb').read())
        def names_none(function, at):
            section = source.section('.text.' + function)
            d = mc.decode(75, int.from_bytes(section.data[at:at + 16], 'little')) if section is not None else None
            return d is not None and not any(x[0] == 'U' for x in d.reads + d.writes)
        for k, row in enumerate(lives or []):
            if len(row) < 5 or row[3] != 'home' or len(row[4]) < 3:
                continue
            inside = [at for at in row[4][1:-1] if names_none(row[0], at)]
            if inside:
                lives[k] = tuple(row[:4]) + ([at for at in row[4] if at != inside[0]],)
                break
        else:
            return None, ''
    elif control == 'home_unsaid':
        # The first register that is the home of two uniform registers: the lines of the second.
        by_register = collections.defaultdict(list)
        for row in lives or []:
            if len(row) >= 5 and row[3] == 'home' and row[1] not in by_register[(row[0], row[2])]:
                by_register[(row[0], row[2])].append(row[1])
        twice = next(((name, r, us[1]) for (name, r), us in by_register.items() if len(us) > 1), None)
        if twice is None:
            return None, ''
        lives[:] = [row for row in lives if not (len(row) >= 5 and row[3] == 'home' and (row[0], row[2], row[1]) == twice)]
    elif control == 'pool_in_home_unsaid':
        # The lines that say a pair is taken where one of its registers is a home declared there
        # are left out.
        homes = collections.defaultdict(set)
        for row in lives or []:
            if len(row) >= 5 and row[3] == 'home':
                homes[(row[0], row[2])].update(row[4])
        kept = []
        for row in lives or []:
            if len(row) >= 5 and row[3] == 'pool' and any(at in homes.get((row[0], row[2] + k), ()) for at in row[4] for k in range(row[1])):
                continue
            kept.append(row)
        if len(kept) == len(lives or []):
            return None, ''
        lives[:] = kept
    elif control == 'missing_pool_line':
        # Remove the last pool line from the .lives output that names a register of the program:
        # one above the highest that the original function names is the translation's own,
        # which it may write without saying. (Until 2026-11-23 the last pool line, whatever it
        # named; and with no lives at all, the translation as it was, counted as changed.)
        original = elf.Elf(source)
        def highest(function):
            section = original.section('.text.' + function)
            top = 0
            for at in range(0, len(section.data) if section is not None else 0, 16):
                d = mc.decode(75, int.from_bytes(section.data[at:at + 16], 'little'))
                if d is not None:
                    top = max([top] + [r for letter, r, *role in d.reads + d.writes if letter == 'R'])
            return top
        for k in range(len(lives or []) - 1, -1, -1):
            if len(lives[k]) >= 5 and lives[k][3] == 'pool' and 2 <= lives[k][2] <= highest(lives[k][0]):
                lives.pop(k)
                break
        else:
            return None, ''
    elif control == 'room_lane_unsaid':
        # The first lane read into a register of a pool line at an instruction: that register is
        # declared put away there (a room line) instead of taken.
        container, changed = elf.Elf(out), False
        for name, new_off, old_off, what in listing:
            if what != 'lane' or old_off == 0:
                continue
            word = int.from_bytes(container.section('.text.' + name).data[new_off:new_off + 16], 'little')
            d70 = mc.decode(70, word)
            r = d70.ops.get('Rd') if d70 is not None and d70.mnemonic == 'S2R' else None
            k = next((k for k, row in enumerate(lives or []) if len(row) >= 5 and row[0] == name and row[3] == 'pool'
                      and old_off in row[4] and r is not None and row[2] <= r < row[2] + row[1]), None)
            if k is None:
                continue
            _, width, first, _, offsets = lives[k]
            lives[k:k + 1] = ([(name, r - first, first, 'pool', offsets)] if r > first else []) + \
                ([(name, first + width - r - 1, r + 1, 'pool', offsets)] if r + 1 < first + width else []) + \
                [(name, 1, r, 'room', [old_off])]
            changed = True
            break
        if not changed:
            return None, ''
    # ---- room controls (VF_BORROW>=5, room made) ----
    elif control == 'room_reads_orig':
        # A register put away that the original instruction reads: replace the first room: away's
        # register with one the original reads. In the binary: change the STL's source register.
        container = elf.Elf(out)
        orig_data = open(path, 'rb').read()
        orig_e = elf.Elf(orig_data)
        changed = False
        for row in listing:
            name, new_off, old_off, what = row
            if not what.startswith('room: R') or 'away' not in what:
                continue
            # Find what the original reads.
            sec = orig_e.section('.text.' + name)
            if sec is None or old_off + 16 > len(sec.data):
                continue
            d75 = mc.decode(75, int.from_bytes(sec.data[old_off:old_off + 16], 'little'))
            if d75 is None:
                continue
            orig_reads = {r for l, r, *_ in d75.reads if l == 'R' and r != 255 and r >= 2}
            if not orig_reads:
                continue
            victim = min(orig_reads)
            # Patch the STL to store this register instead.
            section = container.section('.text.' + name)
            data = bytearray(section.data)
            word = int.from_bytes(data[new_off:new_off + 16], 'little')
            d70 = mc.decode(70, word)
            if d70 and d70.mnemonic == 'STL':
                new_word = mc.assemble('stl__sImmOffset', Ra=d70.ops['Ra'], Rb=victim,
                                       Ra_offset=d70.ops.get('Ra_offset', 0))
                sched_mask = (63 << WAIT) | (7 << READ_BARRIER) | (7 << WRITE_BARRIER) | (0xF << machine.USCHED)
                new_word = (new_word & ~sched_mask) | (word & sched_mask)
                data[new_off:new_off + 16] = new_word.to_bytes(16, 'little')
                section.data = bytes(data)
                changed = True
                break
        if not changed:
            return None, ''
        out = container.write()
    elif control == 'room_home_alive':
        # A register put away that is a home alive at that instruction: add a fake home line
        # to .lives at the room instruction's offset for the room register itself. The validator's
        # rule S should report "register Rn is a home alive at this instruction".
        changed = False
        room_entries = [(k, e) for k, e in enumerate(lives) if len(e) >= 4 and e[3] == 'room']
        if room_entries:
            k, entry = room_entries[0]
            func_name, _, reg, _, offsets = entry
            # Add a home line for this register at this offset.
            lives.append((func_name, 99, reg, 'home', offsets))
            changed = True
        if not changed:
            return None, ''
    elif control == 'room_no_back':
        # The back (LDL) left out: replace with NOP.
        container = elf.Elf(out)
        nop = machine.NOP.to_bytes(16, 'little')
        changed = False
        for row in listing:
            name, new_off, old_off, what = row
            if what.startswith('room: R') and 'back' in what:
                section = container.section('.text.' + name)
                data = bytearray(section.data)
                data[new_off:new_off + 16] = nop
                section.data = bytes(data)
                changed = True
                break
        if not changed:
            return None, ''
        out = container.write()
    elif control == 'room_guarded_away':
        # The first away (STL) under @P0.
        container = elf.Elf(out)
        plain = mc.assemble('stl__sImmOffset', Ra=1, Rb=2, Ra_offset=0)
        under = mc.assemble('stl__sImmOffset', Ra=1, Rb=2, Ra_offset=0, Pg=0, Pg_not=0)
        changed = False
        for name, new_off, old_off, what in listing:
            if what.startswith('room: R') and 'away' in what:
                section = container.section('.text.' + name)
                word = int.from_bytes(section.data[new_off:new_off + 16], 'little')
                d70 = mc.decode(70, word)
                if d70 is not None and d70.mnemonic == 'STL' and d70.ops.get('Pg', 7) == 7:
                    section.data = section.data[:new_off] + (word ^ plain ^ under).to_bytes(16, 'little') + section.data[new_off + 16:]
                    changed = True
                    break
        if not changed:
            return None, ''
        out = container.write()
    elif control == 'room_lane_first':
        # At the first instruction whose lane is read into a register put away there, right
        # after the stores that put registers away: the S2R before them, with the waits of the
        # first of them, which waits for the S2R; what followed the S2R waits for the last store.
        container, changed = elf.Elf(out), False
        rooms = {(row[0], row[2], at) for row in lives or [] if len(row) >= 5 and row[3] == 'room' for at in row[4]}
        by_old = collections.defaultdict(list)
        for k, (name, new_off, old_off, what) in enumerate(listing):
            by_old[(name, old_off)].append((new_off, what, k))
        for (name, old_off), parts in sorted(by_old.items()):
            parts.sort()
            lane = next(((n, k) for n, w, k in parts if w == 'lane'), None)
            if lane is None:
                continue
            section = container.section('.text.' + name)
            data = bytearray(section.data)
            p = lane[0]
            d70 = mc.decode(70, int.from_bytes(data[p:p + 16], 'little'))
            if d70 is None or d70.mnemonic != 'S2R' or (name, d70.ops.get('Rd'), old_off) not in rooms:
                continue
            block = [(n, w, k) for n, w, k in parts if n < p]
            aways = [(n, w, k) for n, w, k in block if w.startswith('room: R') and w.endswith(' away')]
            if not aways or 'room: R%d away' % d70.ops['Rd'] not in [w for n, w, k in aways]:
                continue
            a0 = aways[0][0]
            rows = [(n, w, k) for n, w, k in parts if a0 <= n <= p]
            if [n for n, w, k in rows] != list(range(a0, p + 16, 16)) or rows[:-1] != [x for x in aways if x[0] >= a0]:
                continue
            words = [int.from_bytes(data[n:n + 16], 'little') for n in range(a0, p + 16, 16)]
            board, last = (words[-1] >> WRITE_BARRIER) & 7, (words[-2] >> READ_BARRIER) & 7
            s2r = (words[-1] & ~(63 << WAIT)) | (words[0] & (63 << WAIT))
            first = (words[0] & ~(63 << WAIT)) | (1 << (WAIT + board))
            for n, word in zip(range(a0, p + 16, 16), [s2r, first] + words[1:-1]):
                data[n:n + 16] = word.to_bytes(16, 'little')
            if last != 7 and p + 32 <= len(data):
                after = int.from_bytes(data[p + 16:p + 32], 'little') | 1 << (WAIT + last)
                data[p + 16:p + 32] = after.to_bytes(16, 'little')
            whats = ['lane'] + [w for n, w, k in rows[:-1]]
            for (n, w, k), what in zip(rows, whats):
                listing[k] = (name, n, old_off, what)
            section.data = bytes(data)
            changed = True
            break
        if not changed:
            return None, ''
        out = container.write()
    elif control == 'room_back_wrong_word':
        # The back loads from another word (offset + 4).
        container = elf.Elf(out)
        changed = False
        for row in listing:
            name, new_off, old_off, what = row
            if what.startswith('room: R') and 'back' in what:
                section = container.section('.text.' + name)
                data = bytearray(section.data)
                word = int.from_bytes(data[new_off:new_off + 16], 'little')
                d70 = mc.decode(70, word)
                if d70 and d70.mnemonic == 'LDL':
                    offset = d70.ops.get('Ra_offset', 0)
                    new_word = mc.assemble('ldl__sImmOffset', Rd=d70.ops['Rd'], Ra=d70.ops['Ra'],
                                           Ra_offset=(offset + 4) & 0xFFFFFF,
                                           dst_wr_sb=d70.ops.get('dst_wr_sb', 7))
                    sched_mask = (63 << WAIT) | (0xF << machine.USCHED)
                    new_word = (new_word & ~sched_mask) | (word & sched_mask)
                    data[new_off:new_off + 16] = new_word.to_bytes(16, 'little')
                    section.data = bytes(data)
                    changed = True
                    break
        if not changed:
            return None, ''
        out = container.write()
    elif control == 'room_away_after_write':
        # The away (STL) moved after the first instruction that writes the room register:
        # swap the away with the first core instruction of the same original.
        container = elf.Elf(out)
        by_old = collections.defaultdict(list)
        for row in listing:
            by_old[(row[0], row[2])].append((row[1], row[3]))
        changed = False
        for (fname, old_off), parts in by_old.items():
            aways = [(n, w) for n, w in sorted(parts) if w.startswith('room: R') and 'away' in w]
            core = [(n, w) for n, w in sorted(parts) if not w.startswith('room:') and not w.startswith('mem:') and w not in ('wait', 'save', 'restore', 'late predicate', 'lane', 'kept', 'held: back', 'held: kept', 'settled', 'to pairs', 'from pairs')]
            if aways and core:
                # Swap the first away with the first core instruction.
                section = container.section('.text.' + fname)
                data = bytearray(section.data)
                a_off, c_off = aways[0][0], core[0][0]
                data[a_off:a_off+16], data[c_off:c_off+16] = data[c_off:c_off+16], data[a_off:a_off+16]
                # Also swap in the listing.
                for k, row in enumerate(listing):
                    if row[0] == fname and row[1] == a_off:
                        listing[k] = (fname, a_off, old_off, core[0][1])
                    elif row[0] == fname and row[1] == c_off:
                        listing[k] = (fname, c_off, old_off, aways[0][1])
                section.data = bytes(data)
                changed = True
                break
        if not changed:
            return None, ''
        out = container.write()
    elif control == 'room_same_word':
        # Two registers put away in the same word: set the second away's offset to the first's.
        container = elf.Elf(out)
        by_old = collections.defaultdict(list)
        for row in listing:
            by_old[(row[0], row[2])].append((row[1], row[3]))
        changed = False
        for (fname, old_off), parts in by_old.items():
            aways = [(n, w) for n, w in sorted(parts) if w.startswith('room: R') and 'away' in w]
            if len(aways) >= 2:
                section = container.section('.text.' + fname)
                data = bytearray(section.data)
                # Read offset of first away.
                w1 = int.from_bytes(data[aways[0][0]:aways[0][0]+16], 'little')
                d1 = mc.decode(70, w1)
                if d1 and d1.mnemonic == 'STL':
                    off1 = d1.ops.get('Ra_offset', 0)
                    # Set the second away to the same offset.
                    w2 = int.from_bytes(data[aways[1][0]:aways[1][0]+16], 'little')
                    d2 = mc.decode(70, w2)
                    if d2 and d2.mnemonic == 'STL':
                        new_word = mc.assemble('stl__sImmOffset', Ra=d2.ops['Ra'], Rb=d2.ops['Rb'], Ra_offset=off1)
                        sched_mask = (63 << WAIT) | (7 << READ_BARRIER) | (7 << WRITE_BARRIER) | (0xF << machine.USCHED)
                        new_word = (new_word & ~sched_mask) | (w2 & sched_mask)
                        data[aways[1][0]:aways[1][0]+16] = new_word.to_bytes(16, 'little')
                        section.data = bytes(data)
                        changed = True
                        break
        if not changed:
            return None, ''
        out = container.write()
    elif control == 'room_at_branch':
        # A register is declared put away at an instruction where nothing puts it away: a room
        # line is added to what the translator says, for R2 at the first branch or exit of a
        # function that has room. (A register declared so may be written by the sequence of its
        # instruction while it holds something: the validator must ask for the store and the
        # load, rule U of validate.py. Until 2026-11-20 it took the declaration on trust.)
        orig_data = open(path, 'rb').read()
        orig_e = elf.Elf(orig_data)
        added = False
        for entry in lives:
            if len(entry) >= 4 and entry[3] == 'room':
                # Found a room entry; now find a branch in the same function.
                func_name = entry[0]
                sec = orig_e.section('.text.' + func_name)
                if sec is None:
                    continue
                for at in range(0, len(sec.data), 16):
                    d75 = mc.decode(75, int.from_bytes(sec.data[at:at+16], 'little'))
                    if d75 is not None and d75.kind == 'BRU' and d75.mnemonic in ('BRA', 'EXIT'):
                        # Add a room line at this branch instruction.
                        lives.append((func_name, 1, 2, 'room', [at]))
                        added = True
                        break
                if added:
                    break
        if not added:
            return None, ''
    elif control == 'room_depth_wrong':
        # Depth wrong by the program's own frame (k_param_table): change the offset of the first
        # room away STL to use depth=0 instead of the actual depth.
        container = elf.Elf(out)
        orig_data = open(path, 'rb').read()
        orig_e = elf.Elf(orig_data)
        changed = False
        for row in listing:
            name, new_off, old_off, what = row
            if what.startswith('room: R') and 'away' in what:
                section = container.section('.text.' + name)
                data = bytearray(section.data)
                word = int.from_bytes(data[new_off:new_off + 16], 'little')
                d70 = mc.decode(70, word)
                if d70 and d70.mnemonic == 'STL':
                    offset = d70.ops.get('Ra_offset', 0)
                    # Subtract the depth to make it wrong (pretend depth is 0).
                    # Find the original depth at this instruction.
                    sec = orig_e.section('.text.' + name)
                    if sec is None:
                        continue
                    # Compute the original's R1 movements: find IADD3 R1, R1, imm before old_off.
                    depth = 0
                    for at in range(16, min(old_off + 16, len(sec.data)), 16):
                        w75 = int.from_bytes(sec.data[at:at+16], 'little')
                        d75 = mc.decode(75, w75)
                        if d75 is not None and d75.c.name == 'iadd3_imm__RsIR_RIR' and d75.ops.get('Rd') == 1 and d75.ops.get('Ra') == 1 and d75.ops.get('Rc', 255) == 255:
                            imm = d75.ops['Sb'] - (1 << 32) if d75.ops['Sb'] >> 31 else d75.ops['Sb']
                            depth -= imm
                    if depth == 0:
                        continue
                    # Change offset to what it would be with depth=0 (subtract depth).
                    new_offset = offset - depth
                    if new_offset < 0:
                        new_offset = 0
                    new_word = mc.assemble('stl__sImmOffset', Ra=d70.ops['Ra'], Rb=d70.ops['Rb'],
                                           Ra_offset=new_offset)
                    sched_mask = (63 << WAIT) | (7 << READ_BARRIER) | (7 << WRITE_BARRIER) | (0xF << machine.USCHED)
                    new_word = (new_word & ~sched_mask) | (word & sched_mask)
                    data[new_off:new_off + 16] = new_word.to_bytes(16, 'little')
                    section.data = bytes(data)
                    changed = True
                    break
        if not changed:
            return None, ''
        out = container.write()
    elif control == 'room_frame_short':
        # Frame not raised by the words of room: set frame attributes back to just the slot size.
        container, before = elf.Elf(out), elf.Elf(source)
        # Compute the slot-only frame size.
        n_slots = 0
        for entry in lives:
            if len(entry) >= 4 and entry[3] == 'slot':
                n_slots += 1
        slot_frame = (4 * n_slots + 15) & ~15 if n_slots > 0 else 0
        has_room = any(len(e) >= 4 and e[3] == 'room' for e in lives)
        if not has_room or slot_frame == 0:
            return None, ''
        # Find the full frame size.
        full_frame = 0
        for entry in lives:
            if len(entry) >= 4 and entry[3] == 'frame':
                full_frame = entry[1]
                break
        if full_frame <= slot_frame:
            return None, ''
        # Change the frame entry in lives to the slot-only size.
        for k, entry in enumerate(lives):
            if len(entry) >= 4 and entry[3] == 'frame':
                lives[k] = (entry[0], slot_frame, entry[2], entry[3], entry[4])
                break
    elif control == 'room_missing_lives':
        # Remove all room lines from .lives.
        before_len = len(lives)
        lives[:] = [e for e in lives if not (len(e) >= 4 and e[3] == 'room')]
        if len(lives) == before_len:
            return None, ''
    elif control == 'room_no_back_wait':
        # Drop the wait for the last back's scoreboard: clear it on the instruction after.
        container = elf.Elf(out)
        by_old = collections.defaultdict(list)
        for row in listing:
            by_old[(row[0], row[2])].append((row[1], row[3]))
        changed = False
        for (fname, old_off), parts in by_old.items():
            backs = [(n, w) for n, w in sorted(parts) if w.startswith('room: R') and 'back' in w]
            if not backs:
                continue
            last_back = backs[-1][0]
            section = container.section('.text.' + fname)
            if section is None:
                continue
            data = bytearray(section.data)
            # Find the write scoreboard of the last back.
            bw = int.from_bytes(data[last_back:last_back + 16], 'little')
            wb = (bw >> WRITE_BARRIER) & 7
            if wb == 7:
                continue
            # Clear the wait on the next instruction.
            next_new = last_back + 16
            if next_new + 16 <= len(data):
                nw = int.from_bytes(data[next_new:next_new + 16], 'little')
                if (nw >> WAIT) >> wb & 1:
                    nw &= ~(1 << (WAIT + wb))
                    data[next_new:next_new + 16] = nw.to_bytes(16, 'little')
                    section.data = bytes(data)
                    changed = True
                    break
        if not changed:
            return None, ''
        out = container.write()
    elif control == 'room_scratch_busy':
        # A scratch register of a matrix load at a room instruction is neither free nor put away:
        # patch the 'thread' LOP3 (which writes the scratch register) to write R14 instead,
        # and patch the LEA instructions that read it. R14 is a register the program uses, not
        # put away. The validator's whole-truth check should report it.
        container = elf.Elf(out)
        by_old = collections.defaultdict(list)
        for row in listing:
            by_old[(row[0], row[2])].append((row[1], row[3]))
        changed = False
        for (fname, old_off), parts in by_old.items():
            has_room = any(w.startswith('room:') for _, w in parts)
            thread_parts = [(n, w) for n, w in sorted(parts) if w == 'thread']
            if not has_room or not thread_parts:
                continue
            section = container.section('.text.' + fname)
            if section is None:
                continue
            data = bytearray(section.data)
            # Read the thread instruction to find the destination register.
            tn = thread_parts[0][0]
            tw = int.from_bytes(data[tn:tn + 16], 'little')
            td = mc.decode(70, tw)
            if td is None or td.mnemonic != 'LOP3':
                continue
            old_rd = td.ops.get('Rd', -1)
            if old_rd < 2:
                continue
            # Replace the destination with R14 (a busy register).
            new_rd = 14
            # Patch the LOP3 destination: replace old_rd in the word's register field.
            # Use the decoded instruction to find the field and patch it directly.
            for kind, fields, field_data in mc.d70.plan(td.c):
                if kind == 'operand' and field_data == 'Rd':
                    tw = fields[0].put(tw, new_rd)
                    break
            data[tn:tn + 16] = tw.to_bytes(16, 'little')
            # Patch every LEA/IADD3 of this instruction that reads old_rd: replace with new_rd.
            address_parts = [(n, w) for n, w in sorted(parts) if w in ('address', 'address of the row')]
            for an, aw in address_parts:
                aw_word = int.from_bytes(data[an:an + 16], 'little')
                ad = mc.decode(70, aw_word)
                if ad is None:
                    continue
                did_patch = False
                for kind, fields, field_data in mc.d70.plan(ad.c):
                    if kind == 'operand' and isinstance(field_data, str) and field_data.startswith('R') and field_data != 'Rd':
                        if ad.ops.get(field_data) == old_rd:
                            aw_word = fields[0].put(aw_word, new_rd)
                            did_patch = True
                if did_patch:
                    data[an:an + 16] = aw_word.to_bytes(16, 'little')
            section.data = bytes(data)
            changed = True
            break
        if not changed:
            return None, ''
        out = container.write()
    elif control == 'copy_after_write':
        # Control for fix 3 of the exact-call check: in the original sm_75 binary, swap the
        # copy of the return-address register (IMAD.MOV.U32 R', RZ, RZ, mov_reg) with the
        # write that follows it (IMAD.MOV.U32 mov_reg, RZ, RZ, 0x0), so the write happens
        # first. The call must become inexact, and the pool registers must be reported as not
        # free (or the register liveness marked as not computed).
        orig_data = open(path, 'rb').read()
        orig_e = elf.Elf(orig_data)
        changed = False
        for s in orig_e.sections:
            if not s.name.startswith('.text.') or s.type != 1:
                continue
            data = bytearray(s.data)
            n = len(data) // 16
            # Find a RET.REL.NODEC in this function.
            for i in range(n):
                d = mc.decode(75, int.from_bytes(data[16*i:16*i+16], 'little'))
                if d is None or d.c.name != 'ret__REL':
                    continue
                rp = d.ops.get('Ra', -1)
                if rp < 0:
                    continue
                # Walk backward from the RET to find the copy (IMAD.MOV.U32 R', RZ, RZ, R).
                for j in range(i - 1, max(i - 32, -1), -1):
                    dj = mc.decode(75, int.from_bytes(data[16*j:16*j+16], 'little'))
                    if dj is None:
                        break
                    # Is this the copy? IMAD.MOV (or MOV) where Rd = rp, Ra = RZ, Rb = RZ.
                    # On sm_75, IMAD.MOV.U32 R24, RZ, RZ, R25 is class imad__RRR_RRR with
                    # Ra=255, Rb=255, Rc=source. MOV R24, R25 is mov__RR.
                    is_imad_copy = (dj.ops.get('Rd') == rp and dj.ops.get('Ra') == 255
                                    and dj.ops.get('Rb') == 255 and 'imad' in dj.c.name)
                    is_mov_copy = (dj.c.name.startswith('mov__') and dj.ops.get('Rd') == rp)
                    if is_imad_copy or is_mov_copy:
                        # The source register: Rc for IMAD, Rb or Ra for MOV.
                        if is_imad_copy:
                            mov_src = dj.ops.get('Rc', -1)
                        else:
                            mov_src = next((r for l, r, *_ in dj.reads if l == 'R' and r != 255), -1)
                        if mov_src < 2 or mov_src == 255:
                            continue
                        # Find the next write to mov_src after the copy.
                        for k in range(j + 1, min(j + 8, n)):
                            dk = mc.decode(75, int.from_bytes(data[16*k:16*k+16], 'little'))
                            if dk is None:
                                break
                            if any(l == 'R' and r == mov_src for l, r in dk.writes):
                                # Swap instructions j and k (put the write before the copy).
                                data[16*j:16*j+16], data[16*k:16*k+16] = data[16*k:16*k+16], data[16*j:16*j+16]
                                s.data = bytes(data)
                                changed = True
                                break
                        break
                    if any(l == 'R' and r == rp for l, r in dj.writes):
                        break
                if changed:
                    break
            if changed:
                break
        if not changed:
            return None, ''
        # Re-translate the patched original and validate against it.
        patched_orig = os.path.join(out_dir, 'patched.' + os.path.basename(path))
        open(patched_orig, 'wb').write(orig_e.write())
        listing2, lives2 = [], []
        out2, _ = translate.translate(open(patched_orig, 'rb').read(), mc, listing2, lives2)
        given = patched_orig
        out, listing, lives = out2, listing2, lives2
    elif control == 'product_busy_take_l3':
        # Add a pool entry at an HMMA instruction for a register that the program uses there
        # (Ra: the A operand, which IS busy): the translator says it took a register that holds
        # something.
        orig_data = open(path, 'rb').read()
        orig_e = elf.Elf(orig_data)
        added = False
        for entry in lives:
            if len(entry) >= 4 and entry[3] == 'pool':
                func_name = entry[0]
                for at in entry[4]:
                    sec = orig_e.section('.text.' + func_name)
                    if sec is None or at + 16 > len(sec.data):
                        continue
                    d75 = mc.decode(75, int.from_bytes(sec.data[at:at+16], 'little'))
                    if d75 is not None and 'hmma' in d75.c.name:
                        ra = d75.ops.get('Ra', -1)
                        if ra >= 2 and ra != 255:
                            lives.append((func_name, 1, ra, 'pool', [at]))
                            added = True
                            break
                if added:
                    break
        if not added:
            return None, ''
    elif control == 'product_no_s2r_wait_l3':
        # Clear the SEQUENCE wait from the first 'kept: low' instruction after a 'lane' at a
        # level-3 product: the first reader of the lane no longer waits for the S2R.
        container = elf.Elf(out)
        changed = False
        by_old = collections.defaultdict(list)
        for row in listing:
            by_old[(row[0], row[2])].append((row[1], row[3]))
        for (fname, old_off), parts in by_old.items():
            if not any(w == 'kept: low' for _, w in parts):
                continue
            # Find the first 'kept: low' instruction.
            for n, w in sorted(parts):
                if w == 'kept: low':
                    section = container.section('.text.' + fname)
                    if section is None:
                        break
                    data = bytearray(section.data)
                    word = int.from_bytes(data[n:n+16], 'little')
                    if ((word >> WAIT) & 63) >> translate.SEQUENCE & 1:
                        word &= ~(1 << (WAIT + translate.SEQUENCE))
                        data[n:n+16] = word.to_bytes(16, 'little')
                        section.data = bytes(data)
                        changed = True
                    break
            if changed:
                break
        if not changed:
            return None, ''
        out = container.write()
    elif control == 'load_ur_wrong_slot_l3':
        # At level 3: change the offset of the first LDL (mem: load URn) that is followed by a
        # matrix load sequence: add 4 to the offset (loads from the wrong slot). The validator's
        # B2 check should catch it.
        container = elf.Elf(out)
        changed = False
        by_old = collections.defaultdict(list)
        for row in listing:
            by_old[(row[0], row[2])].append((row[1], row[3]))
        for (fname, old_off), parts in by_old.items():
            has_load_ur = any(w.startswith('mem: load UR') for _, w in parts)
            has_group = any(w == 'group' for _, w in parts)
            if not has_load_ur or not has_group:
                continue
            for n, w in sorted(parts):
                if w.startswith('mem: load UR'):
                    section = container.section('.text.' + fname)
                    if section is None:
                        break
                    data = bytearray(section.data)
                    word = int.from_bytes(data[n:n+16], 'little')
                    d70 = mc.decode(70, word)
                    if d70 and d70.mnemonic == 'LDL':
                        offset = d70.ops.get('Ra_offset', 0)
                        new_word = mc.assemble('ldl__sImmOffset', Rd=d70.ops['Rd'], Ra=d70.ops['Ra'],
                                               Ra_offset=(offset + 4) & 0xFFFFFF,
                                               dst_wr_sb=d70.ops.get('dst_wr_sb', 7))
                        sched_mask = (63 << WAIT) | (0xF << machine.USCHED)
                        new_word = (new_word & ~sched_mask) | (word & sched_mask)
                        data[n:n+16] = new_word.to_bytes(16, 'little')
                        section.data = bytes(data)
                        changed = True
                    break
            if changed:
                break
        if not changed:
            return None, ''
        out = container.write()
    elif control == 'load_ur_no_wait_l3':
        # At level 3: clear the write scoreboard on the last LDL (mem: load URn) before a matrix
        # load sequence, so the next instruction does not wait for it. The validator's B2 check
        # (the wait on the last load) should catch it.
        container = elf.Elf(out)
        changed = False
        by_old = collections.defaultdict(list)
        for row in listing:
            by_old[(row[0], row[2])].append((row[1], row[3]))
        for (fname, old_off), parts in by_old.items():
            loads_ur = [(n, w) for n, w in sorted(parts) if w.startswith('mem: load UR')]
            has_group = any(w == 'group' for _, w in parts)
            if not loads_ur or not has_group:
                continue
            last_load = loads_ur[-1][0]
            section = container.section('.text.' + fname)
            if section is None:
                continue
            data = bytearray(section.data)
            word = int.from_bytes(data[last_load:last_load+16], 'little')
            wb = (word >> WRITE_BARRIER) & 7
            if wb != 7:
                # Clear the write scoreboard.
                word |= 7 << WRITE_BARRIER
                data[last_load:last_load+16] = word.to_bytes(16, 'little')
                # Also clear the wait on the next instruction.
                next_inst = last_load + 16
                if next_inst + 16 <= len(data):
                    nw = int.from_bytes(data[next_inst:next_inst+16], 'little')
                    nw &= ~(1 << (WAIT + wb))
                    data[next_inst:next_inst+16] = nw.to_bytes(16, 'little')
                section.data = bytes(data)
                changed = True
            if changed:
                break
        if not changed:
            return None, ''
        out = container.write()
    elif out == untouched[path]:
        return None, ''
    if control != 'as it is':
        rest = untouched[(path, 'rest')]
        if out == untouched[path] and listing == rest['listing'] and [frozen(row) for row in lives] == rest['lives']:
            return None, ''                 # (code, map and lives as they are: a control that found nothing to change)
        newly = refused_only(out, listing, lives, report, untouched[path], rest)
        if newly:
            return 'refused', '%d functions refused by the translator, among them %s' % (len(newly), min(newly)[:60])
    new = os.path.join(out_dir, os.path.basename(path))
    open(new, 'wb').write(out)
    with open(new + '.map', 'w') as f:
        for row in listing:
            f.write('%s\t%x\t%x\t%s\n' % row)
    translate.write_lives(new + '.map.lives', lives)
    said = subprocess.run([sys.executable, '-I', os.path.join(HERE, 'validate.py'), given, new, new + '.map'],
                          capture_output=True, text=True).stdout
    m = re.search(r'(\d+) instructions of (\d+) functions checked, (\d+) problems', said)
    if m is None:
        return -1, (said.strip().splitlines() or ['validate.py said nothing'])[-1]
    first = next((line.strip() for line in said.splitlines()[1:] if not re.match(r'\s*\d+ \(', line)), '')
    return int(m.group(3)), first


def main():
    args, chosen, paths, unchecked = sys.argv[1:], [], [], False
    while args:
        if args[0] == '-c' and len(args) > 1 and (args[1] in CONTROLS or args[1] in HALF_CONTROLS or args[1] in LEPC_CONTROLS):
            chosen.append(args[1])
            args = args[2:]
        elif args[0] == '-u':
            unchecked, args = True, args[1:]
        else:
            paths.append(args.pop(0))
    if not paths or any(p.startswith('-') for p in paths):
        print(__doc__)
        return 2
    mc = machine.Machine()
    ok, untouched = True, {}
    with tempfile.TemporaryDirectory(prefix='vf-controls-') as out_dir:
        print('%-10s %s' % ('control', '  '.join('%14s' % os.path.basename(p)[:14] for p in paths)))
        for control in CONTROLS + [c for c in HALF_CONTROLS + LEPC_CONTROLS if c not in CONTROLS]:
            if chosen and control not in chosen and control != 'as it is':
                continue
            if control in PAIR_CONTROLS and not translate.PAIRS:
                continue
            if control in BORROW_CONTROLS and not translate.BORROW:
                continue
            if control in FRAME_CONTROLS and translate.BORROW < 2:
                continue
            if control in PRODUCT_L3_CONTROLS and translate.BORROW < 7:
                continue
            if control == 'as it is' and unchecked:
                for path in paths:
                    run(mc, control, path, out_dir, untouched, validated=False)
                print('%-10s (made here, not validated: -u)' % control)
                continue
            results = [run(mc, control, path, out_dir, untouched) for path in paths]
            counts = [problems for problems, _ in results]
            changed = [c for c in counts if isinstance(c, int)]
            holes, right = unobjected(control, setting(), [os.path.basename(p) for p in paths], counts)
            good = (all(c == 0 for c in counts) if control == 'as it is'
                    else all(c >= 0 for c in changed) if control in CHANGES_THE_PROGRAM and changed
                    else (bool(changed) and all(c >= 0 for c in changed) and not holes))
            ok = ok and good
            verdict = 'as expected' if good else 'changes nothing in these cubins: name one that has what it takes away' if not changed else 'NOT AS EXPECTED'
            print('%-10s %s   %s' % (control, '  '.join('%14s' % ('=' if c is None else c if isinstance(c, str) else 'failed' if c < 0 else c) for c in counts), verdict))
            example = next((first for problems, first in results if problems and problems != 'refused' and first), None)
            if example:
                print('           e.g. %s' % re.sub(r'_Z\w+', '<function>', example)[:150])
            if control != 'as it is' and control not in CHANGES_THE_PROGRAM:
                if holes:
                    print('           not objected to in %s' % ', '.join(holes))
                if right:
                    print('           (and rightly not in %s: RIGHT_THERE)' % ', '.join(right))
    if not translate.PAIRS:
        print('(the controls of pairs on whole rows were not run: the translator makes none with VF_PAIRS=0)')
    if not translate.BORROW:
        print('(the controls of borrowed registers were not run: the translator does not borrow with VF_BORROW=0)')
    print('every control is objected to, and the translation as it is to nothing' if ok else 'NOT EVERY CONTROL IS OBJECTED TO (a hole in the check, a fault in the translation, or cubins that lack what a control takes away): see above')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
