#!/usr/bin/env python3
"""validate.py ORIGINAL.cubin TRANSLATED.cubin MAP: check a translation with nvdisasm, instruction by
instruction, without trusting the translator.

MAP (of translate.py) says, for each instruction of the result, which instruction of the original
it comes from and what was done. The map is the translator's own, so it is checked first: it must
have every instruction of the result once and every instruction of the original, for every
function of the result that is not what a refused function becomes (a trap, exits and nothing
else); in the order of the result the instructions must come from the original in its order; a
kind of rewriting that is not known here is a problem. (Until 2026-11-17 an instruction that the
map left out was not looked at, and two instructions exchanged passed: `reference/controls.py`
has a control for each.)

Whatever its kind, an instruction of the result that reads or writes memory must have the guard
of the instruction it comes from. (But what loads a uniform register from the frame before an
instruction and stores it after it: under no guard, rules B2 to B4.)

The original is read as sm_75 and the result as sm_70, and each instruction of the result must
read as its kind allows:
- `same`: as the original, with a branch target moved to where its instruction now is;
- `ordinary registers`: as the original with each uniform register replaced by an ordinary one,
  always the same one, above the registers the original function uses, and the mnemonic without U;
- the sequences (upper half of a product, minimum and maximum, stride, constant): the instructions
  they must be, on the operands of the original;
- what stands for a matrix instruction of Turing (a load, a product): the sequence its recipe
  gives, on registers above those of the original, each for one thing. It goes by the lane of
  the thread, read into such a register when the function is entered, and a product by four
  values made of the lane, computed there too: nothing else in the function writes those
  registers, and no global function symbol points into the function elsewhere than at its start. A
  product may take the elements of B from the registers where an earlier product gathered them,
  and not gather them again: then that product is the last that gathered, from the same register
  of B into the registers this one reads; nothing from it on, itself included, writes that
  register of B in the original, and nothing after its gathering writes those registers in the
  result; no instruction after it up to this product is entered otherwise than from the
  instruction before, no branch of the function leads nobody knows where, and no call is
  between them.

A uniform predicate may have several ordinary ones stand for it, one for each of its lives; the
translator says which (MAP.lives, next to MAP: function, uniform predicate, ordinary predicate,
`free` or `saved`, the offsets of the life), and what it says is checked against the lives of
predicates that nvdisasm works out itself for the original: every place where the uniform
predicate holds something is in a life; the ordinary one holds nothing of the program all along a
life (`free`), or is read and written by nothing there and is saved before the life and given back
after it (`saved`); no value goes from one life to another; an instruction uses the predicate of
the life it is in.
A life may also be `held` in a register of the translation (MAP.lives: the register, the offsets
of the whole life): then each instruction of it that names the uniform predicate has a life of
its own, of that instruction alone, and the value goes between them in the register: read back
into the instruction's predicate before every instruction that reads the uniform one or is under
a guard (`ISETP.NE`, the register against zero), written down after every one that writes it
(`P2R`, that predicate's bit alone); the register holds one uniform predicate at an instruction,
and nothing that the translation adds writes it along a held life but what writes the predicate
down (rule HW). It is above the program's, and nothing else of the translation uses it; or it is
taken among the registers that hold nothing: it then holds nothing of the program all along the
held life (rule H1), and may hold another uniform predicate along other lives.

Five things of the schedule are checked too, on the words of the result (nvdisasm does not print
them), with the translator's tables only for what kind an instruction is and what it reads:
- what an instruction of the original waited for, the first instruction of what it became waits
  for (a wait holds for everything after it: the compiler may put on one instruction the wait for
  what a later one reads);
- an instruction waits for a scoreboard two cycles or more after the instruction that names it
  (native code has no closer pair);
- an instruction that reads its operands some time after it is issued (a memory access, a
  conversion, arithmetic on doubles) and reads a register the translation gave it (one above
  those of the original) must say on a scoreboard when it has read, or when its result is there,
  and the very next instruction must wait for that (after instructions that do nothing, if the
  schedule put some there: `waited_after`, for this rule and for the ones of the frame, of
  room and of borrowed registers that ask the next instruction to wait).
- a predicate given back from a register of the translation (R2P) comes 9 cycles or more after
  the instructions before it, in straight-line code, that read that predicate;
- a register that a memory instruction reads as data (what a store stores, what a shuffle
  exchanges) is not written by arithmetic after it, in straight-line code, before a barrier or a
  wait for a scoreboard that it or a later memory instruction names (`data_overwritten`).
- a result of double precision (what DADD, DMUL, DFMA or DSETP writes) is read, along every
  path that the code's own branches give (not through a register, a BSYNC or a return), no
  earlier than Volta's own code reads one, whether or not a scoreboard announces it and
  the reader waits: 8 cycles later by an instruction of the same pipe (double or half
  precision), 9 by any other, and a predicate of DSETP 15 as a guard or by a branch
  (`double_read_early`; code for sm_80 has 6 and 14, docs/validation.md, section 8).
Not checked: that an instruction comes late enough after one of the translation whose result it
reads and that names no scoreboard (the translator's table of distances is taken on trust).

A translation may also take registers of the program where they hold nothing, keep uniform
registers in a frame of the thread's local memory, put a register away for the time of an
instruction, take everything a product needs at its instruction, and do two products of Turing
as a pair on whole rows. What is checked of each is said where it is checked (the rules 1 to 6
of borrowed registers, A to D of the frame, P to U of room, those of pairs, whose plan is given
to qcheck.py), and in docs/validation.md, section 2, with what is not. A register is the home
of a uniform register for the life that the translator declares, and may serve otherwise
outside it: the lives of uniform registers are worked out here (`register_lives`), the declared
life must be the whole life (rule HL), and nothing that the translation adds writes a home
during it but what the instruction that writes the uniform register became (HW).

Prints what does not hold. Exit status: 0 when everything was checked and holds, 1 if anything
does not hold, 2 if nothing was found but a check could not be made (nvdisasm gave no chart of the
lives of predicates for a function: the verdict says how many).
"""
import collections, os, re, struct, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import elf
import machine
from check import disassemble
from machine import READ_BARRIER, USCHED, WAIT, WRITE_BARRIER
from remap import plain

HEX = re.compile(r'0x[0-9a-f]+$')
GUARD = re.compile(r'@!?U?P\d ')
MEMORY = re.compile(r'(LD|ST|ATOM)[A-Z]*[. ]')        # as nvdisasm names what reads or writes memory, of any kind
REG = re.compile(r'\b(UR\d+|URZ|UP\d|UPT|R\d+|RZ|P\d|PT)\b')
# A number moved into a uniform register, and into an ordinary one (what a relocation of half an address is on).
UMOV_NUMBER = re.compile(r'(?:@!?UP\d )?UMOV UR\d+, (0x[0-9a-f]+)$')
MOV_NUMBER = re.compile(r'(?:@!?P\d )?MOV R\d+, (0x[0-9a-f]+)$')


def texts(code, sm):
    words = [int.from_bytes(code[i:i + 16], 'little') for i in range(0, len(code), 16)]
    # The whole function at once: nvdisasm prints the targets of branches as addresses in what it is given.
    out, err = disassemble(words, sm)
    if any(t is None for t in out):
        # It stopped at an instruction it does not read: the rest, one instruction at a time, in place.
        for i, t in enumerate(out):
            if t is None:
                out[i] = disassemble([0x000fc000000000000000000000007918] * i + [words[i]], sm)[0][-1]
    if sm == 75:
        # Turing's F2FP (a packed conversion) is an encoding of the sm_75 description that
        # NVIDIA's disassembler does not read for SM75; it reads the same encoding for SM80.
        for i, t in enumerate(out):
            if t is None:
                out[i] = disassemble([0x000fc000000000000000000000007918] * i + [words[i]], 80)[0][-1]
    return out


def moved(text, at, new_at, new_of, relative):
    """`text` of the instruction that was at `at`, as it must read at `new_at`."""
    body = text.split(' ', 1)[1] if text.startswith('@') else text
    if not relative or not body.startswith(('BRA', 'BSSY', 'CALL.REL', 'RET.REL', 'JMP', 'BRX', 'BREAK', 'PBK', 'PCNT', 'PEXIT', 'PLONGJMP', 'PRET', 'SSY')):
        return text
    m = HEX.search(text)
    if not m:
        return text
    target = int(m.group(0), 16)
    return text[:m.start()] + ('0x%x' % new_of.get(target, -1)) if target in new_of else text


def unreused(text):
    """Without the marks of register reuse (the translation does not keep them all), and with negative
    numbers as the 32 bits they are (SEL prints its immediate without a sign, IMNMX with one)."""
    text = re.sub(r'\.reuse\b', '', text)
    return re.sub(r'-0x([0-9a-f]+)\b', lambda m: '0x%x' % ((1 << 32) - int(m.group(1), 16)), text)


def fewer_marks(want, got):
    """`got` is `want` with some of its marks of register reuse gone, and none added."""
    def marks(text):
        out, at, gone = set(), text.find('.reuse'), 0
        while at >= 0:
            out.add(at - gone)
            gone += len('.reuse')
            at = text.find('.reuse', at + 1)
        return out
    plain = lambda text: text.replace('.reuse', '')
    return plain(want) == plain(got) and marks(got) < marks(want)


BATCHES = {}


def batch(mc, sm, word):
    """The batch that an instruction names in the field that holds its stall (0: none), by
    NVIDIA's table for its class; None when its bits are no combination that the table has."""
    d = mc.decode(sm, word)
    if d is None:
        return None
    key = next((k for k in d.ops if k.startswith('TABLES_opex')), None)
    if key is None:
        return 0
    name = key.split('(')[0]
    if (sm, name) not in BATCHES:
        md = mc.md70 if sm == 70 else mc.md75
        BATCHES[(sm, name)] = {int(value): int(arguments[0]) for arguments, value in md.tables[name]}
    return BATCHES[(sm, name)].get(d.ops[key])


def relocations(container):
    """The relocations of a cubin outside its debug sections, each as (section, where in it, the
    function whose code it is on or None, the offset it applies at, the symbol and the kind, its
    addend if it has one, the function whose code it names or None)."""
    symbols = container.symbols()
    code = {i: x.name[len('.text.'):] for i, x in enumerate(container.sections) if x.name.startswith('.text.') and x.type == 1}
    out = []
    for s in container.sections:
        if s.type not in (elf.SHT_REL, elf.SHT_RELA) or s.info >= len(container.sections):
            continue
        if container.sections[s.info].name.startswith(('.debug', '.nv_debug')):
            continue
        step = 24 if s.type == elf.SHT_RELA else 16
        for at in range(0, len(s.data) - step + 1, step):
            offset, info = struct.unpack_from('<QQ', s.data, at)
            place = symbols[info >> 32][3] if info >> 32 < len(symbols) else None
            out.append((s.name, at, code.get(s.info), offset, info, s.data[at + 16:at + step], code.get(place)))
    return out


def _recount(data, count):
    """A copy of a cubin's bytes with every function that declares more than `count` registers
    changed to declare `count`. The register count is in the section header's sh_info field
    (the high byte, bits 24-31) and in the .nv.info attribute 0x2f (EIATTR_MAXREG_COUNT).
    Nothing of the code changes: the chart of predicate lives does not depend on the count."""
    container = elf.Elf(data)
    mc = machine.Machine()
    changed = False
    for s in container.sections:
        if s.name.startswith('.text.') and s.type == 1:
            # sh_info high byte is the register count.
            declared = (s.info >> 24) & 0xff
            if declared > count:
                s.info = (s.info & 0x00ffffff) | (count << 24)
                changed = True
    # Also patch the .nv.info and .nv.info.<function> attribute 0x2f.
    for s in container.sections:
        if s.name.startswith('.nv.info'):
            d = bytearray(s.data)
            pos = 0
            while pos + 4 <= len(d):
                fmt, attr = d[pos], d[pos + 1]
                val = struct.unpack_from('<H', d, pos + 2)[0]
                if fmt == 4 and attr == 0x2f and pos + 12 <= len(d):
                    reg = struct.unpack_from('<I', d, pos + 8)[0]
                    if reg > count:
                        struct.pack_into('<I', d, pos + 8, count)
                        changed = True
                pos += 4 + (val if fmt == 4 else 0)
            s.data = bytes(d)
    return container.write() if changed else None


def alive(path):
    """{function: {offset: (predicates, uniform predicates) that hold something alive at that
    instruction, then the predicates that the instruction reads or writes, the uniform predicates
    that hold something when it starts, the predicates it writes, the predicates it reads, the
    uniform predicates it reads, and those it writes)} of a cubin, from the lives of registers
    that nvdisasm (the newer one: it reads the newer container) works out itself; {} when it
    cannot. And {function: {offset of a branch through a register: the offsets it may go to}},
    which the same listing gives (BRANCH_TARGETS, as labels).

    The line of a call is left empty: nvdisasm takes a call to write every predicate, which is a
    convention that code does not hold to (`completed`).

    When nvdisasm refuses a cubin for its register count (255: "Invalid register count"),
    a temporary copy that declares 254 registers is given to it instead. The chart of
    predicate lives does not depend on the register count: nothing of the code changes,
    and nvdisasm's dataflow reads the instructions, not the declared count. No instruction
    of these functions names R254 (the highest is R253: a function of 255 registers uses
    R0 through R253 and R255 is RZ)."""
    # (A container in the format of CUDA 12, which some libraries of CUDA 13 still have: the newer
    # nvdisasm reads it and works out no lives for it; the one of its time does.)
    with open(path, 'rb') as f:
        version = '12.9.2' if f.read(8)[7:8] == b'\x33' else '13.5.0'
    tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../toolkits/cuda-%s/bin/nvdisasm' % version)
    if not os.path.exists(tool):
        sys.exit('%s is not there: this needs NVIDIA\'s disassembler, which scripts/fetch-all.sh redist fetches with the toolkits' % os.path.normpath(tool))
    r = subprocess.run([tool, '-plr', '-lrm', 'narrow', '-c', path], capture_output=True, text=True, errors='replace')
    # When nvdisasm refuses ("Invalid register count"): give it a copy declaring 254.
    recounted = False
    if 'Invalid register count' in r.stderr and not r.stdout.strip():
        recount_data = _recount(open(path, 'rb').read(), 254)
        if recount_data is not None:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as tmp:
                tmp.write(recount_data)
                tmp_path = tmp.name
            try:
                r = subprocess.run([tool, '-plr', '-lrm', 'narrow', '-c', tmp_path], capture_output=True, text=True, errors='replace')
                recounted = True
            finally:
                os.unlink(tmp_path)
    out, name, columns, boxes = {}, None, None, None
    labels, pending, listed, tables = {}, [], {}, {}
    for line in r.stdout.splitlines():
        m = re.search(r'\.section\s+\.text\.([^,\s]+)', line)
        if m:
            name, columns, boxes = m.group(1), None, None
            out[name] = {}
            labels[name], pending, listed[name] = {}, [], {}
            continue
        if name is None:
            continue
        m = re.match(r'\s*(\.L_\w+):', line)
        if m:
            pending.append(m.group(1))
            continue
        m = re.search(r'/\*([0-9a-f]+)\*/', line)
        if m:
            for label in pending:
                labels[name][label] = int(m.group(1), 16)
            pending = []
            t = re.search(r'BRANCH_TARGETS ([^"]*)"', line)
            if t:
                listed[name][int(m.group(1), 16)] = [x.strip() for x in t.group(1).split(',') if x.strip()]
        if '//' not in line:
            continue
        fields = line.split('//', 1)[1].split('|')
        if len(fields) < 3:
            continue
        # The boxes of the chart, by their names: the registers (GPR), the predicates (PRED),
        # the uniform registers (UGPR) and the uniform predicates (UPRED). nvdisasm leaves out
        # a box that it has nothing to draw in: a function that has uniform predicates and no
        # ordinary one has GPR and UPRED side by side. (Taken by their position until
        # 2026-11-20: the uniform predicates of such a function were read as ordinary ones.)
        if fields[1].strip() == 'GPR':
            boxes = [f.strip() for f in fields[1:]]
            continue
        if boxes is None:
            continue
        box = lambda n: fields[1 + boxes.index(n)] if n in boxes and 1 + boxes.index(n) < len(fields) else ''
        groups = [box('PRED'), box('UPRED')]
        if '#' in fields[1]:
            # The header: under each digit, the column of that predicate.
            columns = [[i for i, ch in enumerate(group) if ch.isdigit()] for group in groups]
            continue
        m = re.search(r'/\*([0-9a-f]+)\*/', line)
        if m and columns is not None:
            # At a call nvdisasm marks everything a callee may change by convention, uniform
            # predicates among it.
            call = re.search(r'\*/\s+(@!?U?P\d\s+)?CALL\b', line) is not None
            def marked(g, marks):
                return set() if call else {k for k, at in enumerate(columns[g]) if at < len(groups[g]) and groups[g][at] in marks}
            # (v: read, ^: written, x: both, `:`: alive and neither.)
            out[name][int(m.group(1), 16)] = (marked(0, ':v^x'), marked(1, ':v^x'), marked(0, 'v^x'), marked(1, ':vx'), marked(0, '^x'),
                                              marked(0, 'vx'), marked(1, 'vx'), marked(1, '^x'))
    for name, branches in listed.items():
        # (A branch with a target that is no label of the function is left out: it then leads anywhere.)
        tables[name] = {at: [labels[name][x] for x in targets] for at, targets in branches.items() if all(x in labels[name] for x in targets)}
    return out, tables, recounted


JUMPS = ('BRA', 'BSSY', 'CALL.REL', 'JMP', 'BREAK', 'PBK', 'PCNT', 'PEXIT', 'PLONGJMP', 'PRET', 'SSY')


def flow(texts, tables):
    """Where each instruction of a function may lead, from its text: {offset: [offsets]}; the
    places that are entered otherwise than from the instruction before them; and the instructions
    that may lead anywhere in the function (a branch through a register for which nvdisasm lists
    no targets).

    A branch through a register leads where nvdisasm's listing says (`tables`, from `alive`); a
    return within the section, to after any call within the section."""
    leads, inside, anywhere = collections.defaultdict(list), set(), set()
    bodies = [(t or '').split(' ', 1)[1] if (t or '').startswith('@') else (t or '') for t in texts]
    returns = [16 * k + 16 for k, body in enumerate(bodies) if body.startswith('CALL.REL') and k + 1 < len(bodies)]
    for k, (t, body) in enumerate(zip(texts, bodies)):
        at = 16 * k
        always = not (t or '').startswith('@') and (re.match(r'(BRA|JMP) 0x', body) or body.startswith(('EXIT', 'RET', 'BRX', 'JMX')))
        if not always and k + 1 < len(bodies):
            leads[at].append(at + 16)
        if body.startswith(('BRX', 'JMX')):
            if at in tables:
                leads[at] += tables[at]
                inside.update(tables[at])
            else:
                anywhere.add(at)
        elif body.startswith('RET.REL'):
            leads[at] += returns
            inside.update(returns)
        elif body.startswith(JUMPS):
            m = HEX.search(body)
            if m:
                leads[at].append(int(m.group(0), 16))
                inside.add(int(m.group(0), 16))
    return leads, inside, anywhere


def completed(chart, texts, leads, anywhere):
    """Add to `chart`, nvdisasm's lives of the predicates of one function, the lives it leaves out.

    Its lives stop at every call: it takes a call to write every predicate. Code does not hold
    to that (a predicate set before a call within the function is read after it, in PyTorch's
    kernels that divide: the compiler knows what its own helpers touch), so between the last read
    before a call and the call the chart has nothing alive where something is. And it does not
    take P2R to read predicates nor R2P to write them: their masks are numbers.

    So the lives are worked out again here, from what nvdisasm says each instruction reads and
    writes (its marks), what the text says of P2R and R2P, and the flow of the function (`leads`,
    `anywhere`: `flow`), and added to nvdisasm's own: a predicate holds something at every
    instruction from which a read of it is reached without it being written for sure on the way.
    With care wherever the flow is not known: a call writes nothing and may be stepped over, a
    return within the section goes to after any call, a return out of it has everything alive,
    an instruction under a predicate may not write, and R2P does not end a life."""
    n = len(texts)
    uses, kills, ends = [0] * n, [0] * n, [0] * n
    for k, t in enumerate(texts):
        there = chart.get(16 * k)
        if there is None:
            continue
        body = (t or '').split(' ', 1)[1] if (t or '').startswith('@') else (t or '')
        reads = sum(1 << p for p in there[5]) | sum(1 << (7 + q) for q in there[6])
        writes = sum(1 << p for p in there[4]) | sum(1 << (7 + q) for q in there[7])
        if body.startswith(('P2R', 'R2P')):
            number = re.search(r', 0x([0-9a-f]+)$', body)
            mask = int(number.group(1), 16) & 0x7f if number else 0x7f
            there[2].update(p for p in range(7) if mask >> p & 1)
            if body.startswith('P2R'):
                reads |= mask
            else:
                there[4].update(p for p in range(7) if mask >> p & 1)
        uses[k] = reads
        if not (t or '').startswith('@') and not body.startswith('R2P'):
            kills[k] = writes
        if body.startswith('RET') and not body.startswith('RET.REL'):
            ends[k] = (1 << 14) - 1
    successors = [[to // 16 for to in leads.get(16 * k, ()) if 0 <= to < 16 * n] for k in range(n)]
    lost = {at // 16 for at in anywhere}
    live_in, live_out = [0] * n, list(ends)
    changed = True
    while changed:
        changed = False
        everywhere = 0
        if lost:
            for value in live_in:
                everywhere |= value
        for k in range(n - 1, -1, -1):
            out = ends[k] | (everywhere if k in lost else 0)
            for to in successors[k]:
                out |= live_in[to]
            into = uses[k] | (out & ~kills[k])
            if out != live_out[k] or into != live_in[k]:
                live_out[k], live_in[k], changed = out, into, True
    for k in range(n):
        there = chart.get(16 * k)
        if there is None:
            continue
        both = live_in[k] | live_out[k]
        there[0].update(p for p in range(7) if both >> p & 1)
        there[1].update(q for q in range(7) if both >> (7 + q) & 1)
        there[3].update(q for q in range(7) if live_in[k] >> (7 + q) & 1)


def register_lives(mc, code, texts75, tables, problems=None, name=''):
    """The lives of registers in the sm_75 original, by the validator's own means: backward
    dataflow from the decoded reads and writes, with control flow built for REGISTER liveness
    (not the same as the predicate-lives flow).

    Returns (live_in, live_out, reads, writes, never) as lists of bitmasks (bit k = register
    Rk, bit 256+k = register URk), one per instruction, plus `never` (the mask of registers
    that some instruction writes late without saying when), or None.

    Flow rules for register liveness (differ from the predicate flow in `completed`):
    - BSSY, PBK, SSY, PCNT: go to the NEXT instruction only (they record a place, lanes do not
      go there; every BSYNC sits right after the place a BSSY names, census of 500 cubins).
    - BREAK, BRK, CONT, PEXIT, PRET, PLONGJMP, LONGJMP: go to the next instruction AND to
      every place that any BSSY/PBK/SSY/PCNT of the function names. Conservative: a BREAK
      goes to the place its region names, but without tracking which region, every named place
      is taken.
    - BRA unconditional: taken only. BRA conditional: taken and falls through.
    - BRX with a table: targets and falls through (unless unconditional). Without: not known.
    - CALL.REL: falls through; the called code's reads and writes are the caller's.
    - CALL outside, RET (not REL), JMP/JMX without table: everything alive.
    - EXIT unguarded: nothing follows. Guarded: falls through.
    - A write under a predicate ends no life. A call kills nothing.
    - R0, R1, R255 are tracked like any other: the borrowing rules exclude R0 and R1 separately.
    """
    n = len(code) // 16
    if n == 0:
        return None
    words = [int.from_bytes(code[i:i + 16], 'little') for i in range(0, len(code), 16)]
    decoded = [mc.decode(75, w) for w in words]
    if any(d is None for d in decoded):
        return None

    import translate as TR

    def reg_bits(pairs):
        s = 0
        for entry in pairs:
            l, x = entry[0], entry[1]
            if l == 'R' and x != 255:
                s |= 1 << x
            elif l == 'U' and x != 63:
                s |= 1 << (256 + x)
        return s

    reads_raw = [reg_bits(d.reads) for d in decoded]
    writes_raw = [reg_bits(d.writes) for d in decoded]

    every = 0
    for i in range(n):
        every |= reads_raw[i] | writes_raw[i]

    def _unguarded(d):
        ops = d.ops
        return (ops.get('Pg', 7) == 7 and not ops.get('Pg@not', 0)
                and (ops.get('UPg', 7) == 7 and not ops.get('UPg@not', 0)))

    # Build successor relation for register liveness.
    # Collect meeting places (targets of BSSY, PBK, SSY, PCNT).
    meets = set()
    for i, d in enumerate(decoded):
        if d.mnemonic in ('BSSY', 'PBK', 'SSY', 'PCNT'):
            rel = TR.relative(mc, 75, d) if hasattr(TR, 'relative') else None
            if rel is not None:
                t = 16 * i + 16 + TR.signed(rel[0], d.word) * rel[1]
                if t % 16 == 0 and 0 <= t < 16 * n:
                    meets.add(t // 16)
    LEAVES = ('BREAK', 'BRK', 'CONT', 'PEXIT', 'PRET', 'PLONGJMP', 'LONGJMP')
    succ = [[] for _ in range(n)]
    lost = set()         # instructions that may go anywhere
    for i, d in enumerate(decoded):
        ops = d.ops
        ug = _unguarded(d)
        after = [i + 1] if i + 1 < n else []
        m = d.mnemonic
        if m == 'EXIT':
            succ[i] = [] if ug else after
            continue
        if m == 'RET':
            if d.c.name == 'ret__REL':
                # Return within the section: goes to after each CALL.REL.
                targets = [j + 1 for j in range(n) if decoded[j].c.name.startswith('call_rel') and j + 1 < n]
                succ[i] = ([] if ug else after) + targets
            else:
                succ[i] = [] if ug else after
            continue
        if m in ('BRX', 'JMX', 'JMP'):
            if m == 'BRX' and d.c.name == 'brx_' and 16 * i in tables:
                succ[i] = [t // 16 for t in tables[16 * i] if t % 16 == 0 and 0 <= t < 16 * n] + ([] if ug else after)
            elif m in ('JMX', 'JMP'):
                lost.add(i)
                succ[i] = after
            else:
                lost.add(i)
                succ[i] = after
            continue
        if m == 'BRA':
            _true = lambda p: ops.get(p, 7) == 7 and not ops.get(p + '@not', 0)
            always = ug and _true('Pp') and ops.get('cond', 0) in (0, 1)
            rel = TR.relative(mc, 75, d)
            target = None
            if rel is not None:
                t = 16 * i + 16 + TR.signed(rel[0], d.word) * rel[1]
                if t % 16 == 0 and 0 <= t < 16 * n:
                    target = t // 16
            if always:
                succ[i] = [target] if target is not None else after
            else:
                succ[i] = after + ([target] if target is not None else [])
            continue
        if m in ('BSSY', 'PBK', 'SSY', 'PCNT'):
            # For register lives: goes to the NEXT instruction only.
            succ[i] = after
            continue
        if m in LEAVES:
            # Goes to next instruction AND every meeting place.
            succ[i] = after + sorted(meets)
            continue
        # CALL
        if d.c.name.startswith('call_rel'):
            succ[i] = after           # falls through; callee handled below
            continue
        if d.c.name.startswith(('call', 'jcal')):
            succ[i] = after
            continue
        # Branches with a relative operand (CALL.REL already handled).
        rel = TR.relative(mc, 75, d)
        if rel is not None:
            t = 16 * i + 16 + TR.signed(rel[0], d.word) * rel[1]
            target = t // 16 if t % 16 == 0 and 0 <= t < 16 * n else None
            succ[i] = after + ([target] if target is not None else [])
        else:
            succ[i] = after

    # Reads and writes, with call handling.
    reads = list(reads_raw)
    writes = list(writes_raw)
    kills = [writes_raw[i] if _unguarded(decoded[i]) else 0 for i in range(n)]

    # Calls within the section: exact calls vs merged calls.
    # An exact call: (a) a MOV of the return address into a register R in a straight line
    # before the call; (b) every RET.REL of the called code names R; (c) the called code
    # does not write R; (d) the called code calls nothing. For exact calls, the routine
    # returns to the instruction after the call and nothing else.
    # For exact calls, the RET.REL inside the routine should NOT go to after every call
    # in the section; it should go only to the return address of calls that target it.
    exact_calls, inexact_calls = 0, 0
    inexact_reasons = collections.Counter()
    # Branch targets for the exact-call search: instructions entered from elsewhere.
    branch_targets = set()
    for x in range(n):
        if decoded[x] is not None and decoded[x].mnemonic in ('BRA', 'BRX', 'JMP', 'JMX'):
            branch_targets.update(succ[x])
    # First pass: identify exact calls and build routine-to-returns map.
    routine_returns = collections.defaultdict(list)       # {routine target: [return instructions]}
    for i in range(n):
        d = decoded[i]
        if not d.c.name.startswith(('call', 'jcal')):
            continue
        if d.c.name.startswith('call_rel'):
            # An adjacent LEPC gives this call its return address (what the translator links):
            # the callee must return through the LEPC's register with RET.ABS, and a branch may
            # not arrive at the call (it would skip the LEPC whose value the callee uses).
            lepc_reg = None
            if i > 0:
                prev = (texts75[i - 1] or '')
                prev_body = prev.split(' ', 1)[1] if prev.startswith('@') else prev
                m_lepc = re.match(r'LEPC R(\d+)$', prev_body)
                if m_lepc is not None and decoded[i - 1].mnemonic == 'LEPC':
                    lepc_reg = int(m_lepc.group(1))
            body_text = (texts75[i] or '').split(' ', 1)[1] if (texts75[i] or '').startswith('@') else (texts75[i] or '')
            m_hex = HEX.search(body_text)
            if not m_hex:
                reads[i] |= every
                kills[i] = 0
                inexact_calls += 1
                continue
            target = int(m_hex.group(0), 16) // 16
            if not 0 <= target < n:
                reads[i] |= every
                kills[i] = 0
                inexact_calls += 1
                continue
            # Check for exact call: MOV of 16*(i+1) into a register, reaching the call
            # in a straight line, the routine's RETs name that register, nothing writes it.
            return_addr = 16 * (i + 1)
            mov_reg = None
            # Walk backward in a straight line from the call: stop at an instruction
            # that is a branch target (from BRA/BRX/JMP/JMX) or is a branch/call/return
            # itself (not BSSY/BMOV: they fall through). No distance limit. The MOV may
            # be AT a target; check it even there, stopping AFTER the check.
            # `meets` (targets of BSSY/PBK/SSY/PCNT) are NOT a stop condition:
            # those are structured-control-flow convergence points, not entries from
            # elsewhere. A BREAK that goes to a meeting place between the MOV and the
            # CALL does not invalidate the MOV: the BREAK leaves the region, and the
            # lanes that reconverge at the meeting place come through the BSSY's region
            # (which includes the MOV). `branch_targets` (from BRA/BRX/JMP/JMX) is.
            for j in range(i - 1, -1, -1):
                dj = decoded[j]
                if dj is None:
                    break
                tj = texts75[j] or ''
                body_j = tj.split(' ', 1)[1] if tj.startswith('@') else tj
                # Also recognise IMAD.MOV.U32 R, RZ, RZ, 0xN as a MOV of a return address.
                m_mov = re.match(r'MOV R(\d+), 0x([0-9a-f]+)$', body_j)
                if not m_mov:
                    m_mov = re.match(r'IMAD\.MOV\.U32 R(\d+), RZ, RZ, 0x([0-9a-f]+)$', body_j)
                if m_mov and int(m_mov.group(2), 16) == return_addr:
                    mov_reg = int(m_mov.group(1))
                    # Check straight line: nothing between j+1 and i-1 writes R or
                    # is a branch target (not meets: see above).
                    straight = True
                    for k in range(j + 1, i):
                        if k in branch_targets:
                            straight = False
                            break
                        if any(l == 'R' and r == mov_reg for l, r in decoded[k].writes):
                            straight = False
                            break
                    if not straight:
                        mov_reg = None
                    break
                # Stop at a branch target (but check the MOV first).
                if j in branch_targets:
                    break
                # Stop at a branch, call, or return (not BSSY/BMOV).
                if dj.kind == 'BRU' and dj.mnemonic not in ('BSSY', 'BMOV'):
                    break
            # Walk the callee's body.
            body, work, simple = set(), [target], True
            while work:
                k = work.pop()
                if k in body:
                    continue
                body.add(k)
                dk = decoded[k]
                if dk.c.name.startswith(('call', 'jcal')):
                    simple = False
                    break
                for t in succ[k]:
                    if t in body:
                        continue
                    if dk.mnemonic == 'RET':
                        continue          # RET ends the walk
                    if 0 <= t < n:
                        work.append(t)
            # The LEPC's register is the callee's return address when the callee's RETs are
            # RET.ABS on it and nothing of the callee writes it (a routine that calls others is
            # not checked here: those calls may write the register; it must save and restore it
            # around them, which this does not follow).
            lepc_ok = False
            if lepc_reg is not None and simple:
                rets = [decoded[k] for k in body if decoded[k].mnemonic == 'RET']
                written = any(any(l == 'R' and r != 255 and r in (lepc_reg, lepc_reg + 1) for l, r in decoded[k].writes) for k in body)
                returns_via_lepc = bool(rets) and not written and all(
                    r.c.name in ('ret__ABS', 'ret__ABS_UR') and r.ops.get('Ra') == lepc_reg
                    and r.ops.get('Ra_offset') == 0x20 for r in rets)
                if returns_via_lepc and i in branch_targets and problems is not None:
                    problems.append('%s+0x%x: a branch arrives at a call whose callee returns through R%d, the register of the LEPC before it, so a path skips the LEPC' % (name[:40], 16 * i, lepc_reg))
                # When the callee does not return through the LEPC's register, the call is an
                # ordinary one: its return address is a MOV (the translator's rule), and the
                # LEPC before it is not this call's. Nothing is wrong with the pair as such.
                lepc_ok = returns_via_lepc and i not in branch_targets
            # Check exact: all RET.REL name mov_reg (or a copy of it), nothing
            # writes mov_reg, no calls. A RET.REL may name R' where the last write
            # of R' before the RET in a straight line is `MOV R', mov_reg`; if the
            # RET reads a pair (R':R'+1), the upper half must be `MOV R'+1, 0x0`.
            exact = simple and (mov_reg is not None or lepc_ok)
            inexact_reason = None
            if exact:
                # For each RET, either (a) no instruction in the routine writes mov_reg,
                # or (b) the RET's operand is a copy of mov_reg AND in the backward search from
                # the RET, the copy comes before any write to mov_reg (the copy was made first).
                # Case (b): the routine copies mov_reg, then overwrites it (e.g. for the upper
                # half of a 64-bit return pair). The copy must exist before the overwrite on the
                # way to the RET. We check this conservatively: backward from the RET in a
                # straight line, the copy must be found before any write to mov_reg.
                writes_mov_reg = any(any(l == 'R' and r == mov_reg for l, r in decoded[k].writes) for k in body)
                if writes_mov_reg:
                    # Check every RET: its operand must not be mov_reg, and the copy must precede
                    # any write to mov_reg in a backward walk from the RET.
                    for k in body:
                        dk = decoded[k]
                        if dk.c.name != 'ret__REL':
                            continue
                        ret_operand = dk.ops.get('Ra', -1)
                        if ret_operand == mov_reg:
                            exact = False
                            inexact_reason = 'R written in the routine'
                            break
                        # Flow-graph-based verification. Using `succ` and `body`:
                        # 1. Every way from the entry to the RET passes the copy (remove
                        #    the copy from the graph: the RET is not reachable from entry).
                        # 2. No instruction from which the copy is reachable writes mov_reg
                        #    (the entry's own incoming state apart).
                        # 3. No instruction reachable from the copy and from which the RET
                        #    is reachable writes rp, the upper-half setter apart.
                        rp = ret_operand
                        # Find the copy instruction in the routine's body.
                        copy_at = None
                        for j in sorted(body):
                            dj3 = decoded[j]
                            tj3 = texts75[j] or ''
                            body_j3 = tj3.split(' ', 1)[1] if tj3.startswith('@') else tj3
                            is_copy = (body_j3 == 'MOV R%d, R%d' % (rp, mov_reg)
                                       or body_j3 == 'IMAD.MOV.U32 R%d, RZ, RZ, R%d' % (rp, mov_reg))
                            if is_copy:
                                copy_at = j
                        ok_copy = copy_at is not None
                        if ok_copy:
                            # Condition 1: without the copy, the RET is not reachable from entry.
                            reachable = set()
                            work2 = [target]
                            while work2:
                                x = work2.pop()
                                if x in reachable or x == copy_at:
                                    continue
                                reachable.add(x)
                                if decoded[x].mnemonic == 'RET':
                                    continue
                                for t2 in succ[x]:
                                    if t2 in body and t2 not in reachable:
                                        work2.append(t2)
                            if k in reachable:
                                ok_copy = False    # RET reachable without the copy
                        if ok_copy:
                            # Condition 2: no instruction from which the copy is reachable
                            # writes mov_reg (entry apart). Compute backward reachability
                            # from the copy within the body.
                            # Build predecessors within the body.
                            pred = {x: [] for x in body}
                            for x in body:
                                if decoded[x].mnemonic == 'RET':
                                    continue
                                for t2 in succ[x]:
                                    if t2 in body:
                                        pred[t2].append(x)
                            before_copy = set()
                            work2 = [copy_at]
                            while work2:
                                x = work2.pop()
                                if x in before_copy:
                                    continue
                                before_copy.add(x)
                                for p in pred.get(x, []):
                                    if p not in before_copy:
                                        work2.append(p)
                            for x in before_copy:
                                if x == target:
                                    continue   # entry's incoming state
                                if any(l == 'R' and r == mov_reg for l, r in decoded[x].writes):
                                    ok_copy = False
                                    break
                        if ok_copy:
                            # Condition 3: no instruction between copy and RET writes rp.
                            # Forward from the copy, intersect with backward from the RET.
                            after_copy = set()
                            work2 = list(succ[copy_at])
                            while work2:
                                x = work2.pop()
                                if x not in body or x in after_copy:
                                    continue
                                after_copy.add(x)
                                if decoded[x].mnemonic == 'RET':
                                    continue
                                for t2 in succ[x]:
                                    if t2 not in after_copy:
                                        work2.append(t2)
                            before_ret = set()
                            work2 = [k]
                            while work2:
                                x = work2.pop()
                                if x in before_ret:
                                    continue
                                before_ret.add(x)
                                for p in pred.get(x, []):
                                    if p not in before_ret:
                                        work2.append(p)
                            between = after_copy & before_ret
                            # Find the upper-half setter (if rp+1 is in ret_reads).
                            upper_setter = None
                            if rp + 1 in [r for l, r, *_ in dk.reads if l == 'R']:
                                for x in sorted(between):
                                    if any(l == 'R' and r == rp + 1 for l, r in decoded[x].writes):
                                        upper_setter = x
                            for x in between:
                                if x == upper_setter:
                                    continue
                                if any(l == 'R' and r == rp for l, r in decoded[x].writes):
                                    ok_copy = False
                                    break
                        if not ok_copy:
                            exact = False
                            inexact_reason = 'R written in the routine'
                            break
                if exact:
                    for k in body:
                        dk = decoded[k]
                        if dk.c.name != 'ret__REL':
                            continue
                        # Does the RET name mov_reg directly?
                        ret_reads = [r for l, r, *_ in dk.reads if l == 'R']
                        if mov_reg in ret_reads:
                            continue
                        # Does it name R' that is a copy of mov_reg?
                        ok = False
                        for rp in ret_reads:
                            if rp == 255:
                                continue
                            # Walk backward in a straight line from the RET to find MOV R', mov_reg.
                            # No distance limit (was 32); stop at branch targets
                            # (not meets: see the backward search from the CALL above).
                            found_mov = False
                            for j in range(k - 1, -1, -1):
                                if j not in body:
                                    break
                                dj2 = decoded[j]
                                tj2 = texts75[j] or ''
                                body_t = tj2.split(' ', 1)[1] if tj2.startswith('@') else tj2
                                # Also recognise IMAD.MOV.U32 R', RZ, RZ, R as a copy.
                                is_mov = (body_t == 'MOV R%d, R%d' % (rp, mov_reg)
                                          or body_t == 'IMAD.MOV.U32 R%d, RZ, RZ, R%d' % (rp, mov_reg))
                                # Is j a branch target? (not straight line; but accept the MOV even there)
                                if not is_mov and j in branch_targets:
                                    break
                                if is_mov:
                                    found_mov = True
                                    # If the RET reads a pair, check upper half = MOV R'+1, 0x0
                                    if rp + 1 in ret_reads:
                                        upper_ok = False
                                        # When rp+1 == mov_reg, the upper half IS the
                                        # return address register, which the routine copies into rp:
                                        # R24 = R25, RET R24:R25 where R25 holds the return address.
                                        if rp + 1 == mov_reg:
                                            upper_ok = True
                                        def _is_upper_zero(jj):
                                            tjj = texts75[jj] or ''
                                            body_jj = tjj.split(' ', 1)[1] if tjj.startswith('@') else tjj
                                            return (body_jj == 'MOV R%d, RZ' % (rp + 1) or body_jj == 'MOV R%d, 0x0' % (rp + 1)
                                                    or re.match(r'IMAD\.MOV\.U32 R%d, RZ, RZ, 0x0$' % (rp + 1), body_jj))
                                        # Search backward from the RET first (the routine
                                        # may use the upper-half register for computation and set
                                        # it to zero just before returning), then forward from the
                                        # copy, then backward from the copy.
                                        for jj in range(k - 1, -1, -1):
                                            if jj not in body:
                                                break
                                            if _is_upper_zero(jj):
                                                upper_ok = True
                                                break
                                            if any(l == 'R' and r == rp + 1 for l, r in decoded[jj].writes):
                                                break
                                            if decoded[jj].kind == 'BRU' and decoded[jj].mnemonic not in ('BSSY', 'BMOV', 'WARPSYNC'):
                                                break
                                        if not upper_ok:
                                            for jj in range(j + 1, k):
                                                if _is_upper_zero(jj):
                                                    upper_ok = True
                                                    break
                                                if any(l == 'R' and r == rp + 1 for l, r in decoded[jj].writes):
                                                    break
                                        if not upper_ok:
                                            for jj in range(j - 1, -1, -1):
                                                if jj not in body:
                                                    break
                                                if _is_upper_zero(jj):
                                                    upper_ok = True
                                                    break
                                                if any(l == 'R' and r == rp + 1 for l, r in decoded[jj].writes):
                                                    break
                                                if decoded[jj].kind == 'BRU' and decoded[jj].mnemonic not in ('BSSY', 'BMOV', 'WARPSYNC'):
                                                    break
                                        if not upper_ok:
                                            found_mov = False
                                    break
                                if any(l == 'R' and r == rp for l, r in dj2.writes):
                                    break
                                # WARPSYNC falls through like BSSY/BMOV.
                                if dj2.kind == 'BRU' and dj2.mnemonic not in ('BSSY', 'BMOV', 'WARPSYNC'):
                                    break
                            if found_mov:
                                ok = True
                                break
                        if not ok:
                            exact = False
                            inexact_reason = 'RET names another register'
                            break
            if not simple:
                reads[i] |= every
                kills[i] = 0
                inexact_calls += 1
                inexact_reasons['the routine calls'] += 1
                continue
            if mov_reg is None and not exact:
                inexact_reason = 'no MOV of the return address'
            if exact:
                exact_calls += 1
                routine_returns[target].append(i + 1)
            else:
                inexact_calls += 1
                inexact_reasons[inexact_reason or 'unknown'] += 1
                routine_returns[target].append(None)      # marks an inexact call
            # Backward dataflow on the callee's body.
            # For exact calls: nothing alive at RET (the caller handles what comes after).
            # For inexact calls: at RET, what is alive after every call of the section.
            callee_leaves = {}
            for k in body:
                if decoded[k].mnemonic == 'RET':
                    if exact:
                        callee_leaves[k] = 0
                    else:
                        callee_leaves[k] = every          # conservative
            callee_live = {k: callee_leaves.get(k, 0) for k in body}
            changed_flag = True
            while changed_flag:
                changed_flag = False
                for k in sorted(body, reverse=True):
                    after = callee_leaves.get(k, 0)
                    for t in succ[k]:
                        if t in callee_live:
                            after |= callee_live[t]
                    ck = writes_raw[k] if _unguarded(decoded[k]) else 0
                    now = reads_raw[k] | (after & ~ck)
                    if now != callee_live[k]:
                        callee_live[k], changed_flag = now, True
            callee_writes = 0
            for k in body:
                callee_writes |= writes_raw[k]
            reads[i] |= callee_live.get(target, 0)
            writes[i] |= callee_writes
            # A call kills nothing: the callee may write under a guard, and the caller's
            # value may survive. This is the safe (conservative) direction.
            kills[i] = 0
        else:
            reads[i] |= every
            kills[i] = 0
            inexact_calls += 1

    # For routines where ALL calls are exact: disconnect them from the main flow.
    # The callee is handled separately via reads[call] and writes[call].
    # Its RET.REL has no successors in the main flow, and its body's reads/kills
    # don't participate in the main backward dataflow.
    for target, returns in routine_returns.items():
        if None in returns:
            continue          # at least one inexact call: leave succ as-is
        body, work = set(), [target]
        while work:
            k = work.pop()
            if k in body:
                continue
            body.add(k)
            if decoded[k].mnemonic == 'RET':
                continue
            for t in succ[k]:
                if 0 <= t < n:
                    work.append(t)
        for k in body:
            if decoded[k].c.name == 'ret__REL':
                succ[k] = []              # disconnect from the main flow

    leaves = [0] * n
    for i in range(n):
        if decoded[i].mnemonic == 'RET' and not decoded[i].c.name.startswith('ret__REL'):
            leaves[i] = every

    # Backward dataflow.
    live_in = [0] * n
    changed = True
    while changed:
        changed = False
        everywhere = 0
        if lost:
            for v in live_in:
                everywhere |= v
        for i in range(n - 1, -1, -1):
            out = leaves[i] | (everywhere if i in lost else 0)
            for t in succ[i]:
                out |= live_in[t]
            new = reads[i] | (out & ~kills[i])
            if new != live_in[i]:
                live_in[i], changed = new, True
    live_out = [0] * n
    for i in range(n):
        out = leaves[i] | (everywhere if i in lost else 0)
        for t in succ[i]:
            out |= live_in[t]
        live_out[i] = out

    # `never`: registers that some instruction writes late without naming a scoreboard.
    never = 0
    for d in decoded:
        if d.kind in ('MIO_FAST', 'MIO_SLOW') and (d.word >> WRITE_BARRIER) & 7 == 7:
            never |= reg_bits(d.writes)

    return live_in, live_out, reads_raw, writes_raw, never, exact_calls, inexact_calls, inexact_reasons


def predicate_lives(mc, code, texts75, tables):
    """The lives of predicates P0 to P6 in the sm_75 original, by the validator's own means:
    backward dataflow from the decoded reads and writes, with the same control-flow graph
    as register_lives.

    What each instruction reads: its predicate operands (the decoder's reads with letter 'P'),
    its guard predicate (Pg, unless PT; both @P3 and @!P3 read P3), and P2R (reads the
    predicates named by its mask). What each instruction writes: its predicate outputs (the
    decoder's writes with letter 'P') and R2P (writes the predicates named by its mask).
    A guarded instruction that writes a predicate does NOT end that predicate's life (the write
    may not happen). R2P does not end a life either (same as completed()).

    Calls to a routine of the section: exact calls handled as in register_lives (the routine's
    reads before writes); anything else: everything read. EXIT and RET (not REL) ends.

    Returns {offset: mask of predicates P0-P6 alive there (bit p = predicate Pp)} for every
    instruction of the function, or None when a decode fails."""
    n = len(code) // 16
    if n == 0:
        return None
    words = [int.from_bytes(code[i:i + 16], 'little') for i in range(0, len(code), 16)]
    decoded = [mc.decode(75, w) for w in words]
    if any(d is None for d in decoded):
        return None

    import translate as TR

    ALL_PREDS = (1 << 7) - 1   # P0 to P6

    def pred_reads(i):
        """Bitmask of predicates read by instruction i."""
        d = decoded[i]
        t = texts75[i] or ''
        body = t.split(' ', 1)[1] if t.startswith('@') else t
        s = 0
        # Predicate operands from the decoder.
        for letter, register, *_ in d.reads:
            if letter == 'P' and register < 7:
                s |= 1 << register
        # Guard predicate: Pg (and UPg is a uniform predicate, not P0-P6).
        pg = d.ops.get('Pg', 7)
        if pg < 7:
            s |= 1 << pg
        # P2R reads the predicates named by its mask.
        if body.startswith('P2R'):
            m = re.search(r', 0x([0-9a-f]+)$', body)
            mask = int(m.group(1), 16) & 0x7f if m else 0x7f
            s |= mask
        return s

    def pred_writes(i):
        """Bitmask of predicates written by instruction i."""
        d = decoded[i]
        t = texts75[i] or ''
        body = t.split(' ', 1)[1] if t.startswith('@') else t
        s = 0
        for letter, register in d.writes:
            if letter == 'P' and register < 7:
                s |= 1 << register
        # R2P writes the predicates named by its mask.
        if body.startswith('R2P'):
            m = re.search(r', 0x([0-9a-f]+)$', body)
            mask = int(m.group(1), 16) & 0x7f if m else 0x7f
            s |= mask
        return s

    def _unguarded(d):
        ops = d.ops
        return (ops.get('Pg', 7) == 7 and not ops.get('Pg@not', 0)
                and (ops.get('UPg', 7) == 7 and not ops.get('UPg@not', 0)))

    reads_raw = [pred_reads(i) for i in range(n)]
    writes_raw = [pred_writes(i) for i in range(n)]

    every = ALL_PREDS   # for external calls: all predicates may be touched

    # Build the same successor relation as register_lives.
    meets = set()
    for i, d in enumerate(decoded):
        if d.mnemonic in ('BSSY', 'PBK', 'SSY', 'PCNT'):
            rel = TR.relative(mc, 75, d) if hasattr(TR, 'relative') else None
            if rel is not None:
                t = 16 * i + 16 + TR.signed(rel[0], d.word) * rel[1]
                if t % 16 == 0 and 0 <= t < 16 * n:
                    meets.add(t // 16)
    LEAVES = ('BREAK', 'BRK', 'CONT', 'PEXIT', 'PRET', 'PLONGJMP', 'LONGJMP')
    succ = [[] for _ in range(n)]
    lost = set()
    for i, d in enumerate(decoded):
        ops = d.ops
        ug = _unguarded(d)
        after = [i + 1] if i + 1 < n else []
        m = d.mnemonic
        if m == 'EXIT':
            succ[i] = [] if ug else after
            continue
        if m == 'RET':
            if d.c.name == 'ret__REL':
                targets = [j + 1 for j in range(n) if decoded[j].c.name.startswith('call_rel') and j + 1 < n]
                succ[i] = ([] if ug else after) + targets
            else:
                succ[i] = [] if ug else after
            continue
        if m in ('BRX', 'JMX', 'JMP'):
            if m == 'BRX' and d.c.name == 'brx_' and 16 * i in tables:
                succ[i] = [t // 16 for t in tables[16 * i] if t % 16 == 0 and 0 <= t < 16 * n] + ([] if ug else after)
            elif m in ('JMX', 'JMP'):
                lost.add(i)
                succ[i] = after
            else:
                lost.add(i)
                succ[i] = after
            continue
        if m == 'BRA':
            _true = lambda p: ops.get(p, 7) == 7 and not ops.get(p + '@not', 0)
            always = ug and _true('Pp') and ops.get('cond', 0) in (0, 1)
            rel = TR.relative(mc, 75, d)
            target = None
            if rel is not None:
                t = 16 * i + 16 + TR.signed(rel[0], d.word) * rel[1]
                if t % 16 == 0 and 0 <= t < 16 * n:
                    target = t // 16
            if always:
                succ[i] = [target] if target is not None else after
            else:
                succ[i] = after + ([target] if target is not None else [])
            continue
        if m in ('BSSY', 'PBK', 'SSY', 'PCNT'):
            succ[i] = after
            continue
        if m in LEAVES:
            # For predicate liveness: BREAK removes threads from the barrier's set;
            # where they go next is the BRA that follows. The conservative treatment
            # (every meeting place) used in register_lives takes more ways than there
            # are, which causes false alarms but no missed faults; for predicates,
            # BREAK goes to the next instruction only.
            succ[i] = after
            continue
        if d.c.name.startswith('call_rel'):
            succ[i] = after
            continue
        if d.c.name.startswith(('call', 'jcal')):
            succ[i] = after
            continue
        rel = TR.relative(mc, 75, d)
        if rel is not None:
            t = 16 * i + 16 + TR.signed(rel[0], d.word) * rel[1]
            target = t // 16 if t % 16 == 0 and 0 <= t < 16 * n else None
            succ[i] = after + ([target] if target is not None else [])
        else:
            succ[i] = after

    # Reads and writes for the dataflow.
    reads = list(reads_raw)
    writes = list(writes_raw)
    # Kills: an unguarded write kills the predicate; R2P does NOT kill (same as completed);
    # a guarded write does not kill.
    kills = [0] * n
    for i in range(n):
        d = decoded[i]
        t = texts75[i] or ''
        body = t.split(' ', 1)[1] if t.startswith('@') else t
        if _unguarded(d) and not body.startswith('R2P'):
            kills[i] = writes_raw[i]

    # Calls within the section: exact calls vs external.
    # Branch targets for the exact-call search.
    branch_targets = set()
    for x in range(n):
        if decoded[x] is not None and decoded[x].mnemonic in ('BRA', 'BRX', 'JMP', 'JMX'):
            branch_targets.update(succ[x])

    routine_returns = collections.defaultdict(list)
    for i in range(n):
        d = decoded[i]
        if not d.c.name.startswith(('call', 'jcal')):
            continue
        if d.c.name.startswith('call_rel'):
            body_text = (texts75[i] or '').split(' ', 1)[1] if (texts75[i] or '').startswith('@') else (texts75[i] or '')
            m_hex = HEX.search(body_text)
            if not m_hex:
                reads[i] |= every
                kills[i] = 0
                continue
            target = int(m_hex.group(0), 16) // 16
            if not 0 <= target < n:
                reads[i] |= every
                kills[i] = 0
                continue
            # Exact-call check: same logic as register_lives.
            return_addr = 16 * (i + 1)
            mov_reg = None
            for j in range(i - 1, -1, -1):
                dj = decoded[j]
                if dj is None:
                    break
                tj = texts75[j] or ''
                body_j = tj.split(' ', 1)[1] if tj.startswith('@') else tj
                m_mov = re.match(r'MOV R(\d+), 0x([0-9a-f]+)$', body_j)
                if not m_mov:
                    m_mov = re.match(r'IMAD\.MOV\.U32 R(\d+), RZ, RZ, 0x([0-9a-f]+)$', body_j)
                if m_mov and int(m_mov.group(2), 16) == return_addr:
                    mov_reg = int(m_mov.group(1))
                    straight = True
                    for k in range(j + 1, i):
                        if k in branch_targets:
                            straight = False
                            break
                        if any(l == 'R' and r == mov_reg for l, r in decoded[k].writes):
                            straight = False
                            break
                    if not straight:
                        mov_reg = None
                    break
                if j in branch_targets:
                    break
                if dj.kind == 'BRU' and dj.mnemonic not in ('BSSY', 'BMOV'):
                    break
            # Walk the callee's body.
            body_set, work, simple = set(), [target], True
            while work:
                k = work.pop()
                if k in body_set:
                    continue
                body_set.add(k)
                dk = decoded[k]
                if dk.c.name.startswith(('call', 'jcal')):
                    simple = False
                    break
                for t in succ[k]:
                    if t in body_set:
                        continue
                    if dk.mnemonic == 'RET':
                        continue
                    if 0 <= t < n:
                        work.append(t)
            exact = simple and mov_reg is not None
            if exact:
                # Check RET names mov_reg (same logic as register_lives for the register part).
                writes_mov_reg = any(any(l == 'R' and r == mov_reg for l, r in decoded[k].writes) for k in body_set)
                if writes_mov_reg:
                    for k in body_set:
                        dk = decoded[k]
                        if dk.c.name != 'ret__REL':
                            continue
                        ret_operand = dk.ops.get('Ra', -1)
                        if ret_operand == mov_reg:
                            exact = False
                            break
                        rp = ret_operand
                        copy_at = None
                        for j in sorted(body_set):
                            dj3 = decoded[j]
                            tj3 = texts75[j] or ''
                            body_j3 = tj3.split(' ', 1)[1] if tj3.startswith('@') else tj3
                            is_copy = (body_j3 == 'MOV R%d, R%d' % (rp, mov_reg)
                                       or body_j3 == 'IMAD.MOV.U32 R%d, RZ, RZ, R%d' % (rp, mov_reg))
                            if is_copy:
                                copy_at = j
                        if copy_at is None:
                            exact = False
                            break
                if exact:
                    for k in body_set:
                        dk = decoded[k]
                        if dk.c.name != 'ret__REL':
                            continue
                        ret_reads = [r for l, r, *_ in dk.reads if l == 'R']
                        if mov_reg in ret_reads:
                            continue
                        ok = False
                        for rp in ret_reads:
                            if rp == 255:
                                continue
                            found_mov = False
                            for j in range(k - 1, -1, -1):
                                if j not in body_set:
                                    break
                                dj2 = decoded[j]
                                tj2 = texts75[j] or ''
                                body_t = tj2.split(' ', 1)[1] if tj2.startswith('@') else tj2
                                is_mov = (body_t == 'MOV R%d, R%d' % (rp, mov_reg)
                                          or body_t == 'IMAD.MOV.U32 R%d, RZ, RZ, R%d' % (rp, mov_reg))
                                if not is_mov and j in branch_targets:
                                    break
                                if is_mov:
                                    found_mov = True
                                    if rp + 1 in ret_reads:
                                        if rp + 1 != mov_reg:
                                            found_mov = False  # simplified: do not verify upper half
                                    break
                                if any(l == 'R' and r == rp for l, r in dj2.writes):
                                    break
                                if dj2.kind == 'BRU' and dj2.mnemonic not in ('BSSY', 'BMOV', 'WARPSYNC'):
                                    break
                            if found_mov:
                                ok = True
                                break
                        if not ok:
                            exact = False
                            break
            if not simple or not exact:
                reads[i] |= every
                kills[i] = 0
                routine_returns[target].append(None)
                continue
            routine_returns[target].append(i + 1)
            # Backward dataflow on the callee's body for predicates.
            callee_leaves = {}
            for k in body_set:
                if decoded[k].mnemonic == 'RET':
                    callee_leaves[k] = 0
            callee_live = {k: callee_leaves.get(k, 0) for k in body_set}
            changed_flag = True
            while changed_flag:
                changed_flag = False
                for k in sorted(body_set, reverse=True):
                    after = callee_leaves.get(k, 0)
                    for t in succ[k]:
                        if t in callee_live:
                            after |= callee_live[t]
                    ck = writes_raw[k] if _unguarded(decoded[k]) else 0
                    # R2P also does not kill for predicate liveness.
                    tk = texts75[k] or ''
                    bk = tk.split(' ', 1)[1] if tk.startswith('@') else tk
                    if bk.startswith('R2P'):
                        ck = 0
                    now = reads_raw[k] | (after & ~ck)
                    if now != callee_live[k]:
                        callee_live[k], changed_flag = now, True
            callee_pred_writes = 0
            for k in body_set:
                callee_pred_writes |= writes_raw[k]
            reads[i] |= callee_live.get(target, 0)
            writes[i] |= callee_pred_writes
            kills[i] = 0
        else:
            reads[i] |= every
            kills[i] = 0

    # For routines where ALL calls are exact: disconnect from the main flow.
    for target, returns in routine_returns.items():
        if None in returns:
            continue
        body_set, work = set(), [target]
        while work:
            k = work.pop()
            if k in body_set:
                continue
            body_set.add(k)
            if decoded[k].mnemonic == 'RET':
                continue
            for t in succ[k]:
                if 0 <= t < n:
                    work.append(t)
        for k in body_set:
            if decoded[k].c.name == 'ret__REL':
                succ[k] = []

    leaves = [0] * n
    for i in range(n):
        if decoded[i].mnemonic == 'RET' and not decoded[i].c.name.startswith('ret__REL'):
            leaves[i] = every
        if decoded[i].mnemonic == 'EXIT':
            pass  # EXIT: nothing alive after

    # Backward dataflow.
    live_in = [0] * n
    changed = True
    while changed:
        changed = False
        everywhere = 0
        if lost:
            for v in live_in:
                everywhere |= v
        for i in range(n - 1, -1, -1):
            out = leaves[i] | (everywhere if i in lost else 0)
            for t in succ[i]:
                out |= live_in[t]
            new = reads[i] | (out & ~kills[i])
            if new != live_in[i]:
                live_in[i], changed = new, True
    live_out = [0] * n
    for i in range(n):
        out = leaves[i] | (everywhere if i in lost else 0)
        for t in succ[i]:
            out |= live_in[t]
        live_out[i] = out

    # Return as a dict {offset: mask} for consistency with the chart format.
    result = {}
    for i in range(n):
        result[16 * i] = live_in[i] | live_out[i]
    return result


def shape_of(t):
    """An instruction's text without its registers, the mnemonic of a uniform instruction made ordinary."""
    t = REG.sub('#', t)
    head = t.split(' ')[1] if t.startswith('@') else t.split(' ')[0]
    name = head.split('.')[0]
    names = {'VOTEU': 'VOTE', 'S2UR': 'S2R'}
    return t.replace(name, names.get(name, name[1:] if name.startswith('U') else name), 1)


def matrix_load_expected(source):
    """The instructions that stand for the matrix load `source` (LDSM, as nvdisasm prints it), as
    nvdisasm prints them, with {names} where a register of the translation goes; None when the
    text is not a load this knows. Written from the recipe (crates/vf-ptx/src/lower/,
    measured on a V100), not from the translator: the address of a row comes by a shuffle from the lane
    that has it, into the destination itself; the destination that is the address register last.
    What writes a destination first is a shuffle: the lanes of the rows are computed elsewhere, and
    those of the first rows of a transposed load after the shuffles of the second rows (what writes
    a destination of a load right after a store of it is checked apart: `data_overwritten`)."""
    m = re.match(r'(@!?P\d )?LDSM\.16\.(M88|MT88)(?:\.([24]))? R(\d+), \[R(\d+)(?:\+(UR\d+))?([+-]0x[0-9a-f]+)?\]$', source)
    if not m:
        return None
    guard, transposed, count = m.group(1) or '', m.group(2) == 'MT88', int(m.group(3) or 1)
    rd, ra, uniform, offset = int(m.group(4)), int(m.group(5)), m.group(6), m.group(7) or ''
    order = [k for k in range(count) if rd + k != ra] + [k for k in range(count) if rd + k == ra]
    to = lambda k: 'R%d' % (rd + k)
    out = ['SHF.R.U32.HI {group}, RZ, 0x2, {lane}']
    if not transposed:
        out.append('LOP3.LUT {part}, {lane}, 0x3, RZ, 0xc0, !PT')
        lanes = {}
        for k in order:
            if k == 0:
                lanes[k] = '{group}'
            else:
                lanes[k] = '{row%d}' % k
                out.append('IADD3 {row%d}, {group}, 0x%x, RZ' % (k, 8 * k))
        out += [guard + 'SHFL.IDX PT, %s, R%d, %s, 0x1f' % (to(k), ra, lanes[k]) for k in order]
        if uniform:
            out.append('LEA {sum}, {part}, {%s}, 0x2' % uniform)
            out += [guard + 'IADD3 %s, %s, {sum}, RZ' % (to(k), to(k)) for k in order]
        else:
            out += [guard + 'LEA %s, {part}, %s, 0x2' % (to(k), to(k)) for k in order]
        out += [guard + 'LDS.U %s, [%s%s]' % (to(k), to(k), offset) for k in order]
        return out
    out += ['SHF.L.U32 {part}, {lane}, 0x1, RZ', 'LOP3.LUT {part}, {part}, 0x6, RZ, 0xc0, !PT']
    first = {}
    out += ['IADD3 {second%d}, {part}, 0x%x, RZ' % (k, 8 * k + 1) for k in order]
    out += ['SHFL.IDX PT, {second%d}, R%d, {second%d}, 0x1f' % (k, ra, k) for k in order]
    for k in order:
        if k == 0:
            first[k] = '{part}'
        elif rd + k == ra:
            first[k] = '{other}'
            out.append('IADD3 {other}, {part}, 0x%x, RZ' % (8 * k))
        else:
            first[k] = to(k)
            out.append(guard + 'IADD3 %s, {part}, 0x%x, RZ' % (to(k), 8 * k))
    out += [guard + 'SHFL.IDX PT, %s, R%d, %s, 0x1f' % (to(k), ra, first[k]) for k in order]
    if uniform:
        out.append('LEA {sum}, {group}, {%s}, 0x1' % uniform)
    for k in order:
        if uniform:
            out += ['IADD3 {second%d}, {second%d}, {sum}, RZ' % (k, k), guard + 'IADD3 %s, %s, {sum}, RZ' % (to(k), to(k))]
        else:
            out += ['LEA {second%d}, {group}, {second%d}, 0x1' % (k, k), guard + 'LEA %s, {group}, %s, 0x1' % (to(k), to(k))]
    for k in order:
        # (Both loads under the guard: the one into a register of the translation reads memory too.)
        out += [guard + 'LDS.U.U16 {second%d}, [{second%d}%s]' % (k, k, offset), guard + 'LDS.U.U16 %s, [%s%s]' % (to(k), to(k), offset)]
    out += [guard + 'LEA %s, {second%d}, %s, 0x10' % (to(k), k, to(k)) for k in order]
    return out


# What a function with a product keeps of the lane, computed after the lane is read when the
# function is entered: {low} is (lane >> 4) * 8 + (lane & 3), the lane that holds what the thread
# gathers of B for the columns 0 to 3 of its eight threads, and {high} the one for the columns 4 to
# 7, sixteen lanes further; {half} is all ones in the upper half of each sixteen lanes (bit 3 of
# the lane), where the other parity of k comes first; {odd} is all ones in the threads with an
# odd number, whose results are in the last four of the eight registers that Volta's product writes.
KEPT = [
    'SHF.R.U32.HI {low}, RZ, 0x1, {lane}',
    'LOP3.LUT {low}, {low}, 0x8, RZ, 0xc0, !PT',
    'LOP3.LUT {low}, {low}, 0x3, {lane}, 0xf8, !PT',
    'LOP3.LUT {high}, {low}, 0x10, RZ, 0xfc, !PT',
    'SHF.L.U32 {half}, {lane}, 0x1c, RZ',
    'SHF.R.S32.HI {half}, RZ, 0x1f, {half}',
    'SHF.L.U32 {odd}, {lane}, 0x1f, RZ',
    'SHF.R.S32.HI {odd}, RZ, 0x1f, {odd}',
]
AROUND = ('wait', 'save', 'restore', 'lane', 'kept', 'settled', 'to pairs', 'from pairs', 'held: back', 'held: kept', 'late predicate')   # what is put around an instruction, and is not what it became
MEM_RE = re.compile(r'^mem: ')    # level 3: instructions around the core for the frame


def is_mem(what):
    """Whether `what` is a level-3 frame instruction (mem: base, mem: load URn, mem: store URn, mem: R1 -= N)."""
    return what.startswith('mem: ')

# ---- pairs of products on whole rows (docs/pairs.md) ---------------------------------------------
# Written from the design, not from the translator: what each kept value is as a function of the
# lane, which bytes of which row a thread loads, what a half of a pair is. The instructions that
# compute or load them are run here on the 32 lanes, and what they leave is compared with these.
def _bits(lane):
    return lane >> 4, lane >> 3 & 1, lane >> 2 & 1, lane >> 1 & 1, lane & 1        # h, sr, sc, k1, k0
ROW_VALUES = {
    'kA': [8 * k1 + 4 * sr + 2 * h + k0 for h, sr, sc, k1, k0 in map(_bits, range(32))],
    'kB': [8 * h + 4 * sc + 2 * k1 + k0 for h, sr, sc, k1, k0 in map(_bits, range(32))],
    'kBT': [8 * h + 2 * k1 + k0 for h, sr, sc, k1, k0 in map(_bits, range(32))],
    'cB': [8 * sc for h, sr, sc, k1, k0 in map(_bits, range(32))],
    'pi': [16 * sr + 8 * h + 4 * k0 + 2 * sc + k1 for h, sr, sc, k1, k0 in map(_bits, range(32))],
}
ROW_VALUES['inv'] = [ROW_VALUES['pi'].index(lane) for lane in range(32)]
ROW_VALUES['kA2'], ROW_VALUES['kB2'] = [x + 16 for x in ROW_VALUES['kA']], [x + 16 for x in ROW_VALUES['kB']]


def run_on_lanes(lines, state, written=None):
    """Run the instructions `lines` (as nvdisasm prints them) on the 32 lanes at once. `state`:
    {register: 32 values}, changed in place; `written`, a set, gets the registers they write. A
    value is a number; or ('row', lane, byte, parts): the address that `lane` has for its row (or
    none, when lane is None), plus `byte`, plus `parts` times the uniform part of the address; or
    ('word', lane, byte, parts): the four bytes of memory at such an address. The index of the
    first instruction this does not know, or None."""
    written = set() if written is None else written
    def value(operand, lane):
        if operand == 'RZ':
            return 0
        if operand.startswith('0x'):
            return int(operand, 16)
        return state.get(operand, [None] * 32)[lane]
    def add(x, y):
        if isinstance(x, int) and isinstance(y, int):
            return (x + y) & 0xffffffff
        x, y = (('row', None, v, 0) if isinstance(v, int) else v for v in (x, y))
        if not (isinstance(x, tuple) and isinstance(y, tuple) and x[0] == y[0] == 'row') or (x[1] is not None and y[1] is not None):
            return None
        return ('row', x[1] if x[1] is not None else y[1], x[2] + y[2], x[3] + y[3])
    for k, line in enumerate(lines):
        target = re.match(r'\S+ (?:PT, )?(R\d+),', line)
        if target:
            written.add(target.group(1))
            if line.startswith('LDS.U.64 '):
                written.add('R%d' % (int(target.group(1)[1:]) + 1))
        m = re.match(r'SHF\.L\.U32 (R\d+), (R\d+), (0x[0-9a-f]+), RZ$', line)
        if m:
            out = [None if not isinstance(value(m.group(2), l), int) else (value(m.group(2), l) << int(m.group(3), 16)) & 0xffffffff for l in range(32)]
            state[m.group(1)] = out
            continue
        m = re.match(r'SHF\.R\.(U|S)32\.HI (R\d+), RZ, (0x[0-9a-f]+), (R\d+)$', line)
        if m:
            def shifted(x):
                if not isinstance(x, int):
                    return None
                if m.group(1) == 'S' and x & 0x80000000:
                    x -= 1 << 32
                return (x >> int(m.group(3), 16)) & 0xffffffff
            state[m.group(2)] = [shifted(value(m.group(4), l)) for l in range(32)]
            continue
        m = re.match(r'LOP3\.LUT (R\d+), (R\d+|RZ), (R\d+|RZ|0x[0-9a-f]+), (R\d+|RZ), (0x[0-9a-f]+), !PT$', line)
        if m:
            table = int(m.group(5), 16)
            def looked(l):
                a, b, c = (value(m.group(n), l) for n in (2, 3, 4))
                if not all(isinstance(x, int) for x in (a, b, c)):
                    return None
                return sum(((table >> ((a >> bit & 1) << 2 | (b >> bit & 1) << 1 | (c >> bit & 1))) & 1) << bit for bit in range(32))
            state[m.group(1)] = [looked(l) for l in range(32)]
            continue
        m = re.match(r'IADD3 (R\d+), (R\d+), (R\d+|0x[0-9a-f]+), RZ$', line)
        if m:
            state[m.group(1)] = [add(value(m.group(2), l), value(m.group(3), l)) for l in range(32)]
            continue
        m = re.match(r'SHFL\.IDX PT, (R\d+), (R\d+), (R\d+), 0x1f$', line)
        if m:
            index = [value(m.group(3), l) for l in range(32)]
            state[m.group(1)] = [None if not isinstance(index[l], int) else value(m.group(2), index[l] & 31) for l in range(32)]
            continue
        m = re.match(r'LDS\.U(\.64)? (R\d+), \[(R\d+)(?:\+(0x[0-9a-f]+))?\]$', line)
        if m:
            for word in range(2 if m.group(1) else 1):
                at = [add(value(m.group(3), l), int(m.group(4) or '0', 16) + 4 * word) for l in range(32)]
                state['R%d' % (int(m.group(2)[1:]) + word)] = [None if not isinstance(x, tuple) or x[0] != 'row' or x[1] is None else ('word',) + x[1:] for x in at]
            continue
        return k
    return None


def rows_loaded(lines, source, kept, highest, uniform):
    """What the instructions `lines` load for the matrix load `source`, as rows: (role, {(fragment
    or pair, half of k): the first of its two registers}), or a sentence that says what is wrong.
    `kept`: {name of ROW_VALUES: register}; `uniform`: the register that stands for the uniform
    part of the address, or None."""
    m = re.match(r'LDSM\.16\.(M88|MT88)\.4 R(\d+), \[R(\d+)(?:\+(UR\d+))?([+-]0x[0-9a-f]+)?\]$', source)
    if not m:
        return 'not a load of four matrices under no predicate'
    transposed, rd, ra, off = m.group(1) == 'MT88', int(m.group(2)), 'R' + m.group(3), int(m.group(5) or '0', 16)
    state = {ra: [('row', lane, 0, 0) for lane in range(32)]}
    for name, register in kept.items():
        state[register] = list(ROW_VALUES[name])
    if uniform is not None:
        state[uniform] = [('row', None, 0, 1)] * 32
    written = set()
    at = run_on_lanes(lines, state, written)
    if at is not None:
        return 'instruction %d of it is not one of a whole-row load' % at
    parts = 1 if m.group(4) else 0
    loaded = {r: state[r] for r in written if all(isinstance(x, tuple) and x[0] == 'word' for x in state[r])}
    for role in ('BT',) if transposed else ('A', 'B'):
        where, used = {}, set()
        for q in range(2):
            for half in range(2):
                for word in range(2):
                    if role == 'BT':
                        want = [('word', 16 * q + ROW_VALUES['kBT'][l] + 4 * half, off + ROW_VALUES['cB'][l] + 4 * word, parts) for l in range(32)]
                    else:
                        want = [('word', 16 * q + ROW_VALUES['kA' if role == 'A' else 'kB'][l], off + 8 * half + 4 * word, parts) for l in range(32)]
                    found = [r for r, v in loaded.items() if v == want]
                    if len(found) != 1:
                        where = None
                        break
                    used.add(found[0])
                    if word == 0:
                        where[(q, half)] = int(found[0][1:])
                    elif int(found[0][1:]) != where[(q, half)] + 1:
                        where = None
                        break
                if where is None:
                    break
            if where is None:
                break
        if where is None:
            continue
        # Nothing of the program is written but, for the first half of k, the registers where the
        # program has the fragment; everything else is new registers, each for one thing.
        own = all(where[(q, 0)] == rd + 2 * q for q in range(2))
        new = [r for (q, half), r in where.items() if not (own and half == 0)]
        if any(r <= highest or r % 2 for r in new) or len(set(where.values())) != 4:
            return 'its rows are not in new registers (or, for the first half, where the program has them)'
        others = {int(r[1:]) for r in written - used}
        if any(r <= highest for r in others):
            return 'it writes a register of the program that is not one of its rows'
        return role, where, others
    return 'what it loads is not the rows of the design'



def matrix_product_expected(source, gathers):
    """The same for the product `source` (HMMA.1688 with accumulators in floats or in halves):
    two of Volta's products of 8x8x4, the first on the registers of A as they are and the second
    on them with their halves swapped ({x0}, {x1}); the elements of B in {y0} to {y3}; with
    accumulators in floats the results of the threads with an even number in the first four of
    Volta's eight registers and of the others in the last four; with accumulators in halves one
    exchange with the thread two lanes away before and one after.

    `gathers`: the sequence gathers the elements of B, from the neighbour (the lane 4 further)
    and from the lanes that hold what the thread gives its eight threads ({low}, {high}, {half},
    {odd}: what the function keeps of the lane, `KEPT`); or it takes them as an earlier product
    left them in {y0} to {y3} (what must hold then is checked where this is called). While B is
    gathered, three of the four registers of the sums ({s0} to {s3}) hold what is being
    gathered: the first step of a product, which writes them, waits for the shuffles that read
    them (`sequence_schedule`)."""
    m = re.match(r'HMMA\.1688\.F(32|16) R(\d+), R(\d+), R(\d+), (?:R(\d+)|RZ)$', source)
    if not m or (m.group(5) is None and m.group(1) != '32'):
        return None
    floats, rd, ra, rb = m.group(1) == '32', int(m.group(2)), int(m.group(3)), int(m.group(4))
    R = lambda n: 'R%d' % n
    # (The accumulators may be the zero register, with accumulators in floats: nothing to add to.)
    c0, c2 = ('RZ', 'RZ') if m.group(5) is None else (R(int(m.group(5))), R(int(m.group(5)) + 2))
    rc = None if m.group(5) is None else int(m.group(5))
    choose = lambda to, a, b, mask: 'LOP3.LUT %s, %s, %s, %s, 0xe4, !PT' % (to, a, b, mask)
    swapped = ['SHF.L.W.U32.HI {x0}, %s, 0x10, %s' % (R(ra), R(ra)), 'SHF.L.W.U32.HI {x1}, %s, 0x10, %s' % (R(ra + 1), R(ra + 1))]
    out = swapped
    if gathers:
        out = [
            'SHFL.BFLY PT, {s0}, %s, 0x4, 0x1f' % R(rb),
            'SHF.L.U32 {s1}, %s, 0x10, RZ' % R(rb),
        ] + swapped + [
            'SHF.R.U32.HI {s2}, RZ, 0x10, {s0}',
            'SHF.R.W.U32 {s1}, {s1}, 0x10, {s0}',
            'SHF.L.W.U32.HI {s2}, %s, 0x10, {s2}' % R(rb),
            choose('{s0}', '{s2}', '{s1}', '{half}'),
            choose('{s1}', '{s1}', '{s2}', '{half}'),
            'SHFL.IDX PT, {y0}, {s0}, {low}, 0x1f',
            'SHFL.IDX PT, {y1}, {s0}, {high}, 0x1f',
            'SHFL.IDX PT, {y2}, {s1}, {low}, 0x1f',
            'SHFL.IDX PT, {y3}, {s1}, {high}, 0x1f',
        ]
    if floats:
        step = 'HMMA.884.F32.F32.STEP%d %s, %s.COL, %s.ROW, %s'
        out = out + [step % (0, '{s0}', R(ra), '{y0}', c0), step % (1, '{s2}', R(ra), '{y0}', c2),
                     step % (2, R(rd), R(ra), '{y0}', c0), step % (3, R(rd + 2), R(ra), '{y0}', c2),
                     step % (0, '{s0}', '{x0}', '{y2}', '{s0}'), step % (1, '{s2}', '{x0}', '{y2}', '{s2}'),
                     step % (2, R(rd), '{x0}', '{y2}', R(rd)), step % (3, R(rd + 2), '{x0}', '{y2}', R(rd + 2))]
        return out + [choose(R(rd + k), R(rd + k), '{s%d}' % k, '{odd}') for k in range(4)]
    step = 'HMMA.884.F16.F16.STEP%d %s, %s.COL, %s.ROW, %s'
    return out + [
        'SHF.L.U32 {pair}, {lane}, 0x1e, RZ',
        'SHF.R.S32.HI {pair}, RZ, 0x1f, {pair}',
        choose('{s2}', R(rc), R(rc + 1), '{pair}'),
        'SHFL.BFLY PT, {s3}, {s2}, 0x2, 0x1f',
        choose('{s0}', '{s3}', R(rc), '{pair}'),
        choose('{s1}', R(rc + 1), '{s3}', '{pair}'),
        choose('{s2}', '{s3}', R(rc), '{pair}'),
        choose('{s3}', R(rc + 1), '{s3}', '{pair}'),
        step % (0, '{s0}', R(ra), '{y0}', '{s0}'), step % (1, '{s2}', R(ra), '{y0}', '{s2}'),
        step % (0, '{s0}', '{x0}', '{y2}', '{s0}'), step % (1, '{s2}', '{x0}', '{y2}', '{s2}'),
        choose('{s0}', '{s2}', '{s0}', '{odd}'),
        choose('{s1}', '{s3}', '{s1}', '{odd}'),
        choose('{s2}', '{s0}', '{s1}', '{pair}'),
        'SHFL.BFLY PT, {s3}, {s2}, 0x2, 0x1f',
        choose(R(rd), '{s3}', '{s0}', '{pair}'),
        choose(R(rd + 1), '{s1}', '{s3}', '{pair}'),
    ]


def matrix_product_level3_expected(source):
    """The expected sequence for a level-3 matrix product (HMMA.1688), where everything is taken
    at the instruction: the lane, the four values of the lane (low, high, half; odd into half's
    register after gathering), the gathering of B (always), the HMMA steps, and the selections.
    With accumulators in halves, also the pair computation and the exchanges before and after.

    At level 3 the scratch registers are taken individually: sums as two pairs from even (s0:s1
    and s2:s3), swapped as a pair from even (x0:x1), gathered B as two pairs from even (y0:y1
    and y2:y3). They need not be contiguous. `odd` is computed after gathering into the register
    that held `half` (which is dead once the choose operations of the gathering are done)."""
    m = re.match(r'HMMA\.1688\.F(32|16) R(\d+), R(\d+), R(\d+), (?:R(\d+)|RZ)$', source)
    if not m or (m.group(5) is None and m.group(1) != '32'):
        return None
    floats = m.group(1) == '32'
    rd, ra, rb = int(m.group(2)), int(m.group(3)), int(m.group(4))
    R = lambda n: 'R%d' % n
    c0, c2 = ('RZ', 'RZ') if m.group(5) is None else (R(int(m.group(5))), R(int(m.group(5)) + 2))
    rc = None if m.group(5) is None else int(m.group(5))
    choose = lambda to, a, b, mask: 'LOP3.LUT %s, %s, %s, %s, 0xe4, !PT' % (to, a, b, mask)
    # Kept computation from the lane (the same instructions as `kept()` for low, high, half):
    out = [
        'SHF.R.U32.HI {low}, RZ, 0x1, {lane}',
        'LOP3.LUT {low}, {low}, 0x8, RZ, 0xc0, !PT',
        'LOP3.LUT {low}, {low}, 0x3, {lane}, 0xf8, !PT',
        'LOP3.LUT {high}, {low}, 0x10, RZ, 0xfc, !PT',
        'SHF.L.U32 {half}, {lane}, 0x1c, RZ',
        'SHF.R.S32.HI {half}, RZ, 0x1f, {half}',
    ]
    # Gathering (always at level 3): B from the neighbour and the lanes that hold the columns.
    out += [
        'SHFL.BFLY PT, {s0}, %s, 0x4, 0x1f' % R(rb),
        'SHF.L.U32 {s1}, %s, 0x10, RZ' % R(rb),
        'SHF.L.W.U32.HI {x0}, %s, 0x10, %s' % (R(ra), R(ra)),
        'SHF.L.W.U32.HI {x1}, %s, 0x10, %s' % (R(ra + 1), R(ra + 1)),
        'SHF.R.U32.HI {s2}, RZ, 0x10, {s0}',
        'SHF.R.W.U32 {s1}, {s1}, 0x10, {s0}',
        'SHF.L.W.U32.HI {s2}, %s, 0x10, {s2}' % R(rb),
        choose('{s0}', '{s2}', '{s1}', '{half}'),
        choose('{s1}', '{s1}', '{s2}', '{half}'),
        'SHFL.IDX PT, {y0}, {s0}, {low}, 0x1f',
        'SHFL.IDX PT, {y1}, {s0}, {high}, 0x1f',
        'SHFL.IDX PT, {y2}, {s1}, {low}, 0x1f',
        'SHFL.IDX PT, {y3}, {s1}, {high}, 0x1f',
    ]
    # `odd` computed after gathering into the register that held `half` (dead after the two
    # choose operations above, whose last reader is the SHFL.IDX instructions):
    out += [
        'SHF.L.U32 {half}, {lane}, 0x1f, RZ',
        'SHF.R.S32.HI {half}, RZ, 0x1f, {half}',
    ]
    if floats:
        step = 'HMMA.884.F32.F32.STEP%d %s, %s.COL, %s.ROW, %s'
        out += [step % (0, '{s0}', R(ra), '{y0}', c0), step % (1, '{s2}', R(ra), '{y0}', c2),
                step % (2, R(rd), R(ra), '{y0}', c0), step % (3, R(rd + 2), R(ra), '{y0}', c2),
                step % (0, '{s0}', '{x0}', '{y2}', '{s0}'), step % (1, '{s2}', '{x0}', '{y2}', '{s2}'),
                step % (2, R(rd), '{x0}', '{y2}', R(rd)), step % (3, R(rd + 2), '{x0}', '{y2}', R(rd + 2))]
        # Selections: each thread picks its own result by `odd` (which is in {half}'s register).
        return out + [choose(R(rd + k), R(rd + k), '{s%d}' % k, '{half}') for k in range(4)]
    # Halves: pair computation, exchanges before, HMMA steps, exchanges after, selections.
    step = 'HMMA.884.F16.F16.STEP%d %s, %s.COL, %s.ROW, %s'
    return out + [
        'SHF.L.U32 {pair}, {lane}, 0x1e, RZ',
        'SHF.R.S32.HI {pair}, RZ, 0x1f, {pair}',
        choose('{s2}', R(rc), R(rc + 1), '{pair}'),
        'SHFL.BFLY PT, {s3}, {s2}, 0x2, 0x1f',
        choose('{s0}', '{s3}', R(rc), '{pair}'),
        choose('{s1}', R(rc + 1), '{s3}', '{pair}'),
        choose('{s2}', '{s3}', R(rc), '{pair}'),
        choose('{s3}', R(rc + 1), '{s3}', '{pair}'),
        step % (0, '{s0}', R(ra), '{y0}', '{s0}'), step % (1, '{s2}', R(ra), '{y0}', '{s2}'),
        step % (0, '{s0}', '{x0}', '{y2}', '{s0}'), step % (1, '{s2}', '{x0}', '{y2}', '{s2}'),
        # `odd` is in {half}'s register:
        choose('{s0}', '{s2}', '{s0}', '{half}'),
        choose('{s1}', '{s3}', '{s1}', '{half}'),
        choose('{s2}', '{s0}', '{s1}', '{pair}'),
        'SHFL.BFLY PT, {s3}, {s2}, 0x2, 0x1f',
        choose(R(rd), '{s3}', '{s0}', '{pair}'),
        choose(R(rd + 1), '{s1}', '{s3}', '{pair}'),
    ]


def unify(templates, got, bound):
    """Give the {names} of `templates` the registers that the texts `got` have in their places
    (`bound`: {name: register}, which may hold some already). The index of the first text that
    does not fit, or None when all do."""
    for k, template in enumerate(templates):
        if k >= len(got):
            return k
        pattern, names = '', []
        for piece in re.split(r'(\{\w+\})', template):
            if piece.startswith('{'):
                pattern += r'(R\d+)'
                names.append(piece[1:-1])
            else:
                pattern += re.escape(piece)
        m = re.match(pattern + '$', got[k])
        if not m:
            return k
        for name, value in zip(names, m.groups()):
            if bound.setdefault(name, value) != value:
                return k
    return None if len(got) == len(templates) else len(templates)


def sequence_schedule(mc, words, highest, product, borrowed=frozenset()):
    """What does not hold of the schedule of the sequence `words` that stands for a matrix
    instruction (`product`: 0 for a load, else the number of steps of one of Volta's products).

    An instruction of it that reads late a register of the translation: a later instruction of the
    sequence waits for a scoreboard that it names, or (products are done in order) that a later
    step names, and nothing before that wait writes a register of the translation that it reads.
    The steps of a product are as the compiler has them: two cycles apart (four on halves); the
    last of the first product names a scoreboard that the first of the second waits for; after
    the last of all, eight cycles (thirteen) before the next instruction, which waits for the
    scoreboard it names. On a V100, selections that wait for that scoreboard two cycles after the
    last step read registers that are not written yet."""
    out = []
    decoded = [mc.decode(70, w) for w in words]
    named = lambda w: {(w >> WRITE_BARRIER) & 7, (w >> READ_BARRIER) & 7} - {7}
    waits = lambda w, boards: any((w >> WAIT) & 63 >> b & 1 if False else ((w >> WAIT) & 63) >> b & 1 for b in boards)
    for k, d in enumerate(decoded):
        if d is None or d.kind not in machine.LATE:
            continue
        mine = {register for letter, register, role in d.reads if letter == 'R' and register > highest}
        if not mine:
            continue
        boards, ok = set(named(words[k])), False
        for j in range(k + 1, len(words)):
            if waits(words[j], boards):
                ok = True
                break
            if decoded[j] is None or any(letter == 'R' and register in mine for letter, register in decoded[j].writes):
                break
            if d.kind == 'HMMA' and decoded[j].kind == 'HMMA':
                boards |= named(words[j])
        if not ok:
            out.append((k, 'reads late a register of the translation, and nothing of its sequence waits for it before that register is written again'))
    if product:
        steps = [k for k, d in enumerate(decoded) if d is not None and d.kind == 'HMMA']
        apart, after = (2, 8) if product == 4 else (4, 13)
        if len(steps) != 2 * product:
            return out + [(0, 'not two products of %d steps' % product)]
        for k in steps[:-1]:
            if (words[k] >> USCHED) & 15 < apart:
                out.append((k, 'a step of a product less than %d cycles before the next' % apart))
        first, last = steps[product - 1], steps[-1]
        if (words[first] >> WRITE_BARRIER) & 7 == 7 or not waits(words[steps[product]], {(words[first] >> WRITE_BARRIER) & 7}):
            out.append((first, 'the second product does not wait for the last step of the first'))
        if (words[last] >> USCHED) & 15 < after:
            out.append((last, 'less than %d cycles after the last step of a product' % after))
        if (words[last] >> WRITE_BARRIER) & 7 == 7 or last + 1 >= len(words) or not waits(words[last + 1], {(words[last] >> WRITE_BARRIER) & 7}):
            out.append((last, 'what follows the last step of a product does not wait for it'))
    return out


def rows_schedule(mc, words, after, named):
    """What does not hold of the schedule of a matrix load done as whole rows (`words`; `after`:
    the instruction that follows it; `named`: the scoreboard on which the program's load said its
    result arrives, 7 for none). The shuffles that bring the addresses say on one scoreboard when
    they are there, and the first instruction that reads one waits for it; every load says on
    one scoreboard when it has read its address (a register of the translation, which the next
    sequence writes again), and what follows the last load waits for that; every load says where
    what it loads arrives: on the program's scoreboard, which the program's readers wait for, or,
    when the program named none, on one that the halves of pairs wait for (checked with them)."""
    out = []
    decoded = [mc.decode(70, w) for w in words]
    shuffles = [k for k, d in enumerate(decoded) if d is not None and d.mnemonic == 'SHFL']
    loads = [k for k, d in enumerate(decoded) if d is not None and d.mnemonic == 'LDS']
    waits = lambda w, board: ((w >> WAIT) & 63) >> board & 1
    if not shuffles or not loads or shuffles[-1] > loads[0]:
        return [(0, 'not shuffles of addresses, then loads')]
    boards = {(words[k] >> WRITE_BARRIER) & 7 for k in shuffles}
    if len(boards) != 1 or 7 in boards:
        out.append((shuffles[0], 'the shuffles of addresses do not say on one scoreboard when they are there'))
    elif not waits(words[shuffles[-1] + 1], next(iter(boards))):
        out.append((shuffles[-1] + 1, 'reads a shuffled address without waiting for the shuffles'))
    reads = {(words[k] >> READ_BARRIER) & 7 for k in loads}
    if len(reads) != 1 or 7 in reads:
        out.append((loads[0], 'the loads of rows do not say on one scoreboard when they have read their address'))
    elif not waits(words[loads[-1] + 1] if loads[-1] + 1 < len(words) else after, next(iter(reads))):
        out.append((loads[-1], 'what follows the last load of a row does not wait until the loads have read their address'))
    for k in loads:
        board = (words[k] >> WRITE_BARRIER) & 7
        if board == 7 or (named != 7 and board != named):
            out.append((k, 'a load of a row does not say what it loads is there on the scoreboard of the program\'s load'))
    return out


def halves_schedule(words):
    """The same for a product that is half of a pair (four steps): two cycles or more after each
    step, as the compiler has them; the last step says when the result is there, and the first
    waits for that scoreboard: for the products before it, which may have written what it reads."""
    out = [(k, 'a step of a product less than 2 cycles before what follows') for k in range(4) if (words[k] >> USCHED) & 15 < 2]
    board = (words[3] >> WRITE_BARRIER) & 7
    if board == 7:
        out.append((3, 'the last step of a half of a pair does not say when its result is there'))
    elif not ((words[0] >> WAIT) & 63) >> board & 1:
        out.append((0, 'the first step of a half of a pair does not wait for the products before it'))
    return out


def plan_checked(mc, code, tables, plan):
    """What the checker of plans (qcheck.py, written from the design apart from the planner,
    pairing.py) objects to in `plan` for the original function `code`: [(instruction, what)];
    None when that checker is not here."""
    try:
        import qcheck
    except ImportError:
        return None
    src = [mc.decode(75, int.from_bytes(code[at:at + 16], 'little')) for at in range(0, len(code), 16)]
    if None in src:
        return [(None, 'cannot check: an instruction of the original is not known')]
    try:
        return qcheck.check(mc, src, tables, plan)
    except Exception as e:
        # (A plan that the checker cannot even follow, such as a half of a pair without its
        # other half, which the validator has said above: no plan that the design allows.)
        return [(None, 'the checker of plans stopped on it (%s: %s)' % (type(e).__name__, e))]


def does_nothing(word):
    """Whether `word` is an instruction that does nothing (a NOP, whatever its schedule)."""
    return word & ~(0x3ffff << USCHED) == machine.NOP & ~(0x3ffff << USCHED)


def waited_after(code, at, board):
    """Whether what follows the instruction at `at` waits for scoreboard `board` before anything
    is done: the very next instruction, or the first one after instructions that do nothing.
    (The schedule puts such instructions where it needs more time than a stall count holds, and
    they do not wait: a wait holds for everything after it, and an instruction that does nothing
    reads and writes no register.) Returns that, and the offset of the instruction that should
    have waited (`at` itself when nothing follows)."""
    k = at + 16
    while k + 16 <= len(code):
        word = int.from_bytes(code[k:k + 16], 'little')
        if ((word >> WAIT) & 63) >> board & 1:
            return True, k
        if not does_nothing(word):
            return False, k
        k += 16
    return False, at


def schedule(mc, code, highest, sequences=frozenset(), borrowed=frozenset(), added=frozenset()):
    """What does not hold of the schedule of the translated `code` of a function whose original uses
    registers up to `highest`: (offset, what) for each, and how many late readers were looked at.
    `sequences`: the offsets of the instructions that stand for matrix instructions, whose late
    readers `sequence_schedule` looks at. `borrowed`: registers of the program (numbers <= highest)
    that the translation uses for its own purposes; `added`: the offsets of the instructions that
    are not the program's own, unchanged (in those, a borrowed register is one of the
    translation's: the kept copy of a scaled index at level 3, for one)."""
    words = [int.from_bytes(code[i:i + 16], 'little') for i in range(0, len(code) - 15, 16)]
    out, late = [], 0
    for k, w in enumerate(words):
        mask = (w >> WAIT) & 63
        distance = 0
        for j in range(k - 1, -1, -1):
            if not mask:
                break
            distance += (words[j] >> USCHED) & 15
            if distance >= 2:
                break
            if any(sb != 7 and mask >> sb & 1 for sb in ((words[j] >> WRITE_BARRIER) & 7, (words[j] >> READ_BARRIER) & 7)):
                out.append((16 * k, 'waits for a scoreboard named %d cycle before, at 0x%x' % (distance, 16 * j)))
                break
        d = mc.decode(70, w)
        if 16 * k in sequences:
            continue
        # A register of the program that the translation borrows, read late by an instruction that
        # is not the program's own: what next writes it, in the code that follows, comes after a
        # wait for that read. (Until 2026-11-23 these were left out: at level 3 an STS at the kept
        # copy of its scaled index, in a borrowed register that the next copy writes, could go
        # without the wait, unseen; the control `nowait` changed two Triton cubins so and was
        # objected to in neither. Not the rule below, the next instruction waiting: the homes of
        # uniform registers are read late and written again only much later.)
        if d is not None and d.kind in machine.LATE and 16 * k in added and not any(
                letter == 'R' and register > highest for letter, register, role in d.reads):
            mine = {register for letter, register, role in d.reads if letter == 'R' and register in borrowed}
            boards = {sb for sb in ((w >> WRITE_BARRIER) & 7, (w >> READ_BARRIER) & 7) if sb != 7}
            for j in range(k + 1, len(words)):
                x = words[j]
                if any(((x >> WAIT) & 63) >> sb & 1 for sb in boards):
                    break
                e = mc.decode(70, x)
                if e is None or e.kind == 'BRU':
                    break
                if mine & {register for letter, register in e.writes if letter == 'R'}:
                    late += 1
                    out.append((16 * k, 'reads late a register that the translation borrows, and the instruction at 0x%x writes it before a wait until it has read' % (16 * j)))
                    break
        if d is not None and d.kind in machine.LATE and any(
                letter == 'R' and register > highest for letter, register, role in d.reads):
            late += 1
            # (Instructions that do nothing and do not wait may come between, where the schedule
            # needed more time than a stall count holds.)
            waited = False
            for x in words[k + 1:]:
                mask = (x >> WAIT) & 63
                waited = any(sb != 7 and mask >> sb & 1 for sb in ((w >> WRITE_BARRIER) & 7, (w >> READ_BARRIER) & 7))
                if waited or not does_nothing(x):
                    break
            if not waited:
                out.append((16 * k, 'reads late a register of the translation, and the next instruction does not wait until it has read'))
        # A predicate read through a mask, by a P2R or an R2P (their operands do not name it):
        # what wrote it last, in the code that goes on to here, is
        # as far before as native code keeps the pair (4 cycles for plain arithmetic, 5 for an
        # R2P), or announced it on a scoreboard that is waited for by then. R2P reads the seven
        # predicates and writes them all back, those outside its mask as it read them: one
        # written closer gets its former value back (measured on a V100, 2026-11-18; the
        # translation had its restores one cycle apart and lost the first). A
        # P2R reads those of its mask. (Every P2R and every R2P, the program's own too: what
        # is before one may be an instruction of the translation.)
        if d is not None and d.mnemonic in ('R2P', 'P2R'):
            mask = d.ops['Sb'] & 0x7f if 'Sb' in d.ops else (0 if d.mnemonic == 'R2P' else 0x7f)
            wanted = {p for p in range(7) if bool(mask >> p & 1) == (d.mnemonic == 'P2R')}
            waited, distance = (w >> WAIT) & 63, 0
            for j in range(k - 1, -1, -1):
                if not wanted:
                    break
                distance += (words[j] >> USCHED) & 15
                before = mc.decode(70, words[j])
                if before is None:
                    break
                if before.kind == 'BRU' and before.mnemonic not in ('BSSY', 'BMOV'):
                    ops = before.ops
                    always = (ops.get('Pg', 7) == 7 and not ops.get('Pg@not') and ops.get('Pp', 7) == 7
                              and not ops.get('Pp@not') and ops.get('cond', 0) in (0, 1))
                    if before.mnemonic != 'BRA' or always:
                        break                        # what is before it did not run just before
                written = {register for letter, register in before.writes if letter == 'P' and register < 7}
                if before.mnemonic == 'R2P':
                    written |= {p for p in range(7) if 'Sb' not in before.ops or before.ops['Sb'] >> p & 1}
                board = (words[j] >> WRITE_BARRIER) & 7
                for p in sorted(written & wanted):
                    if board != 7:
                        if not waited >> board & 1:
                            out.append((16 * k, 'reads predicate P%d, which the instruction at 0x%x announces on scoreboard %d, and nothing waits for it by then' % (p, 16 * j, board)))
                    else:
                        least = mc.least('P', 'data', before, 'FXU')
                        if distance < least:
                            out.append((16 * k, 'reads predicate P%d %d cycles after the instruction at 0x%x writes it (%d or more)' % (p, distance, 16 * j, least)))
                wanted -= written
                waited |= (words[j] >> WAIT) & 63
        # A predicate given back from a register of the translation: 9 cycles or more after any
        # instruction that read what stood in it (as an operand, as a guard, or to copy it to a
        # register). (First taken, the day PyTorch's f32 attention for heads of 65 to 128
        # faulted, for what the reader needs: it is not, a reader does not go by what an R2P
        # issued after it gives back. It keeps the R2P nine cycles into its block: far from any
        # writer of a predicate that ran before the block on a path the check above does not
        # follow.)
        # (RZ is 255, not a register of the translation: a program's own R2P that restores from
        # RZ, or clears predicates, is not one of the translation's restores. cuDNN 9.24's sm_75
        # kernels have such R2Ps; they were counted as the translation's, 1,040 times.)
        if (d is not None and d.mnemonic == 'R2P' and d.ops.get('Ra', 0) > highest
                and d.ops.get('Ra', 0) != 255 and 'Sb' in d.ops):
            given, distance = d.ops['Sb'] & 0x7f, 0
            for j in range(k - 1, -1, -1):
                distance += (words[j] >> USCHED) & 15
                if distance >= 9:
                    break
                before = mc.decode(70, words[j])
                if before is None:
                    break
                read = {register for letter, register, role in before.reads if letter == 'P'}
                if before.mnemonic == 'P2R' and 'Sb' in before.ops:
                    read |= {p for p in range(7) if before.ops['Sb'] >> p & 1}
                if any(given >> p & 1 for p in read):
                    out.append((16 * k, 'gives a predicate back %d cycles after an instruction that reads it, at 0x%x' % (distance, 16 * j)))
                    break
                # (Not further back than a branch: what is before it may not have run just
                # before. BSSY and BMOV are of the branch unit and go on to the next instruction:
                # looked through.)
                if before.kind == 'BRU' and before.mnemonic not in ('BSSY', 'BMOV'):
                    break
    return out, late


def predicates_written(d):
    """The predicates (0 to 6) that the decoded instruction `d` may write: those its operands name,
    and for R2P those of its mask (all seven when the mask is not a number)."""
    out = {register for letter, register in d.writes if letter == 'P' and register < 7}
    if d.mnemonic == 'R2P':
        out |= {p for p in range(7) if 'Sb' not in d.ops or d.ops['Sb'] >> p & 1}
    return out


def data_overwritten(mc, code):
    """What does not hold of the registers that memory instructions read as data: (offset, what) for
    each, how many such registers were followed, and how many writes of them under the complement
    of a store's guard were left out.

    A store, an atomic operation and a shuffle read their data (what is stored, combined or
    exchanged) some time after they are issued, and native code counts on the memory instructions
    after them, which go in order, to write such a register only after it has been read: it
    writes one with a memory instruction at once (`ptxas`: a load from shared memory or a matrix
    load 4 cycles after a store of the register), and with any other instruction only after a
    barrier or a wait for a scoreboard that the reader itself or a later memory instruction names
    (none otherwise among 44,052 such writes in 821 sm_75 cubins, PyTorch's 475 and 346 of
    Triton, 2026-11-23). The address of a
    memory access is read when it is issued (native code writes it with arithmetic 6 cycles
    later, nothing between), and is not followed. So: from such an instruction on, in
    straight-line code, nothing but a memory instruction writes a register that it reads as data
    before a barrier or such a wait. (Until 2026-11-23 the sequence of a matrix load wrote the
    lanes of its rows into its destinations first, and on a V100 a store of one of them just
    before the load stored the lane. Not followed: past a branch, and from the end of a loop to
    its start.)

    But a store under a guard (an atomic operation or a reduction too) runs in the lanes where its
    predicate holds, as it held when the store was issued, and reads nothing in the others: an
    instruction under the complement of that guard, on the same predicate, which nothing between
    them writes (in straight-line code, as everything here), runs in those others alone, and may
    write the data there at once. `ptxas` does (`@P2 STS.64 [R6], R4` and then `@!P2 IMAD.MOV.U32
    R5, RZ, RZ, RZ`, no wait: 1,270 writes in 14 cubins of the held-out corpus, of cuSOLVER, NCCL
    and vLLM, 2026-11-23). Such a write is not objected to, and the register is still followed:
    in the lanes where the store ran it holds what the store may not have read yet. Not for a
    shuffle, which reads the register in other lanes than its own. (Until 2026-11-23 such a write
    was objected to.)"""
    words = [int.from_bytes(code[i:i + 16], 'little') for i in range(0, len(code) - 15, 16)]
    decoded = [mc.decode(70, w) for w in words]
    out, followed, complement = [], 0, 0
    for k, d in enumerate(decoded):
        if d is None or not d.kind.startswith('MIO'):
            continue
        address = re.match(r'(LD|ST|ATOM|RED)', d.mnemonic)
        if not (address and d.mnemonic[:2] != 'LD' or d.mnemonic == 'SHFL'):
            continue
        uses = mc._uses.get((70, d.c.name))
        if uses is None:
            continue
        data = set()
        for letter, operand, span in uses.reads:
            if letter == 'R' and operand in d.ops and d.ops[operand] != 255 and not (address and operand == 'Ra'):
                data |= {d.ops[operand] + n for n in range(machine.registers(span, d.ops))}
        if not data:
            continue
        followed += len(data)
        boards = {(words[k] >> WRITE_BARRIER) & 7, (words[k] >> READ_BARRIER) & 7} - {7}
        # The guard of a memory access, (predicate, negated), while nothing has written its predicate since.
        guard = (d.ops.get('Pg', 7), bool(d.ops.get('Pg@not'))) if address else None
        if guard is not None and guard[0] == 7:
            guard = None
        for j in range(k + 1, len(words)):
            y = decoded[j]
            if y is None or ((words[j] >> WAIT) & 63) & sum(1 << b for b in boards) or y.mnemonic == 'BAR':
                break
            written = data & {register for letter, register in y.writes if letter == 'R'}
            if written and not y.kind.startswith('MIO'):
                if guard is not None and y.ops.get('Pg', 7) == guard[0] and bool(y.ops.get('Pg@not')) != guard[1]:
                    complement += len(written)
                    written = set()         # (still followed: the lanes where the store ran)
                else:
                    out += [(16 * j, 'writes R%d, which the memory instruction at 0x%x reads as data and may not have read yet (nothing waited since for a scoreboard that it or a later memory instruction names, and no barrier)' % (r, 16 * k)) for r in sorted(written)]
            data -= written
            if guard is not None and guard[0] in predicates_written(y):
                guard = None
            if not data or y.kind == 'BRU' and y.mnemonic not in ('BSSY', 'BMOV'):
                break
            if y.kind.startswith('MIO'):
                boards |= {(words[j] >> WRITE_BARRIER) & 7, (words[j] >> READ_BARRIER) & 7} - {7}
    return out, followed, complement


# What Volta's own code keeps between an instruction of double precision and a reader of its
# result (docs/validation.md, section 8): written here, not taken from the translator.
DOUBLE = ('DADD', 'DMUL', 'DFMA', 'DSETP')
DOUBLE_PIPE, DOUBLE_OTHER, DOUBLE_GUARD = 8, 9, 15


def code_successors(mc, words, decoded):
    """For each instruction of sm_70 code, where control goes next that the code itself says:
    [(index, through a branch)]. The next instruction, unless the instruction always leaves (an
    unguarded BRA with no condition, JMP, BRX, JMX, RET or EXIT), and the target of a branch or
    jump whose operand is a distance in the code. Not known here, and not listed: where a branch
    through a register goes, where BSYNC and BREAK go back to, where a call returns from."""
    out = []
    for j, d in enumerate(decoded):
        nexts = []
        if d is None:
            out.append(nexts)
            continue
        target = None
        names = [n for n, o in d.c.operands.items() if o.type == 'RSImm']
        if names and d.mnemonic in ('BRA', 'BRX', 'JMP', 'JMX'):
            for step, fields, data in mc.d70.plan(d.c):
                if step == 'scaled' and data[0] == names[0]:
                    v = fields[0].get(d.word)
                    if v >> (len(fields[0]) - 1):
                        v -= 1 << len(fields[0])
                    target = j + 1 + v * data[1] // 16
        ops = d.ops
        always = (ops.get('Pg', 7) == 7 and not ops.get('Pg@not') and ops.get('Pp', 7) == 7
                  and not ops.get('Pp@not') and ops.get('cond', 0) in (0, 1))
        if not (d.mnemonic in ('BRA', 'EXIT', 'RET', 'JMP', 'BRX', 'JMX') and always) and j + 1 < len(words):
            nexts.append((j + 1, False))
        if target is not None and 0 <= target < len(words):
            nexts.append((target, True))
        out.append(nexts)
    return out


def double_read_early(mc, code):
    """What does not hold of the readers of results of double precision: (offset, what) for each,
    and how many readers were looked at. From each DADD, DMUL, DFMA or DSETP, along every path
    that the code says (the next instruction and the targets of its branches: `code_successors`),
    to the instructions that read what it wrote, until something writes it again: each comes as
    many cycles later as native sm_70 code keeps (never fewer in 54,085 pairs of NVIDIA's own
    code for sm_70, with a scoreboard or without; sm_80 code keeps fewer, and its translation
    must not). The cycles are those of the instructions passed on the path. (Until 2026-11-25 in
    straight-line code only: the reader at a branch's target was not looked at, and the
    translator did not keep it away either.)"""
    words = [int.from_bytes(code[i:i + 16], 'little') for i in range(0, len(code) - 15, 16)]
    decoded = [mc.decode(70, w) for w in words]
    nexts = code_successors(mc, words, decoded)
    stall = [(w >> USCHED) & 15 for w in words]
    out, looked = [], 0
    for k, d in enumerate(decoded):
        if d is None or d.mnemonic not in DOUBLE or not d.writes:
            continue
        # (instruction, what is still as the writer left it) -> the fewest cycles to there
        least, found = {}, {}
        stack = [(j, stall[k], frozenset(d.writes)) for j, branch in nexts[k]]
        while stack:
            j, distance, written = stack.pop()
            y = decoded[j]
            if y is None or distance >= DOUBLE_GUARD or least.get((j, written), DOUBLE_GUARD) <= distance:
                continue
            least[(j, written)] = distance
            for letter, register, role in y.reads:
                if (letter, register) not in written:
                    continue
                need = (DOUBLE_GUARD if letter == 'P' and (role == 'guard' or y.kind == 'BRU')
                        else DOUBLE_PIPE if y.kind == 'FMALITE' else DOUBLE_OTHER)
                key = (j, letter, register, role)
                if key not in found or distance < found[key][0]:
                    found[key] = (distance, need)
            written -= set(y.writes)
            if written:
                stack += [(n, distance + stall[j], written) for n, branch in nexts[j]]
        looked += len(found)
        for (j, letter, register, role), (distance, need) in sorted(found.items()):
            if distance < need:
                out.append((16 * j, 'reads %s%d %d cycles after the %s at 0x%x wrote it: Volta\'s own code never reads a result of double precision before %d' % (
                    letter, register, distance, d.mnemonic, 16 * k, need)))
    return out, looked


# ---- level 3: validation of "everything at the instruction" (the frame) ----

def check_frame_attributes(original, result, name, slots, frame_bytes, problems):
    """Rule A1: attributes 0x11 and 0x12 of the kernel in the translation = the original's + the
    frame; 0x23 too where present. Established by reading the .nv.info sections of both cubins.
    The attributes are in the global .nv.info section, keyed by symbol index."""
    # Find the symbol index of the kernel entry point.
    orig_syms = original.symbols()
    text_idx = next((k for k, s in enumerate(original.sections) if s.name == '.text.' + name and s.type == 1), None)
    if text_idx is None:
        return
    # The entry point: a function symbol with the kernel mark (st_other & 0x10).
    func_sym = next((k for k, sym in enumerate(orig_syms) if sym[3] == text_idx and sym[1] & 15 == 2 and sym[2] & 0x10), None)
    if func_sym is None:
        return
    def find_attrs(container, attr_id, sym_idx):
        """All values of `attr_id` for symbol `sym_idx` in the global .nv.info and per-function section."""
        found = []
        for section_name in ('.nv.info', '.nv.info.' + name):
            info = container.section(section_name)
            if info is None:
                continue
            d = info.data
            pos = 0
            while pos + 4 <= len(d):
                fmt, a, val = d[pos], d[pos + 1], struct.unpack_from('<H', d, pos + 2)[0]
                if fmt == 4 and a == attr_id and pos + 12 <= len(d):
                    sym = struct.unpack_from('<I', d, pos + 4)[0]
                    value = struct.unpack_from('<I', d, pos + 8)[0]
                    if sym == sym_idx:
                        found.append(value)
                pos += 4 + val if fmt == 4 else 4
        return found
    for attr in (0x11, 0x12, 0x23):
        orig_vals = find_attrs(original, attr, func_sym)
        new_vals = find_attrs(result, attr, func_sym)
        for val_o, val_n in zip(orig_vals, new_vals):
            if val_n != val_o + frame_bytes:
                problems.append('%s: A1: attribute 0x%02x is %d in the translation, expected %d (original %d + frame %d)' % (
                    name[:40], attr, val_n, val_o + frame_bytes, val_o, frame_bytes))
        if not orig_vals and attr in (0x11, 0x12):
            problems.append('%s: A1: attribute 0x%02x not found in the original' % (name[:40], attr))


def check_frame_slots(slots, frame_bytes, name, problems):
    """Rule A2: each slot is a multiple of 4, inside the frame, distinct; the two of a pair are
    adjacent with the lower register at the lower address.
    `slots`: {uniform register: bytes under the top of the thread's local memory}."""
    seen = set()
    for u, bytes_under in sorted(slots.items()):
        if bytes_under <= 0 or bytes_under > frame_bytes:
            problems.append('%s: A2: slot of UR%d (%d bytes under top) is outside the frame of %d bytes' % (name[:40], u, bytes_under, frame_bytes))
        if bytes_under % 4 != 0:
            problems.append('%s: A2: slot of UR%d is not at a multiple of 4 bytes' % (name[:40], u))
        if bytes_under in seen:
            problems.append('%s: A2: two uniform registers share the slot at %d bytes under the top' % (name[:40], bytes_under))
        seen.add(bytes_under)
    # Pairs: the lower register at the lower address (= higher bytes_under, since address = top - bytes_under).
    for u in sorted(slots):
        if u + 1 in slots:
            if abs(slots[u] - slots[u + 1]) != 4:
                problems.append('%s: A2: pair UR%d and UR%d are not in adjacent slots (at %d and %d bytes under top)' % (
                    name[:40], u, u + 1, slots[u], slots[u + 1]))
            elif slots[u] < slots[u + 1]:
                # bytes_under(u) < bytes_under(u+1) means address(u) > address(u+1):
                # u is at higher address, but it is the lower register and must be at lower address.
                problems.append('%s: A2: pair UR%d is at a higher address than UR%d+1 (the lower register must be at the lower address)' % (
                    name[:40], u, u))


def frame_completeness(mc, old, original_code, slots, loads, stores, at, say):
    """Rule B4, completeness. Every uniform register of the frame (`slots`) that the original
    instruction at `old` READS is loaded (`loads`); every one it WRITES is stored (`stores`).
    Where the original is under a guard, what it writes is loaded too: the store after it, under
    no guard, then puts back what was there when the instruction did not run. Asked also where
    nothing moves the frame around the instruction (until 2026-11-23 it was not: with every store
    of a function left out, the instructions that only write its uniform registers passed, and
    the loads after them read words that nothing stored). `at`: where to say it."""
    d75 = mc.decode(75, int.from_bytes(original_code[old:old + 16], 'little')) if old + 16 <= len(original_code) else None
    if d75 is not None and slots:
        uses = mc._uses.get((75, d75.c.name))
        if uses is not None:
            read_urs, written_urs = set(), set()
            for letter, operand, span in uses.reads:
                if letter == 'U' and operand in d75.ops and d75.ops[operand] != 63:
                    u, count = d75.ops[operand], machine.registers(span, d75.ops)
                    for k in range(count):
                        if u + k in slots:
                            read_urs.add(u + k)
            for letter, operand, span in uses.writes:
                if letter == 'U' and operand in d75.ops and d75.ops[operand] != 63:
                    u, count = d75.ops[operand], machine.registers(span, d75.ops)
                    for k in range(count):
                        if u + k in slots:
                            written_urs.add(u + k)
            loaded_urs = {int(re.match(r'mem: load UR(\d+)$', w).group(1)) for new, w in loads if re.match(r'mem: load UR(\d+)$', w)}
            stored_urs = {int(re.match(r'mem: store UR(\d+)$', w).group(1)) for new, w in stores if re.match(r'mem: store UR(\d+)$', w)}
            for ur in read_urs - loaded_urs:
                say(at, old, 'mem: completeness', 'B4: UR%d is read by the original and not loaded from memory' % ur)
            for ur in written_urs - stored_urs:
                say(at, old, 'mem: completeness', 'B4: UR%d is written by the original and not stored to memory' % ur)
            guarded = d75.ops.get('UPg', 7) != 7 or d75.ops.get('UPg@not') or d75.ops.get('Pg', 7) != 7 or d75.ops.get('Pg@not')
            if guarded:
                for ur in sorted(written_urs - loaded_urs):
                    say(at, old, 'mem: completeness', 'B4: the original is under a guard, and the old value of UR%d, which it writes, is not loaded from memory' % ur)


def check_frame_sequence(mc, name, old, parts, tb, ta, result_code, original_code,
                         slots, frame_bytes, highest, place, by_origin, listing,
                         problems, kinds, pool, borrowed_regs, borrow_pools_declared,
                         reg_lives, sequences, say, r1_depths=None):
    """Rules B and C for one original instruction that has mem: instructions around it.
    Returns (core_parts, mem_standins) where core_parts is what remains after stripping
    mem: instructions, and mem_standins is {uniform_register: ordinary_register} for this
    instruction's loads; or None if something is fatally wrong."""
    source = ta[old // 16]
    mem_parts = [(new, w) for new, w in parts if is_mem(w)]
    core_parts = [(new, w) for new, w in parts if not is_mem(w)]

    if not mem_parts:
        return core_parts, {}

    # Parse the mem: instructions.
    bases = [(new, w) for new, w in mem_parts if w == 'mem: base']
    loads = [(new, w) for new, w in mem_parts if w.startswith('mem: load ')]
    stores = [(new, w) for new, w in mem_parts if w.startswith('mem: store ')]
    r1_lower = [(new, w) for new, w in mem_parts if w.startswith('mem: R1 -=')]

    # B1: at most one mem: base, a MOV from c[0x0][0x28] into a register declared at this instruction.
    base_reg = None
    if len(bases) > 1:
        say(bases[1][0], old, 'mem: base', 'B1: more than one base load for a single instruction')
    for new, w in bases:
        t = tb[new // 16] or ''
        m = re.match(r'MOV (R\d+), c\[0x0\]\[0x28\]$', t)
        if not m:
            say(new, old, w, 'B1: not MOV Rb, c[0x0][0x28]')
        else:
            base_reg = m.group(1)
            base_rn = int(base_reg[1:])
            if base_rn <= highest and base_rn >= 2:
                borrowed_regs.add(base_rn)

    # B2: each load is LDL, 4 bytes, from the base at the slot of that uniform register.
    standins = {}    # {uniform register number: ordinary register name}
    load_boards = set()  # scoreboards named by loads
    last_load_new = None
    for new, w in loads:
        t = tb[new // 16] or ''
        m_load = re.match(r'mem: load UR(\d+)$', w)
        if not m_load:
            say(new, old, w, 'B2: cannot parse uniform register from kind')
            continue
        ur = int(m_load.group(1))
        # R1-addressed frame: expected offset is depth + frame_bytes - bytes_under_top.
        if base_reg is not None:
            expected_offset = -slots[ur] if ur in slots else None
        elif r1_depths is not None and ur in slots:
            depth_here = r1_depths.get(old // 16)
            expected_offset = (depth_here + frame_bytes - slots[ur]) if depth_here is not None else None
        else:
            expected_offset = None
        m = re.match(r'LDL(?:\.LU)? (R\d+), \[(R\d+)(?:\+?(-?0x[0-9a-f]+))?\]$', t)
        if not m:
            say(new, old, w, 'B2: not LDL Rv, [Rb-D]')
            continue
        rv, rb_got, offset_str = m.group(1), m.group(2), m.group(3) or '0x0'
        offset_val = int(offset_str, 16) if not offset_str.startswith('-') else -int(offset_str[1:], 16)
        if base_reg is not None and rb_got != base_reg:
            say(new, old, w, 'B2: load from %s, not the base %s' % (rb_got, base_reg))
        elif base_reg is None and rb_got != 'R1':
            say(new, old, w, 'B2: load from %s, not R1 (R1-addressed frame)' % rb_got)
        elif base_reg is None and expected_offset is None:
            # (Passed without a word until 2026-11-20: the offset was compared only where the
            # depth was known.)
            say(new, old, w, 'B2: load from the frame through R1 where the depth of R1 is not known')
        if expected_offset is not None and offset_val != expected_offset:
            say(new, old, w, 'B2: load at offset %d, expected %d for UR%d (slot %d bytes under top, frame %d)' % (
                offset_val, expected_offset, ur, slots.get(ur, -1), frame_bytes))
        rv_n = int(rv[1:])
        if rv_n <= highest and rv_n >= 2:
            borrowed_regs.add(rv_n)
        standins[ur] = rv
        # Check write scoreboard
        word = int.from_bytes(result_code[new:new + 16], 'little')
        wb = (word >> WRITE_BARRIER) & 7
        if wb == 7:
            say(new, old, w, 'B2: load does not name a write scoreboard')
        else:
            load_boards.add(wb)
        last_load_new = new

    # B2 (continued): every later reader of the loaded register in the sequence comes at or after
    # an instruction that waits for that scoreboard.
    if last_load_new is not None and load_boards:
        # The instruction right after the last load (`waited_after`), whatever it is of what
        # the instruction became: a wait holds for everything after it. (Until 2026-11-20 the
        # first of the parts given here, from which the save of a predicate has been taken
        # out: where a save stands between the loads and their reader and waits, the reader
        # was objected to; and where nothing of these parts followed, nothing was looked at.)
        for sb in sorted(load_boards):
            waited, where = waited_after(result_code, last_load_new, sb)
            if not waited:
                say(where, old, 'mem: load', 'B2: what follows the last load does not wait for scoreboard %d' % sb)

    # B3: each store is STL, 4 bytes, to the slot of the uniform register, of the register that
    # stands for it, under no guard, saying on a scoreboard when it has read. (Under the
    # original's guard the store would go by what the predicate holds after the instruction, and
    # an instruction may write the predicate that guards it: `@UP0 UIADD3 UR0, UP0, URZ, 0x0,
    # URZ` leaves its carry there. Without a guard the store is right whether the instruction ran
    # or not, the old value having been loaded before a guarded one: B4.)
    for new, w in stores:
        t = tb[new // 16] or ''
        m_store = re.match(r'mem: store UR(\d+)$', w)
        if not m_store:
            say(new, old, w, 'B3: cannot parse uniform register from kind')
            continue
        ur = int(m_store.group(1))
        # R1-addressed frame: expected offset is depth + frame_bytes - bytes_under_top.
        if base_reg is not None:
            expected_offset = -slots[ur] if ur in slots else None
        elif r1_depths is not None and ur in slots:
            depth_here = r1_depths.get(old // 16)
            expected_offset = (depth_here + frame_bytes - slots[ur]) if depth_here is not None else None
        else:
            expected_offset = None
        m = re.match(r'(@!?P\d )?STL \[(R\d+)(?:\+?(-?0x[0-9a-f]+))?\], (R\d+)$', t)
        if not m:
            say(new, old, w, 'B3: not STL [Rb-D], Rv')
            continue
        guard_got, rb_got, offset_str, rv = m.group(1), m.group(2), m.group(3) or '0x0', m.group(4)
        offset_val = int(offset_str, 16) if not offset_str.startswith('-') else -int(offset_str[1:], 16)
        if base_reg is not None and rb_got != base_reg:
            say(new, old, w, 'B3: store to %s, not the base %s' % (rb_got, base_reg))
        elif base_reg is None and rb_got != 'R1':
            say(new, old, w, 'B3: store to %s, not R1 (R1-addressed frame)' % rb_got)
        elif base_reg is None and expected_offset is None:
            say(new, old, w, 'B3: store to the frame through R1 where the depth of R1 is not known')
        if expected_offset is not None and offset_val != expected_offset:
            say(new, old, w, 'B3: store at offset %d, expected %d for UR%d' % (offset_val, expected_offset, ur))
        # Check the register is the stand-in for this UR. Where the instruction only writes the
        # uniform register, nothing was loaded and the register that is stored stands for it:
        # the rules of each kind then ask that what the instruction became writes that one.
        # (Until 2026-11-20 nothing tied the two: a store of any register passed.)
        if ur in standins and rv != standins[ur]:
            say(new, old, w, 'B3: stores %s but the stand-in of UR%d is %s' % (rv, ur, standins[ur]))
        standins.setdefault(ur, rv)
        if guard_got is not None:
            say(new, old, w, 'B3: store under a guard (%s): it would go by what the predicate holds after the instruction' % guard_got.strip())
        # Check read scoreboard: the store says when it has read.
        word = int.from_bytes(result_code[new:new + 16], 'little')
        rb_sb = (word >> READ_BARRIER) & 7
        if rb_sb == 7:
            say(new, old, w, 'B3: store does not say on a scoreboard when it has read')
    # B3 (continued): the instruction right after the last store waits for the store's read
    # scoreboard (`waited_after`: instructions that do nothing may stand between. Until
    # 2026-11-20 the very next one only: where the schedule had put a wait of its own after a
    # store, the instruction that does nothing was objected to, and the one after it waits).
    if stores:
        last_store_new = stores[-1][0]
        rb_sb = (int.from_bytes(result_code[last_store_new:last_store_new + 16], 'little') >> READ_BARRIER) & 7
        if rb_sb != 7:
            waited, where = waited_after(result_code, last_store_new, rb_sb)
            if not waited:
                say(where, old, 'mem: store', 'B3: the next instruction does not wait for the store\'s read scoreboard %d' % rb_sb)

    # B4: completeness (`frame_completeness`).
    frame_completeness(mc, old, original_code, slots, loads, stores, parts[0][0], say)

    # R1 lowering is handled separately.
    for new, w in r1_lower:
        # Handled by check_r1_lowered.
        pass

    # C2: no added instruction reads a declared register of its instruction before an instruction
    # of the same sequence has written it. This checks that mem: load / S2R instructions are
    # present before the core instruction reads the stand-in register.
    declared_regs = set(standins.values())
    if base_reg:
        declared_regs.add(base_reg)
    if declared_regs:
        all_sorted = sorted(parts, key=lambda x: x[0])
        written_so_far = set()
        for new, w in all_sorted:
            if is_mem(w) or w in AROUND:
                # These are added instructions that write declared registers.
                t = tb[new // 16] or ''
                target_m = re.match(r'\S+ (?:PT, )?(R\d+),', t)
                if target_m:
                    written_so_far.add(target_m.group(1))
                # Also LDL writes its destination.
                m_ldl = re.match(r'LDL(?:\.LU)? (R\d+),', t)
                if m_ldl:
                    written_so_far.add(m_ldl.group(1))
                # S2R writes its destination.
                m_s2r = re.match(r'S2R (R\d+),', t)
                if m_s2r:
                    written_so_far.add(m_s2r.group(1))
                continue
            # This is a core instruction or another kind. Check that declared registers it reads were written.
            d70 = mc.decode(70, int.from_bytes(result_code[new:new + 16], 'little'))
            if d70 is not None:
                for letter, register, *role in d70.reads:
                    r_name = 'R%d' % register if letter == 'R' else None
                    if r_name and r_name in declared_regs and r_name not in written_so_far:
                        say(new, old, w, 'C2: reads declared register %s before it is written by an instruction of the sequence' % r_name)
                # (What it writes is written for the instructions after it: the register that
                # stands for a uniform one the instruction only writes is among the declared.)
                written_so_far.update('R%d' % register for letter, register in d70.writes if letter == 'R')

    return core_parts, standins


LDL_RE = re.compile(r'LDL(?:\.LU)? (R\d+), \[(R\d+)(?:\+?(-?0x[0-9a-f]+))?\]$')
STL_RE = re.compile(r'(@!?P\d )?STL \[(R\d+)(?:\+?(-?0x[0-9a-f]+))?\], (R\d+)$')


def predicate_frame_moves(parts, tb, code, old, live_slots, depth, frame_bytes, temps, say):
    """Connect an actual frame load/store to its adjacent predicate conversion. Check the
    persistent address, transient register, direction, scoreboard and position around the core."""
    moves, converted, seen = set(), {}, set()
    core = [new for new, what in parts if not is_mem(what) and what not in AROUND and not is_room(what)]
    for i, (new, what) in enumerate(parts):
        action = re.fullmatch(r'mem: (load|store) UP(\d+)', what)
        if not action:
            continue
        load, q = action[1] == 'load', int(action[2])
        moves.add(new)
        slot = live_slots.get(q)
        if slot is None or depth is None:
            say(new, old, what, 'no live predicate frame slot or known R1 depth')
            continue
        pair = i + 1 if load else i - 1
        if (not 0 <= pair < len(parts) or parts[pair][1] != ('held: back' if load else 'held: kept')
                or not core or (parts[pair][0] >= min(core) if load else parts[pair][0] <= max(core))):
            say(new, old, what, 'predicate frame move is not paired with its conversion on the proper side of the instruction')
            continue
        conv = parts[pair][0]
        mem = (LDL_RE if load else STL_RE).fullmatch(tb[new // 16] or '')
        if not mem or mem[2] != 'R1' or (not load and mem[1]) or int(mem[3] or '0', 0) != depth + frame_bytes - slot:
            say(new, old, what, 'predicate frame move must access its own word through R1 without a guard')
            continue
        temp = mem[1] if load else mem[4]
        if int(temp[1:]) not in temps:
            say(new, old, what, 'predicate temporary %s was neither borrowed nor put away here' % temp)
        conversion = tb[conv // 16] or ''
        if not (re.fullmatch(r'ISETP\.NE\.U32\.AND P[0-6], PT, %s, RZ, PT' % temp, conversion) if load
                else re.fullmatch(r'P2R %s, PR, RZ, 0x[0-9a-f]+' % temp, conversion)):
            say(new, old, what, 'predicate frame move and conversion use different registers')
        if (q, load) in seen:
            say(new, old, what, 'repeated predicate frame move')
        seen.add((q, load))
        word = int.from_bytes(code[new:new + 16], 'little')
        board = (word >> (WRITE_BARRIER if load else READ_BARRIER)) & 7
        if board >= 6 or not waited_after(code, new, board)[0]:
            say(new, old, what, 'predicate frame move requires a scoreboard and a following wait')
        converted[conv] = 'frame-%d' % slot
    return moves, converted


def check_frame_values(mc, name, old, parts, tb, result_code, original_code, homes_here, held_here, say):
    """Rule V: what the moves of the frame carry. Rules B, P and R say what each load, store,
    register put away and register brought back is by itself: the word, the register, the
    scoreboard. This one follows the values through everything the instruction at `old` became
    (`parts`: all of it, the room made included), in the order of the code.

    V1, the registers. The register that stands for a uniform register of the frame here (the
    one its load writes, or the one its store reads when nothing is loaded) holds nothing else
    that the instruction needs: it is not a register that the instruction itself reads or
    writes, not the home of a uniform register declared there (`homes_here`: {register:
    uniform register}), not a register that holds a predicate there (`held_here`), and not
    the register of another uniform register of the frame at the same instruction. The same of
    the register that the top of the frame is loaded in, and of the temporary register that a
    uniform predicate kept in the frame is loaded in or stored from (`mem: load UP<n>`,
    `mem: store UP<n>`).

    V2, the order. A value has a name: what a register held before the sequence, what the
    frame holds of a uniform register, what an instruction of the sequence made. A load gives
    its register the value of its word, a store gives its word the value of its register, and
    so do what puts a register away and brings it back. Then:
    - what the instruction became reads, in the register of a uniform register, that register
      as loaded, or what the instruction made there; in any other register, nothing that a
      move of the frame put there, but for the conversion that reads a uniform predicate
      back from its temporary register;
    - a store is of a uniform register that the instruction writes, of the value that the
      instruction made in its register, and the last it makes there; once for each; a store
      of a uniform predicate is of what its conversion wrote down;
    - a register put away ends with what it held;
    - a load or a store of the frame, and what puts a register away or brings it back, takes
      its address from the stack pointer or from a register that holds, at that instruction,
      the top of the frame as `mem: base` loaded it.

    (Until 2026-11-20 none of this was asked: a store placed before the instruction that
    computes its value, or a load into the home of a uniform register that the instruction
    still read, passed. Until 2026-11-21 the values were followed and the addresses were not:
    a store placed before the load of the base it is addressed from passed, and wrote the
    word of another uniform register. Until 2026-11-22 the moves of uniform predicates kept in
    the frame were not followed: a load of a predicate into a register that the instruction
    read, or into the register of a uniform register loaded before it, passed.) Returns the
    registers of V1."""
    seq = sorted(parts)
    d75 = mc.decode(75, int.from_bytes(original_code[old:old + 16], 'little')) if old + 16 <= len(original_code) else None
    if d75 is None:
        return set()
    own = {r for l, r, *_ in d75.reads if l == 'R'} | {r for l, r in d75.writes if l == 'R'}
    written_u = {r for l, r in d75.writes if l == 'U'}
    # What stands where.
    standin, where, loaded, base, room, temporaries = {}, {}, set(), None, set(), []
    for new, w in seq:
        t = tb[new // 16] or ''
        move = re.match(r'mem: (load|store) UR(\d+)$', w)
        predicate = re.match(r'mem: (load|store) UP(\d+)$', w)
        if predicate:
            m = (LDL_RE if predicate.group(1) == 'load' else STL_RE).match(t)
            if m:               # (`predicate_frame_moves` says so when it is not)
                temporaries.append((int(m.group(1 if predicate.group(1) == 'load' else 4)[1:]), int(predicate.group(2)), new, w))
        elif move:
            load, u = move.group(1) == 'load', int(move.group(2))
            m = (LDL_RE if load else STL_RE).match(t)
            if not m:
                continue        # (rules B2 and B3 say so)
            x = int(m.group(1 if load else 4)[1:])
            if load:
                if u in loaded:
                    say(new, old, w, 'V: UR%d is loaded twice for one instruction' % u)
                loaded.add(u)
                standin[u], where[u] = x, (new, w)
            elif u not in standin:
                standin[u], where[u] = x, (new, w)
        elif w == 'mem: base':
            m = re.match(r'MOV (R\d+), c\[0x0\]\[0x28\]$', t)
            if m:
                base = (int(m.group(1)[1:]), new)
        elif is_room(w):
            m = re.match(r'room: R(\d+) (?:away|back)$', w)
            if m:
                room.add(int(m.group(1)))
    by_register = collections.defaultdict(list)
    for u, x in sorted(standin.items()):
        by_register[x].append(u)
    for x, us in sorted(by_register.items()):
        new, w = where[us[0]]
        if len(us) > 1:
            say(new, old, w, 'V: R%d stands for %s at once' % (x, ' and '.join('UR%d' % u for u in us)))
        if x in own:
            say(new, old, w, 'V: R%d stands for UR%d here, and the instruction itself reads or writes R%d' % (x, us[0], x))
        if x in homes_here:
            say(new, old, w, 'V: R%d stands for UR%d here, and is the home of UR%d there' % (x, us[0], homes_here[x]))
        if x in held_here:
            say(new, old, w, 'V: R%d stands for UR%d here, and holds a predicate there' % (x, us[0]))
    if base is not None:
        x, new = base
        if x in by_register or x in own or x in homes_here or x in held_here:
            say(new, old, 'mem: base', 'V: R%d, where the top of the frame is loaded, holds something else that the instruction needs' % x)
    for x, q, new, w in temporaries:
        if x in by_register:
            say(new, old, w, 'V: R%d is the temporary of UP%d here, and stands for UR%d' % (x, q, by_register[x][0]))
        if x in own:
            say(new, old, w, 'V: R%d is the temporary of UP%d here, and the instruction itself reads or writes R%d' % (x, q, x))
        if x in homes_here:
            say(new, old, w, 'V: R%d is the temporary of UP%d here, and is the home of UR%d there' % (x, q, homes_here[x]))
        if x in held_here:
            say(new, old, w, 'V: R%d is the temporary of UP%d here, and holds a predicate there' % (x, q))
        if base is not None and x == base[0]:
            say(new, old, w, 'V: R%d is the temporary of UP%d here, and the top of the frame is loaded in it' % (x, q))
    owner = {x: us[0] for x, us in by_register.items() if len(us) == 1}
    base = base[0] if base is not None else None

    def said(value):
        if value[0] == 'in':
            return 'what R%d held before the instruction' % value[1]
        if value[0] == 'ur':
            return 'UR%d as the frame has it' % value[1]
        if value[0] == 'up':
            return 'UP%d as the frame has it' % value[1]
        if value[0] == 'base':
            return 'the top of the frame'
        if value[0] == 'none':
            return 'a word in which nothing was put away'
        return 'what the instruction at 0x%x left there' % value[1]

    # The values, in the order of the code.
    reg, mem, stored, made = {}, {}, {}, {}
    kind = dict(seq)
    held = lambda r: reg.get(r, ('in', r))

    def addressed(new, w, register):
        # (From the stack pointer: rules B and P say how far under it. From any other register:
        # the top of the frame must be in it by now.)
        a = int(register[1:])
        if a != 1 and held(a) != ('base',):
            say(new, old, w, 'V: takes its address from R%d, which holds %s then, not the top of the frame' % (a, said(held(a))))
    for new, w in seq:
        t = tb[new // 16] or ''
        away = re.match(r'room: R(\d+) (away|back)$', w)
        move = re.match(r'mem: (load|store) UR(\d+)$', w)
        predicate = re.match(r'mem: (load|store) UP(\d+)$', w)
        if predicate:
            load, q = predicate.group(1) == 'load', int(predicate.group(2))
            m = (LDL_RE if load else STL_RE).match(t)
            if not m:
                continue
            addressed(new, w, m.group(2))
            if load:
                reg[int(m.group(1)[1:])] = ('up', q)
            else:
                x = int(m.group(4)[1:])
                value = held(x)
                if not (value[0] == 'made' and kind.get(value[1]) == 'held: kept'):
                    say(new, old, w, 'V: stores for UP%d what R%d holds then: %s, not what its conversion wrote down' % (q, x, said(value)))
        elif away:
            v = int(away.group(1))
            m = (STL_RE if away.group(2) == 'away' else LDL_RE).match(t)
            if not m:
                continue        # (rule P says so)
            addressed(new, w, m.group(2))
            if away.group(2) == 'away':
                mem['room', v] = held(int(m.group(4)[1:]))
            else:
                reg[int(m.group(1)[1:])] = mem.get(('room', v), ('none',))
        elif move:
            load, u = move.group(1) == 'load', int(move.group(2))
            m = (LDL_RE if load else STL_RE).match(t)
            if not m:
                continue
            addressed(new, w, m.group(2))
            if load:
                if u in stored:
                    say(new, old, w, 'V: UR%d is loaded after the instruction stored it' % u)
                reg[int(m.group(1)[1:])] = mem.get(('slot', u), ('ur', u))
            else:
                x = int(m.group(4)[1:])
                if u in stored:
                    say(new, old, w, 'V: UR%d is stored twice for one instruction' % u)
                stored[u] = (new, w, x, held(x))
                mem['slot', u] = held(x)
        elif w == 'mem: base':
            # (The register that this instruction loads, when it loads the top of the frame:
            # rule B1 says so when it does not.)
            m = re.match(r'MOV (R\d+), c\[0x0\]\[0x28\]$', t)
            if m:
                reg[int(m.group(1)[1:])] = ('base',)
        elif is_mem(w):
            continue            # (the stack pointer lowered at the first instruction: rule A3)
        else:
            d70 = mc.decode(70, int.from_bytes(result_code[new:new + 16], 'little'))
            if d70 is None:
                continue
            core = w not in AROUND
            for letter, r, *role in d70.reads:
                if letter != 'R':
                    continue
                value = held(r)
                if value[0] == 'up' and w == 'held: back':
                    continue        # (the conversion: `predicate_frame_moves` pairs it with its load)
                if r in owner:
                    if not (value == ('ur', owner[r]) or (value[0] == 'made' and value[2])):
                        say(new, old, w, 'V: reads R%d, which stands for UR%d here, when it holds %s' % (r, owner[r], said(value)))
                elif r == base:
                    if value != ('base',):
                        say(new, old, w, 'V: reads R%d for the top of the frame when it holds %s' % (r, said(value)))
                elif value[0] in ('ur', 'up', 'base', 'none') or (value[0] == 'in' and value[1] != r):
                    say(new, old, w, 'V: reads R%d, where a move of the frame has put %s' % (r, said(value)))
            for letter, r in d70.writes:
                if letter == 'R':
                    reg[r] = ('made', new, core)
                    if core:
                        made[r] = reg[r]
    for v in sorted(room):
        if held(v) != ('in', v):
            say(seq[-1][0], old, 'room: R%d' % v, 'V: R%d, put away for the instruction, is left with %s' % (v, said(held(v))))
    for u, (new, w, x, value) in sorted(stored.items()):
        if u not in written_u:
            say(new, old, w, 'V: stores UR%d, which the instruction does not write' % u)
        elif not (value[0] == 'made' and value[2]):
            say(new, old, w, 'V: stores for UR%d what R%d holds then: %s, not what the instruction made' % (u, x, said(value)))
        elif made.get(x) != value:
            say(new, old, w, 'V: stores UR%d from R%d before the last instruction that writes it there' % (u, x))
    return set(by_register) | ({base} if base is not None else set()) | {x for x, q, new, w in temporaries if w.startswith('mem: load ')}


def check_r1_lowered(mc, name, old, parts, tb, ta, result_code, original_code, frame_bytes, highest, listing, problems, say):
    """Rule A3: the original's first instruction sets R1 from c[0x0][0x28] unguarded, no other
    original instruction reads c[0x0][0x28], its translation is followed by exactly
    IADD3 R1, R1, -frame, RZ, and no other added instruction writes R1."""
    r1_parts = [(new, w) for new, w in parts if w.startswith('mem: R1 -=')]
    if not r1_parts:
        if frame_bytes > 0 and old == 0:
            say(parts[0][0], old, 'mem: R1', 'A3: frame of %d bytes, and R1 is not lowered' % frame_bytes)
        return
    # Check the original's first instruction sets R1 from c[0x0][0x28] unguarded.
    if old == 0 and old + 16 <= len(original_code):
        d75 = mc.decode(75, int.from_bytes(original_code[old:old + 16], 'little'))
        if d75 is not None:
            if ('R', 1) not in d75.writes:
                say(r1_parts[0][0], old, 'mem: R1', 'A3: the first instruction does not write R1')
            # Check unguarded.
            ops = d75.ops
            if ops.get('UPg', 7) != 7 or ops.get('UPg@not') or ops.get('Pg', 7) != 7 or ops.get('Pg@not'):
                say(r1_parts[0][0], old, 'mem: R1', 'A3: the first instruction is guarded')
        # Check no other original instruction reads c[0x0][0x28].
        for at in range(16, len(original_code), 16):
            t = ta[at // 16] or ''
            if 'c[0x0][0x28]' in t:
                say(r1_parts[0][0], old, 'mem: R1', 'A3: another original instruction at 0x%x reads c[0x0][0x28]' % at)
                break
    # Check the R1 lowering instruction: IADD3 R1, R1, -frame, RZ.
    for new, w in r1_parts:
        t = tb[new // 16] or ''
        # nvdisasm may print the negative as unsigned or signed.
        neg_unsigned = (1 << 32) - frame_bytes
        ok = t in ('IADD3 R1, R1, 0x%x, RZ' % neg_unsigned,
                    'IADD3 R1, R1, -0x%x, RZ' % frame_bytes)
        if not ok:
            say(new, old, w, 'A3: expected IADD3 R1, R1, -0x%x, RZ, got %s' % (frame_bytes, t))
    # No other added instruction writes R1.
    for new_at, old_at, what in listing:
        if what in ('same',) or what.startswith('mem: R1 -='):
            continue
        if what in AROUND:
            continue
        if is_mem(what):
            # mem: base, load, store: should not write R1.
            d70 = mc.decode(70, int.from_bytes(result_code[new_at:new_at + 16], 'little'))
            if d70 is not None and ('R', 1) in d70.writes:
                problems.append('%s+0x%x: A3: a mem: instruction writes R1: %s' % (name[:40], new_at, what))
            continue
        d70 = mc.decode(70, int.from_bytes(result_code[new_at:new_at + 16], 'little'))
        if d70 is not None and ('R', 1) in d70.writes and what != 'same':
            problems.append('%s+0x%x: A3: an added instruction writes R1: %s' % (name[:40], new_at, what))


# ---- rules P through U (room made, frame from R1) ----
# Rule P: the validator's own dataflow on the ORIGINAL code to track how far R1 is under where
# the first instruction sets it. The program moves R1 by IADD3 R1, R1, imm (unguarded) only;
# anything else that writes R1 makes it unknown; code that is called has the depth of its calls;
# where two paths disagree it is unknown.

def compute_r1_depth(mc, original_code, texts75, tables):
    """The depth of R1 at each instruction of the original, by the validator's own dataflow:
    how far R1 is, before each instruction, under where the first instruction puts it. The
    first instruction sets R1 (instruction 0: depth None). Returns {instruction_index: depth}
    where depth is None when unknown, 0 or a positive integer otherwise. Also returns whether
    the depth is known everywhere the first instruction leads (r1_fixed)."""
    n = len(original_code) // 16
    if n <= 1:
        return {}, True
    words = [int.from_bytes(original_code[i:i + 16], 'little') for i in range(0, len(original_code), 16)]
    decoded = [mc.decode(75, w) for w in words]

    import translate as TR

    # Build successor relation (same as register_lives: register liveness flow).
    meets = set()
    for i, d in enumerate(decoded):
        if d is not None and d.mnemonic in ('BSSY', 'PBK', 'SSY', 'PCNT'):
            rel = TR.relative(mc, 75, d) if hasattr(TR, 'relative') else None
            if rel is not None:
                t = 16 * i + 16 + TR.signed(rel[0], d.word) * rel[1]
                if t % 16 == 0 and 0 <= t < 16 * n:
                    meets.add(t // 16)
    succ = [[] for _ in range(n)]
    for i, d in enumerate(decoded):
        if d is None:
            succ[i] = [i + 1] if i + 1 < n else []
            continue
        ops = d.ops
        ug = (ops.get('Pg', 7) == 7 and not ops.get('Pg@not', 0)
              and ops.get('UPg', 7) == 7 and not ops.get('UPg@not', 0))
        after = [i + 1] if i + 1 < n else []
        m = d.mnemonic
        if m == 'EXIT':
            succ[i] = [] if ug else after
        elif m == 'RET':
            if d.c.name == 'ret__REL':
                targets = [j + 1 for j in range(n) if decoded[j] is not None and decoded[j].c.name.startswith('call_rel') and j + 1 < n]
                succ[i] = ([] if ug else after) + targets
            else:
                succ[i] = [] if ug else after
        elif m in ('BRX', 'JMX', 'JMP'):
            # A branch through a register leads where nvdisasm's listing says (`tables`), as
            # in `register_lives`. One that may lead anywhere leaves nothing known. (Taken to
            # lead to the next instruction only until 2026-11-20: what is reached through a
            # table alone had no depth, and room made there was objected to; and a depth
            # could be taken for known that another way to the same place did not have.)
            if m == 'BRX' and d.c.name == 'brx_' and 16 * i in tables:
                succ[i] = [t // 16 for t in tables[16 * i] if t % 16 == 0 and 0 <= t < 16 * n] + ([] if ug else after)
            else:
                return {}, False
        elif m == 'BRA':
            _true = lambda p: ops.get(p, 7) == 7 and not ops.get(p + '@not', 0)
            always = ug and _true('Pp') and ops.get('cond', 0) in (0, 1)
            rel = TR.relative(mc, 75, d)
            target = None
            if rel is not None:
                t = 16 * i + 16 + TR.signed(rel[0], d.word) * rel[1]
                if t % 16 == 0 and 0 <= t < 16 * n:
                    target = t // 16
            if always:
                succ[i] = [target] if target is not None else after
            else:
                succ[i] = after + ([target] if target is not None else [])
        elif m in ('BSSY', 'PBK', 'SSY', 'PCNT'):
            succ[i] = after
        elif m in ('BREAK', 'BRK', 'CONT', 'PEXIT', 'PRET', 'PLONGJMP', 'LONGJMP'):
            succ[i] = after + sorted(meets)
        elif d.c.name.startswith('call_rel'):
            succ[i] = after
        else:
            succ[i] = after

    # Identify internal calls and their targets.
    call_targets = {}   # {instruction index of CALL.REL: target instruction index}
    for i, d in enumerate(decoded):
        if d is not None and d.c.name.startswith('call_rel'):
            body_text = (texts75[i] or '').split(' ', 1)[1] if (texts75[i] or '').startswith('@') else (texts75[i] or '')
            m_hex = HEX.search(body_text)
            if m_hex:
                target = int(m_hex.group(0), 16) // 16
                if 0 <= target < n:
                    call_targets[i] = target

    # Forward dataflow: propagate depth from successors of instruction 0.
    depth = [None] * n
    r1_fixed = True
    work = [(t, 0) for t in succ[0]]
    while work and r1_fixed:
        i, at = work.pop()
        if depth[i] is not None:
            r1_fixed = depth[i] == at
            continue
        depth[i] = at
        d = decoded[i]
        after = at
        if d is not None:
            writes_r1 = any(l == 'R' and r == 1 for l, r in d.writes)
            if writes_r1:
                ops = d.ops
                ug = (ops.get('Pg', 7) == 7 and not ops.get('Pg@not', 0)
                      and ops.get('UPg', 7) == 7 and not ops.get('UPg@not', 0))
                if (d.c.name == 'iadd3_imm__RsIR_RIR' and ops.get('Rd') == 1 and ops.get('Ra') == 1
                        and ops.get('Rc', 255) == 255 and ug
                        and not any(ops.get(k) for k in ('Ra@negate', 'Rc@negate'))):
                    imm = ops['Sb'] - (1 << 32) if ops['Sb'] >> 31 else ops['Sb']
                    after = at - imm
                else:
                    r1_fixed = False
        work += [(t, after) for t in succ[i]]
        # Code that is called has the depth of its calls.
        if d is not None and i in call_targets:
            work.append((call_targets[i], at))
    # (Two ways to one place with two depths, or R1 written otherwise than by adding a number
    # to it: no depth is known anywhere. Until 2026-11-20 what had been found up to there was
    # returned, the place with two depths at the first of them.)
    if not r1_fixed:
        return {}, False
    return {i: depth[i] for i in range(n) if depth[i] is not None}, True


def is_room(what):
    """Whether `what` is a room instruction (room: Rn away or room: Rn back)."""
    return what.startswith('room: ')


def check_room(mc, name, old, parts, tb, result_code, original_code,
               frame_slots, frame_bytes, highest, reg_lives, r1_depths,
               rooms_declared, borrow_homes_list, helds_list, borrow_never,
               problems, kinds, say):
    """Rules P, R, S, T for one original instruction that has room: instructions.
    `parts` are the room instructions only (room: Rn away and room: Rn back).
    `r1_depths` is {instruction_index: depth} from compute_r1_depth.
    `rooms_declared` is [(register, [offsets])] from the .lives file for this function.
    Returns the registers that were put away (for the whole-truth check, rule U)."""
    i75 = old // 16
    n75 = len(original_code) // 16

    aways = [(new, w) for new, w in parts if 'away' in w]
    backs = [(new, w) for new, w in parts if 'back' in w]

    # Parse registers from the kind strings.
    def parse_room_reg(w):
        m = re.match(r'room: R(\d+) (away|back)', w)
        return int(m.group(1)) if m else -1

    away_regs = [parse_room_reg(w) for _, w in aways]
    back_regs = [parse_room_reg(w) for _, w in backs]
    all_regs = set(away_regs) | set(back_regs)

    # Rule S: which registers may be put away.
    d75 = mc.decode(75, int.from_bytes(original_code[old:old + 16], 'little')) if old + 16 <= len(original_code) else None
    if d75 is not None:
        orig_reads = {r for l, r, *_ in d75.reads if l == 'R' and r != 255}
        orig_writes = {r for l, r, *_ in d75.writes if l == 'R' and r != 255}
        # S: not read and not written by the original instruction.
        for v in all_regs:
            if v in orig_reads:
                say(parts[0][0], old, 'room: R%d' % v, 'S: register R%d is read by the original instruction' % v)
            if v in orig_writes:
                say(parts[0][0], old, 'room: R%d' % v, 'S: register R%d is written by the original instruction' % v)
        # S: not R0 or R1.
        for v in all_regs:
            if v <= 1:
                say(parts[0][0], old, 'room: R%d' % v, 'S: register R%d is R0 or R1' % v)
        # S: not the home of a home line whose offsets include this instruction.
        home_regs_here = set()
        for u, r, offsets in borrow_homes_list:
            if old in offsets:
                home_regs_here.add(r)
        for v in all_regs:
            if v in home_regs_here:
                say(parts[0][0], old, 'room: R%d' % v, 'S: register R%d is a home alive at this instruction' % v)
        # S: not a held register here.
        held_regs_here = set()
        for q, register, offsets in helds_list:
            rn = int(register[1:]) if register.startswith('R') else -1
            if old in offsets and rn >= 2:
                held_regs_here.add(rn)
        for v in all_regs:
            if v in held_regs_here:
                say(parts[0][0], old, 'room: R%d' % v, 'S: register R%d is held at this instruction' % v)
        # S: the original instruction always goes on to the next one and nowhere else
        # (no branch, call, return, exit) and does not write R1.
        ops = d75.ops
        ug = (ops.get('Pg', 7) == 7 and not ops.get('Pg@not', 0)
              and ops.get('UPg', 7) == 7 and not ops.get('UPg@not', 0))
        is_branch = d75.kind == 'BRU' and d75.mnemonic not in ('BSSY', 'BMOV')
        is_call = d75.c.name.startswith(('call', 'jcal'))
        is_ret = d75.mnemonic in ('RET', 'EXIT')
        if is_branch or is_call or is_ret:
            say(parts[0][0], old, 'room', 'S: room made at an instruction that is a branch, call, return or exit')
        if not ug and (is_branch or is_call or is_ret):
            pass   # already reported
        elif not ug and d75.kind == 'BRU':
            pass   # BSSY/BMOV under a guard: still goes to next
        writes_r1 = any(l == 'R' and r == 1 for l, r in d75.writes)
        if writes_r1:
            say(parts[0][0], old, 'room', 'S: room made at an instruction that writes R1')
        # S: not a register the function writes late without naming a scoreboard ("never").
        if borrow_never is not None:
            for v in all_regs:
                if borrow_never >> v & 1:
                    say(parts[0][0], old, 'room: R%d' % v, 'S: register R%d is written late without a scoreboard somewhere in the function' % v)

    # Rule R: what is put away comes back.
    # For each register: exactly one away, exactly one back, from the same word.
    away_counts = collections.Counter(away_regs)
    back_counts = collections.Counter(back_regs)
    for v in all_regs:
        if away_counts[v] != 1:
            say(parts[0][0], old, 'room: R%d' % v, 'R: register R%d has %d away instructions (expected 1)' % (v, away_counts[v]))
        if back_counts[v] != 1:
            say(parts[0][0], old, 'room: R%d' % v, 'R: register R%d has %d back instructions (expected 1)' % (v, back_counts[v]))

    # Rule P: check the offsets. The depth at this instruction must be known.
    depth = r1_depths.get(i75)
    if depth is None and all_regs:
        say(parts[0][0], old, 'room', 'P: room made at instruction 0x%x where the depth of R1 is not known' % old)
    n_slots = len(frame_slots)
    if depth is not None:
        # Check away (STL) instructions.
        for k, (new, w) in enumerate(aways):
            v = away_regs[k]
            t = tb[new // 16] or ''
            # Expected: STL [R1 + (depth + 4*(n_slots + index_of_v_among_aways))], Rv
            # The index among aways for this register.
            indices_so_far = [j for j in range(k + 1) if away_regs[j] == v]
            room_index = len([j for j, (_, ww) in enumerate(aways) if j < k and parse_room_reg(ww) != v]) + (indices_so_far[-1] if indices_so_far else 0)
            # Actually: the room words go at positions n_slots, n_slots+1, etc. in order of the aways.
            expected_offset = depth + 4 * (n_slots + k)
            m = re.match(r'(@!?P\d )?STL \[(R\d+)(?:\+?(-?0x[0-9a-f]+))?\], (R\d+)$', t)
            if not m:
                say(new, old, w, 'P: not STL [R1+offset], Rv')
            else:
                rb_got = m.group(2)
                offset_str = m.group(3) or '0x0'
                offset_val = int(offset_str, 16) if not offset_str.startswith('-') else -int(offset_str[1:], 16)
                rv = int(m.group(4)[1:]) if m.group(4).startswith('R') else -1
                if rb_got != 'R1':
                    say(new, old, w, 'P: away from %s, not R1' % rb_got)
                if offset_val != expected_offset:
                    say(new, old, w, 'P: away offset 0x%x, expected 0x%x (depth %d + 4*(%d+%d))' % (offset_val, expected_offset, depth, n_slots, k))
                if rv != v:
                    say(new, old, w, 'P: stores R%d but should store R%d' % (rv, v))
                # Under no guard: where it does not hold, the register would not be put away, and
                # it is brought back all the same (a load under a guard is not the LDL above).
                # (Until 2026-11-23 a guard here passed: rule V follows values, not lanes.)
                if m.group(1):
                    say(new, old, w, 'R: puts R%d away under a guard: where it does not hold, R%d is not put away and is brought back all the same' % (v, v))
                # Check the STL says when it has read (read scoreboard).
                word = int.from_bytes(result_code[new:new + 16], 'little')
                rb_sb = (word >> READ_BARRIER) & 7
                if rb_sb == 7:
                    say(new, old, w, 'R: away does not say on a scoreboard when it has read')
        # Check back (LDL) instructions.
        for k, (new, w) in enumerate(backs):
            v = back_regs[k]
            t = tb[new // 16] or ''
            # The back must be from the same word as the away of the same register.
            away_index = next((j for j, (_, ww) in enumerate(aways) if parse_room_reg(ww) == v), None)
            if away_index is None:
                say(new, old, w, 'R: register R%d has back without away' % v)
                continue
            expected_offset = depth + 4 * (n_slots + away_index)
            m = re.match(r'LDL(?:\.LU)? (R\d+), \[(R\d+)(?:\+?(-?0x[0-9a-f]+))?\]$', t)
            if not m:
                say(new, old, w, 'P: not LDL Rv, [R1+offset]')
            else:
                rv = int(m.group(1)[1:]) if m.group(1).startswith('R') else -1
                rb_got = m.group(2)
                offset_str = m.group(3) or '0x0'
                offset_val = int(offset_str, 16) if not offset_str.startswith('-') else -int(offset_str[1:], 16)
                if rb_got != 'R1':
                    say(new, old, w, 'P: back from %s, not R1' % rb_got)
                if offset_val != expected_offset:
                    say(new, old, w, 'P: back offset 0x%x, expected 0x%x (must match away)' % (offset_val, expected_offset))
                if rv != v:
                    say(new, old, w, 'R: loads into R%d but should load R%d' % (rv, v))
                # Check the LDL names a write scoreboard.
                word = int.from_bytes(result_code[new:new + 16], 'little')
                wb = (word >> WRITE_BARRIER) & 7
                if wb == 7:
                    say(new, old, w, 'R: back does not name a write scoreboard')
        # Rule R: the instruction after the last back waits for its scoreboard.
        if backs:
            last_back_new = backs[-1][0]
            last_back_word = int.from_bytes(result_code[last_back_new:last_back_new + 16], 'little')
            back_sb = (last_back_word >> WRITE_BARRIER) & 7
            if back_sb != 7:
                waited, where = waited_after(result_code, last_back_new, back_sb)
                if not waited:
                    say(where, old, 'room: back', 'R: the instruction after the last back does not wait for scoreboard %d' % back_sb)
        # Rule R: the away must come before writes, the back after reads/writes.
        # Two registers put away must each be at a different word (different offsets).
        offsets_used = set()
        for k in range(len(aways)):
            expected_offset = depth + 4 * (n_slots + k)
            if expected_offset in offsets_used:
                say(aways[k][0], old, aways[k][1], 'P: two registers put away in the same word (offset 0x%x)' % expected_offset)
            offsets_used.add(expected_offset)
    # Check the next instruction after the last STL (away) waits for its read scoreboard.
    if aways:
        last_away_new = aways[-1][0]
        last_away_word = int.from_bytes(result_code[last_away_new:last_away_new + 16], 'little')
        away_sb = (last_away_word >> READ_BARRIER) & 7
        if away_sb != 7:
            waited, where = waited_after(result_code, last_away_new, away_sb)
            if not waited:
                say(where, old, 'room: away', 'R: the instruction after the last away does not wait for the store\'s read scoreboard %d' % away_sb)
    return all_regs


def check_room_declared(rooms_declared, name, old, all_regs, problems, say):
    """Rule U supplement: every register written by room instructions at this offset has a room
    line in the .lives file for it."""
    declared_at = set()
    for r, offsets in rooms_declared:
        if old in offsets:
            declared_at.add(r)
    for v in all_regs:
        if v not in declared_at:
            say(0, old, 'room: R%d' % v, 'U: register R%d put away at 0x%x but no room line in .lives' % (v, old))
    return declared_at


def check_frame_room_size(frame_slots, frame_bytes, rooms_declared, name, problems):
    """Rule Q supplement: the frame holds every slot and every word of room used. The frame must
    be at least 4*(number of slots + max registers put away at once) bytes, rounded up to 16."""
    n_slots = len(frame_slots)
    # Group room lines by offset to find the max number at one instruction.
    by_offset = collections.defaultdict(set)
    for r, offsets in rooms_declared:
        for off in offsets:
            by_offset[off].add(r)
    max_room = max((len(regs) for regs in by_offset.values()), default=0)
    needed = (4 * (n_slots + max_room) + 15) & ~15
    if frame_bytes < needed:
        problems.append('%s: Q: frame is %d bytes, but needs %d (slots %d + room %d, rounded to 16)' % (
            name[:40], frame_bytes, needed, n_slots, max_room))


def check_r1_frame_offsets(mc, name, old, parts, tb, result_code, frame_slots,
                           frame_bytes, r1_depths, problems, say):
    """Rule P for mem: load and mem: store instructions when the frame is addressed from R1
    (no mem: base instruction). The expected offset is depth + frame_bytes - bytes_under_top."""
    i75 = old // 16
    depth = r1_depths.get(i75)
    if depth is None:
        return
    loads = [(new, w) for new, w in parts if w.startswith('mem: load ')]
    stores = [(new, w) for new, w in parts if w.startswith('mem: store ')]
    for new, w in loads + stores:
        m_ur = re.match(r'mem: (?:load|store) UR(\d+)$', w)
        if not m_ur:
            continue
        ur = int(m_ur.group(1))
        if ur not in frame_slots:
            continue
        bytes_under_top = frame_slots[ur]
        expected_offset = depth + frame_bytes - bytes_under_top
        t = tb[new // 16] or ''
        if w.startswith('mem: load'):
            m = re.match(r'LDL(?:\.LU)? (R\d+), \[(R\d+)\+?(-?0x[0-9a-f]+)\]$', t)
        else:
            m = re.match(r'(@!?P\d )?STL \[(R\d+)\+?(-?0x[0-9a-f]+)\], (R\d+)$', t)
        if not m:
            continue
        if w.startswith('mem: load'):
            rb_got = m.group(2)
            offset_str = m.group(3)
        else:
            rb_got = m.group(2)
            offset_str = m.group(3)
        offset_val = int(offset_str, 16) if not offset_str.startswith('-') else -int(offset_str[1:], 16)
        if rb_got == 'R1' and offset_val != expected_offset:
            say(new, old, w, 'P: offset 0x%x from R1, expected 0x%x (depth %d + frame %d - slot %d)' %
                (offset_val, expected_offset, depth, frame_bytes, bytes_under_top))


def main():
    original, result = elf.Elf(open(sys.argv[1], 'rb').read()), elf.Elf(open(sys.argv[2], 'rb').read())
    mc = machine.Machine()
    rows = collections.defaultdict(list)
    for line in open(sys.argv[3]):
        name, new, old, what = line.rstrip('\n').split('\t')
        rows[name].append((int(new, 16), int(old, 16), what))
    problems, checked, kinds = [], 0, collections.Counter()
    lives, tables, recounted = alive(sys.argv[1])
    if recounted:
        recounted_funcs = sum(1 for name, chart in lives.items() if chart)
    # (`held`: the lives that the translator keeps in a register between the instructions that
    # name the uniform predicate: {function: [(uniform predicate, register, offsets)]}.)
    claims, helds = collections.defaultdict(list), collections.defaultdict(list)
    held_slots = collections.defaultdict(list)
    borrow_homes, borrow_pools = collections.defaultdict(list), collections.defaultdict(list)
    # Level 3: slots and frame from .lives.
    borrow_slots = collections.defaultdict(dict)   # {function: {uniform register: bytes under top}}
    borrow_frames = {}                              # {function: frame bytes}
    lives_unknown = collections.Counter()
    # ---- room lines from .lives ----
    borrow_rooms = collections.defaultdict(list)       # {function: [(register, [original_offset])]}
    if os.path.exists(sys.argv[3] + '.lives'):
        for line in open(sys.argv[3] + '.lives'):
            name, q, p, how, offsets = line.rstrip('\n').split('\t')
            off_list = [int(x, 16) for x in offsets.split(',') if x] if offsets else []
            if how == 'held':
                helds[name].append((int(q), 'R%s' % p, off_list))
            elif how == 'held-slot':
                held_slots[name].append((int(q), int(p), off_list))
            elif how == 'home':
                borrow_homes[name].append((int(q), int(p), off_list))
            elif how == 'pool':
                borrow_pools[name].append((int(q), int(p), off_list))
            elif how in ('free', 'saved'):
                claims[name].append((int(q), int(p), how == 'saved', off_list))
            elif how == 'slot':
                borrow_slots[name][int(q)] = int(p)
            elif how == 'frame':
                borrow_frames[name] = int(q)
            elif how == 'room':
                # Room line: width=int(q), register=int(p), at original offsets.
                borrow_rooms[name].append((int(p), off_list))
            else:
                lives_unknown[how] += 1
    # Every function of the result is either in the map, or what a refused function becomes: a
    # trap and a branch to itself, exits where the function had them, and nothing else. (The map
    # is the translator's own: what it leaves out would not be looked at.)
    code_of = lambda container: {x.name for x in container.sections if x.name.startswith('.text.') and x.type == 1}
    if code_of(original) != code_of(result):
        problems.append('the result does not have the functions of the original: %s' % sorted(code_of(original) ^ code_of(result))[:3])
    for section in result.sections:
        if section.name.startswith('.text.') and section.type == 1 and section.name[len('.text.'):] not in rows:
            said = texts(section.data, 70)
            stub = len(section.data) % 16 == 0 and bool(said) and said[0] in ('BPT.TRAP 0x1', 'EXIT')
            stub = stub and all(t in ('NOP', 'EXIT', 'BPT.TRAP 0x1') or (k == 1 and t == 'BRA 0x10') for k, t in enumerate(said))
            if not stub:
                problems.append('%s: a function of the result that the map does not have and that is not a refused one' % section.name[len('.text.'):][:60])
    # Relocations: what the driver, or a linker, writes into the instructions and the data they
    # name once it knows the addresses. The result has those of the original, and nothing of
    # them changed but where those on translated code apply (checked with each function below).
    # A place in code that one names would have moved with the translation, and nothing carries
    # that over: no function that one names, and no function that holds such a one, may be
    # translated. (Until 2026-11-17 nothing here looked at relocations, and those on code stayed
    # where they were, on other instructions.)
    on_code = collections.defaultdict(list)
    before, after = relocations(original), relocations(result)
    if [r[:3] + r[4:] for r in before] != [r[:3] + r[4:] for r in after]:
        problems.append('the relocations of the result are not those of the original, where they apply apart')
    else:
        for (section, at, on, offset, info, addend, names), (_, _, _, now, _, _, _) in zip(before, after):
            for f in sorted({on, names} - {None}) if names is not None else ():
                if f in rows:
                    problems.append('%s+0x%x: a relocation names a place in code, and %s, which it %s, is translated' % (
                        section[:40], at, f[:40], 'is in' if f == on else 'names'))
            if on in rows:
                on_code[on].append((section, at, offset, info & 0xffffffff, now))
            elif now != offset:
                problems.append('%s+0x%x: a relocation applies elsewhere than in the original, and its code is not translated' % (section[:40], at))
    # {index of a section: where global function symbols point in it, other than its start}: the
    # places where other code, or the driver, may enter it. (The helpers that the compiler puts at
    # the end of a section are weak symbols, which the section's own calls go to; anything else
    # that names a place in code does it by a relocation, and is refused above.)
    entered_at = collections.defaultdict(list)
    for symbol in original.symbols():
        if symbol[1] & 15 == 2 and symbol[1] >> 4 == 1 and symbol[4] != 0:
            entered_at[symbol[3]].append(symbol[4])
    for name, listing in rows.items():
        a, b = original.section('.text.' + name), result.section('.text.' + name)
        if a is None or b is None:
            problems.append('%s: in the map, and not a function of %s' % (name[:60], 'the original' if a is None else 'the result'))
            continue
        # The map has every instruction of the result once, and every instruction of the
        # original (which may have become several).
        news, olds = sorted(new for new, old, what in listing), {old for new, old, what in listing}
        if news != list(range(0, len(b.data), 16)) or olds != set(range(0, len(a.data), 16)):
            problems.append('%s: the map does not have every instruction of the result once and every instruction of the original' % name[:60])
            continue
        # In the order of the result (not of the lines of the map), the instructions come from
        # the original in its order: what one instruction became is in one piece, before what the
        # next became. The translator moves nothing, and each instruction is checked below by
        # itself: two that are exchanged, with a map that says so, would each read as it must.
        # (They passed until 2026-11-17.)
        listing.sort()
        back = next((k for k in range(len(listing) - 1) if listing[k][1] > listing[k + 1][1]), None)
        if back is not None:
            problems.append('%s+0x%x: the result is not in the order of the original: from 0x%x, and after it from 0x%x' % (
                name[:40], listing[back][0], listing[back][1], listing[back + 1][1]))
            continue
        ta, tb = texts(a.data, 75), texts(b.data, 70)
        # (Before anything is checked against the lives: the first version put this after the
        # instructions were checked, and a minimum whose comparison went into a predicate that a
        # P2R of the program reads later passed unseen. controls.py, `masks`.)
        leads, inside, anywhere = flow(ta, tables.get(name, {}))
        if lives.get(name) is not None:
            completed(lives[name], ta, leads, anywhere)
            kinds['(lives) branches through a register, led where nvdisasm lists'] += len(tables.get(name, {}))
            kinds['(lives) branches through a register without a list: taken to lead anywhere'] += len(anywhere)
        # Register liveness of the sm_75 original, by the validator's own means (for borrowed
        # register checks: rules 1 through 6). None when the flow is not known.
        reg_lives = register_lives(mc, a.data, ta, tables.get(name, {}), problems, name)
        # Predicate lives of the validator's own (P0-P6), by backward dataflow on the original.
        pred_own = predicate_lives(mc, a.data, ta, tables.get(name, {}))
        # nvdisasm prints a relative target as an address counted from the start of what it is given:
        # each piece of 4096 instructions starts again at 0, so pieces are compared piece by piece only
        # for functions that fit in one; larger functions have their targets left out of the comparison.
        whole = True
        new_of = {}
        for new, old, what in listing:
            new_of.setdefault(old, new)
        new_of[len(a.data)] = len(b.data)
        # A direct call whose return address is the program counter of the LEPC immediately
        # before it: both move with the code, so they must stay adjacent in the result (the
        # callee returns to what follows the call, and the LEPC's value is the call's address).
        for k in range(len(ta) - 1):
            first = (ta[k] or '')
            second = (ta[k + 1] or '')
            first = first.split(' ', 1)[1] if first.startswith('@') else first
            second = second.split(' ', 1)[1] if second.startswith('@') else second
            if re.match(r'LEPC R\d+$', first) and second.startswith('CALL.REL'):
                old = 16 * k
                if new_of.get(old, -1) + 16 != new_of.get(old + 16, -2):
                    problems.append('%s+0x%x: the LEPC and the call it gives the return address of are not adjacent in the result' % (name[:40], old))
                else:
                    moved_text = tb[new_of[old] // 16] or ''
                    moved_text = moved_text.split(' ', 1)[1] if moved_text.startswith('@') else moved_text
                    if moved_text != first:
                        problems.append('%s+0x%x: the LEPC of the call is not the instruction it was (the callee returns through its register)' % (name[:40], old))
        place, highest, pool = {}, 0, set()
        borrowed_regs = set()           # registers <= highest that the translation uses (homes and pool)
        # The ordinary predicate an instruction has for a uniform one: {(uniform predicate, old
        # offset): ordinary}; the predicates saved before an instruction and given back after it:
        # {old offset: [(register, mask)]}; the instructions whose own comparison has one saved.
        ups, saves, restores, pred_borrowed, savers = {}, collections.defaultdict(list), collections.defaultdict(list), set(), set()
        # A uniform predicate held in a register: where it is read back into a predicate before
        # an instruction and written down after one, {old offset: [(register, predicate)]}; the
        # registers that hold.
        backs, downs, holders = collections.defaultdict(list), collections.defaultdict(list), set()
        def stands_for(u, r, old):
            """The ordinary predicate `r` (P3) stands for the uniform one `u` (UP1) at `old`."""
            return ups.setdefault((int(u[2:]), old), int(r[1:])) == int(r[1:])
        def guarded(a, b, old):
            """The guard of the result, `b` (`@!P3 ` or None), is that of the original, `a` (`@!UP1 `)."""
            if a is None or b is None:
                return a is None and b is None
            return ('!' in a) == ('!' in b) and stands_for(a.strip('@! '), b.strip('@! '), old)
        for t in ta:
            for r in REG.findall(t or ''):
                if r[0] == 'R' and r != 'RZ':
                    highest = max(highest, int(r[1:]))
        for at in range(0, len(a.data) - 15, 16):
            # (A value of several registers is printed as its first one: the tables know the others.)
            d = mc.decode(75, int.from_bytes(a.data[at:at + 16], 'little'))
            if d is not None:
                highest = max([highest] + [register for letter, register, *role in d.reads + d.writes if letter == 'R'])
        # The instructions that stand for a matrix instruction: by the first thing done for it.
        firsts, sequences = {}, set()
        for new, old, what in listing:
            if what not in AROUND and not is_mem(what):
                firsts.setdefault(old, what)
        # (Whole-row loads, halves of pairs and conversions have rules of their own: `rows_schedule`,
        # `halves_schedule`, and for the whole function below. A whole-row load may be followed
        # by the sequence that stands for the load today.)
        stands = collections.defaultdict(set)
        for new, old, what in listing:
            stands[old].add(what)
        for new, old, what in listing:
            if stands[old] & {'group', 'neighbour', 'halves swapped'} and what not in AROUND and not is_mem(what) and not is_room(what):
                sequences.add(new)
            if what in ('settled', 'to pairs', 'from pairs', 'half of a pair') or what.startswith('whole row'):
                sequences.add(new)
        # ---- level 3: per-function frame checks (rules A1, A2) ----
        frame_slots = borrow_slots.get(name, {})
        frame_bytes = borrow_frames.get(name, 0)
        predicate_words, slots_at = {}, collections.defaultdict(dict)
        for q, word, offsets in held_slots.get(name, []):
            if not 0 <= q < 7 or not 0 < word <= frame_bytes or word % 4:
                problems.append('%s: UP%d has invalid frame location %d' % (name[:40], q, word))
            if predicate_words.setdefault(q, word) != word:
                problems.append('%s: UP%d has inconsistent frame locations' % (name[:40], q))
            for at in offsets:
                slots_at[at][q] = word
        all_words = list(frame_slots.values()) + list(predicate_words.values())
        if len(set(all_words)) != len(all_words):
            problems.append('%s: uniform registers or predicates occupy the same frame word' % name[:40])
        if predicate_words and set(all_words) != {frame_bytes - 4 * k for k in range(len(all_words))}:
            problems.append('%s: named frame words must precede the space for temporarily saved registers' % name[:40])
        temps_at = collections.defaultdict(set)
        if predicate_words:
            for width, first, offsets in borrow_pools.get(name, []):
                for at in offsets:
                    temps_at[at].update(range(first, first + width))
            for r, offsets in borrow_rooms.get(name, []):
                for at in offsets:
                    temps_at[at].add(r)
        # ---- R1 depth and room (rules P, Q, R, S, T, U) ----
        rooms_declared = borrow_rooms.get(name, [])
        room_made = {}          # {offset in the original: the registers that its sequence puts away}
        pair_writes = {}        # {offset in the original of a half of a pair: the registers its steps write}
        r1_depths, r1_fixed = {}, True
        has_room = bool(rooms_declared) or any(is_room(what) for _, _, what in listing)
        if frame_bytes > 0 or has_room:
            r1_depths, r1_fixed = compute_r1_depth(mc, a.data, ta, tables.get(name, {}))
        if frame_bytes > 0:
            check_frame_attributes(original, result, name, frame_slots, frame_bytes, problems)
            check_frame_slots(frame_slots, frame_bytes, name, problems)
            kinds['(level3) functions with a frame'] += 1
            kinds['(level3) uniform registers in the frame'] += len(frame_slots)
        # ---- rule Q supplement: frame holds slots and room ----
        if rooms_declared and frame_bytes > 0:
            check_frame_room_size(all_words, frame_bytes, rooms_declared, name, problems)
            kinds['(room) functions with room made'] += 1
            kinds['(room) room lines'] += len(rooms_declared)
        original_code = a.data         # saved before `a` and `b` are overwritten by the return-address code
        result_code = b.data
        # (The schedule check is deferred until after the instruction loop, when borrowed_regs is complete.)
        def say(new, old, what, problem):
            problems.append('%s+0x%x (from 0x%x, %s): %s: %s  ->  %s' % (name[:40], new, old, what, problem, ta[old // 16], tb[new // 16]))
        by_origin = collections.defaultdict(list)
        for new, old, what in listing:
            by_origin[old].append((new, what))
        # For rule V: the homes and the registers that hold predicates, by instruction of the
        # original, as the translator declares them; and the registers that the loads of the
        # frame write (looked at with the registers of the whole function, after the loop).
        homes_at, held_at, frame_regs = collections.defaultdict(dict), collections.defaultdict(set), set()
        for u, r, offsets in borrow_homes.get(name, []):
            for at in offsets:
                homes_at[at][r] = u
        for q, register, offsets in helds.get(name, []):
            if register.startswith('R') and register[1:].isdigit():
                for at in offsets:
                    held_at[at].add(int(register[1:]))
        for section, at, offset, kind, now in on_code.get(name, ()):
            # A relocation on this code is on what its instruction became: the instruction itself;
            # or, for the two kinds that write half of an address into the 32 bits of a number
            # (56 and 57), the one instruction that moves that number into an ordinary register
            # where the original moved it into a uniform one. Anything else the relocation would
            # be written into is not what the original had it written into.
            kinds['(relocations) on the code of translated functions'] += 1
            parts = [(new, what) for new, what in by_origin.get(offset & ~15, ()) if what not in AROUND and not is_mem(what) and not is_room(what)]
            fine = len(parts) == 1 and now == parts[0][0] + offset % 16
            if fine and parts[0][1] != 'same':
                was, became_ = UMOV_NUMBER.match(ta[offset // 16] or ''), MOV_NUMBER.match(tb[parts[0][0] // 16] or '')
                fine = kind in (56, 57) and parts[0][1] == 'ordinary registers' and None not in (was, became_) and was.group(1) == became_.group(1)
            if not fine:
                problems.append('%s+0x%x (on %s+0x%x): the relocation is not on what its instruction became: %s' % (
                    section[:30], at, name[:40], offset, ta[offset // 16] if offset // 16 < len(ta) else 'outside the code'))
        numbers, became = {}, {}        # MOVs of a number that changed; where each instruction itself now is
        lane, scratch = None, set()     # the register of the lane; the registers that matrix sequences compute into
        # What the function keeps of the lane: {name of `KEPT`: register}; and the products that
        # gathered the elements of B, in order: (old offset, register of B, the registers they
        # were gathered into, the offset in the result of the last shuffle that gathers).
        lanes_kept, gatherings = None, []
        has_level3_products = False
        borrowed_by_loads = set()  # registers of the program that level-3 matrix loads borrow
        # Pairs on whole rows: {name of ROW_VALUES: register}; the whole-row loads {old offset:
        # (role, destination, {(fragment or pair, half): register}, also)}; the halves of pairs
        # [(old offset, accumulators x and y, register of A, register of B, B by rows)]; the
        # conversions [(old offset, 'before' or 'after', register, 'Q' or 'T')].
        rows_kept, row_loads, halves, conversions, row_scratch = {}, {}, [], [], set()
        for old, parts in by_origin.items():
            source = ta[old // 16]
            checked += len(parts)
            per_inst_lane = None     # level 3: lane register for this instruction only
            # Whatever was done to an instruction, what reads or writes memory in what it became
            # has its guard: an instruction that does not run reads and writes nothing. Checked
            # here for every kind, apart from what each kind must read as below (the sequence
            # expected for a transposed matrix load had the same load without its guard as the
            # translator, until 2026-11-17).
            before = GUARD.match(source or '')
            for new, w in parts:
                # Level 3: mem: base, mem: load and mem: store are under no guard (they load from
                # the thread's local memory and store there: rules B2 to B4 of
                # `check_frame_sequence`); the guard check applies to core instructions. The
                # mem: R1 instruction is also unconditional.
                if w in ('mem: base',) or w.startswith(('mem: load ', 'mem: store ', 'mem: R1')) or is_room(w):
                    continue
                t = tb[new // 16] or ''
                after = GUARD.match(t)
                if not MEMORY.match(t[after.end():] if after else t):
                    continue
                if before is not None and 'U' in before.group(0):
                    same = after is not None and guarded(before.group(0), after.group(0), old)
                else:
                    same = (before and before.group(0)) == (after and after.group(0))
                if not same and w.startswith('zero fill: load') and after is not None:
                    # A Turing load that names the predicate where it loads: sm_70 has no such
                    # operand, and what it becomes loads under it (and zeroes the rest).
                    fill = re.search(r', (!?P\d+)$', source or '')
                    same = bool(fill) and after.group(0).strip() == '@' + fill.group(1)
                if not same:
                    say(new, old, w, 'reads or writes memory under another guard than the instruction')
            # ---- rule V: the values that the moves of the frame carry, in the whole sequence ----
            if any(is_room(w) or w == 'mem: base' or w.startswith(('mem: load ', 'mem: store ')) for _, w in parts):
                frame_regs.update(check_frame_values(mc, name, old, parts, tb, result_code, original_code,
                                                     homes_at.get(old, {}), held_at.get(old, ()), say))
            # ---- strip room: instructions early (before the lane check) ----
            room_regs_here = set()
            if any(is_room(w) for _, w in parts):
                room_parts = [(new, w) for new, w in parts if is_room(w)]
                borrow_never_mask = None
                if reg_lives is not None:
                    borrow_never_mask = reg_lives[4]
                room_regs_here = check_room(mc, name, old, room_parts, tb, result_code, original_code,
                               all_words, frame_bytes, highest, reg_lives, r1_depths,
                               rooms_declared, borrow_homes.get(name, []), helds.get(name, []),
                               borrow_never_mask, problems, kinds, say)
                check_room_declared(rooms_declared, name, old, room_regs_here, problems, say)
                room_made[old] = set(room_regs_here)
                # Check R1-addressed frame offsets for mem: instructions (rule P).
                has_base = any(w == 'mem: base' for _, w in parts)
                if not has_base and any(is_mem(w) and not w.startswith('mem: R1') for _, w in parts):
                    check_r1_frame_offsets(mc, name, old, [(n, w) for n, w in parts if is_mem(w)],
                                           tb, result_code, frame_slots, frame_bytes, r1_depths, problems, say)
                # Mark room instructions as sequences for the schedule check.
                for n, w in room_parts:
                    sequences.add(n)
                # Strip room instructions.
                parts = [(new, w) for new, w in parts if not is_room(w)]
                kinds['(room) instructions with room made'] += 1
                if not parts:
                    continue
            if parts[0][1] == 'lane':
                at = parts[0][0]
                m = re.match(r'S2R (R\d+), SR_LANEID$', tb[at // 16] or '')
                word, after = (int.from_bytes(b.data[x:x + 16], 'little') for x in (at, at + 16))
                board = (word >> WRITE_BARRIER) & 7
                if old == 0:
                    # Before the first instruction: the lane of the thread, read once into a new register,
                    # which the next instruction waits for.
                    if at != 0 or not m or int(m.group(1)[1:]) <= highest or board == 7 or not ((after >> WAIT) & 63) >> board & 1:
                        say(at, old, 'lane', 'not the lane read into a new register at the start, with the next instruction waiting for it')
                    else:
                        lane = m.group(1)
                else:
                    # Rule D: level 3, per-instruction lane read. An LDSM sequence in a function that has
                    # no lane register for the whole function begins by S2R Rl, SR_LANEID into a declared
                    # register, on a scoreboard, the next instruction waits.
                    rn_lane = int(m.group(1)[1:]) if m else -1
                    if not m or board == 7 or not ((after >> WAIT) & 63) >> board & 1:
                        say(at, old, 'lane', 'D: not S2R into a register on a scoreboard with the next instruction waiting')
                    else:
                        # Set a per-instruction lane (not the function-wide one) so the
                        # matrix-load sequence check below can use it. `per_inst_lane`
                        # is checked after `lane` by the sequence code.
                        per_inst_lane = m.group(1)
                        if rn_lane <= highest and rn_lane >= 2:
                            borrowed_regs.add(rn_lane)
                            # A register of the program: declared taken at the instruction (a pool
                            # of MAP.lives) or put away there (a room line), as for what the
                            # sequence computes into. ('lane' is put around the instruction: the
                            # check of undeclared writes below leaves it out. Until 2026-11-23 its
                            # declaration was not looked at.) Taken, it must hold nothing there
                            # (rule 1 of borrowed registers); put away, the sequence must put it
                            # away and bring it back (rule U the other way round, below), each
                            # once, from one word, under no guard, on scoreboards that what
                            # follows waits for (rule R), and it must end with what it held, the
                            # store before the lane is read into it (rule V). (Until 2026-11-23 a
                            # room line was not taken: the lane of a product at level 3 in a
                            # register put away there was objected to, 1,443 times in 12 cubins
                            # of the held-out corpus, cuDNN's and vLLM's.)
                            if not any(first <= rn_lane < first + width and old in offsets
                                       for width, first, offsets in borrow_pools.get(name, [])) \
                                    and not any(r == rn_lane and old in offsets for r, offsets in borrow_rooms.get(name, [])):
                                say(at, old, 'lane', 'D: the lane goes into a register of the program that is not declared taken at the instruction')
                    kinds['(level3) per-instruction lane reads'] += 1
                parts = parts[1:]
            if parts and parts[0][1] == 'kept':
                # After the lane, in a function with a product: the four values made of the lane
                # that its products go by (`KEPT`), each computed as it must be, into new
                # registers. (That nothing else writes them is checked for the whole function,
                # below.)
                mine = [new for new, w in parts if w == 'kept']
                said = [tb[new // 16] or '' for new in mine]
                bound = {'lane': lane} if lane else {}
                contiguous = lane is not None and old == 0 and mine == list(range(16, 16 + 16 * len(mine), 16)) and len(parts) > len(mine)
                # First the four of the products done one by one, when the function has such
                # (`KEPT`); then the values that pairs on whole rows go by (`ROW_VALUES`), which
                # are found by running the instructions on the 32 lanes.
                first = len(KEPT) if contiguous and len(said) >= len(KEPT) and unify(KEPT, said[:len(KEPT)], bound) is None else 0
                if first:
                    values = [bound.get(n) for n in ('low', 'high', 'half', 'odd')]
                    if len(set(values)) != 4 or any(int(r[1:]) <= highest or r == lane for r in values):
                        say(parts[0][0], old, 'kept', 'not the four values that a function with a product keeps of the lane, computed after it into new registers')
                    else:
                        lanes_kept = dict(zip(('low', 'high', 'half', 'odd'), values))
                rest = said[first:]
                if not contiguous or (not first and not rest):
                    say(parts[0][0], old, 'kept', 'not values kept of the lane, computed after it into new registers')
                elif rest:
                    state = {lane: list(range(32))}
                    stopped = run_on_lanes(rest, state)
                    made = {r: v for r, v in state.items() if r != lane}
                    held = set((lanes_kept or {}).values())
                    if stopped is not None or any(int(r[1:]) <= highest or r in held for r in made):
                        say(mine[first + (stopped or 0)], old, 'kept', 'not a value of the lane computed into a new register')
                    else:
                        for value_name, values in ROW_VALUES.items():
                            found = [r for r, v in made.items() if v == values]
                            if len(found) == 1:
                                rows_kept[value_name] = found[0]
                        # (The other registers it writes are scratch: checked with the sequences that use them.)
                        row_scratch.update(r for r in made if r not in rows_kept.values())
                        if not rows_kept:
                            say(mine[first], old, 'kept', 'computes none of the values that pairs on whole rows go by')
                parts = [(new, w) for new, w in parts if w != 'kept']
                if not parts:
                    continue
            # What the instruction waited for, the first instruction of what it became waits for:
            # a wait is for everything after it, and the compiler may have put on this
            # instruction the wait for what a later one reads.
            # When room instructions were stripped, the wait was on the first room
            # instruction (the first of the full sequence); check the full sequence's first.
            waited = (int.from_bytes(a.data[old:old + 16], 'little') >> WAIT) & 63
            first_new = by_origin[old][0][0] if room_regs_here else parts[0][0]
            first = (int.from_bytes(b.data[first_new:first_new + 16], 'little') >> WAIT) & 63
            if waited & ~first:
                say(first_new, old, by_origin[old][0][1] if room_regs_here else parts[0][1], 'does not wait for a scoreboard that the instruction waited for')
            # Accumulator registers of pairs shuffled between lanes, before what the instruction
            # became or after it: runs that start with an instruction that does nothing (it waits
            # and stalls: `pairs_schedule`), then one shuffle a register, in place, by the kept
            # permutation or its inverse. Where they are, and which registers, is the plan's
            # (checked for the whole function, below).
            moves = ('settled', 'to pairs', 'from pairs')
            body = [k for k, (new, w) in enumerate(parts) if w not in moves and w not in ('wait', 'save', 'restore')]
            for k, (new, w) in enumerate(parts):
                if w not in moves:
                    continue
                side = 'before' if not body or k < body[0] else 'after' if k > body[-1] else None
                t = tb[new // 16] or ''
                if side is None:
                    say(new, old, w, 'a conversion in the middle of what the instruction became')
                elif w == 'settled':
                    if t != 'NOP' or (k + 1 < len(parts) and parts[k + 1][1] not in ('to pairs', 'from pairs')) or k + 1 == len(parts):
                        say(new, old, w, 'not a NOP before conversions')
                else:
                    m = re.match(r'SHFL\.IDX PT, (R\d+), (R\d+), (R\d+), 0x1f$', t)
                    key = rows_kept.get('pi' if w == 'to pairs' else 'inv')
                    if not m or m.group(1) != m.group(2) or key is None or m.group(3) != key or int(m.group(1)[1:]) > highest:
                        say(new, old, w, 'not a register of the program shuffled in place by the kept permutation')
                    elif k == 0 or parts[k - 1][1] not in moves:
                        say(new, old, w, 'conversions that do not start with the instruction that waits for the products')
                    else:
                        conversions.append((old, side, int(m.group(1)[1:]), 'Q' if w == 'to pairs' else 'T'))
            parts = [(new, w) for new, w in parts if w not in moves]
            if not parts:
                say(by_origin[old][0][0], old, 'conversions', 'nothing left of the instruction itself')
                continue
            for new, w in parts:
                if w in ('wait', 'late predicate') and not (tb[new // 16] or '').startswith('NOP'):
                    say(new, old, w, 'not a NOP')
                # A predicate saved in a register before the instruction, given back after it.
                if w in ('save', 'restore'):
                    m = re.match(r'P2R (R\d+), PR, RZ, 0x([0-9a-f]+)$' if w == 'save' else r'R2P PR, (R\d+), 0x([0-9a-f]+)$', tb[new // 16] or '')
                    if not m or int(m.group(1)[1:]) <= highest:
                        say(new, old, w, 'not a predicate saved in a new register' if w == 'save' else 'not a predicate given back from a new register')
                    else:
                        (saves if w == 'save' else restores)[old].append((m.group(1), int(m.group(2), 16)))
                        savers.add(m.group(1))
            # (Saved first, given back last, around what the instruction became. What loads
            # uniform registers from the frame before it and stores them after it stands on
            # either side: under no guard, and with no predicate among its operands (rules B1
            # to B3), it neither reads nor changes what is saved. Counted with the rest until
            # 2026-11-20: a minimum or maximum with an operand from the frame, whose predicate
            # is saved around the comparison and the selection only, was objected to.)
            kept = [w for _, w in parts if w not in ('wait', 'late predicate') and not is_mem(w)]
            first, last = sum(1 for w in kept if w == 'save'), sum(1 for w in kept if w == 'restore')
            if kept[:first] != ['save'] * first or kept[len(kept) - last:] != ['restore'] * last or first + last >= len(kept):
                say(parts[0][0], old, ' + '.join(kept), 'predicates saved and given back elsewhere than around the instruction')
                continue
            parts = [(new, w) for new, w in parts if w not in ('wait', 'save', 'restore', 'late predicate')]
            frame_moves, frame_conversions = predicate_frame_moves(
                parts, tb, result_code, old, slots_at.get(old, {}), r1_depths.get(old // 16),
                frame_bytes, temps_at.get(old, ()), say) if any(
                    w.startswith(('mem: load UP', 'mem: store UP')) for _, w in parts) else (set(), {})
            sequences.update(frame_moves)
            # A uniform predicate held in a register: read back into a predicate before what the
            # instruction became (whether the register is not zero), written down after it (the
            # bit of that predicate, alone), in a register above the program's.
            for new, w in parts:
                if w == 'held: back':
                    m = re.match(r'ISETP\.NE\.U32\.AND P(\d), PT, (R\d+), RZ, PT$', tb[new // 16] or '')
                    rn_back = int(m.group(2)[1:]) if m else -1
                    # The register may be one of the program's (borrowed, where the translation
                    # takes its registers among those that hold nothing): rules H1 and HW then
                    # say that it holds nothing of the program along the held life and that
                    # nothing else writes it there. (Accepted only in a function with a frame
                    # until 2026-11-20, and without rule H1.)
                    if not m or rn_back < 2:
                        say(new, old, w, 'not a predicate read back from a new register')
                    else:
                        backs[old].append((frame_conversions.get(new, m.group(2)), int(m.group(1))))
                        holders.add(m.group(2))
                        if rn_back <= highest and rn_back >= 2:
                            borrowed_regs.add(rn_back)
                elif w == 'held: kept':
                    m = re.match(r'P2R (R\d+), PR, RZ, 0x([0-9a-f]+)$', tb[new // 16] or '')
                    rn_kept = int(m.group(1)[1:]) if m else -1
                    if not m or rn_kept < 2 or bin(int(m.group(2), 16)).count('1') != 1 or int(m.group(2), 16) > 0x40:
                        say(new, old, w, 'not one predicate written down in a new register')
                    else:
                        downs[old].append((frame_conversions.get(new, m.group(1)), int(m.group(2), 16).bit_length() - 1))
                        holders.add(m.group(1))
                        if rn_kept <= highest and rn_kept >= 2:
                            borrowed_regs.add(rn_kept)
            parts = [(new, w) for new, w in parts if new not in frame_moves and new not in frame_conversions]
            names = [w for _, w in parts]
            first = next((k for k, w in enumerate(names) if w != 'held: back'), len(names))
            last = next((k for k, w in enumerate(reversed(names)) if w != 'held: kept'), len(names))
            if first + last >= len(names) or any(w.startswith('held: ') for w in names[first:len(names) - last]):
                say(parts[0][0], old, ' + '.join(names), 'held predicates read back or written down elsewhere than around the instruction')
                continue
            parts = parts[first:len(parts) - last]
            # ---- level 3: validate and strip mem: instructions (rules A3, B, C) ----
            mem_standins = {}      # {uniform register number: ordinary register name} for this instruction
            if frame_slots and parts and not any(is_mem(w) for _, w in parts):
                # (Nothing moves the frame around the instruction: it reads and writes no uniform
                # register of the frame, rule B4.)
                frame_completeness(mc, old, original_code, frame_slots, [], [], parts[0][0], say)
            if any(is_mem(w) for _, w in parts):
                result_core, mem_standins = check_frame_sequence(
                    mc, name, old, parts, tb, ta, result_code, original_code,
                    frame_slots, frame_bytes, highest, place, by_origin, listing,
                    problems, kinds, pool, borrowed_regs, borrow_pools.get(name, []),
                    reg_lives, sequences, say, r1_depths=r1_depths)
                if result_core is None:
                    continue
                # Rule A3: check R1 lowering at offset 0.
                if old == 0 and frame_bytes > 0:
                    check_r1_lowered(mc, name, old, parts, tb, ta, result_code, original_code,
                                     frame_bytes, highest, listing, problems, say)
                # Mark mem: instructions as part of sequences for the schedule check.
                for n, w in parts:
                    if is_mem(w):
                        sequences.add(n)
                parts = result_core
                kinds['(level3) instructions with mem: frame operations'] += 1
                # Level 3 with a uniform register in the address: the mem: instructions come
                # before the lane and load sequence. After stripping mem:, the lane may be first.
                if parts and parts[0][1] == 'lane':
                    at_lane = parts[0][0]
                    m_lane = re.match(r'S2R (R\d+), SR_LANEID$', tb[at_lane // 16] or '')
                    word_lane, after_lane = (int.from_bytes(b.data[x:x + 16], 'little') for x in (at_lane, at_lane + 16))
                    board_lane = (word_lane >> WRITE_BARRIER) & 7
                    if old != 0:
                        rn_lane = int(m_lane.group(1)[1:]) if m_lane else -1
                        if not m_lane or board_lane == 7 or not ((after_lane >> WAIT) & 63) >> board_lane & 1:
                            say(at_lane, old, 'lane', 'D: not S2R into a register on a scoreboard with the next instruction waiting')
                        else:
                            per_inst_lane = m_lane.group(1)
                            if rn_lane <= highest and rn_lane >= 2:
                                borrowed_regs.add(rn_lane)
                        kinds['(level3) per-instruction lane reads'] += 1
                    parts = parts[1:]
            became[old] = parts[-1][0] if parts else (by_origin[old][-1][0] if by_origin[old] else 0)
            if not parts:
                # All mem: instructions (e.g. the R1 lowering at offset 0 with 'same' as core that
                # was already validated, or a pure store/load with no core instruction): nothing else to check.
                continue
            whats = [w for _, w in parts]
            got = [tb[new // 16] for new, w in parts]
            kinds[' + '.join(whats)] += 1
            if source is None or any(g is None for g in got):
                say(parts[0][0], old, whats[0], 'nvdisasm does not read it')
                continue
            new = parts[0][0]
            # The field that holds the stall of an instruction also says which batch it is of, and
            # with some stalls the same bits are marks of register reuse: whatever the schedule
            # of the translation did to a stall, the batch is the one of the original (none, for
            # what the translation added), and the bits are a combination NVIDIA's table has.
            # (An instruction that lost a stall that ends its group had its marks read as
            # BATCH_START until 2026-11-17: found on code the translator had never seen, 42.)
            named = batch(mc, 75, int.from_bytes(a.data[old:old + 16], 'little'))
            for part, w in parts:
                has = batch(mc, 70, int.from_bytes(b.data[part:part + 16], 'little'))
                if has is None:
                    say(part, old, w, 'its stall, batch and marks of reuse are no combination the instruction has')
                elif has != (named if w in ('same', 'ordinary registers') else 0):
                    say(part, old, w, 'names another batch than the instruction (%s, and %s)' % (named, has))
            if whats[0].startswith('whole row'):
                # A matrix load done as whole rows (docs/pairs.md, sections 2, 4 and 5): what its
                # instructions load, run on the 32 lanes, is the rows of the design for an A or a
                # B, in new registers (or, for the first half of k, where the program has the
                # fragment). When what the load gives in Turing's layout may be read elsewhere,
                # the sequence that stands for the load today follows: checked as such, below.
                mine = [k for k, w in enumerate(whats) if w.startswith('whole row')]
                rest = [k for k, w in enumerate(whats) if not w.startswith('whole row')]
                ok = mine == list(range(len(mine)))
                lines = [unreused(g) for g in got[:len(mine)]]
                under = re.search(r'\+(UR\d+)', unreused(source))
                stand, why = None, 'its instructions are not in one piece'
                if under and ok:
                    # The register that stands for the uniform part of the address: the one
                    # register these instructions add that they did not make and that is neither
                    # kept of the lane nor the address register.
                    made = {x.group(1) for x in (re.match(r'\S+ (?:PT, )?(R\d+),', line) for line in lines) if x}
                    added = {r for line in lines if line.startswith('IADD3 ') for r in REG.findall(line)[1:] if r != 'RZ'}
                    known = set(rows_kept.values()) | {'R' + re.search(r'\[R(\d+)', source).group(1)}
                    candidates = added - made - known
                    stand = next(iter(candidates)) if len(candidates) == 1 else None
                    stand_r = int(stand[1:]) if stand is not None else -1
                    ok = stand is not None and place.setdefault(under.group(1), stand) == stand and (stand_r > highest or stand_r >= 2)
                    if ok and stand_r <= highest:
                        borrowed_regs.add(stand_r)
                    why = 'the uniform part of its address is not added once from the register that stands for it'
                found = rows_loaded(lines, unreused(source), rows_kept, highest, stand) if ok else why
                if isinstance(found, str):
                    say(new, old, 'whole rows', found)
                else:
                    role, where, others = found
                    destination = int(re.search(r' R(\d+), \[', source).group(1))
                    row_scratch.update('R%d' % r for r in others)
                    span = range(parts[0][0], parts[len(mine) - 1][0] + 16, 16)
                    words = [int.from_bytes(b.data[x:x + 16], 'little') for x in span]
                    after = int.from_bytes(b.data[span[-1] + 16:span[-1] + 32], 'little') if span[-1] + 32 <= len(b.data) else 0
                    named = (int.from_bytes(a.data[old:old + 16], 'little') >> WRITE_BARRIER) & 7
                    # (When the program's load named no scoreboard: the one its loads name, to be
                    # the one the halves of pairs wait for.)
                    arrive = {(w >> WRITE_BARRIER) & 7 for w in words if mc.decode(70, w) is not None and mc.decode(70, w).mnemonic == 'LDS'} if named == 7 else set()
                    row_loads[old] = (role, destination, where, bool(rest), arrive)
                    for k, what in rows_schedule(mc, words, after, named):
                        problems.append('%s+0x%x (from 0x%x): %s: %s' % (name[:40], span[k], old, what, tb[span[k] // 16]))
                    kinds['(pairs) matrix loads done as whole rows' + (', with Turing\'s layout as well' if rest else '')] += 1
                if not rest:
                    continue
                parts = [parts[k] for k in rest]
                whats, got, new = [w for _, w in parts], [tb[x // 16] for x, w in parts], parts[0][0]
            if whats == ['half of a pair'] * 4:
                # A product that is half of a pair (docs/pairs.md, sections 2 and 5): the four steps of
                # one of Volta's products, A by rows, B by columns or by rows, on two registers of
                # A and two of B, into the destinations of two products: the instruction's own
                # are the first two steps' or the last two's. What the steps add to: the
                # destinations themselves, or for each product four other registers or zero
                # (which it must be is known when both halves are: for the whole function).
                m = re.match(r'HMMA\.1688\.F32 R(\d+), R(\d+), R(\d+), (R\d+|RZ)$', unreused(source))
                steps = [re.match(r'HMMA\.884\.F32\.F32\.STEP(\d) R(\d+), R(\d+)\.ROW, R(\d+)\.(COL|ROW), (R\d+|RZ)$', unreused(g)) for g in got]
                ok = bool(m) and all(steps)
                if ok:
                    to = [int(x.group(2)) for x in steps]
                    onto = [None if x.group(6) == 'RZ' else int(x.group(6)[1:]) for x in steps]
                    ok = [int(x.group(1)) for x in steps] == [0, 1, 2, 3]
                    ok = ok and len({(x.group(3), x.group(4), x.group(5)) for x in steps}) == 1
                    ok = ok and to[1] == to[0] + 2 and to[3] == to[2] + 2 and to[0] % 2 == 0 and to[2] % 2 == 0 and abs(to[0] - to[2]) >= 4
                    ok = ok and int(m.group(1)) in (to[0], to[2])
                    # (Each product's accumulator: zero twice, or two registers two apart, even,
                    # that are the destination or apart from both destinations.)
                    both = set(range(to[0], to[0] + 4)) | set(range(to[2], to[2] + 4))
                    for k in (0, 2):
                        c, c2 = onto[k], onto[k + 1]
                        ok = ok and ((c is None and c2 is None) or (c is not None and c2 == c + 2 and c % 2 == 0 and c + 3 <= highest
                                                                    and (c == to[k] or not set(range(c, c + 4)) & both)))
                if not ok:
                    say(new, old, 'half of a pair', 'not four steps of one product of Volta on the accumulators of a pair')
                else:
                    span = range(parts[0][0], parts[-1][0] + 16, 16)
                    words = [int.from_bytes(b.data[x:x + 16], 'little') for x in span]
                    halves.append((old, to[0], to[2], int(steps[0].group(3)), int(steps[0].group(4)), steps[0].group(5) == 'ROW',
                                   int(m.group(2)), int(m.group(3)), int(m.group(1)), (words[3] >> WRITE_BARRIER) & 7,
                                   onto[0], onto[2], None if m.group(4) == 'RZ' else int(m.group(4)[1:])))
                    for k, what in halves_schedule(words):
                        problems.append('%s+0x%x (from 0x%x): %s: %s' % (name[:40], span[k], old, what, tb[span[k] // 16]))
                    kinds['(pairs) products done as halves of pairs'] += 1
            elif whats == ['same']:
                want = moved(source, old, new, new_of, True)
                if got[0] != want and fewer_marks(want, got[0]):
                    # Marks of register reuse are hints, and go with a stall that ends a group: an
                    # instruction that the schedule gave another stall may have lost some.
                    kinds['(schedule) instructions without marks of register reuse they had'] += 1
                elif got[0] != want and (whole or HEX.sub('', got[0]) != HEX.sub('', want)):
                    # A number that is a code offset (a return address) follows the code.
                    m, n = HEX.search(source), HEX.search(got[0])
                    if not (m and n and source[:m.start()] == got[0][:n.start()] and 'MOV R' in source):
                        say(new, old, 'same', 'reads otherwise')
                    elif new_of.get(int(m.group(0), 16)) != int(n.group(0), 16):
                        # Or one of the two numbers whose difference, added to where a LEPC is,
                        # is a return address: looked at with the other one, below.
                        numbers[old] = (new, int(m.group(0), 16), int(n.group(0), 16))
            elif whats == ['ordinary registers']:
                want = unreused(source)
                pairs = list(zip(REG.findall(want), REG.findall(got[0])))
                ok = len(REG.findall(want)) == len(REG.findall(got[0]))
                got_ur = {}
                for u, r in pairs:
                    if u.startswith('UR') and u != 'URZ':
                        un = int(u[2:])
                        rn = int(r[1:]) if r[0] == 'R' and r != 'RZ' else -1
                        if un in mem_standins:
                            # Level 3: the stand-in is per-instruction, loaded from memory.
                            ok = ok and rn >= 0 and r == mem_standins[un] and (rn > highest or rn >= 2)
                        elif un in frame_slots:
                            # Level 3: a UR in the frame but not loaded (it is being written).
                            ok = ok and rn >= 0 and (rn > highest or rn >= 2)
                        else:
                            ok = ok and rn >= 0 and place.setdefault(u, r) == r and (rn > highest or rn >= 2)
                        if ok and rn >= 0:
                            got_ur[un] = rn
                        if ok and rn <= highest and rn >= 2:
                            borrowed_regs.add(rn)
                    elif u == 'URZ':
                        ok = ok and r == 'RZ'
                    elif u.startswith('UP') and u != 'UPT':
                        ok = ok and r[0] == 'P' and r != 'PT' and stands_for(u, r, old)
                    elif u == 'UPT':
                        ok = ok and r == 'PT'
                    else:
                        ok = ok and u == r
                # A 64-bit uniform operand (e.g. UIMAD.WIDE, UIMAD.WIDE.X): its upper register
                # UR(u+1) stands for R(r+1) on an aligned even pair, and counts if borrowed.
                d75 = mc.decode(75, int.from_bytes(a.data[old:old + 16], 'little'))
                uses75 = mc._uses.get((75, d75.c.name)) if d75 is not None else None
                if ok and uses75 is not None:
                    for letter, operand, span in uses75.reads + uses75.writes:
                        if letter == 'U' and operand in d75.ops and d75.ops[operand] != 63 and machine.registers(span, d75.ops) == 2:
                            un = d75.ops[operand]
                            rn = got_ur.get(un, -1)
                            ok = ok and rn >= 0 and rn % 2 == 0
                            if not ok:
                                break
                            r_hi = 'R%d' % (rn + 1)
                            if un + 1 in mem_standins:
                                ok = ok and mem_standins[un + 1] == r_hi and (rn + 1 > highest or rn + 1 >= 2)
                            elif un + 1 in frame_slots:
                                ok = ok and (rn + 1 > highest or rn + 1 >= 2)
                            else:
                                ok = ok and place.setdefault('UR%d' % (un + 1), r_hi) == r_hi and (rn + 1 > highest or rn + 1 >= 2)
                            if ok and rn + 1 <= highest and rn + 1 >= 2:
                                borrowed_regs.add(rn + 1)
                # Everything but the registers: the same once the uniform mnemonic and names are made ordinary.
                if not ok or plain(shape_of(want)) != plain(shape_of(unreused(got[0]))):
                    say(new, old, whats[0], 'not the same on ordinary registers')
            elif whats == ['product', 'upper half']:
                m = re.match(r'(@!?P\d )?IMAD\.HI(\.U32)? (R\d+|RZ), (.*)$', unreused(source))
                p = re.match(r'(@!?P\d )?IMAD\.WIDE(\.U32)? (R\d+), (.*)$', unreused(got[0]))
                q = re.match(r'(@!?P\d )?MOV (R\d+|RZ), (R\d+)$', got[1])
                pool_r = int(p.group(3)[1:]) if p else -1
                ok = m and p and q and m.group(2) == p.group(2) and q.group(2) == m.group(3) and int(q.group(3)[1:]) == pool_r + 1 \
                    and (pool_r > highest or pool_r >= 2) and m.group(1) == p.group(1) == q.group(1)
                if ok:
                    pool.update((p.group(3), q.group(3)))
                    if pool_r <= highest:
                        borrowed_regs.update((pool_r, pool_r + 1))
                    rest_a, rest_b = REG.findall(m.group(4)), REG.findall(p.group(4))
                    def stands(x, y):
                        # A uniform operand: the register it was loaded into from the frame for
                        # this instruction (rule B2 says which), or else its register in the
                        # whole function. (Until 2026-11-20 the second only: a product of a
                        # function with a frame was objected to.)
                        if not x.startswith('UR'):
                            return x == y
                        un = int(x[2:])
                        if un in mem_standins:
                            return y == mem_standins[un]
                        return un not in frame_slots and place.setdefault(x, y) == y
                    ok = REG.sub('#', m.group(4)) == REG.sub('#', p.group(4)) and all(stands(x, y) for x, y in zip(rest_a, rest_b))
                if not ok:
                    say(new, old, 'upper half of a product', 'not IMAD.WIDE into two new registers then MOV of the upper one')
            elif whats == ['comparison', 'selection']:
                m = re.match(r'(@!?P\d )?IMNMX(\.U32)? (R\d+|RZ), (.*), (!?)PT$', unreused(source))
                p = re.match(r'ISETP\.LT(\.U32)?\.AND (P\d), PT, (.*), PT$', unreused(got[0]))
                q = re.match(r'(@!?P\d )?SEL (R\d+|RZ), (.*), (!?)(P\d)$', unreused(got[1]))
                ok = m and p and q and m.group(2) == p.group(1) and m.group(1) == q.group(1) and m.group(3) == q.group(2) and m.group(5) == q.group(4) and p.group(2) == q.group(5)
                if ok:
                    # The predicate is one the function does not mention, or one the translator found
                    # to hold nothing alive there: the second is counted, not checked here. Or one
                    # that holds something, saved before the comparison and given back after the
                    # selection.
                    used = set(re.findall(r'\bP\d\b', ' '.join(t for t in ta if t)))
                    bit = 1 << int(p.group(2)[1:])
                    if saves[old] or restores[old]:
                        pred_borrowed.add(old)
                        ok = len(saves[old]) == 1 and saves[old] == restores[old] and saves[old][0][1] == bit
                    elif p.group(2) in used:
                        # One of the function's own: nvdisasm's analysis of the original must find
                        # nothing alive in it at that instruction.
                        at = lives.get(name, {}).get(old)
                        if at is None:
                            # nvdisasm gave no chart: use the validator's own predicate lives.
                            if pred_own is not None and old in pred_own:
                                if pred_own[old] >> int(p.group(2)[1:]) & 1:
                                    ok = False
                                kinds['(own lives) a predicate of the function, taken as dead, for a minimum or maximum'] += 1
                            else:
                                kinds['(not checked) a predicate of the function, taken as dead, for a minimum or maximum'] += 1
                        elif int(p.group(2)[1:]) in at[0]:
                            ok = False
                    ok = ok and p.group(2) not in set(re.findall(r'\bP\d\b', source))
                    def ur_matches(x, y):
                        if not x.startswith('UR'):
                            return x == y
                        un = int(x[2:])
                        if un in mem_standins:
                            return y == mem_standins[un]
                        if un in frame_slots:
                            return True
                        return place.setdefault(x, y) == y
                    for operands in (p.group(3), q.group(3)):
                        ok = ok and REG.sub('#', m.group(4)) == REG.sub('#', operands) and all(ur_matches(x, y) for x, y in zip(REG.findall(m.group(4)), REG.findall(operands)))
                if not ok:
                    say(new, old, 'minimum or maximum', 'not a comparison into a predicate the function does not use, then a selection by it')
            elif whats[0] in ('group', 'neighbour', 'halves swapped'):
                # A matrix load, a product: the sequence that the recipe gives for it, on registers
                # of the translation that are no stand-in, with the lane where it was read and,
                # for a product, what the function keeps of the lane where it was computed. A
                # product has two forms: it gathers the elements of B (its first thing is the
                # shuffle with the neighbour), or takes them where an earlier one gathered them
                # (its first thing is A with its halves swapped).
                load, gathers = whats[0] == 'group', whats[0] == 'neighbour'
                clean = unreused(source)
                want = matrix_load_expected(clean) if load else matrix_product_expected(clean, gathers)
                effective_lane = per_inst_lane or lane
                given = {'lane': effective_lane} if effective_lane else {}
                if not load:
                    given.update(lanes_kept or {})
                # Level 3: when the address has a uniform register loaded from the frame,
                # bind it from mem_standins so the template match uses the per-instruction register.
                if load and mem_standins:
                    m_ur = re.search(r'\+(UR\d+)', clean)
                    if m_ur:
                        un = int(m_ur.group(1)[2:])
                        if un in mem_standins:
                            given[m_ur.group(1)] = mem_standins[un]
                bound = dict(given)
                known = effective_lane is not None and (load or lanes_kept is not None)
                at = 0 if want is None or not known else unify(want, [unreused(g) for g in got], bound)
                if at is not None:
                    say(parts[min(at, len(parts) - 1)][0], old, 'matrix load' if load else 'matrix product', 'not the sequence that stands for it (instruction %d of it)' % at)
                else:
                    mine = {n: r for n, r in bound.items() if n not in given and not n.startswith('UR')}
                    ok = len(set(mine.values())) == len(mine) and not set(mine.values()) & set(given.values()) and all(int(r[1:]) > highest or int(r[1:]) >= 2 for r in mine.values())
                    # Level 3 with mem_standins: UR bindings are per-instruction, not function-wide.
                    if mem_standins:
                        ok = ok and all(r == given.get(n, r) for n, r in bound.items() if n.startswith('UR'))
                    else:
                        ok = ok and all(place.setdefault(n, r) == r for n, r in bound.items() if n.startswith('UR'))
                    # Level 3: borrowed registers for the matrix sequence.
                    for r in mine.values():
                        rn = int(r[1:])
                        if rn <= highest and rn >= 2:
                            borrowed_regs.add(rn)
                            if load:
                                borrowed_by_loads.add(r)
                    number = lambda n: int(bound[n][1:])
                    if not load:
                        ok = ok and number('x1') == number('x0') + 1 and all(number(n) % 2 == 0 for n in ('y0', 'y2', 'x0', 's0'))
                        ok = ok and [number('s%d' % k) for k in range(4)] == list(range(number('s0'), number('s0') + 4))
                        ok = ok and (not gathers or (number('y1') == number('y0') + 1 and number('y3') == number('y2') + 1))
                    if not ok:
                        say(new, old, 'matrix load' if load else 'matrix product', 'the registers of its sequence are not new ones, each for one thing')
                    scratch.update(mine.values())
                    span = range(parts[0][0], parts[-1][0] + 16, 16)
                    words = [int.from_bytes(b.data[x:x + 16], 'little') for x in span]
                    steps = 0 if load else (4 if ' HMMA.1688.F32 ' in ' ' + clean else 2)
                    for k, what in sequence_schedule(mc, words, highest, steps, borrowed_regs):
                        problems.append('%s+0x%x (from 0x%x): %s: %s' % (name[:40], span[k], old, what, tb[span[k] // 16]))
                    kinds['(schedule) instructions of matrix sequences'] += len(words)
                    of_b = None if load else int(re.match(r'HMMA\.1688\.F\d+ R\d+, R\d+, R(\d+), ', clean).group(1))
                    if gathers and ok:
                        gatherings.append((old, of_b, (number('y0'), number('y2')), parts[max(k for k, t in enumerate(want) if t.startswith('SHFL.IDX'))][0]))
                    elif not load and not gathers:
                        # It takes the elements of B where a product gathered them. They are
                        # those it would gather itself when: that product is the last that
                        # gathered, from this register of B and into the registers this one
                        # reads; every way to this product goes through that one and straight on
                        # from it (nothing after it up to this one is entered otherwise than
                        # from the instruction before, and no branch through a register leads
                        # nobody knows where); no call between them (what is called may gather
                        # for a product of its own, or write the register); nothing from that
                        # product on, itself included, writes the register of B in the
                        # original; and nothing after its gathering writes the gathered
                        # registers in the result. (None of this was looked at when the form
                        # was first written, 2026-11-17: only that the sequence read as it must.)
                        kinds['(matrix) products that take the elements of B where an earlier one gathered them'] += 1
                        why = None
                        if not gatherings:
                            why = 'no product before it gathered them'
                        else:
                            there, register, into, last = gatherings[-1]
                            if register != of_b or into != (number('y0'), number('y2')):
                                why = 'the last product that gathered (0x%x) did it from R%d into R%d and R%d' % ((there, register) + into)
                            elif anywhere:
                                why = 'a branch through a register may lead anywhere in the function'
                            elif any(x in inside for x in range(there + 16, old + 16, 16)):
                                why = 'something after the product that gathered them (0x%x) is entered otherwise than from the instruction before' % there
                            for x in range(there, old, 16) if why is None else ():
                                t = ta[x // 16] or ''
                                d75 = mc.decode(75, int.from_bytes(a.data[x:x + 16], 'little'))
                                if d75 is None or not t:
                                    why = 'the instruction at 0x%x, since they were gathered (0x%x), is not read' % (x, there)
                                elif (t.split(' ', 1)[1] if t.startswith('@') else t).startswith(('CALL', 'JCAL')):
                                    why = 'a call at 0x%x comes after the product that gathered them (0x%x)' % (x, there)
                                elif ('R', of_b) in d75.writes:
                                    why = 'R%d is written at 0x%x, after they were gathered from it (0x%x)' % (of_b, x, there)
                                if why is not None:
                                    break
                            held = {into[0], into[0] + 1, into[1], into[1] + 1}
                            for x in range(last + 16, parts[0][0], 16) if why is None else ():
                                d70 = mc.decode(70, int.from_bytes(b.data[x:x + 16], 'little'))
                                if d70 is None or any(letter == 'R' and register in held for letter, register in d70.writes):
                                    why = 'the registers they were gathered into are written at 0x%x of the result' % x
                                    break
                        if why is not None:
                            say(parts[0][0], old, 'matrix product', 'takes the elements of B where an earlier product gathered them, and ' + why)
            elif whats == ['stride', 'on the product']:
                m = re.search(r'\[(R\d+)\.X(4|8|16)', source)
                p = re.match(r'SHF\.L\.U32 (R\d+), (R\d+), 0x(\d), RZ$', got[0])
                stride_r = int(p.group(1)[1:]) if p else -1
                ok = m and p and p.group(2) == m.group(1) and 1 << int(p.group(3)) == int(m.group(2)) and (stride_r > highest or stride_r >= 2) \
                    and unreused(got[1]) == unreused(source).replace('[%s.X%s' % (m.group(1), m.group(2)), '[' + p.group(1))
                if not ok:
                    say(new, old, 'stride', 'not a shift into a new register then the instruction on it')
                else:
                    if stride_r <= highest:
                        borrowed_regs.add(stride_r)
                    pool.add(p.group(1))
            elif set(whats) == {'constant'}:
                m = re.match(r'(@!?UP\d )?ULDC(\.64)? (UR\d+), c\[(0x[0-9a-f]+)\]\[(-?0x[0-9a-f]+)\]$', source)
                ok = bool(m) and len(got) == (2 if m.group(2) else 1)
                for half, t in enumerate(got if ok else []):
                    p = re.match(r'(@!?P\d )?MOV (R\d+), c\[(0x[0-9a-f]+)\]\[(-?0x[0-9a-f]+)\]$', t)
                    u = 'UR%d' % (int(m.group(3)[2:]) + half)
                    un = int(m.group(3)[2:]) + half
                    const_r = int(p.group(2)[1:]) if p else -1
                    ok = ok and bool(p) and guarded(m.group(1), p.group(1), old) and p.group(3) == m.group(4) and int(p.group(4), 16) & 0xffff == (int(m.group(5), 16) + 4 * half) & 0xffff and (const_r > highest or const_r >= 2)
                    if ok:
                        if un in mem_standins:
                            # Level 3: the register is the per-instruction stand-in loaded from memory.
                            ok = ok and p.group(2) == mem_standins[un]
                        elif un in frame_slots:
                            # Level 3: a UR in the frame, but not loaded (it is being written).
                            # The register is per-instruction; do not register in function-wide `place`.
                            pass
                        else:
                            ok = ok and place.setdefault(u, p.group(2)) == p.group(2)
                    if ok and const_r <= highest:
                        borrowed_regs.add(const_r)
                if not ok:
                    say(new, old, 'constant', 'not MOV of the same constant into the ordinary registers')
            elif whats[-1] in ('on the sum', 'on the uniform part') and set(whats[:-1]) <= {'address'}:
                # A memory instruction whose address has a uniform part: the register it now reads
                # was made just before it (the sum of the two parts), or is the ordinary register
                # of the uniform part when there is no other.
                made, last = [unreused(g) for g in got[:-1]], unreused(got[-1])
                # (Without what a load says to the cache about its neighbours: sm_70 has no such thing.)
                src = re.sub(r'\.LTC\d+B', '', unreused(source))
                ok, shadow = True, None
                def stands(u, r):
                    rn = int(r[1:]) if r[0] == 'R' and r != 'RZ' else -1
                    if u == 'URZ' and r == 'RZ':
                        return True
                    un = int(u[2:]) if u.startswith('UR') else -1
                    if un >= 0 and un in mem_standins:
                        return rn >= 0 and r == mem_standins[un] and (rn > highest or rn >= 2)
                    if un >= 0 and un in frame_slots:
                        # Level 3: a UR in the frame, per-instruction register.
                        if rn < 0 or (rn <= highest and rn < 2):
                            return False
                        if rn <= highest:
                            borrowed_regs.add(rn)
                        return True
                    if rn < 0 or place.setdefault(u, r) != r or (rn <= highest and rn < 2):
                        return False
                    if rn <= highest:
                        borrowed_regs.add(rn)
                    return True
                m = re.search(r'\[(R\d+)\.(U32|64)\+(UR\d+)', src)
                shared = re.search(r'\[(R\d+)(?:\.X(4|8|16))?\+(UR\d+)', src) if not m else None
                if whats[-1] == 'on the uniform part':
                    u, r = re.search(r'\[(UR\d+)', src), re.search(r'\[(R\d+)', last)
                    ok = bool(u and r) and not made and stands(u.group(1), r.group(1)) and last == src.replace('[' + u.group(1), '[' + r.group(1), 1)
                    if ok and not re.match(r'(@!?P\d )?(LDS|STS|ATOMS)\b', src):
                        # An address of 64 bits: in two registers, side by side.
                        ok = stands('UR%d' % (int(u.group(1)[2:]) + 1), 'R%d' % (int(r.group(1)[1:]) + 1))
                elif whats[-1] == 'on the sum' and shared:
                    # In shared memory: the register times its stride plus the uniform part, 32 bits.
                    p = re.match(r'IMAD(?:\.IADD)? (R\d+), (R\d+), 0x(\d+), (R\d+)$', made[0]) if len(made) == 1 else None
                    shared_pool = int(p.group(1)[1:]) if p else -1
                    ok = bool(p) and p.group(2) == shared.group(1) and int(p.group(3), 16) == int(shared.group(2) or 1) and stands(shared.group(3), p.group(4)) and (shared_pool > highest or shared_pool >= 2)
                    if ok:
                        pool.add(p.group(1))
                        if shared_pool <= highest:
                            borrowed_regs.add(shared_pool)
                        ok = last == src.replace(shared.group(0), '[' + p.group(1))
                else:
                    p = re.match(r'IMAD\.WIDE\.U32 (R\d+), (R\d+), 0x1, (R\d+)$', made[0]) if m else None
                    wide_pool = int(p.group(1)[1:]) if p else -1
                    ok = bool(p) and p.group(2) == m.group(1) and stands(m.group(3), p.group(3)) and (wide_pool > highest or wide_pool >= 2) and len(made) == (2 if m.group(2) == '64' else 1)
                    if ok and m.group(2) == '64':
                        t, upper = wide_pool + 1, 'R%d' % (int(m.group(1)[1:]) + 1)
                        ok = made[1] == 'IADD3 R%d, R%d, %s, RZ' % (t, t, upper)
                    if ok:
                        shadow = p.group(1)
                        pool.update((shadow, 'R%d' % (int(shadow[1:]) + 1)))
                        if wide_pool <= highest:
                            borrowed_regs.update((wide_pool, wide_pool + 1))
                        ok = last == src.replace('[%s.%s+%s' % m.groups(), '[' + shadow)
                if not ok:
                    say(new, old, ' + '.join(whats), 'not the plain instruction on registers made from its uniform operand')
            elif whats[0] == 'constant' and set(whats[1:]) == {'bytes'}:
                m = re.match(r'(@!?UP\d )?ULDC\.([US])(8|16) (UR\d+), c\[(0x[0-9a-f]+)\]\[(-?0x[0-9a-f]+)\]$', source)
                ok = bool(m)
                if ok:
                    bits, address = int(m.group(3)), int(m.group(6), 16) & 0xffff
                    up = 32 - bits - 8 * (address % 4)
                    p = re.match(r'(@!?P\d )?MOV (R\d+), c\[(0x[0-9a-f]+)\]\[(-?0x[0-9a-f]+)\]$', got[0])
                    small_r = int(p.group(2)[1:]) if p else -1
                    un_small = int(m.group(4)[2:])
                    if un_small in mem_standins:
                        ok = bool(p) and guarded(m.group(1), p.group(1), old) and p.group(3) == m.group(5) and int(p.group(4), 16) & 0xffff == address - address % 4 and p.group(2) == mem_standins[un_small] and (small_r > highest or small_r >= 2)
                    elif un_small in frame_slots:
                        ok = bool(p) and guarded(m.group(1), p.group(1), old) and p.group(3) == m.group(5) and int(p.group(4), 16) & 0xffff == address - address % 4 and (small_r > highest or small_r >= 2)
                    else:
                        ok = bool(p) and guarded(m.group(1), p.group(1), old) and p.group(3) == m.group(5) and int(p.group(4), 16) & 0xffff == address - address % 4 and place.setdefault(m.group(4), p.group(2)) == p.group(2) and (small_r > highest or small_r >= 2)
                    if ok and small_r <= highest:
                        borrowed_regs.add(small_r)
                    if ok:
                        r, g = p.group(2), p.group(1) or ''
                        want = (['%sSHF.L.U32 %s, %s, 0x%x, RZ' % (g, r, r, up)] if up else []) + ['%sSHF.R.%s32.HI %s, RZ, 0x%x, %s' % (g, m.group(2), r, 32 - bits, r)]
                        ok = got[1:] == want
                if not ok:
                    say(new, old, 'small constant', 'not the word of the constant shifted up and down again')
            elif whats == ['constant at an index', 'until it is there']:
                # A load of constants at the ordinary register of the index, into that of the
                # destination, then an instruction that does nothing (and waits for the load).
                m = re.match(r'(@(!?)(UP\d) )?ULDC(\.\w+)? (UR\d+), c\[(0x[0-9a-f]+)\]\[(UR\d+)(\+-?0x[0-9a-f]+)?\]$', source)
                q = re.match(r'(@(!?)(P\d) )?LDC(\.\w+)? (R\d+), c\[(0x[0-9a-f]+)\]\[(R\d+)(\+-?0x[0-9a-f]+)?\]$', got[0])
                ok = bool(m and q) and got[1] == 'NOP' and (m.group(2), m.group(4), m.group(6), m.group(8)) == (q.group(2), q.group(4), q.group(6), q.group(8))
                if ok:
                    pairs = [(m.group(5), q.group(5)), (m.group(7), q.group(7))]
                    if m.group(4) == '.64':
                        pairs.append(('UR%d' % (int(m.group(5)[2:]) + 1), 'R%d' % (int(q.group(5)[1:]) + 1)))
                    for u, r in pairs:
                        rn = int(r[1:])
                        if rn <= highest and rn >= 2:
                            borrowed_regs.add(rn)
                    def check_pair(u, r):
                        un = int(u[2:])
                        rn = int(r[1:])
                        if un in mem_standins:
                            return r == mem_standins[un] and (rn > highest or rn >= 2)
                        if un in frame_slots:
                            return rn > highest or rn >= 2
                        return place.setdefault(u, r) == r and (rn > highest or rn >= 2)
                    ok = all(check_pair(u, r) for u, r in pairs)
                    ok = ok and (m.group(3) is None) == (q.group(3) is None) and (m.group(3) is None or stands_for(m.group(3), q.group(3), old))
                if not ok:
                    say(new, old, 'constant at an index', 'not a load of the same constant at the ordinary registers, then a NOP')
            elif whats == ['move']:
                m = re.match(r'(@!?P\d )?R2UR (UR\d+), (R\d+)$', source)
                p = re.match(r'(@!?P\d )?MOV (R\d+), (R\d+)$', got[0])
                move_r = int(p.group(2)[1:]) if p else -1
                move_un = int(m.group(2)[2:]) if m else -1
                if m and p and m.group(1) == p.group(1) and m.group(3) == p.group(3) and (move_r > highest or move_r >= 2):
                    if move_un in mem_standins:
                        ok_move = p.group(2) == mem_standins[move_un]
                    elif move_un in frame_slots:
                        ok_move = True
                    else:
                        ok_move = place.setdefault(m.group(2), p.group(2)) == p.group(2)
                else:
                    ok_move = False
                if not ok_move:
                    say(new, old, 'move', 'not a MOV into the ordinary register')
                elif move_r <= highest:
                    borrowed_regs.add(move_r)
            elif whats == ['without its cache hint']:
                if re.sub(r'\.LTC\d+B', '', source) != got[0] or source == got[0]:
                    say(new, old, 'cache hint', 'not the instruction without its hint')
            elif whats[0] == 'zero fill: load where the predicate holds' and set(whats[1:]) <= {'zero fill: zeros elsewhere'}:
                # A Turing load that names the predicate where it loads and zeroes the rest:
                # sm_70 has no such operand, and ptxas writes it for Volta as the load under the
                # predicate and a MOV of RZ into each destination under its complement.
                m = re.match(r'(LDG\S*) (R\d+), (\[[^\]]*\]), (!?P\d+)$', unreused(source))
                ok = bool(m) and len(whats) == len(got)
                if ok:
                    body = re.sub(r'\.LTC\d+B', '', m.group(1))
                    pred = m.group(4)
                    size = re.search(r'\.(32|64|128)(?:\.|$)', body)
                    registers = {'32': 1, '64': 2, '128': 4}.get(size.group(1)) if size else 1
                    other = pred[1:] if pred.startswith('!') else '!' + pred
                    first = int(m.group(2)[1:])
                    want = ['@%s MOV R%d, RZ' % (other, first + k) for k in range(registers or 0)]
                    ok = bool(registers) and got[0] == '@%s %s %s, %s' % (pred, body, m.group(2), m.group(3)) and got[1:] == want
                    if ok:
                        kinds['(load) Turing zero-fill loads'] += 1
                if not ok:
                    say(new, old, 'zero fill', 'not the load under its predicate and a MOV of RZ into each destination under its complement')
            elif whats[0].startswith('f2fp: ') and set(whats) <= {'f2fp: convert high half', 'f2fp: convert low half', 'f2fp: pack halves'}:
                # Turing's F2FP.PACK_AB: two f32 values converted to f16 and packed in one
                # register. (The text is nvdisasm's sm_80 reading: the sm_75 description has the
                # encoding, but NVIDIA's disassembler does not print it for SM75.) Volta has no
                # such instruction: the high half is converted in a register of the translation,
                # the low half into the destination, and a PRMT packs the two halves.
                m = re.match(r'(@!?P\d )?F2FP(?:\.F16\.F32)?\.PACK_AB (R\d+), (R\d+), (R\d+)$', unreused(source))
                ok = bool(m)
                if ok:
                    guard, rd, high, low = m.groups()
                    guard = guard or ''
                    at, upper = 0, 'RZ'
                    if high != 'RZ':
                        p = re.match(r'F2F\.F16\.F32 (R\d+), %s$' % high, got[0])
                        ok = bool(p)
                        at, upper = 1, p.group(1) if p else 'RZ'
                    ok = ok and len(got) == at + 2
                    if ok:
                        q = re.match(r'%sF2F\.F16\.F32 %s, %s$' % (guard, rd, low), got[at])
                        r = re.match(r'%sPRMT %s, %s, 0x5410, %s$' % (guard, rd, rd, upper), got[at + 1])
                        ok = bool(q and r)
                    if ok and upper != 'RZ':
                        rn = int(upper[1:])
                        ok = rn > highest or rn >= 2
                        if rn <= highest and rn >= 2:
                            borrowed_regs.add(rn)
                        elif rn > highest:
                            scratch.add(upper)
                    if ok:
                        kinds['(conversions) Turing packed f32 to f16'] += 1
                if not ok:
                    say(new, old, 'F2FP', 'not two conversions and a PRMT that packs their halves')
            elif whats[0] == 'kept: low':
                # ---- Level-3 matrix product: everything taken at the instruction ----
                # The lane was already stripped; its per-instruction register is in `per_inst_lane`.
                # The sequence: low/high/half from the lane, B gathered, odd into half's register,
                # the HMMA steps, and the selections (floats) or pair/exchanges/rows (halves).
                clean = unreused(source)
                want = matrix_product_level3_expected(clean)
                effective_lane = per_inst_lane or lane
                given_l3 = {'lane': effective_lane} if effective_lane else {}
                bound = dict(given_l3)
                known = effective_lane is not None
                at = 0 if want is None or not known else unify(want, [unreused(g) for g in got], bound)
                if at is not None:
                    say(parts[min(at, len(parts) - 1)][0], old, 'matrix product (level 3)', 'not the sequence that stands for it (instruction %d of it)' % at)
                else:
                    mine = {n: r for n, r in bound.items() if n not in given_l3 and not n.startswith('UR')}
                    number = lambda n: int(bound[n][1:])
                    # Register constraints for level 3: each pair from an even register, each for one thing.
                    ok = len(set(mine.values())) == len(mine) and not set(mine.values()) & set(given_l3.values())
                    ok = ok and all(int(r[1:]) > highest or int(r[1:]) >= 2 for r in mine.values())
                    # Pairs from even: s0:s1, s2:s3, x0:x1, y0:y1, y2:y3.
                    ok = ok and number('s1') == number('s0') + 1 and number('s0') % 2 == 0
                    ok = ok and number('s3') == number('s2') + 1 and number('s2') % 2 == 0
                    ok = ok and number('x1') == number('x0') + 1 and number('x0') % 2 == 0
                    ok = ok and number('y1') == number('y0') + 1 and number('y0') % 2 == 0
                    ok = ok and number('y3') == number('y2') + 1 and number('y2') % 2 == 0
                    if not ok:
                        say(new, old, 'matrix product (level 3)', 'the registers of its sequence are not new ones, each for one thing, pairs from even registers')
                    # Add borrowed registers (those <= highest) to borrowed_regs; add to
                    # scratch only registers above highest (at level 3 with the frame,
                    # the per-instruction overlap check in `frame_bytes > 0` applies; at
                    # level 3 with the top extension (no frame), registers above highest
                    # are per-instruction and would cause false overlaps in the
                    # function-wide check).
                    for r in mine.values():
                        rn = int(r[1:])
                        if rn <= highest and rn >= 2:
                            borrowed_regs.add(rn)
                    if frame_bytes > 0:
                        scratch.update(mine.values())
                    else:
                        scratch.update(r for r in mine.values() if int(r[1:]) <= highest)
                    # Schedule check: the HMMA steps and the gathering, within the full sequence.
                    span = range(parts[0][0], parts[-1][0] + 16, 16)
                    words = [int.from_bytes(b.data[x:x + 16], 'little') for x in span]
                    steps = 0 if ' HMMA.1688.F32 ' in ' ' + clean else 0
                    floats_l3 = ' HMMA.1688.F32 ' in ' ' + clean
                    steps = 4 if floats_l3 else 2
                    l3_borrowed = {int(r[1:]) for r in mine.values()} | borrowed_regs
                    for k, what in sequence_schedule(mc, words, highest, steps, l3_borrowed):
                        problems.append('%s+0x%x (from 0x%x): %s: %s' % (name[:40], span[k], old, what, tb[span[k] // 16]))
                    has_level3_products = True
                    kinds['(matrix) level-3 products'] += 1
                    kinds['(schedule) instructions of matrix sequences'] += len(words)
            else:
                say(new, old, ' + '.join(whats), 'not a way of rewriting that this checks')
        # A call through a register is told where to return by the place of a LEPC plus the
        # difference of two numbers (IADD3 on the three): the place it returns to must be where
        # the instruction it returned to now is, whatever the two numbers are.
        for k, t in enumerate(ta):
            m = re.match(r'LEPC (R\d+)$', t or '')
            add = next((j for j in range(k + 1, min(k + 16, len(ta))) if (ta[j] or '').startswith('IADD3 ') and m and m.group(1) in re.findall(r'-?R\d+', ta[j])[1:]), None)
            if add is None:
                continue
            terms = re.findall(r'-?R\d+', ta[add])[1:]
            minus, plus = [x[1:] for x in terms if x.startswith('-')], [x for x in terms if not x.startswith('-') and x != m.group(1)]
            def number(register):
                """(old offset, number) of the MOV that gives `register` before the addition."""
                for i in range(add - 1, max(k - 16, -1), -1):
                    found = re.match(r'MOV %s, 0x([0-9a-f]+)$' % register, ta[i] or '')
                    if found:
                        return 16 * i, int(found.group(1), 16)
                return None
            a, b = (number(plus[0]), number(minus[0])) if len(minus) == len(plus) == 1 else (None, None)
            if a and b and (a[0] in numbers or b[0] in numbers):
                now = [numbers[at][2] if at in numbers else new_of.get(value) for at, value in (a, b)]
                back = new_of.get(16 * k + a[1] - b[1])
                if None not in now and back is not None and became.get(16 * k, -1) + now[0] - now[1] == back:
                    numbers.pop(a[0], None)
                    numbers.pop(b[0], None)
        for old, (new, was, now) in sorted(numbers.items()):
            say(new, old, 'same', 'reads otherwise')
        registers = {u: r for u, r in place.items() if u.startswith('UR')}
        # A uniform register that has its ordinary one in the whole function (`whole`: the
        # translator declares no home for it there) shares it with nothing: not with another
        # such, and not with the home of a uniform register for a declared life. Two homes may
        # be one register where their declared lives do not meet, whether the register is one
        # of the program's or one left under the kernel's own count: rules R6, HL and HW say
        # what makes that right. (Until 2026-11-20 two homes in one register above the
        # program's were objected to whatever their lives.)
        declared_pairs = {('UR%d' % u, 'R%d' % r) for u, r, offsets in borrow_homes.get(name, [])}
        whole = {u: r for u, r in registers.items() if (u, r) not in declared_pairs}
        above_regs = {u: r for u, r in whole.items() if int(r[1:]) > highest}
        if len(set(above_regs.values())) != len(above_regs):
            problems.append('%s: two uniform registers share an ordinary one above highest: %s' % (name[:40], above_regs))
        both = sorted(set(whole.values()) & {r for u, r in declared_pairs})
        if both:
            problems.append('%s: a register stands for a uniform one in the whole function and is the home of another for a declared life: %s' % (name[:40], both))
        # (Rule V1, for the registers that the translator declares no life of.)
        both = sorted(set(whole.values()) & {'R%d' % r for r in frame_regs})
        if both:
            problems.append('%s: V: a register stands for a uniform one in the whole function and a load of the frame writes it at an instruction: %s' % (name[:40], both))
        # (The pool vs place overlap check is in the borrow-rules section below.)
        apart = ({lane} if lane else set()) | set((lanes_kept or {}).values()) | set(rows_kept.values())
        # (The registers that hold rows: those of the translation, which nothing but whole-row loads writes.)
        row_registers = {'R%d' % (r + k) for role, destination, where, also, arrive in row_loads.values() for r in where.values() for k in range(2) if r > highest}
        scratch |= row_scratch - apart
        # At level 3, the lane, scratch, pool, and held registers are per-instruction and may
        # share the same register at different instructions, so the function-wide overlap check
        # is relaxed.
        # The registers of the program that a matrix load at level 3 takes for its sequence are
        # borrowed at that instruction alone, as the pool is at others: what each holds there is
        # checked instruction by instruction (the whole-truth check of undeclared writes, and the
        # borrow rules), not here. (Until 2026-11-23 a function with level-3 matrix loads and no
        # product, a Triton tl.dot in float32 whose epilogue loads with LDSM and names 252
        # registers, was objected to here for that alone.)
        if frame_bytes == 0 and not has_level3_products:
            if ((scratch - borrowed_by_loads) | apart | row_registers) & (set(registers.values()) | pool | savers) or apart & scratch or row_registers & (scratch | apart):
                problems.append('%s: a register of the matrix sequences, the lane, what is kept of it or a row is used for something else too' % name[:40])
        if row_loads or halves or conversions:
            code = result.section('.text.' + name).data
            held = {int(r[1:]) for r in row_registers}
            for new, old, what in listing:
                d70 = mc.decode(70, int.from_bytes(code[new:new + 16], 'little'))
                if what != 'whole row' and d70 is not None and any(letter == 'R' and register in held for letter, register in d70.writes):
                    problems.append('%s+0x%x: writes a register that holds a row, and is no load of a row: %s' % (name[:40], new, tb[new // 16]))
            # One scoreboard for the halves of pairs: each waits for the ones before it on it; a
            # load whose program's load named none says there that its rows have arrived. The
            # instruction that does nothing before conversions waits for every scoreboard,
            # fifteen cycles before the first shuffle: a conversion reads a register where
            # nothing of the program did, so no wait of the program is known to cover it, and
            # fifteen cycles is the longest an instruction that names no scoreboard is waited
            # for (and more than the eight after which a product's result can be read by
            # something else). The shuffles say on a scoreboard when they are done
            # and what follows them waits for it.
            boards = {half[9] for half in halves}
            if len(boards) > 1:
                problems.append('%s: the halves of pairs do not all say on one scoreboard when their result is there' % name[:40])
            board = next(iter(boards)) if len(boards) == 1 else None
            for old, (role, destination, where, also, arrive) in row_loads.items():
                if arrive and arrive != {board}:
                    problems.append('%s (from 0x%x): a whole-row load whose program\'s load named no scoreboard does not say its rows are there where the halves of pairs wait' % (name[:40], old))
            for k, (new, old, what) in enumerate(listing):
                if what != 'settled':
                    continue
                word = int.from_bytes(code[new:new + 16], 'little')
                last = next((j for j in range(k + 1, len(listing)) if listing[j][2] not in ('to pairs', 'from pairs')), len(listing)) - 1
                follows = int.from_bytes(code[listing[last][0] + 16:listing[last][0] + 32], 'little') if listing[last][0] + 32 <= len(code) else 0
                said = {(int.from_bytes(code[listing[j][0]:listing[j][0] + 16], 'little') >> WRITE_BARRIER) & 7 for j in range(k + 1, last + 1)}
                if (word >> WAIT) & 63 != 63 or (word >> USCHED) & 15 != 15:
                    problems.append('%s+0x%x: conversions that do not wait for every scoreboard, fifteen cycles before the first shuffle' % (name[:40], new))
                if last == k or len(said) != 1 or 7 in said or not ((follows >> WAIT) & 63) >> next(iter(said)) & 1:
                    problems.append('%s+0x%x: what follows conversions does not wait for them' % (name[:40], new))
            # The plan, read back from the result: which load holds which rows where, which
            # product is which half of which pair, which registers are converted where. Whether
            # it is a plan the design allows (the pairs' conditions, the states of the registers
            # on every way, the places that are whole) is another program's to say (qcheck.py),
            # from the original function and this.
            # (A destination's rows are in the same registers whatever the load: a product does
            # not know which of the loads with its operand's destination it comes after. Whether
            # those that reach it are of the role it takes them in is the plan's.)
            by_destination = {}
            for old, (role, destination, where, also, arrive) in sorted(row_loads.items()):
                if by_destination.setdefault(destination, where) != where:
                    problems.append('%s (from 0x%x): the loads with destination R%d do not all put the same rows in the same registers' % (name[:40], old, destination))
            plan = {'loads': {old // 16: {'role': role, 'rd': destination, 'also': also} for old, (role, destination, where, also, arrive) in row_loads.items()},
                    'sites': {}, 'conversions': [(old // 16, side, r, to, None) for old, side, r, to in conversions], 'share': {}}
            for destination, where in by_destination.items():
                same = [d for d, other in by_destination.items() if {r for q, r in other.items() if r > highest} & {r for q, r in where.items() if r > highest}]
                plan['share'][destination] = min(same) if same else destination
            reads_of = {}
            for old, x, y, a_reg, b_reg, by_rows, ra, rb, rd, said, cx, cy, rc in halves:
                reads_of[old // 16] = (rd, rc, cx, cy)
                # (Destinations are multiples of four: the one of an operand register is known.)
                found, da, db = None, ra - ra % 4, rb - rb % 4
                wa, wb = by_destination.get(da), by_destination.get(db)
                for half in range(2) if wa and wb and da != db and ra % 2 == 0 else ():
                    if wa[((ra - da) // 2, half)] == a_reg and wb[((rb - db) // 2, half)] == b_reg and (rd == x) == ((rb - db) % 2 == 0):
                        found = dict(half=half, x=x, y=y, ra=da, a=(ra - da) // 2, rb=db, p=(rb - db) // 2, transposed=by_rows)
                if found is None:
                    problems.append('%s (from 0x%x): a half of a pair whose registers are not the rows of loads of its operands' % (name[:40], old))
                else:
                    plan['sites'][old // 16] = found
            order = sorted(plan['sites'])
            for k, i in enumerate(order):
                site = plan['sites'][i]
                others = [j for j in (order[k + 1:] if site['half'] == 0 else order[:k][::-1]) if (plan['sites'][j]['x'], plan['sites'][j]['y']) == (site['x'], site['y'])]
                if not others or plan['sites'][others[0]]['half'] == site['half']:
                    problems.append('%s (from 0x%x): a half of a pair without its other half' % (name[:40], 16 * i))
                else:
                    site['partner'] = others[0]
            # (A half of a pair writes the destinations of its two products, its own instruction's
            # and its partner's: for the rule on the registers of the program that the translation
            # writes, below, which goes by the destination of the instruction alone.)
            pair_writes = {16 * i: {site[k] + n for k in ('x', 'y') for n in range(4)} for i, site in plan['sites'].items()}
            # What the steps add to: the first half, to the accumulator each of the two products
            # of the program reads (its own instruction's and its partner's); the second half,
            # to the destinations.
            for i, site in plan['sites'].items():
                if 'partner' not in site:
                    continue
                j = site['partner']
                first, second = (i, j) if site['half'] == 0 else (j, i)
                accumulator = {reads_of[k][0]: reads_of[k][1] for k in (first, second)}        # {a product's destination: what it adds to}
                site['cx'], site['cy'] = accumulator.get(site['x']), accumulator.get(site['y'])
                want = (site['cx'], site['cy']) if site['half'] == 0 else (site['x'], site['y'])
                if set(accumulator) != {site['x'], site['y']} or reads_of[i][2:] != want:
                    problems.append('%s (from 0x%x): a half of a pair that does not add to what its products add to' % (name[:40], 16 * i))
            kinds['(pairs) conversions of accumulator registers'] += len(conversions)
            objections = plan_checked(mc, original.section('.text.' + name).data, tables.get(name, {}), plan)
            if objections is None:
                kinds['(not checked) the plan of pairs on whole rows of a function: no checker of plans here'] += 1
            for at, what in objections or ():
                problems.append('%s (from 0x%x): the plan of pairs: %s' % (name[:40], 16 * at if at is not None else 0, what))
        if lane is not None:
            # The lane and what is kept of it are written when the function is entered, and by
            # nothing else: whatever kind the map gives an instruction, none after those writes
            # them. And the function is entered there: no function symbol points further into it.
            # (`a` and `b` are other things by now.)
            held, code = {int(r[1:]) for r in apart}, result.section('.text.' + name).data
            for at in range(16 * (1 + sum(1 for new, old, what in listing if what == 'kept')), len(code) - 15, 16):
                d70 = mc.decode(70, int.from_bytes(code[at:at + 16], 'little'))
                if d70 is not None and any(letter == 'R' and register in held for letter, register in d70.writes):
                    problems.append('%s+0x%x: writes the lane, or what the function keeps of it: %s' % (name[:40], at, tb[at // 16]))
            kinds['(matrix) functions that go by the lane read at their start'] += 1
            inside_it = entered_at.get(original.sections.index(original.section('.text.' + name)))
            if inside_it:
                problems.append('%s: goes by the lane read at its start, and a global function symbol points to 0x%x in it' % (name[:40], inside_it[0]))
        if savers & (set(registers.values()) | pool):
            problems.append('%s: a register that predicates are saved in is used for something else too: %s' % (name[:40], sorted(savers & (set(registers.values()) | pool))))
        # And nothing that the translation adds writes such a register but what saves the
        # predicates in it: what is loaded from the frame for an instruction, or taken for it
        # among the registers left under the kernel's count, is in no set above.
        if savers:
            saver_numbers = {int(r[1:]) for r in savers}
            for new70, old75, what in listing:
                if what in ('same', 'save'):
                    continue
                d70 = mc.decode(70, int.from_bytes(result_code[new70:new70 + 16], 'little'))
                uses70 = mc._uses.get((70, d70.c.name)) if d70 is not None else None
                if uses70 is None:
                    continue
                for letter, operand, span in uses70.writes:
                    if letter == 'R' and operand in d70.ops and d70.ops[operand] != 255:
                        hit = saver_numbers & {d70.ops[operand] + k for k in range(machine.registers(span, d70.ops))}
                        if hit:
                            problems.append('%s+0x%x (from 0x%x, %s): writes R%d, in which predicates are saved: %s' % (name[:40], new70, old75, what, min(hit), tb[new70 // 16]))
        # Where the translation takes its registers one instruction at a time (it declares
        # pools, homes or room), a register may hold a uniform predicate along a held life and
        # be computed into elsewhere: rule HW says what makes that right. Otherwise the
        # register holds the predicate and nothing else in the whole function.
        if not (borrow_pools.get(name) or borrow_homes.get(name) or borrow_rooms.get(name)) and holders & (set(registers.values()) | pool | savers):
            problems.append('%s: a register that holds a uniform predicate is used for something else too: %s' % (name[:40], sorted(holders & (set(registers.values()) | pool | savers))))
        # The lives of the uniform predicates, as the translator says them, against nvdisasm's own
        # lives of the predicates of the original.
        chart, said = lives.get(name), claims.get(name, [])
        def tell(at, what):
            problems.append('%s+0x%x: %s' % (name[:40], at, what))
        life_of, taken = {}, {}
        for k, (q, p, saved, offsets) in enumerate(said):
            for at in offsets:
                if (q, at) in life_of:
                    tell(at, 'UP%d is in two lives there' % q)
                if taken.setdefault((p, at), k) != k:
                    tell(at, 'P%d stands for two uniform predicates there' % p)
                life_of[(q, at)] = k
        # The lives held in a register: {(uniform predicate, offset): which}. Inside one, an
        # instruction that names the uniform predicate has a life of its own, of that
        # instruction alone, for the ordinary predicate it uses there (checked like any life:
        # free there, or saved around it); the value goes from one to the next in the register.
        # A register holds one uniform predicate at an instruction: it may hold another along
        # other held lives, where the translation takes its registers among those that hold
        # nothing. (`holds`. Until 2026-11-20 a register was looked up for the first predicate
        # held in it anywhere in the function: what read back or wrote down a second one was
        # objected to.)
        held_at, holder_of, holds = {}, {}, {}
        held_locations = helds.get(name, []) + [(q, 'frame-%d' % word, offsets)
                                              for q, word, offsets in held_slots.get(name, [])]
        for k, (q, register, offsets) in enumerate(held_locations):
            if holder_of.setdefault(q, register) != register:
                tell(offsets[0], 'UP%d is held in two registers' % q)
            for at in offsets:
                if (q, at) in held_at:
                    tell(at, 'UP%d is in two held lives there' % q)
                held_at[(q, at)] = k
                if holds.setdefault((register, at), q) != q:
                    tell(at, '%s holds UP%d and UP%d there' % (register, holds[(register, at)], q))
        for (q, at), k in sorted(life_of.items()):
            if (q, at) in held_at and said[k][3] != [at]:
                tell(at, 'UP%d is held in a register there and has a life of several instructions too' % q)
        region = lambda q, at: ('held', held_at[(q, at)]) if (q, at) in held_at else ('life', life_of[(q, at)]) if (q, at) in life_of else None
        for at in sorted(set(backs) | set(downs)):
            for register, p in backs[at] + downs[at]:
                q = holds.get((register, at))
                if q is None:
                    tell(at, '%s is read back or written down there, where no uniform predicate is held in it' % register)
                elif ups.get((q, at)) != p:
                    tell(at, 'P%d is read back or written down for UP%d there, and the instruction has %s for it' % (p, q, 'P%d' % ups[(q, at)] if (q, at) in ups else 'none'))
        # HW, for the registers that hold predicates: nothing that the translation adds writes
        # one at an instruction of a held life, but what writes the predicate down there.
        # (What the program does to it there is rule H1's, where the register is the program's.)
        if holds:
            for new70, old75, what in listing:
                if what in ('same', 'held: kept'):
                    continue
                d70 = mc.decode(70, int.from_bytes(result_code[new70:new70 + 16], 'little'))
                uses70 = mc._uses.get((70, d70.c.name)) if d70 is not None else None
                if uses70 is None:
                    continue
                for letter, operand, span in uses70.writes:
                    if letter != 'R' or operand not in d70.ops or d70.ops[operand] == 255:
                        continue
                    for r in range(d70.ops[operand], d70.ops[operand] + machine.registers(span, d70.ops)):
                        if ('R%d' % r, old75) in holds:
                            problems.append('%s+0x%x (from 0x%x, %s): HW: writes R%d, which holds UP%d there: %s' % (
                                name[:40], new70, old75, what, r, holds[('R%d' % r, old75)], tb[new70 // 16]))
        for (q, at), r in sorted(ups.items()):
            k = life_of.get((q, at))
            if k is None:
                tell(at, 'the instruction has UP%d, which no life has there' % q)
            elif said[k][1] != r:
                tell(at, 'the instruction has P%d for UP%d, whose life there is in P%d' % (r, q, said[k][1]))
        saved_cover = {}           # {(predicate, old offset): first offset of the saved run}
        if chart is None and pred_own is not None:
            # nvdisasm gave no chart: check the lives of uniform predicates by the
            # validator's own backward dataflow on the original code.
            kinds['(own lives) lives of uniform predicates checked by own dataflow'] += len(said)
            for q, p, saved, offsets in said:
                wrong = None
                for at in offsets:
                    if at not in pred_own:
                        continue
                    if not saved and pred_own[at] >> p & 1:
                        wrong = wrong or (at, 'P%d stands for UP%d there, where it holds something of the program (own lives)' % (p, q))
                    if saved:
                        # For saved: the original instruction must not read or write the predicate.
                        d75 = mc.decode(75, int.from_bytes(original_code[at:at + 16], 'little'))
                        if d75 is not None:
                            touches = False
                            for letter, register, *_ in d75.reads:
                                if letter == 'P' and register == p:
                                    touches = True
                            for letter, register in d75.writes:
                                if letter == 'P' and register == p:
                                    touches = True
                            if touches:
                                wrong = wrong or (at, 'P%d is saved to stand for UP%d, and the instruction there reads or writes it (own lives)' % (p, q))
                if saved:
                    first, last = offsets[0], offsets[-1]
                    here = [(register, mask) for register, mask in saves[first] if mask == 1 << p and (register, mask) in restores[last]]
                    if offsets != list(range(first, last + 16, 16)):
                        wrong = wrong or (first, 'P%d is saved for a life of UP%d that is not a run of instructions' % (p, q))
                    elif not here:
                        wrong = wrong or (first, 'P%d stands for UP%d and is not saved before its life and given back after it' % (p, q))
                    elif any(at in inside for at in offsets[1:]) or any(leads[at] != [at + 16] for at in offsets) or (anywhere and len(offsets) > 1):
                        wrong = wrong or (first, 'P%d is saved for a life of UP%d that can be entered or left in the middle' % (p, q))
                if wrong:
                    tell(*wrong)
            # Every predicate saved is for a life, or for the comparison of a minimum.
            for at in {at for at, there in list(saves.items()) + list(restores.items()) if there}:
                if at not in pred_borrowed and not any(saved and at in (offsets[0], offsets[-1]) for q, p, saved, offsets in said):
                    tell(at, 'a predicate is saved or given back there for no life')
            # Build saved_cover for the slow-predicate check that follows.
            for q, p, saved, offsets in said:
                if not saved or len(offsets) < 2:
                    continue
                first, last = offsets[0], offsets[-1]
                if offsets != list(range(first, last + 16, 16)):
                    continue
                here = [(register, mask) for register, mask in saves[first] if mask == 1 << p and (register, mask) in restores[last]]
                if not here:
                    continue
                if any(at in inside for at in offsets[1:]) or any(leads[at] != [at + 16] for at in offsets) or (anywhere and len(offsets) > 1):
                    continue
                run_ok = True
                for at in offsets:
                    d75 = mc.decode(75, int.from_bytes(original_code[at:at + 16], 'little'))
                    if d75 is not None:
                        for letter, register, *_ in d75.writes:
                            if letter == 'P' and register == p:
                                run_ok = False
                        for letter, register, *_ in d75.reads:
                            if letter == 'P' and register == p:
                                run_ok = False
                    if not run_ok:
                        break
                if not run_ok:
                    continue
                for at in offsets[1:]:
                    saved_cover[(p, at)] = first
        elif chart is None:
            kinds['(not checked) the lives of uniform predicates'] += len(said)
        else:
            kinds['(lives) lives of uniform predicates checked with nvdisasm'] += len(said)
            # (Where an instruction leads: `leads`, `inside`, `anywhere`, from `flow` above.)
            entered = {q: [to for to, there in chart.items() if q in there[3]] for q in {q for q, _, _, _ in said}}
            # (A uniform predicate that no instruction of the function reads or writes needs no
            # life: nothing stands for it. In a function that returns to its caller every
            # predicate is taken to hold something all along, those it never names too.)
            named = {q for there in chart.values() for q in there[6] | there[7]}
            missing = set()
            for at, there in sorted(chart.items()):
                uniform = there[1] & named
                for q in uniform:
                    if (q, at) not in life_of and (q, at) not in held_at and q not in missing:
                        missing.add(q)
                        tell(at, 'UP%d holds something there in the original, and no life has it' % q)
                # In a held life: the instruction that reads the uniform predicate (or may leave
                # it as it was, under a guard) has it read back before; the one that writes it
                # has it written down after.
                for q in there[6] | there[7]:
                    if (q, at) not in held_at:
                        continue
                    register = holder_of[q]
                    guarded = ta[at // 16] is not None and ta[at // 16].lstrip().startswith('@')
                    if (q in there[6] or guarded) and (register, ups.get((q, at))) not in backs[at]:
                        tell(at, 'UP%d is held in %s and is not read back before the instruction that reads it' % (q, register))
                    if q in there[7] and (register, ups.get((q, at))) not in downs[at]:
                        tell(at, 'UP%d is held in %s and is not written down after the instruction that writes it' % (q, register))
                # No value goes from one life to another. (From a branch that may lead anywhere: to
                # no other life at all.)
                for q in uniform:
                    for to in (entered.get(q, []) if at in anywhere else leads[at]):
                        if to in chart and q in chart[to][3] and region(q, at) is not None and region(q, to) is not None and region(q, at) != region(q, to):
                            tell(at, 'UP%d goes from one life to another, at 0x%x' % (q, to))
            for q, p, saved, offsets in said:
                wrong = None
                for at in offsets:
                    if at not in chart:
                        continue
                    if not saved and p in chart[at][0]:
                        wrong = wrong or (at, 'P%d stands for UP%d there, where it holds something of the program' % (p, q))
                    if saved and p in chart[at][2]:
                        wrong = wrong or (at, 'P%d is saved to stand for UP%d, and the instruction there reads or writes it' % (p, q))
                if saved:
                    first, last = offsets[0], offsets[-1]
                    here = [(register, mask) for register, mask in saves[first] if mask == 1 << p and (register, mask) in restores[last]]
                    if offsets != list(range(first, last + 16, 16)):
                        wrong = wrong or (first, 'P%d is saved for a life of UP%d that is not a run of instructions' % (p, q))
                    elif not here:
                        wrong = wrong or (first, 'P%d stands for UP%d and is not saved before its life and given back after it' % (p, q))
                    elif any(at in inside for at in offsets[1:]) or any(leads[at] != [at + 16] for at in offsets) or (anywhere and len(offsets) > 1):
                        wrong = wrong or (first, 'P%d is saved for a life of UP%d that can be entered or left in the middle' % (p, q))
                if wrong:
                    tell(*wrong)
            # Every predicate saved is for a life, or for the comparison of a minimum.
            for at in {at for at, there in list(saves.items()) + list(restores.items()) if there}:
                if at not in pred_borrowed and not any(saved and at in (offsets[0], offsets[-1]) for q, p, saved, offsets in said):
                    tell(at, 'a predicate is saved or given back there for no life')
            # ---- saved_cover: for a slow predicate in a valid saved life, the
            # wait at the save covers every instruction of the life. Build a map
            # {(predicate, instruction offset) -> first offset of the saved run} for
            # the saved lives that passed all checks above.
            # Conditions (checked independently above at lines 3322-3339 of the original):
            # 1. The life is `saved` (from the .lives file).
            # 2. The offsets form a contiguous run.
            # 3. The save (P2R) and restore (R2P) exist at the first and last.
            # 4. No instruction of the run is a branch target entered from elsewhere.
            # 5. Every instruction of the run goes on to the next (straight line).
            # 6. No original instruction in the run reads or writes the predicate.
            # A write to the predicate within such a run is covered by the wait at the
            # save: the late writer was issued before the save, and the save waits for all
            # scoreboards plus enough cycles.
            saved_cover = {}           # {(predicate, old offset): first offset of the saved run}
            for q, p, saved, offsets in said:
                if not saved or len(offsets) < 2:
                    continue
                first, last = offsets[0], offsets[-1]
                # Check the same conditions the validator already checked for this life.
                if offsets != list(range(first, last + 16, 16)):
                    continue
                here = [(register, mask) for register, mask in saves[first] if mask == 1 << p and (register, mask) in restores[last]]
                if not here:
                    continue
                if any(at in inside for at in offsets[1:]) or any(leads[at] != [at + 16] for at in offsets) or (anywhere and len(offsets) > 1):
                    continue
                # Also verify independently that no original instruction writes or reads the predicate.
                run_ok = True
                for at in offsets:
                    d75 = mc.decode(75, int.from_bytes(original_code[at:at + 16], 'little'))
                    if d75 is not None:
                        for letter, register, *_ in d75.writes:
                            if letter == 'P' and register == p:
                                run_ok = False
                        for letter, register, *_ in d75.reads:
                            if letter == 'P' and register == p:
                                run_ok = False
                    if not run_ok:
                        break
                if not run_ok:
                    continue
                for at in offsets[1:]:            # the first has the wait; the rest are covered
                    saved_cover[(p, at)] = first
        # ---- slow predicates: a predicate written by DSETP or by an instruction
        # of kind MIO_FAST or MIO_SLOW. Computed from the ORIGINAL.
        # `slow`: the set of such predicates. `never`: those written by MIO_SLOW under no
        # scoreboard (never taken at all). `wait_cycles[p]`: the minimum stall the translation
        # must put before an instruction that writes or saves predicate p.
        # Distances from the table: DSETP 13, MIO_FAST (FCHK etc.) 26, with scoreboard 1.
        slow_preds, never_preds = set(), set()
        wait_cycles = {}              # {predicate: cycles needed}
        for at in range(0, len(original_code), 16):
            d75 = mc.decode(75, int.from_bytes(original_code[at:at + 16], 'little'))
            if d75 is None:
                continue
            if d75.mnemonic == 'DSETP' or d75.kind in ('MIO_FAST', 'MIO_SLOW'):
                board = (int.from_bytes(original_code[at:at + 16], 'little') >> WRITE_BARRIER) & 7
                for letter, register in d75.writes:
                    if letter != 'P':
                        continue
                    slow_preds.add(register)
                    if d75.kind == 'MIO_SLOW' and board == 7:
                        never_preds.add(register)
                    elif board != 7:
                        wait_cycles[register] = max(wait_cycles.get(register, 0), 1)
                    elif d75.mnemonic == 'DSETP':
                        wait_cycles[register] = max(wait_cycles.get(register, 0), 13)
                    elif d75.kind == 'MIO_FAST':
                        wait_cycles[register] = max(wait_cycles.get(register, 0), 26)
                    else:
                        # MIO_SLOW with scoreboard: handled above (board != 7)
                        pass
        if slow_preds:
            # Rule 3: no predicate in `never` is used at all.
            for (q, at), p in ups.items():
                if p in never_preds:
                    problems.append('%s+0x%x: a never-predicate P%d (MIO_SLOW under no scoreboard) stands for UP%d' % (name[:40], at, p, q))
            for at, pairs in saves.items():
                for reg, mask in pairs:
                    for p in range(7):
                        if mask >> p & 1 and p in never_preds:
                            problems.append('%s+0x%x: a never-predicate P%d is saved by the translation' % (name[:40], at, p))
            for at, pairs in backs.items():
                for reg, p in pairs:
                    if p in never_preds:
                        problems.append('%s+0x%x: a never-predicate P%d is read back for a held uniform predicate' % (name[:40], at, p))
            for new70, old75, what in listing:
                if what == 'comparison':
                    d70 = mc.decode(70, int.from_bytes(result_code[new70:new70 + 16], 'little'))
                    if d70 is not None:
                        for letter, register in d70.writes:
                            if letter == 'P' and register in never_preds:
                                problems.append('%s+0x%x: a never-predicate P%d is used for a minimum' % (name[:40], new70, register))
            # Rule 2: a slow predicate may be taken only behind a wait. For every instruction
            # of the original whose translation writes or saves a slow predicate: the sequence
            # must begin with `late predicate` NOPs, the first waiting for all six scoreboards,
            # with stalls adding up to at least `wait_cycles[p]`.
            # Exception: an instruction inside a saved run (not at its first
            # instruction) needs no wait of its own when the save at the first instruction
            # already has it, because:
            #  - the late writer was issued before the save;
            #  - the save waits for all scoreboards and enough cycles;
            #  - no original instruction in the run writes the predicate;
            #  - no entry from elsewhere, straight line (all checked above for the life).
            # "Writes or saves": writes a UR whose life stands in p; saves p (the first of a
            # life); reads back p for a held UR; computes a minimum into p.
            for old75, parts_all in by_origin.items():
                parts = [(n, w) for n, w in parts_all if w not in ('wait',)]
                # Find slow predicates this instruction writes or saves, AND collect them.
                needed = 0
                needed_preds = set()          # the predicates that need a wait here
                for n, w in parts:
                    if w == 'save':
                        m_save = re.match(r'P2R (R\d+), PR, RZ, 0x([0-9a-f]+)$', tb[n // 16] or '')
                        if m_save:
                            mask = int(m_save.group(2), 16)
                            for p in range(7):
                                if mask >> p & 1 and p in slow_preds and p not in never_preds:
                                    needed = max(needed, wait_cycles.get(p, 0))
                                    needed_preds.add(p)
                    elif w == 'held: back':
                        m_back = re.match(r'ISETP\.NE\.U32\.AND P(\d), PT, (R\d+), RZ, PT$', tb[n // 16] or '')
                        if m_back and int(m_back.group(1)) in slow_preds - never_preds:
                            needed = max(needed, wait_cycles.get(int(m_back.group(1)), 0))
                            needed_preds.add(int(m_back.group(1)))
                    elif w == 'comparison':
                        d70 = mc.decode(70, int.from_bytes(result_code[n:n + 16], 'little'))
                        if d70 is not None:
                            for letter, register in d70.writes:
                                if letter == 'P' and register in slow_preds - never_preds:
                                    needed = max(needed, wait_cycles.get(register, 0))
                                    needed_preds.add(register)
                    elif w == 'ordinary registers' or w == 'constant' or w == 'move':
                        # Writes a UR whose life stands in a slow predicate.
                        for (q, off), p in ups.items():
                            if off == old75 and p in slow_preds - never_preds:
                                d75 = mc.decode(75, int.from_bytes(original_code[old75:old75 + 16], 'little'))
                                if d75 is not None and any(l == 'Q' and x == q for l, x in d75.writes):
                                    needed = max(needed, wait_cycles.get(p, 0))
                                    needed_preds.add(p)
                if needed == 0:
                    continue
                # Check: the sequence begins with `late predicate` NOPs.
                late_parts = [(n, w) for n, w in parts_all if w == 'late predicate']
                if not late_parts:
                    # Check if covered by a saved run. Every predicate that needs a
                    # wait must be in saved_cover for this instruction.
                    covered = all((p, old75) in saved_cover for p in needed_preds)
                    if covered:
                        # Verify the save at the first instruction of each covering run has
                        # the wait with enough cycles. (Each predicate may be in a different
                        # saved run, though in practice they share one.)
                        for p in needed_preds:
                            run_first = saved_cover[(p, old75)]
                            run_parts = [(n, w) for n, w in by_origin.get(run_first, ()) if w == 'late predicate']
                            if not run_parts:
                                problems.append('%s+0x%x: writes a slow predicate P%d in a saved run whose save at 0x%x has no wait' % (name[:40], old75, p, run_first))
                                covered = False
                                break
                            first_word = int.from_bytes(result_code[run_parts[0][0]:run_parts[0][0] + 16], 'little')
                            if (first_word >> WAIT) & 63 != 63:
                                problems.append('%s+0x%x: writes a slow predicate P%d in a saved run whose save does not wait for all six scoreboards' % (name[:40], old75, p))
                                covered = False
                                break
                            total_stalls = sum((int.from_bytes(result_code[n:n + 16], 'little') >> USCHED) & 15 for n, w in run_parts)
                            if total_stalls < wait_cycles.get(p, 0):
                                problems.append('%s+0x%x: writes a slow predicate P%d in a saved run whose save waits %d cycles, need %d' % (name[:40], old75, p, total_stalls, wait_cycles.get(p, 0)))
                                covered = False
                                break
                    if not covered:
                        problems.append('%s+0x%x: writes or saves a slow predicate without the wait before it' % (name[:40], old75))
                    continue
                # First NOP waits for all six scoreboards.
                first_word = int.from_bytes(result_code[late_parts[0][0]:late_parts[0][0] + 16], 'little')
                if (first_word >> WAIT) & 63 != 63:
                    problems.append('%s+0x%x: the first NOP of a slow-predicate wait does not wait for all six scoreboards' % (name[:40], old75))
                # Stalls add up to at least `needed`.
                total_stalls = sum((int.from_bytes(result_code[n:n + 16], 'little') >> USCHED) & 15 for n, w in late_parts)
                if total_stalls < needed:
                    problems.append('%s+0x%x: the slow-predicate wait is %d cycles, need %d' % (name[:40], old75, total_stalls, needed))
            kinds['(slow) predicates written by DSETP or late instructions'] += len(slow_preds)
        # Rule U the other way round: what the translator declares put away at an instruction is
        # put away there. A register declared so may be written by the sequence of that instruction
        # while it holds something of the program, and only its store before and its load after
        # make that right. (Until 2026-11-20 a declaration with neither was taken on trust.)
        for r, offsets in rooms_declared:
            for at in offsets:
                if r not in room_made.get(at, ()):
                    problems.append('%s (from 0x%x): U: R%d is declared put away at this instruction, and nothing of its sequence puts it away' % (name[:40], at, r))
        # ---- schedule check (deferred from above: needs borrowed_regs) ----
        wrong, late = schedule(mc, result_code, highest, sequences, borrowed_regs,
                               {new70 for new70, old75, what in listing if what != 'same'})
        kinds['(schedule) instructions that read late a register of the translation'] += late
        for at, what in wrong:
            problems.append('%s+0x%x: %s: %s' % (name[:40], at, what, tb[at // 16]))
        wrong, followed, complement = data_overwritten(mc, result_code)
        kinds['(schedule) registers that a memory instruction reads as data, followed to what writes them next'] += followed
        kinds['(schedule) writes of such a register under the complement of the guard of the store that reads it'] += complement
        for at, what in wrong:
            problems.append('%s+0x%x: %s: %s' % (name[:40], at, what, tb[at // 16]))
        wrong, looked = double_read_early(mc, result_code)
        kinds['(schedule) readers of a result of double precision, each as far after it as Volta keeps them'] += looked
        for at, what in wrong:
            problems.append('%s+0x%x: %s: %s' % (name[:40], at, what, tb[at // 16]))
        # ---- even-register alignment: an operand of two registers begins on an even one ----
        # Check on the TRANSLATION (sm_70 result), every instruction.
        for new70, old75, what in listing:
            d70 = mc.decode(70, int.from_bytes(result_code[new70:new70+16], 'little'))
            if d70 is None:
                continue
            uses70 = mc._uses.get((70, d70.c.name))
            if uses70 is None:
                continue
            for letter, operand, span in uses70.reads + uses70.writes:
                if letter == 'R' and operand in d70.ops:
                    reg = d70.ops[operand]
                    if reg == 255:
                        continue
                    count = machine.registers(span, d70.ops)
                    if count == 2 and reg % 2 == 1:
                        problems.append('%s+0x%x: a two-register operand begins on odd R%d: %s' % (name[:40], new70, reg, tb[new70 // 16]))
        # ---- what the sequences compute into is not what stands for a uniform register ----
        # In the whole function, for a uniform register that has its register from one end of the
        # function to the other (the translator declares no life of it). A register that is the
        # home of a uniform register for its declared life may be computed into outside that
        # life: rules HL and HW below say what makes that right. (Until 2026-11-20 this rule
        # took every home above the program's registers for one held all along: a function
        # translated within its own count, where the registers left under the count are taken
        # one instruction at a time like the program's, was objected to.)
        pool_place_overlap = pool & set(whole.values())
        if pool_place_overlap:
            problems.append('%s: a register that the sequences compute into stands for a uniform one too: %s' % (name[:40], sorted(pool_place_overlap)))
        # ---- rules for borrowed registers (1 through 6) ----
        # Declared borrowings from the .lives file: homes and pools.
        homes_declared = borrow_homes.get(name, [])        # [(UR, R, [offsets])]
        pools_declared = borrow_pools.get(name, [])        # [(width, R, [offset])]
        B = borrowed_regs
        has_borrows = bool(homes_declared) or bool(pools_declared) or bool(B)
        if has_borrows:
            # Rule 5: nothing is borrowed in a function that is not a kernel entry.
            orig_section_idx = next((k for k, s in enumerate(original.sections) if s.name == '.text.' + name and s.type == 1), None)
            is_entry = orig_section_idx is not None and any(sym[3] == orig_section_idx and sym[1] & 15 == 2 and sym[2] & 0x10 for sym in original.symbols())
            if not is_entry:
                problems.append('%s: registers are borrowed in a function that is not a kernel entry' % name[:40])
            if reg_lives is not None:
                rl_in, rl_out, rl_reads, rl_writes, rl_never, rl_exact, rl_inexact, rl_reasons = reg_lives
                kinds['(borrow) exact calls'] += rl_exact
                kinds['(borrow) inexact calls'] += rl_inexact
                for reason, count in rl_reasons.items():
                    kinds['(borrow) inexact: ' + reason] += count
                n75 = len(original_code) // 16
                # R1 for homes: at each declared instruction, the register holds nothing.
                for u, r, offsets in homes_declared:
                    for at in offsets:
                        i = at // 16
                        if i >= n75:
                            continue
                        busy = (rl_in[i] | rl_out[i] | rl_reads[i] | rl_writes[i] | rl_never) & ((1 << 256) - 1)
                        if busy >> r & 1:
                            problems.append('%s+0x%x: borrowed home R%d (UR%d) is not free' % (name[:40], at, r, u))
                            break
                # R1 for pools: at each declared instruction, each register holds nothing.
                for width, r, offsets in pools_declared:
                    for at in offsets:
                        i = at // 16
                        if i >= n75:
                            continue
                        busy = (rl_in[i] | rl_out[i] | rl_reads[i] | rl_writes[i] | rl_never) & ((1 << 256) - 1)
                        for k in range(width):
                            if busy >> (r + k) & 1:
                                problems.append('%s+0x%x: borrowed pool R%d is not free' % (name[:40], at, r + k))
                                break
                # HL: the life that the translator declares for the home of a uniform register is
                # its whole life, by the validator's own lives of the uniform registers (bit
                # 256 + u of `register_lives`): every instruction before or after which it holds
                # something that may still be read, and every instruction that reads or writes
                # it. And its homes are one register. (Until 2026-11-20 the declared life was
                # taken on trust: homes declared at fewer instructions than the value lives
                # passed, and with them whatever was computed into the register meanwhile.)
                home_of, home_life = {}, collections.defaultdict(set)
                for u, r, offsets in homes_declared:
                    if home_of.setdefault(u, r) != r:
                        problems.append('%s: HL: UR%d has two homes, R%d and R%d' % (name[:40], u, home_of[u], r))
                    home_life[u].update(offsets)
                for u in sorted(home_of):
                    bit = 1 << (256 + u)
                    for i in range(n75):
                        if (rl_in[i] | rl_out[i] | rl_reads[i] | rl_writes[i]) & bit and 16 * i not in home_life[u]:
                            problems.append('%s+0x%x: HL: UR%d holds something there, or is read or written there, and its home R%d is not declared there' % (name[:40], 16 * i, u, home_of[u]))
                            break
                kinds['(borrow) homes whose declared life is the whole life of the uniform register'] += len(home_of)
                # H1: a register of the program that holds a uniform predicate between the
                # instructions that name it holds nothing of the program all along that life.
                for q, register, offsets in helds.get(name, []):
                    rn = int(register[1:]) if register.startswith('R') else -1
                    if not 2 <= rn <= highest:
                        continue
                    for at in offsets:
                        i = at // 16
                        if i < n75 and (rl_in[i] | rl_out[i] | rl_reads[i] | rl_writes[i] | rl_never) >> rn & 1:
                            problems.append('%s+0x%x: H1: borrowed holder R%d (UP%d) is not free' % (name[:40], at, rn, q))
                            break
                # HW: nothing that the translation adds writes the home of a uniform register at an
                # instruction of its declared life, but what an instruction that writes that
                # uniform register became. (The same of the registers that hold predicates is
                # asked of every function, with the held lives.)
                homes_at = collections.defaultdict(dict)
                for u, r, offsets in homes_declared:
                    for at in offsets:
                        homes_at[at][r] = u
                for new70, old75, what in listing:
                    if what == 'same' or old75 not in homes_at:
                        continue
                    d70 = mc.decode(70, int.from_bytes(result_code[new70:new70 + 16], 'little'))
                    uses70 = mc._uses.get((70, d70.c.name)) if d70 is not None else None
                    if uses70 is None:
                        continue
                    written = set()
                    for letter, operand, span in uses70.writes:
                        if letter == 'R' and operand in d70.ops and d70.ops[operand] != 255:
                            written.update(d70.ops[operand] + k for k in range(machine.registers(span, d70.ops)))
                    for r in sorted(written):
                        u = homes_at[old75].get(r)
                        if u is None:
                            continue
                        if not rl_writes[old75 // 16] >> (256 + u) & 1:
                            problems.append('%s+0x%x (from 0x%x, %s): HW: writes R%d, the home of UR%d declared there, and the instruction does not write UR%d: %s' % (
                                name[:40], new70, old75, what, r, u, u, tb[new70 // 16]))
                        elif what in AROUND or is_mem(what) or is_room(what):
                            # (What is put around the instruction, a load of the frame: the
                            # instruction may still read what the home holds. Passed until
                            # 2026-11-20 at any instruction that writes the uniform register.)
                            problems.append('%s+0x%x (from 0x%x, %s): HW: writes R%d, the home of UR%d declared there, and is not what the instruction that writes UR%d became: %s' % (
                                name[:40], new70, old75, what, r, u, u, tb[new70 // 16]))
                # Whole-truth check: no instruction ADDED by the translation (not 'same')
                # writes a register of the program (number <= highest) that is not declared
                # as a home or pool at that instruction, nor the destination of the original.
                declared_at = {}           # {sm_75 offset: set of declared registers}
                for u, r, offsets in homes_declared:
                    for at in offsets:
                        declared_at.setdefault(at, set()).add(r)
                for width, r, offsets in pools_declared:
                    for at in offsets:
                        for k in range(width):
                            declared_at.setdefault(at, set()).add(r + k)
                # Held registers: a register that holds a uniform predicate is borrowed there.
                for q, register, offsets in helds.get(name, []):
                    rn = int(register[1:]) if register.startswith('R') else -1
                    if rn >= 2 and rn <= highest:
                        for at in offsets:
                            declared_at.setdefault(at, set()).add(rn)
                # ---- rule U: room registers are declared (NOT required to hold nothing) ----
                for r, offsets in rooms_declared:
                    for at in offsets:
                        declared_at.setdefault(at, set()).add(r)
                for new70, old75, what in listing:
                    if what in ('same',) or what in AROUND or is_room(what):
                        continue
                    d70 = mc.decode(70, int.from_bytes(result_code[new70:new70+16], 'little'))
                    if d70 is None:
                        continue
                    # The original instruction's destination registers.
                    i = old75 // 16
                    orig_dest = set()
                    if i < n75:
                        d75 = mc.decode(75, int.from_bytes(original_code[old75:old75+16], 'little'))
                        if d75 is not None:
                            orig_dest = {r for l, r in d75.writes if l == 'R' and r != 255}
                    declared = declared_at.get(old75, set()) | orig_dest
                    if what == 'half of a pair':
                        declared |= pair_writes.get(old75, set())
                    for l, r in d70.writes:
                        if l == 'R' and r <= highest and r >= 2 and r not in declared:
                            problems.append('%s+0x%x: a register of the program (R%d) written by the translation and not declared' % (name[:40], new70, r))
                # R3a: no MIO_FAST or MIO_SLOW instruction writes a register of B without
                # naming a write scoreboard.
                words70 = [int.from_bytes(result_code[x:x+16], 'little') for x in range(0, len(result_code), 16)]
                for k, w in enumerate(words70):
                    d70 = mc.decode(70, w)
                    if d70 is None or d70.kind not in ('MIO_FAST', 'MIO_SLOW'):
                        continue
                    board = (w >> WRITE_BARRIER) & 7
                    if board == 7 and any(l == 'R' and r in B for l, r in d70.writes):
                        problems.append('%s+0x%x: R3a: a late writer writes a borrowed register without a write scoreboard: %s' % (name[:40], 16 * k, tb[k]))
                # R3b: for each instruction that has a pool or home declaration, the first
                # instruction of its sequence waits for every scoreboard any instruction of
                # the function names for a write into, or a late read of, that register.
                boards_of = {}         # {register: set of scoreboards}
                for k, w in enumerate(words70):
                    d70 = mc.decode(70, w)
                    if d70 is None:
                        continue
                    wb = (w >> WRITE_BARRIER) & 7
                    rb = (w >> READ_BARRIER) & 7
                    for l, r in d70.writes:
                        if l == 'R' and r in B and wb != 7:
                            boards_of.setdefault(r, set()).add(wb)
                    for l, r, *_ in d70.reads:
                        if l == 'R' and r in B and rb != 7:
                            boards_of.setdefault(r, set()).add(rb)
                borrows_at_old = {}       # {sm_75 offset: set of borrowed registers}
                for u, r, offsets in homes_declared:
                    for at in offsets:
                        d75 = mc.decode(75, int.from_bytes(original_code[at:at+16], 'little')) if at + 16 <= len(original_code) else None
                        if d75 is not None and any(l == 'U' and x == u for l, x in d75.writes):
                            borrows_at_old.setdefault(at, set()).add(r)
                for width, r, offsets in pools_declared:
                    for at in offsets:
                        for k in range(width):
                            borrows_at_old.setdefault(at, set()).add(r + k)
                # ---- rule T: room registers added to the C3 / R3b wait check ----
                for r, offsets in rooms_declared:
                    for at in offsets:
                        borrows_at_old.setdefault(at, set()).add(r)
                for old75, regs in borrows_at_old.items():
                    # Find the first instruction of this instruction's sequence in the result.
                    # C3: the sequence starts at the FIRST instruction the map gives to the
                    # original one, whatever its kind (save, held: back, mem: base...).
                    all_seq = [(n, w) for n, o, w in listing if o == old75]
                    if not all_seq:
                        continue
                    first_n70 = all_seq[0][0]
                    waits = 0
                    for n, w in all_seq:
                        waits |= (int.from_bytes(result_code[n:n+16], 'little') >> WAIT) & 63
                    for r in regs:
                        needed = boards_of.get(r, set())
                        for sb in needed:
                            if not waits >> sb & 1:
                                problems.append('%s+0x%x: R3b: writes borrowed R%d without waiting for scoreboard %d' % (name[:40], first_n70, r, sb))
                                break
                # C3: for a register taken at an original instruction (pool, held, home),
                # the first instruction of the translation that writes it must be preceded
                # by a wait for every scoreboard on which some PROGRAM instruction of the
                # function writes that register late. The scoreboards are those of the sm_70
                # result (the program's instructions are translated, and their scoreboards
                # carry over). Accumulate over the whole function's sm_70 instructions that
                # write the register on a scoreboard AND are translations of instructions
                # the ORIGINAL writes that register on.
                # Build: for each borrowed register, the scoreboards from instructions that
                # are 'same' or translated from an original that writes R on a scoreboard.
                c3_boards_of = {}      # {register: set of scoreboards}
                c3_examined, c3_needed_wait = 0, 0
                words75 = [int.from_bytes(original_code[x:x+16], 'little') for x in range(0, len(original_code), 16)]
                for k70, w70 in enumerate(words70):
                    d70 = mc.decode(70, w70)
                    if d70 is None:
                        continue
                    wb70 = (w70 >> WRITE_BARRIER) & 7
                    if wb70 == 7:
                        continue
                    # Find the original instruction this came from.
                    new70_off = 16 * k70
                    orig_off = None
                    for _new, _old, _what in listing:
                        if _new == new70_off:
                            orig_off = _old
                            break
                    if orig_off is None:
                        continue
                    i75 = orig_off // 16
                    if i75 >= len(words75):
                        continue
                    d75 = mc.decode(75, words75[i75])
                    if d75 is None:
                        continue
                    # For each register of B that the original writes (on any scoreboard):
                    for l, r in d75.writes:
                        if l == 'R' and r in B:
                            c3_boards_of.setdefault(r, set()).add(wb70)
                for old75, regs in borrows_at_old.items():
                    all_seq = [(n, w) for n, o, w in listing if o == old75]
                    if not all_seq:
                        continue
                    waits = 0
                    for n, w in all_seq:
                        waits |= (int.from_bytes(result_code[n:n+16], 'little') >> WAIT) & 63
                    for r in regs:
                        c3_examined += 1
                        needed = c3_boards_of.get(r, set())
                        if needed:
                            c3_needed_wait += 1
                        for sb in needed:
                            if not waits >> sb & 1:
                                problems.append('%s+0x%x: C3: first write of borrowed R%d does not wait for scoreboard %d (program writes R%d late on it)' % (name[:40], all_seq[0][0], r, sb, r))
                                break
                kinds['(level3) C3: first writes of borrowed registers examined'] += c3_examined
                kinds['(level3) C3: of those, needing a wait'] += c3_needed_wait
                # R3c: a translation instruction that writes, with a write scoreboard,
                # a declared borrowed register: the NEXT instruction waits for it.
                # Borrowed at an original instruction: named by a pool line, or the home of
                # a home line whose offsets include it, or a held register there; the
                # original's own R destinations never.
                # (At level 3, mem: load instructions have their own wait rule, B2.
                #  Matrix sequence instructions have their own schedule check.)
                for new70, old75, what in listing:
                    if what in ('same',) or what in AROUND or is_mem(what) or is_room(what) or new70 in sequences:
                        continue
                    d70 = mc.decode(70, int.from_bytes(result_code[new70:new70+16], 'little'))
                    if d70 is None:
                        continue
                    wb = (int.from_bytes(result_code[new70:new70+16], 'little') >> WRITE_BARRIER) & 7
                    if wb == 7:
                        continue
                    declared = declared_at.get(old75, set())
                    # Exclude the original instruction's own R destinations.
                    i75 = old75 // 16
                    orig_dest = set()
                    if i75 < n75:
                        d75_r3c = mc.decode(75, int.from_bytes(original_code[old75:old75+16], 'little'))
                        if d75_r3c is not None:
                            orig_dest = {r for l, r in d75_r3c.writes if l == 'R' and r != 255}
                    writes_borrowed = {r for l, r in d70.writes if l == 'R' and r in declared and r not in orig_dest}
                    if writes_borrowed and d70.kind in ('MIO_FAST', 'MIO_SLOW'):
                        if not waited_after(result_code, new70, wb)[0]:
                            problems.append('%s+0x%x: R3c: writes borrowed register with a late result, next does not wait for scoreboard %d' % (name[:40], new70, wb))
                # R4a: every instruction of the result of a kind in machine.LATE that reads
                # a register of B names a read scoreboard.
                for k, w in enumerate(words70):
                    d70 = mc.decode(70, w)
                    if d70 is None or d70.kind not in machine.LATE:
                        continue
                    if any(l == 'R' and r in B for l, r, *_ in d70.reads):
                        if (w >> READ_BARRIER) & 7 == 7:
                            problems.append('%s+0x%x: R4a: reads a borrowed register late without a read scoreboard: %s' % (name[:40], 16 * k, tb[k]))
                # R4b is handled by the schedule check above (which includes borrowed registers).
                # R6: values of the translation do not meet (using declarations).
                # Two UR homes alive at the same instruction must be different registers.
                home_by_r = {}         # {register: [(UR, offsets_set)]}
                for u, r, offsets in homes_declared:
                    home_by_r.setdefault(r, []).append((u, set(offsets)))
                for r, entries in home_by_r.items():
                    if len(entries) < 2:
                        continue
                    for i in range(n75):
                        at = 16 * i
                        present = [u for u, offs in entries if at in offs]
                        if len(present) > 1:
                            problems.append('%s+0x%x: R6: UR%d and UR%d share borrowed R%d and are both declared there' % (name[:40], at, present[0], present[1], r))
                            break
                # Pool at an instruction must not overlap with a home declared there.
                for width, r, offsets in pools_declared:
                    for at in offsets:
                        for k in range(width):
                            for u, hr, hoffsets in homes_declared:
                                if hr == r + k and at in hoffsets:
                                    problems.append('%s+0x%x: R6: pool R%d is also the home of UR%d declared there' % (name[:40], at, r + k, u))
                kinds['(borrow) borrowed registers'] += len(B)
            else:
                kinds['(not checked) borrowed registers: register liveness could not be computed'] += 1
    # What could not be checked (nvdisasm gave no chart of the lives for a function) is said in the
    # verdict and in the exit status: 0 is "everything checked, nothing found", and nothing else.
    for how, count in lives_unknown.items():
        kinds['(not checked) %d lines of kind %s in .lives' % (count, how)] += count
    skipped = sum(n for kind, n in kinds.items() if kind.startswith('(not checked)'))
    print('%s: %d instructions of %d functions checked, %d problems%s' % (
        os.path.basename(sys.argv[2]), checked, len(rows), len(problems), ', %d checks not made' % skipped if skipped else ''))
    for p in problems[:12]:
        print('  ' + p)
    for kind, n in kinds.most_common():
        if n and kind.startswith(('(not checked)', '(schedule)', '(lives)', '(own lives)', '(relocations)', '(matrix)', '(level3)', '(borrow)', '(room)', '(slow)')):
            print('  %d %s' % (n, kind))
    if recounted:
        print('  %d functions got their chart by declaring 254 registers (nvdisasm refuses 255)' % recounted_funcs)
    return 1 if problems else 2 if skipped else 0


if __name__ == '__main__':
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    sys.exit(main())
