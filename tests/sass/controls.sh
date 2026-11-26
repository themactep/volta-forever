#!/bin/bash
# Every control of reference/controls.py, each in a process of its own: translations that are wrong
# on purpose, which reference/validate.py must object to.
#
#   tests/sass/controls.sh OUT [CUBIN...]
#
# Without CUBIN: the sm_75 cubins of tests/kernels; some of PyTorch's when they are where `TORCH`
# says (tests/sass/compare.sh), and six of the held-out corpus when it is where `HELDOUT` says
# (the directory given to tests/sass/heldout-extract.sh): the ones named below have what the
# controls take away. Five passes, because a control needs a translation that has what it takes
# away:
#   the default           every control
#   VF_BORROW=1         the controls of borrowed registers (every function borrows)
#   VF_BORROW=8         the controls of the frame, of products at level 3, and those of matrix
#                         sequences again (level 3 everywhere)
#   VF_BORROW=5         the controls of room and of the frame (room made where a kernel would
#                         need more registers than it has)
#   VF_BORROW=3         the controls of the base of the frame (the frame addressed from its
#                         top, which a register is loaded with: elsewhere these kernels have
#                         their frame under the stack pointer)
# In each pass one process validates the translation as it is, and the others take it unvalidated.
# OUT gets one file a pass and control (what controls.py printed), and `summary.tsv`: for each
# control, in how many cubins it changed the translation, in how many of those the validator
# objected, the passes where it did, its first objection, the cubins whose translation it changed
# and where the validator found nothing, and in how many the translator refused all that it
# changed (`refused` in controls.py's table, not counted as changed: nothing of it reaches the
# validator). The exit status is 1 when the translation as it is has a problem somewhere, when a
# control changes the translation of a cubin and is not objected to there (a hole in the
# validator, or a control that tests nothing there), or when the validator stopped without a
# verdict. Not in these passes: reference/controls.py's HALF_CONTROLS (F2FP and Turing zero-fill
# loads), which no cubin of the repository has; they are run by name on a cubin that has one
# (`-c`). LEPC_CONTROLS runs in these passes (k_lepc.sm75.cubin carries the pair). Apart: the
# controls of CHANGES_THE_PROGRAM in reference/controls.py (no objection is no
# hole), and the cases of RIGHT_THERE there, where what a control changed stays right, each looked
# at (reference/controls.py decides a cubin's verdict for both: unobjected()). (Until 2026-11-23 a control had to be objected to in one cubin only: the validator could
# have lost the objection in all others unnoticed.) A cubin whose translation as it is has
# problems in a pass is left out of the counts of that pass: what a control adds there cannot be
# told apart.
#   JOBS          processes (default 32); 40 minutes with 64 for the cubins named below
#   SUMMARY_ONLY  set: nothing is run, the summary is made again from the files of OUT
set -u
W=$(cd "$(dirname "$0")/../.." && pwd)
[ -f "$W/tests/site.env" ] && . "$W/tests/site.env"
: "${VF_TEST_ROOT:=${XDG_CACHE_HOME:-$HOME/.cache}/volta-forever/tests}"
TORCH=${TORCH:-$VF_TEST_ROOT/torch-cubins}
HELDOUT=${HELDOUT:-$VF_TEST_ROOT/heldout}
JOBS=${JOBS:-32}
out=${1:?usage: controls.sh OUT [CUBIN...]}
shift
mkdir -p "$out"
if [ -z "${SUMMARY_ONLY:-}" ]; then
if [ $# -eq 0 ]; then
    set -- "$W"/tests/kernels/translate/*.sm75.cubin "$W/tests/kernels/translate/triton_add.sm_75.cubin" \
        "$W"/tests/kernels/k_{tensor,pairs,groups,kept,stored}.135.sm75.cubin
    for n in 0097 0142 0150 0158 0161 0175 0176 0255 0382 0433 0434 0439 0442 0445 0448 0467 0469; do
        [ -f "$TORCH/$n.sm_75.cubin" ] && set -- "$@" "$TORCH/$n.sm_75.cubin"
    done
    # (Marks of register reuse that name a batch; relocations on code; within a kernel's own
    # count, a register that holds two uniform predicates in turn, and one that is the home of
    # two uniform registers in turn; what a store reads written under the complement of its guard;
    # the lane of a product at level 3 read into a register put away there.)
    for n in cudnn-libcudnn_engines_precompiled/0086 cuda135-libcufft/9552 cu13-libcublasLt/0066 cu13-libcusparse/0046 \
            nccl-libnccl/0129 cudnn-libcudnn_engines_precompiled/0854; do
        [ -f "$HELDOUT/$n.sm_75.cubin" ] && set -- "$@" "$HELDOUT/$n.sm_75.cubin"
    done
fi
printf '%s\n' "$@" > "$out/cubins"
python3 -I - "$W/reference" "$out/of-the-program" > "$out/todo" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import controls as c
passes = [('default', [x for x in c.CONTROLS if x != 'as it is']),
          ('1', c.BORROW_CONTROLS),
          ('8', c.FRAME_CONTROLS + c.PRODUCT_L3_CONTROLS + ['unguarded', 'lanes', 'unwaited', 'unordered', 'soon']),
          ('5', c.ROOM_CONTROLS + c.FRAME_CONTROLS),
          ('3', c.BASE_CONTROLS)]
for name, controls in passes:
    print(name, 'as-it-is')
for name, controls in passes:
    for control in controls:
        print(name, control)
open(sys.argv[2], 'w').write('\n'.join(c.CHANGES_THE_PROGRAM) + '\n')
PY
export W out
one() {
    local pass=$1 name=$2 cubins
    mapfile -t cubins < "$out/cubins"
    if [ "$pass" = default ]; then unset VF_BORROW; else export VF_BORROW=$pass; fi
    if [ "$name" = as-it-is ]; then python3 -I "$W/reference/controls.py" -c 'as it is' "${cubins[@]}"
    else python3 -I "$W/reference/controls.py" -u -c "$name" "${cubins[@]}"; fi > "$out/$pass.$name.txt" 2>&1
}
export -f one
xargs -a "$out/todo" -P "$JOBS" -L 1 bash -c 'one "$@"' _
fi
python3 -I - "$out" "$W/reference" <<'PY'
import collections, os, re, sys
out = sys.argv[1]
sys.path.insert(0, sys.argv[2])
# (A cubin's verdict, and the cases where a control changes the translation and the validator
# rightly finds nothing, each looked at: reference/controls.py, which decides the same when it is
# run by itself.)
import controls as ctl
cubins = [os.path.basename(l.strip()) for l in open(os.path.join(out, 'cubins'))]
changed, objected, failed, refused = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
where, first, names, bad = collections.defaultdict(list), {}, [], []
unobjected = collections.defaultdict(list)  # {control: ['PASS CUBIN', ...]}: changed, nothing found there, and not in RIGHT_THERE
right = []                                 # the cases of RIGHT_THERE met in this run
dirty = collections.defaultdict(set)       # {pass: the cubins whose translation as it is has problems there}
for line in open(os.path.join(out, 'todo')):
    p, name = line.split()
    text = open(os.path.join(out, '%s.%s.txt' % (p, name)), errors='replace').read()
    row = 'as it is' if name == 'as-it-is' else name
    m = re.search(r'^%s\s+(.*?)\s{3}(as expected|changes nothing|NOT AS EXPECTED)' % re.escape(row), text, re.M)
    if name == 'as-it-is':
        if m is None:
            bad.append('the translation as it is, pass %s: not validated (%s)' % (p, (text.strip().splitlines() or ['nothing'])[-1][:120]))
        elif any(c != '0' for c in m.group(1).split()):
            dirty[p] = {i for i, c in enumerate(m.group(1).split()) if c != '0'}
            bad.append('the translation as it is has problems in pass %s: %s' % (
                p, ', '.join('%s %s' % (cubins[i], c) for i, c in enumerate(m.group(1).split()) if c != '0')))
        continue
    if name not in names:
        names.append(name)
    if m is None:
        if 'every control is objected to' not in text and 'NOT EVERY CONTROL' not in text:
            bad.append('%s in pass %s: did not run (%s)' % (name, p, (text.strip().splitlines() or ['nothing'])[-1][:120]))
        continue                          # not run with this setting of the translator
    clean = [i for i in range(len(m.group(1).split())) if i not in dirty[p]]
    holes, rightly = ctl.unobjected(name, p, [cubins[i] for i in clean], [m.group(1).split()[i] for i in clean])
    right += [(p, name, cubin) for cubin in rightly]
    unobjected[name] += ['%s %s' % (p, cubin) for cubin in holes]
    counts = [x for i, x in enumerate(m.group(1).split()) if i not in dirty[p]]
    c = sum(1 for x in counts if x not in ('=', 'refused'))
    o = sum(1 for x in counts if x not in ('=', 'refused', '0', 'failed'))
    failed[name] += counts.count('failed')
    refused[name] += counts.count('refused')
    changed[name] += c
    objected[name] += o
    if o:
        where[name].append('%s (%d of %d)' % (p, o, c))
        e = re.search(r'^\s+e\.g\. (.*)$', text, re.M)
        if e and name not in first:
            first[name] = re.sub(r'^\S+ ', '', e.group(1))[:110]
with open(os.path.join(out, 'summary.tsv'), 'w') as f:
    f.write('control\tcubins changed\tof which objected to\tpasses\tfirst objection\tchanged and not objected to\t'
            'refused by the translator\n')
    for name in names:
        f.write('%s\t%d\t%d\t%s\t%s\t%s\t%d\n' % (name, changed[name], objected[name], ', '.join(where[name]) or '-', first.get(name, ''),
                                                  ', '.join(['%s %s (RIGHT_THERE)' % (p, c) for p, n, c in right if n == name]
                                                            + unobjected[name]) or '-', refused[name]))
# (A control that changes the program, not the translation: no objection is no hole.)
of_the_program = open(os.path.join(out, 'of-the-program')).read().split()
# (Not counted as changed there: the cases of RIGHT_THERE. Until 2026-11-23 a control met only in
# them was taken for one objected to nowhere, and the run failed on what it accepted.)
exempt = collections.Counter(n for _, n, _ in right)
silent = [n for n in names if changed[n] - exempt[n] and not objected[n] and n not in of_the_program]
# (A control makes the translation wrong: it is objected to wherever it changes it, but in RIGHT_THERE.)
missed = [n for n in names if unobjected[n] and n not in of_the_program and n not in silent]
idle = [n for n in names if not changed[n]]
stopped = [n for n in names if failed[n]]
print('%d controls on %d cubins: %d objected to; %d change translations and are objected to nowhere: %s; %d change nothing in these cubins: %s'
      % (len(names), len(cubins), sum(1 for n in names if objected[n]), len(silent), ' '.join(silent) or '-', len(idle), ' '.join(idle) or '-'))
print('%d are not objected to in a cubin whose translation they change: %s' % (
    len(missed), '; '.join('%s (%s)' % (n, ', '.join(unobjected[n])) for n in missed) or '-'))
gone = sorted(set(ctl.RIGHT_THERE) - set(right))
print('(%d cases where a control changes the translation and the validator rightly finds nothing, of RIGHT_THERE; %d of its cases not met%s)'
      % (len(right), len(gone), ': ' + ', '.join(' '.join(case) for case in gone) if gone else ''))
if stopped:
    print('the validator stopped without a verdict on translations of: %s' % ' '.join(stopped))
for b in bad:
    print(b)
sys.exit(1 if silent or missed or bad or stopped else 0)
PY
