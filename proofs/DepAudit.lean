/-
DEPENDENCY AUDIT (round 26) - a gate, not a claim.

`Gen11Sound.generator_sound` bounds machine 13's spectrum from machine 11's
word.  The whole point is that machine 13's OWN 5,005-slot period scan
(`Machine13.qasm`, and everything unpacked from it) is not used.  That is a
statement about the PROOF TERM, which `#print axioms` cannot see, so this
file computes the transitive constant closure of the theorem and asserts the
forbidden names are absent.

Run it (it is deliberately NOT a `defaultTarget`; it is an audit, like
`AxiomCheck.lean`):

    lake env lean DepAudit.lean

It prints the closure size and the verdict, and FAILS the elaboration if any
forbidden constant is reachable.
-/

import Gen11Sound

open Lean Elab Command in
run_cmd do
  let env ← getEnv
  let target : Name := `Gen11.generator_sound
  -- everything that machine 13's own period scan produces
  let forbidden : List Name :=
    [`Machine13.qasm, `Machine13.qslice, `Machine13.qokAll, `Machine13.chainT,
     `Machine13.chain_facts, `Machine13.spectrum_one, `Machine13.spectrum_two,
     `Machine13.spectrum_three, `Machine13.spectrum_four,
     `Machine13.spectrum_ladder, `Machine13.no_big_run, `Machine13.w11,
     `Machine13.w16, `Machine13.nextOp_le_11, `Machine13.seek_next]
  let mut visited : NameSet := {}
  let mut stack : List Name := [target]
  while !stack.isEmpty do
    match stack with
    | [] => pure ()
    | n :: rest =>
      stack := rest
      if !visited.contains n then
        visited := visited.insert n
        if let some ci := env.find? n then
          let mut ns := ci.type.getUsedConstants
          -- NOTE (round-26 infrastructure fact): `ConstantInfo.value?` does NOT
          -- return a THEOREM's proof term in this toolchain, so an audit written
          -- with it passes vacuously.  Match on `thmInfo` explicitly.
          match ci with
          | .thmInfo v => ns := ns ++ v.value.getUsedConstants
          | .defnInfo v => ns := ns ++ v.value.getUsedConstants
          | .opaqueInfo v => ns := ns ++ v.value.getUsedConstants
          | _ => pure ()
          for m in ns do
            if !visited.contains m then
              stack := m :: stack
  -- POSITIVE CONTROLS: the traversal must find what the proof really does use.
  -- Without these the audit could pass vacuously on a broken walk.
  let required : List Name :=
    [`Machine11.qasm, `Machine11.opSeq_surj, `Machine11.ow_135,
     `Machine11.opSeq_shift, `Gen11.gen_zero, `Gen11.word_check,
     `Gen11.walk_sound, `Periodic.op_shift, `Periodic.next_shift,
     `Machine13.opSeq, `Machine13.exists_exposed_above]
  let missing := required.filter fun f => !visited.contains f
  let hits := forbidden.filter fun f => visited.contains f
  if !missing.isEmpty then
    throwError m!"DEP AUDIT BROKEN: positive controls not reached: {missing}"
  else if hits.isEmpty then
    logInfo m!"DEP AUDIT GREEN: {target} closes over {visited.toList.length} constants; \
all {required.length} positive controls reached; none of the {forbidden.length} \
machine-13-period constants is among them."
  else
    throwError m!"DEP AUDIT RED: {target} depends on {hits}"
