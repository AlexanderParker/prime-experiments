-- Round 30: the CaseCert37 root is back (built TIERED, 35 sub-roots of 11 cases; see
-- lakefile.toml at its lib entry and formalist.md R30).
import CaseCert37
import WordLegal
import WordLegal11
import MachineUp
import CrtSlots
import Increment
import AnchorChain
import AnchorRecord17
import AlternationOrder
import MirrorM11
import CaseCert31
import CaseCert23
import CaseSplit
import Machine23Q
import Machine23Scan
import Machine29
import CoveringCert
import PotentialLadder
import CoveringCert2
import BlockedSlots
import Horizon
import Layer
import Supply
import Census
import Bridge
import Gear
import Placement
import Corridor
import Machine13
import MaxGap
import LiteralCap
import Machine17
import TierA
import PolignacCap
import Spectrum
import LiteralCapTable
import Machine19
import MergeLaw
import TwoTeeth
import Machine19Q
import Machine23
import Machine11
import Machine13Q
import Machine17Q
import Ladder
import DepthSum
import Potential
import Potential19
import Machine31
import Gen11
import Machine37
import Periodic
import Machine11Per
import Machine13Per
import Gen11Sound
import Machine29Cen
import Machine31Cen
import LadderPeriod
import Mirror
open BlockedSlots
#print axioms BlockedSlots.twins_infinite_iff_survivor_in_window
#print axioms BlockedSlots.survivor_in_window_of_gap_bound
#print axioms BlockedSlots.nextGap_spec
#print axioms BlockedSlots.twin_of_centreSurvivor
#print axioms BlockedSlots.covering_of_not_infinite
#print axioms BlockedSlots.survivor_step
#print axioms BlockedSlots.card_blocked_by_le
#print axioms Horizon.exists_prime_factor_lt
#print axioms Horizon.prime_of_no_prime_factor_lt
#print axioms Horizon.twin_of_no_prime_factor_lt
#print axioms Layer.slot_cap
#print axioms Layer.minFac_lt_or_eq
#print axioms Layer.eq_mul_prime_of_minFac_eq
#print axioms Layer.layer_novelty
#print axioms Supply.minFac_mem_gears
#print axioms Supply.card_composites_eq_sum_roots
#print axioms Supply.card_eq_primes_add_sum_roots
#print axioms Supply.roots_ne
#print axioms Census.census_partition
#print axioms Census.comps_eq
#print axioms Census.primes_add_comps
#print axioms Census.primes_eq
#print axioms Census.census_pinned
#print axioms Census.census_pinned_prefix
#print axioms Census.n0_eq_zero_iff
#print axioms Bridge.card_members
#print axioms Bridge.card_comps_members
#print axioms Bridge.card_primes_members
#print axioms Bridge.sum_roots_eq_census
#print axioms Bridge.sum_roots_pinned
#print axioms Bridge.slot_roots_ne
#print axioms Gear.supply_eq_sum_R
#print axioms Gear.sum_R_eq_census
#print axioms Gear.R_le_card_multiples
#print axioms Gear.R_prefix_le
#print axioms Gear.sq_le_of_minFac_eq
#print axioms Gear.R_eq_zero_of_below_sq
#print axioms Gear.semiprime_of_fiber
#print axioms Gear.R_eq_card_partners
#print axioms Gear.mem_partners
#print axioms Placement.prime_mod_six
#print axioms Placement.sign_law
#print axioms Placement.mem_members_iff_slot
#print axioms Placement.slot_injOn_partners
#print axioms Placement.card_slots_of_line
#print axioms Placement.R_slots_eq
#print axioms Corridor.exists_class_in_run
#print axioms Corridor.both_composite_of_class
#print axioms Corridor.both_composite_in_run
#print axioms Corridor.double_slot_in_run
#print axioms Corridor.prime_adjacent_run_le
#print axioms Corridor.product_slotOf
#print axioms Corridor.twin_product_pin
#print axioms Corridor.exposed_iff_mem
#print axioms Corridor.endpoint_law
#print axioms Corridor.endpoint_law_34
#print axioms Corridor.adjacency_law
#print axioms Corridor.no_chain_of_forbidden
#print axioms Corridor.forbidden_first_examples
#print axioms Corridor.forbidden_pairs_count
#print axioms Corridor.n2_packing
#print axioms Machine13.w11
#print axioms Machine13.w16
#print axioms Machine13.exposed13_iff
#print axioms Machine13.gap_le
#print axioms Machine13.pair_sum_le
#print axioms Machine13.gap11_realized
#print axioms Machine13.pair16_realized
#print axioms Machine13.alpha1_certificate
#print axioms Machine13.lemma1_at_13
#print axioms Machine13.tierA_forbidden
#print axioms Machine13.no_11_11_chain
#print axioms MaxGap.uncovered_span_mod_three
#print axioms MaxGap.F_zero_mod_three
#print axioms MaxGap.M_two_mod_three
#print axioms MaxGap.not_max_of_mod_three
#print axioms LiteralCap.no_run_seven
#print axioms LiteralCap.s_eq
#print axioms LiteralCap.literal_chain_le_six
#print axioms LiteralCap.cap_six_classes_sharp
#print axioms Machine17.w18All
#print axioms Machine17.w25All
#print axioms Machine17.exposed17_iff
#print axioms Machine17.gap_le
#print axioms Machine17.pair_sum_le
#print axioms Machine17.alpha1_certificate
#print axioms Machine17.lemma1_at_17
#print axioms TierA.mem_carrier_of_chain
#print axioms TierA.no_chain_of_carrier_empty
#print axioms TierA.no_maximal_flanks
#print axioms TierA.flanks_17_19
#print axioms TierA.flanks_19_23_nonempty
#print axioms TierA.padding_count_le
#print axioms TierA.padding_at_most_one_below_onset
#print axioms TierA.onset_gate
#print axioms TierA.padding_three_not_excluded
#print axioms TierA.no_adjacent_padded_41
#print axioms TierA.equal_padding_forbidden_classes
#print axioms TierA.equal_padding_forbidden_card
#print axioms TierA.padding_shape_dichotomy
#print axioms TierA.no_adjacent_equal_padded
#print axioms PolignacCap.exists_mul_mod_eq
#print axioms PolignacCap.cap_gcd_1
#print axioms PolignacCap.cap_gcd_3
#print axioms PolignacCap.cap_gcd_15
#print axioms PolignacCap.cap_gcd_105
#print axioms PolignacCap.capOf_le_twelve
#print axioms PolignacCap.cap_gcd_5
#print axioms PolignacCap.cap_gcd_7
#print axioms PolignacCap.cap_gcd_21
#print axioms PolignacCap.cap_gcd_35
#print axioms Spectrum.merged_eq
#print axioms Spectrum.merged_le_spectrum
#print axioms Spectrum.merged_le_spectrum_succ
#print axioms Spectrum.merged_le_of_shallow
#print axioms Spectrum.windowSum_mono
#print axioms Spectrum.qual_le_of_suppressed
#print axioms Spectrum.merged_le_of_suppressed
#print axioms Spectrum.qualifying_of_word
#print axioms Spectrum.merged_le_qual
#print axioms Spectrum.merged_le_of_qual_flat
#print axioms Spectrum.merged_le_of_qual_flat_all
#print axioms Spectrum.merged_le_of_corrected
#print axioms Spectrum.alphabet_ge_floor
#print axioms Spectrum.padded_ge_floor
#print axioms Spectrum.jointCount_antitone
#print axioms LiteralCapTable.cap_table_maximal
#print axioms LiteralCapTable.cap_table_realized
#print axioms LiteralCapTable.literal_chain_le_capC
#print axioms LiteralCapTable.word_length_lt_capC
#print axioms LiteralCapTable.hasRunL_mono
#print axioms LiteralCapTable.capC_le_six
#print axioms LiteralCapTable.cap_two_classes
#print axioms LiteralCapTable.cap_three_classes
#print axioms LiteralCapTable.cap_four_classes
#print axioms LiteralCapTable.cap_six_classes
#print axioms LiteralCapTable.no_cap_five
#print axioms LiteralCapTable.cap_spectrum_counts
#print axioms LiteralCapTable.tripled_teeth_antipode
#print axioms Machine19.sliceAll
#print axioms Machine19.gap_le
#print axioms Machine19.pair_sum_le
#print axioms Machine19.quad_sum_le
#print axioms Machine19.alpha1_certificate
#print axioms Machine19.lemma1_at_19
#print axioms Machine19.shallow_flatness
#print axioms Machine19.exists_exposed_above
#print axioms Machine19.spectrum_four
#print axioms Machine19.spectrum_four_flat
#print axioms Machine19.D_of_shallow_word
#print axioms MergeLaw.sub_mod_eq
#print axioms MergeLaw.interior_gap_mod
#print axioms MergeLaw.floor_of_mod
#print axioms MergeLaw.newgap_le
#print axioms MergeLaw.newgap_le_max
#print axioms MergeLaw.D_of_qualmax
#print axioms TwoTeeth.next_kill_of_lo
#print axioms TwoTeeth.next_kill_of_hi
#print axioms TwoTeeth.kill_spacing
#print axioms TwoTeeth.kill_spacing_min
#print axioms TwoTeeth.kill_period
#print axioms TwoTeeth.kill_spacing_gear
#print axioms TwoTeeth.kill_spacing_min_gear
#print axioms TwoTeeth.teeth_letters
#print axioms TwoTeeth.spacing_from_lo
#print axioms TwoTeeth.spacing_from_hi
#print axioms TwoTeeth.kills_gap_ge
#print axioms TwoTeeth.fuel_span_cap
#print axioms TwoTeeth.fuel_le
#print axioms Machine19.qsliceAll
#print axioms Machine19.chain_facts
#print axioms Machine19.no_big_run
#print axioms Machine19.spectrum_one
#print axioms Machine19.spectrum_two
#print axioms Machine19.spectrum_three
#print axioms Machine19.spectrum_five
#print axioms Machine19.spectrum_ladder
#print axioms Machine19.qual_bound_all
#print axioms Machine19.qual_five_flat
#print axioms Machine19.D_of_word
#print axioms Machine19.opSeq_surj
#print axioms Machine23.killed23_iff
#print axioms Machine23.merge_alphabet
#print axioms Machine23.g23_le
#print axioms Machine23.D_at_19_23

-- round 22: the (D) ladder
#print axioms Machine11.qasm
#print axioms Machine11.chain_facts
#print axioms Machine11.spectrum_ladder
#print axioms Machine11.qual_bound_all
#print axioms Machine11.opSeq_surj
#print axioms Machine13.qasm
#print axioms Machine13.chain_facts
#print axioms Machine13.spectrum_ladder
#print axioms Machine13.qual_bound_all
#print axioms Machine13.opSeq_surj
#print axioms Machine17.qsliceAll
#print axioms Machine17.chain_facts
#print axioms Machine17.spectrum_ladder
#print axioms Machine17.qual_bound_all
#print axioms Machine17.opSeq_surj
#print axioms MergeLaw.pos_le_add
#print axioms MergeLaw.windowSum_telescope
#print axioms MergeLaw.newgap_le_step
#print axioms Ladder.g13_le
#print axioms Ladder.g17_le
#print axioms Ladder.g19_le_of_17
#print axioms Ladder.D_at_11_13
#print axioms Ladder.D_at_13_17
#print axioms Ladder.D_at_17_19
#print axioms Ladder.D_ladder
#print axioms Ladder.D_at_23_29
#print axioms Ladder.D_at_37_41
#print axioms Ladder.criterion_arith
#print axioms DepthSum.window_depth_unique
#print axioms DepthSum.depth_partition
#print axioms DepthSum.mem_reachSet
#print axioms DepthSum.local_factor_5
#print axioms DepthSum.local_factor_13
#print axioms DepthSum.depth_sum_at_13
#print axioms DepthSum.depth_sum_hl_form
#print axioms Potential.chain_le_potential
#print axioms Potential.D_of_potential
#print axioms Potential.windowSum_succ_left
#print axioms Potential.tail_le_potential
#print axioms Potential.merged_le_of_potential
#print axioms Potential19.h19_C1
#print axioms Potential19.h19_C2
#print axioms Potential19.h19_C3
#print axioms Potential19.D_of_word_potential

-- round 23

#print axioms Machine23.opSeq23_surj
#print axioms Machine23.opSeq23_strict_mono
#print axioms Machine23.windowSum_g23
#print axioms Machine29.killed29_iff
#print axioms Machine29.merge_alphabet
#print axioms Machine29.D_at_23_29
#print axioms Machine29.g29_le
#print axioms CoveringCert.tot_eq
#print axioms CoveringCert.S5_le
#print axioms CoveringCert.S19_le
#print axioms CoveringCert.P7_ge
#print axioms CoveringCert.P19_ge
#print axioms CoveringCert.cert_signs
#print axioms CoveringCert.kounias
#print axioms CoveringCert.cover_bound
#print axioms CoveringCert.no_cover
#print axioms CoveringCert.no_37_run
#print axioms CoveringCert.F19_le_37
#print axioms CoveringCert.D_17_19_lp
#print axioms PotentialLadder.h11_C3
#print axioms PotentialLadder.h13_C3
#print axioms PotentialLadder.h17_C3
#print axioms PotentialLadder.D_of_word_11
#print axioms PotentialLadder.D_of_word_13
#print axioms PotentialLadder.D_of_word_17
#print axioms PotentialLadder.potential_ladder
#print axioms CoveringCert2.T13_eq
#print axioms CoveringCert2.cert13
#print axioms CoveringCert2.kounias4
#print axioms CoveringCert2.cover13
#print axioms CoveringCert2.F13_le_20
#print axioms CoveringCert2.D_11_13_lp
#print axioms CoveringCert2.cert17
#print axioms CoveringCert2.F17_le_28
#print axioms CoveringCert2.D_13_17_lp
#print axioms CoveringCert2.lp_ladder

-- Round 24: the position-indexed machine-23 scan and the fifth rung
#print axioms Machine23.qsliceIdxAll
#print axioms Machine23.next23_step
#print axioms Machine23.chain_facts23
#print axioms Machine23.spectrum23_one
#print axioms Machine23.spectrum23_two
#print axioms Machine23.qual23_all
#print axioms Machine23.D_23_29
#print axioms Machine23.g29_le_60

-- Round 25: the sixth rung by the DICTIONARY vehicle, and the generator
#print axioms Machine29.D2_ok
#print axioms Machine29.D3_ok
#print axioms Machine29.D4_ok
#print axioms Machine29.D5_ok
#print axioms Machine29.D6_ok
#print axioms Machine29.D7_ok
#print axioms Machine29.opSeq29_surj
#print axioms Machine29.spectrum29_two
#print axioms Machine29.qual29_all
#print axioms Machine29.criterion_29_31
#print axioms Machine31.killed31_iff
#print axioms Machine31.merge_alphabet
#print axioms Machine31.D_at_29_31
#print axioms Machine31.D_29_31
#print axioms Machine31.g31_le_of_census
#print axioms Gen11.gw11_sum
#print axioms Gen11.no_truncation
#print axioms Gen11.gen_zero
#print axioms Gen11.gen_one
#print axioms Gen11.generator_matches_machine13

-- Round 25: the SEVENTH rung, by the same dictionary vehicle one gear up
#print axioms Machine31.D2_ok
#print axioms Machine31.D4_ok
#print axioms Machine31.D5_ok
#print axioms Machine31.D7_ok
#print axioms Machine31.opSeq31_surj
#print axioms Machine31.spectrum31_two
#print axioms Machine31.qual31_five
#print axioms Machine31.qual31_all
#print axioms Machine31.criterion_31_37
#print axioms Machine37.killed37_iff
#print axioms Machine37.merge_alphabet
#print axioms Machine37.D_at_31_37
#print axioms Machine37.D_31_37
#print axioms Machine37.g37_le_of_census

-- Round 26: THE PERIODIC-ENUMERATION LEMMA and the two gaps it closes
#print axioms Periodic.next_shift
#print axioms Periodic.op_shift
#print axioms Periodic.index_reduce
#print axioms Machine11.exposed11_period
#print axioms Machine11.ow_135
#print axioms Machine11.opSeq_shift
#print axioms Machine11.g11_mod
#print axioms Machine13.ow13_1485
#print axioms Machine13.opSeq_shift
#print axioms Machine13.g13_shift

-- Round 26: THE GENERATOR'S SOUNDNESS BRIDGE at 11 -> 13
#print axioms Gen11.word_check
#print axioms Gen11.gAt_succ
#print axioms Gen11.walk_sound
#print axioms Gen11.gen_two
#print axioms Gen11.gen_three
#print axioms Gen11.spectrum_of_gen
#print axioms Gen11.generator_sound

-- Round 26: the census hypothesis shrunk to ONE PERIOD
#print axioms Machine29.exposed29_period
#print axioms Machine29.index_reduce29
#print axioms Machine29.census29_of_period
#print axioms Machine31.exposed31_period
#print axioms Machine31.index_reduce31
#print axioms Machine31.census31_of_period
#print axioms LadderPeriod.D_29_31_period
#print axioms LadderPeriod.D_31_37_period

-- Round 26: the mirror (Lateral's parity laws, the arithmetic halves)
#print axioms Mirror.mirror_gear
#print axioms Mirror.mirror_exposed11
#print axioms Mirror.mirror_exposed29
#print axioms Mirror.antipode_open
#print axioms Mirror.antipode_exposed11
#print axioms Mirror.antipode_exposed29
#print axioms Mirror.self_mirror_unique

-- Round 27: the case-split LP certificates (the 19->23 rung, hypothesis-free)
#print axioms CaseSplit.le_mxr
#print axioms CaseSplit.le_mxr2
#print axioms CaseSplit.lowest6
#print axioms CaseSplit.lowest7
#print axioms CaseSplit.degpos6
#print axioms CaseSplit.ind_low2
#print axioms CaseCert23.cert0
#print axioms CaseCert23.cert1
#print axioms CaseCert23.cert2
#print axioms CaseCert23.cert3
#print axioms CaseCert23.cert4
#print axioms CaseCert23.nocov0
#print axioms CaseCert23.nocov4
#print axioms CaseCert23.blocked
#print axioms CaseCert23.no_run
#print axioms CaseCert23.F_le
#print axioms CaseCert23.D_19_23_case

-- Round 27: the counting half of the mirror lever
#print axioms Mirror.even_card_involution
#print axioms Mirror.window_count_even
#print axioms Mirror.adjacent_equal_even
#print axioms Mirror.none_of_at_most_one
#print axioms CaseCert31.cert0
#print axioms CaseCert31.cert34
#print axioms CaseCert31.nocov0
#print axioms CaseCert31.blocked
#print axioms CaseCert31.no_run
#print axioms CaseCert31.F_le
#print axioms CaseCert31.D_29_31_case

-- Round 28: the mirror lever instantiated at machine 11
#print axioms Machine11.opSeq_133
#print axioms Machine11.opSeq_mirror
#print axioms Machine11.g11_mirror
#print axioms Machine11.mir2_invol
#print axioms Machine11.L2_mirror
#print axioms Machine11.L2_133
#print axioms Machine11.window2_even
#print axioms Machine11.adjacent_max_none_of_at_most_one
#print axioms Machine11.adjacent_max_none
#print axioms CaseSplit.lowest5
#print axioms CaseSplit.degpos5

-- Round 28: the increment law at the six literal steps
#print axioms IncCert23.cert0
#print axioms IncCert23.nocov0
#print axioms IncCert23.blocked
#print axioms IncCert23.no_run
#print axioms IncCert23.F_le
#print axioms IncCert23.inc_19_23
#print axioms IncCert29.cert0
#print axioms IncCert29.nocov0
#print axioms IncCert29.F_le
#print axioms IncCert29.inc_23_29
#print axioms IncCert31.cert0
#print axioms IncCert31.nocov34
#print axioms IncCert31.F_le
#print axioms IncCert31.inc_29_31
#print axioms Increment.f2_11
#print axioms Increment.f2_13
#print axioms Increment.f2_17
#print axioms Increment.f2_19
#print axioms Increment.f2_23
#print axioms Increment.f2_29
#print axioms Increment.increment_11_13
#print axioms Increment.increment_13_17
#print axioms Increment.increment_17_19
#print axioms Increment.increment_19_23
#print axioms Increment.increment_23_29
#print axioms Increment.increment_29_31
#print axioms Increment.increment_law_literal_steps

#print axioms AnchorChain.teeth_eq_phase
#print axioms AnchorChain.chain_law
#print axioms AnchorChain.copy_phase
#print axioms AnchorChain.phase_bijective
#print axioms AnchorChain.no_two_up
#print axioms AnchorChain.no_two_down
#print axioms AnchorChain.neighbour_of_hit
#print axioms AnchorChain.hop_zero
#print axioms AnchorChain.hop_iter
#print axioms AnchorChain.hop_one
#print axioms AnchorRecord17.mg2
#print axioms AnchorRecord17.record_max
#print axioms AnchorRecord17.shift_res
#print axioms AnchorRecord17.surv_shift
#print axioms AnchorRecord17.openT17_iff
#print axioms AnchorRecord17.phase_is_machine
#print axioms AnchorRecord17.gap18_realized
#print axioms AnchorRecord17.F17_eq_18
#print axioms AlternationOrder.surv_downward
#print axioms AlternationOrder.ps_min_le_five
#print axioms AlternationOrder.ps_min_five_iff
#print axioms AlternationOrder.ps_min_four_iff
#print axioms AlternationOrder.ps_min_counts
#print axioms AlternationOrder.ps_max_eq_capC
#print axioms AlternationOrder.ps_max_le_six
#print axioms AlternationOrder.arelax_le_five
#print axioms AlternationOrder.arelax_le_four

-- Round 30: the 31->37 case-split root, assembled through 35 tiers
#print axioms CaseCert37.nocov0
#print axioms CaseCert37.nocov384
#print axioms CaseCert37.nopair0
#print axioms CaseCert37.nopair34
#print axioms CaseCert37.blocked
#print axioms CaseCert37.no_run
#print axioms CaseCert37.F_le
#print axioms CaseCert37.D_31_37_case

-- Round 30: R89 (word reduction) and R90 (same-tooth lemma), abstract and at machine 11
#print axioms WordLegal.legal_iff_noRepeat
#print axioms WordLegal.alt_iff_prefixSum
#print axioms WordLegal.killable_iff
#print axioms WordLegal.same_tooth
#print axioms WordLegal.two_mul_ne_zero
#print axioms WordLegal.val_injective
#print axioms WordLegal.chain_iff_word
#print axioms WordLegal.qstar_iff_word
#print axioms WordLegal.jmax
#print axioms WordLegal.akill
#print axioms WordLegal.middle_span
#print axioms WordLegal.same_tooth_window
#print axioms WordLegal.literal_even_span
#print axioms WordLegal11.L11
#print axioms WordLegal11.jmax11
#print axioms WordLegal11.akill11

-- Round 30: the CRT slots (F_2 lower halves at 37, 41, 53, 59)
#print axioms MachineUp.exposed59_iff
#print axioms CrtSlots.f2_37
#print axioms CrtSlots.five_37
#print axioms CrtSlots.f2_41
#print axioms CrtSlots.five_41
#print axioms CrtSlots.f2_53
#print axioms CrtSlots.five_53
#print axioms CrtSlots.f2_59_A
#print axioms CrtSlots.five_59_A
#print axioms CrtSlots.f2_59_B
#print axioms CrtSlots.five_59_B
#print axioms CrtSlots.mirror_59
